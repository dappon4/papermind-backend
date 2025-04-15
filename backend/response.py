import openai
from google import genai
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
from utils import get_embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

valid_labels = ["title", "abstract", "method", "results", "conclusion", "year", "author"]

class ExtractedAbstract(BaseModel):
    is_abstract_present: bool
    abstract: str

class Labels(BaseModel):
    class Label(BaseModel):
        label: Literal["title", "abstract", "method", "results", "conclusion", "year", "author"]
    
    labels: List[Label]

class Suggestions(BaseModel):
    suggestion: List[str]

def extract_abstract(text, engine):
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": "You are a text extractor. Extract the abstract from the user provided test, and summarize it and set is_abstract_presetn to True. If the abstract is not complete, lacking cruciala information to be compared to other papers, set is_abstract_present to False."},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format=ExtractedAbstract
    )
    message = completion.choices[0].message
    return message.parsed

def extract_abstract_gemini(text):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    """Extracts the abstract from the given text."""
    
    prompt = f"""
    Extract the abstract from the user provided text.
    text: {text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config={"response_mime_type": "application/json",
                "response_schema": ExtractedAbstract,
                }
    )
    
    response_object = response.parsed
    
    # print(type(response_object.abstract))
    # print(response_object.abstract)
    
    return response_object.abstract

def select_labels(text, engine):
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": "You are a label suggestion system. Suggest labels of database so that subsequent sentence completion models can retrieve those columns from the database and complete the sentence."},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format=Labels
    )
    message = completion.choices[0].message
    labels = message.parsed.labels
    labels = [label.label for label in labels if label.label in valid_labels]
    labels.extend(["title", "author", "year"])  # Always include title and author and year
    labels = list(set(labels))  # Remove duplicates
    print(labels)
    return labels

def retrieve_summary(abstract, labels, df, vectors):
    user_embedding = get_embeddings(abstract)
    # Calculate cosine similarities
    similarities = cosine_similarity([user_embedding], vectors)[0]

    # Get the indices of the top 5 most similar vectors
    top_indices = np.argsort(similarities)[-5:]

    retrieved_df = df.iloc[top_indices][labels]
    
    return retrieved_df

def generate_response(text, engine):
    prompt = f"""
    You are a text generator. Continue the sentence, or suggest next sentence if the sentence is already complete. The suggested text will be added to the end of the sentence.
    When mentioning other papers, use MLA format to cite them.
    You should only suggest 1 sentence.
    Give at least 3 and at most 5 suggestions.
    text: {text}
    """
    
    if text[-1] != ".":
        prompt += f"continue after this word: {text.split()[-1]}"
    
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format=Suggestions
    )
    print(completion)
    
    message = completion.choices[0].message
    if len(message.parsed.suggestion) > 5:
        suggestions = message.parsed.suggestion[:5]
    else:
        suggestions = message.parsed.suggestion
        
    return suggestions

def generate_response_with_abstract(retrieved_df, text, engine):
    
    prompt = f"""
    You are a text generator. You are given a table with the following columns: {retrieved_df.columns}.
    Use those information to continue the sentence. The suggested text will be added to the end of the text.
    You should only suggest 1 sentence.
    Give at least 3 and at most 5 suggestions.
    
    table: {retrieved_df.to_dict(orient='records')}
    
    text: {text}
    """
    
    if text[-1] != ".":
        prompt += f"continue after this word: {text.split()[-1]}"
    
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format=Suggestions
    )
    print(completion)
    message = completion.choices[0].message
    if len(message.parsed.suggestion) > 5:
        suggestions = message.parsed.suggestion[:5]
    else:
        suggestions = message.parsed.suggestion
    
    return suggestions

def generate_completion(df, vectors, text, engine="gpt-4o-mini-2024-07-18"):
    
    labels = select_labels(text, engine)
    
    extracted_abstract = extract_abstract(text, engine)
    if extracted_abstract.is_abstract_present:
        abstract = extracted_abstract.abstract
        retrieved_df = retrieve_summary(abstract, labels, df, vectors)
        suggestions = generate_response_with_abstract(retrieved_df, text, engine)
    else:
        suggestions = generate_response(text, engine)
    
    probabilities = [random.random() for _ in range(len(suggestions))]
    engines = [engine for _ in range(len(suggestions))]
    
    return list(zip(suggestions, probabilities, engines))

if __name__ == "__main__":
    text = """
    Abstract: This is a sample abstract.
    This is the introduction to the paper. It provides background information and context. According to Scientist at Meta, the result are comparable to those of OpenAI.
    """
    
    # message = extract_abstract(text)
    # print(message.parsed.is_abstract_present)
    # print(message.parsed.abstract)
    
    from preprocess import extract_info
    search_folder = "../contents/files/"
    contents_path = "../contents/"
    # need to account for removed files
    df, vectors = extract_info(search_folder, contents_path)
    
    print(generate_completion(df, vectors, text))