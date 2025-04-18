from llama_index.core import StorageContext, load_index_from_storage
from openai import OpenAI
import os
from dotenv import load_dotenv
import random

from pydantic import BaseModel, Field
from typing import Literal, List

class Suggestions(BaseModel):
    suggestion: List[str]

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_abstract_summary(text, engine):
    abstract = text.split("\n\n")[0]
    response = client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": f"Summarize the following abstract"},
            {"role": "user", "content": abstract}
        ],
        temperature=0,
    )
    
    abstract_summary = response.choices[0].message.content
    
    return abstract_summary

def generate_response(paragraph, abstract, engine, max_suggestions, retriever):
    
    nodes = retriever.retrieve(abstract + " " + paragraph)
    context = [f"{node.metadata}: " + node.get_content() + "\n\n" for node in nodes]
    
    prompt = f"""
    You are a text generator. Continue the sentence, or suggest next sentence if the sentence is already complete. The suggested text will be added to the end of the sentence.
    When mentioning other papers, use MLA format to cite them.
    
    When making suggestions, use these retrieved documents as context:
    {context}
    
    You should only suggest 1 sentence.
    Give {max_suggestions} suggestions.
    """
    
    message = f"""
    {paragraph}
    """
    
    if paragraph[-1] != ".":
        prompt += f"continue after this word: {message.split()[-1]}"
    
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ],
        temperature=0.2,
        response_format=Suggestions
    )
    print(completion)
    
    message = completion.choices[0].message
    if len(message.parsed.suggestion) > max_suggestions:
        suggestions = message.parsed.suggestion[:max_suggestions]
    else:
        suggestions = message.parsed.suggestion
        
    return suggestions

def generate_completion_rag(text, max_suggestions, retriever, engine = "gpt-4o-mini-2024-07-18"):
    """
    Get the response from the RAG model.
    """
    
    # Get the abstract summary
    abstract_summary = get_abstract_summary(text, engine)
    
    # Generate the response
    current_paragraph = text.split("\n\n")[-1]
    suggestions = generate_response(current_paragraph, abstract_summary, engine, max_suggestions, retriever)
    
    # probabilities = [random.random() for _ in range(len(suggestions))]
    # engines = [engine for _ in range(len(suggestions))]
    
    # return list(zip(suggestions, probabilities, engines))
    
    return suggestions

def generate_completion_zero_shot(text, max_suggestions, engine = "gpt-4o-mini-2024-07-18"):
    """
    Get the response from the zero-shot model.
    """
    
    prompt = f"""
    You are a text generator. Continue the sentence, or suggest next sentence if the sentence is already complete. The suggested text will be added to the end of the sentence.
    
    You should only suggest 1 sentence.
    Give {max_suggestions} suggestions.
    """
    
    message = f"""
    {text}
    """
    
    if text[-1] != ".":
        prompt += f"continue after this word: {message.split()[-1]}"
    
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ],
        temperature=0.2,
        response_format=Suggestions
    )
    
    message = completion.choices[0].message
    if len(message.parsed.suggestion) > max_suggestions:
        suggestions = message.parsed.suggestion[:max_suggestions]
    else:
        suggestions = message.parsed.suggestion
        
    return suggestions

if __name__ == "__main__":
    text = "In this paper, we propose a new method for image classification. The method is based on a convolutional neural network (CNN) architecture. We evaluate the performance of our method on several benchmark datasets."
    engine = "gpt-4o-mini-2024-07-18"
    
    suggestions = generate_completion_rag(text, engine)
    print("Suggestions:")
    for suggestion in suggestions:
        print(suggestion)
        print("===")