from llama_index.core import StorageContext, load_index_from_storage
from openai import OpenAI
import os
from dotenv import load_dotenv
import random

from pydantic import BaseModel, Field
from typing import Literal, List

class Suggestions(BaseModel):
    suggestion: List[str]

class Summaries(BaseModel):
    summary: List[str]

class AnalysisScores(BaseModel):
    relevance: float
    fluency: float
    tonematch: float
    coherence: float
    
class Verification(BaseModel):
    verification: List[bool]

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

def get_section_summary(text, engine):
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": "Summarize each section of the following text in 3-5 sentences."},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        response_format=Summaries
    )
    
    message = completion.choices[0].message
    return message.parsed.summary

def get_stage_1_suggestions(text, engine, retriever):
    print("Generating stage 1 suggestions...")
    section_summaries = get_section_summary(text, engine)
    
    nodes = retriever.retrieve(" ".join(section_summaries))
    metadata_template = "title: {title}, authors: {author}, year: {year}"
    context = ["{ " + metadata_template.format(title=node.metadata["title"], author=node.metadata["author"], year=node.metadata["year"]) + " }: " + node.get_content() + "\n\n" for node in nodes]
    # print(context)
    print(f"retrieved {len(context)} documents")
    prompt = f"""
    Based on the current paragraph and retrieved documents, suggest the next 3 sentences. Focus on only one aspect when making suggestions.
    
    User text summary:
    {section_summaries}
    
    Retrieved documents:
    {context}
    
    You should only suggest up to 3 sentences.
    
    """
    
    if text.strip()[-1] != ".":
        prompt += f"continue after this word: {text.split()[-1]}"
    
    completion = client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Curreet paragraph: " + text.split('\n\n')[-1]}
        ],
        temperature=0.3,
    )
    suggestion = completion.choices[0].message.content
    return suggestion, section_summaries, nodes
    
def verify_retrieval(text, retrieved_content):
    prompt = f"""
    Given the current paragraph and retrieved documents from RAG engine,
    for each document verify if the retrieved documents are relevant to the current paragraph.
    
    Current paragraph:
    {text}
    """
    
    documents = [f"Document {i+1}: {content}" for i, content in enumerate(retrieved_content)]
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"""
            Retrieved documents:
            {documents}
            
            The return value should be a list of boolean values, with length {len(retrieved_content)}. There can be more than one True value.
             """}
        ],
        temperature=0,
        response_format=Verification
    )
    
    message = completion.choices[0].message
    return message.parsed.verification

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
    
    message = completion.choices[0].message
    if len(message.parsed.suggestion) > max_suggestions:
        suggestions = message.parsed.suggestion[:max_suggestions]
    else:
        suggestions = message.parsed.suggestion
        
    return suggestions

def get_thoughts(text, suggestion, engine):
    prompt = f"""
    Given the current paragraph and the suggestion of the next few sentences, critically compare and contrast each source, user text, and suggestion. Create a thinking process
    that serves as the base for the next refined suggestion.
    """
    
    message = client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Current paragraph: {text}\n\nSuggestion: {suggestion}"}
        ],
        temperature=0.2,
    )
    
    return message.choices[0].message.content

def generate_completion_rag(text, max_suggestions, retriever, engine = "gpt-4o-mini-2024-07-18"):
    initial_suggestion, section_summaries, stage_1_nodes = get_stage_1_suggestions(text, engine, retriever) # get initial suggestion
    abstract_summary = section_summaries[0]
    
    # retrieve documents based on initial suggestion and current paragraph
    refined_retrieved_nodes = retriever.retrieve(abstract_summary + " " + text.split("\n\n")[-1] + " " + initial_suggestion)
    
    # print("\n\n".join([node.get_content() for node in refined_retrieved_nodes] + [node.get_content() for node in stage_1_nodes]))
    
    combined_retrieved_nodes = refined_retrieved_nodes + stage_1_nodes
    unique_nodes = []
    unique_node_contents = set()
    # remove duplicates based on content
    for node in combined_retrieved_nodes:
        if node.get_content() not in unique_node_contents:
            unique_nodes.append(node)
            unique_node_contents.add(node.get_content())
    
    # verify if the retrieved documents are relevant to the current paragraph
    verification_results = verify_retrieval(text, [node.get_content() for node in unique_nodes])
    print(verification_results, len(unique_nodes))
    
    # verify if the verification list has the same length as the retrieved nodes
    while len(verification_results) < len(unique_nodes):
        verification_results.append(False)
    verification_results = verification_results[:len(unique_nodes)]
    
    # filter out the verified nodes 
    verified_nodes = [node for node, verified in zip(unique_nodes, verification_results) if verified]
    
    # generate context to pass to the model
    metadata_template = "title: {title}, authors: {author}, year: {year}"
    context = ["{ " + metadata_template.format(title=node.metadata["title"], author=node.metadata["author"], year=node.metadata["year"]) + " }: " + node.get_content() + "\n\n" for node in verified_nodes]
    print(len(context), "used documents")
    
    # generate thinking process
    thoughts = get_thoughts(text, initial_suggestion, engine)
    
    prompt = f"""
    Given the current paragraph, retrieved documents that are refined based on the initial suggestion, and the thought process that supports the next suggestion,
    generate next suggestion that is 2 sentences long continuing from the current paragraph. Your answer must start from the last word of the current paragraph.
    You must generate {max_suggestions} suggestions.
    
    user text summary:
    {" ".join(section_summaries)}
    
    thought process:
    {thoughts}
    
    retrieved documents:
    {context}
    
    generate {max_suggestions} suggestions.
    """
    
    # if the sentence is not complete, continue from the last word
    if text.strip()[-1] != ".":
        prompt += f"continue after this word: {text.split()[-1]}"

    print("Generating stage 2 suggestions...")
    completion = client.beta.chat.completions.parse(
        model=engine,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Current paragraph: " + text.split('\n\n')[-1] + "\n" + f"Continue from the last word so that it becomes a complete sentence. Generate {max_suggestions} suggestions."}
        ],
        temperature=0.2,
        response_format=Suggestions
    )
    
    message = completion.choices[0].message
    if len(message.parsed.suggestion) > max_suggestions:
        suggestions = message.parsed.suggestion[:max_suggestions]
    else:
        suggestions = message.parsed.suggestion
    
    last_word = text.split(" ")[-1]
    for i, suggestion in enumerate(suggestions):
        first_word = suggestion.split(" ")[0]
        if first_word == last_word:
            suggestions[i] = suggestion.replace(first_word, "", 1).strip()
    
    references = [
        {
            "title": node.metadata["title"],
            "author": node.metadata["author"],
            "year": node.metadata["year"],
            "content": node.get_content()
        }
        for node in verified_nodes
    ]
    print(len(references), "references")
    
    return [[thoughts]] * len(suggestions), suggestions, [references] * len(suggestions)

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

def post_analysis(suggestion_texts, text):
    current_paragraph = text.split("\n\n")[-1]
    
    prompt = f"""
    Given current paragraph and the suggestion of the next few sentences,
    Give a score for each metric: relevance, fluency, tonematch, coherence.
    The score should be between 0 and 1, where 0 means not relevant and 1 means very relevant.
    
    current paragraph:
    {current_paragraph}
    """
    
    scores = []
    
    for suggestion in suggestion_texts:
        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Suggestion: {suggestion}"}
            ],
            temperature=0,
            response_format=AnalysisScores
        )
        
        scores.append(completion.choices[0].message.parsed)
    
    return_scores = [{
            "relevance": score.relevance,
            "fluency": score.fluency,
            "tonematch": score.tonematch,
            "coherence": score.coherence
        } for score in scores]
    return return_scores