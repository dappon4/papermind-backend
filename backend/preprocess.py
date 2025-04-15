import os
from PyPDF2 import PdfReader
import pandas as pd
import hashlib
from pydantic import BaseModel
from typing import Literal
import openai
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np

from utils import get_embeddings

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class PaperInfo(BaseModel):
    title: str
    year: int
    author: list[str]
    abstract: str
    method: str
    results: str
    conclusion: str

def calculatee_hash(text):
    h = hashlib.sha256()
    h.update(text.encode('utf-8'))
    return h.hexdigest()

def extract_paper_content(text):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a text extractor. Extract the title, year, author, abstract, method, results, and conclusion from the text, summarize them, and return them in JSON format."},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format=PaperInfo
    )
    message = completion.choices[0].message
    return message

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    whole_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            whole_text += text
    
    return whole_text, calculatee_hash(whole_text)

def extract_info(search_folder, contents_path):
    data_path = os.path.join(contents_path, "data.csv")
    vector_path = os.path.join(contents_path, "vectors.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} does not exist.")
    if os.path.exists(vector_path):
        original_vectors = np.load(vector_path, allow_pickle=True)
    else:
        original_vectors = []
    
    original_df = pd.read_csv(data_path, sep="|")
    
    new_df = pd.DataFrame(columns=original_df.columns)
    new_vectors = []
    # iterate through all files in the search folder
    for file in tqdm(os.listdir(search_folder)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(search_folder, file)
            text, hash_value = extract_text(pdf_path)
            
            # Check if the hash already exists in the original DataFrame
            if not original_df[original_df['hash'] == hash_value].empty:
                idx = original_df[original_df['hash'] == hash_value].index[0]
                new_df.loc[len(new_df)] = original_df.loc[idx]
                if os.path.exists(vector_path):
                    new_vectors.append(original_vectors[idx])
                else:
                    new_vectors.append(get_embeddings(original_df.loc[idx]['abstract']))
                
            else:
                message = extract_paper_content(text)
                # Add new entry to the new DataFrame
                new_entry = {
                    "path": pdf_path,
                    "hash": hash_value,
                    "title": message.parsed.title if message.parsed else "",
                    "year": message.parsed.year if message.parsed else "",
                    "author": message.parsed.author if message.parsed else "",
                    "abstract": message.parsed.abstract if message.parsed else "",
                    "method": message.parsed.method if message.parsed else "",
                    "results": message.parsed.results if message.parsed else "",
                    "conclusion": message.parsed.conclusion if message.parsed else "",
                }
                
                new_df.loc[len(new_df)] = new_entry
                new_vectors.append(get_embeddings(new_entry['abstract']))
                
    new_df.to_csv(data_path, sep="|", index=False)
    
    new_vectors = np.array(new_vectors)
    np.save(vector_path, new_vectors)
    
    return new_df, new_vectors

if __name__ == "__main__":
    search_folder = "../contents/files/"
    contents_path = "../contents/"
    # need to account for removed files
    df, vectors = extract_info(search_folder, contents_path)