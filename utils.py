import os
from PyPDF2 import PdfReader
import pandas as pd
import hashlib
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import json
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_embeddings(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding

def calculate_hash(text):
    h = hashlib.sha256()
    h.update(text.encode('utf-8'))
    return h.hexdigest()

def load_paper_contents(path):
    """
    Load the paper contents from the JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)

    return []
    
def update_metadata(metadata, path, text):
    # update the metadata.json file with the new metadata
    
    paper_contents = load_paper_contents("./contents/metadata.json")
    
    text_hash = calculate_hash(text)
    
    for data in paper_contents:
        if data["hash"] == text_hash:
            # Metadata already exists
            return
    
    paper_contents.append({
        "title": metadata.title,
        "year": metadata.year,
        "author": metadata.author,
        "file_path": path,
        "file_name": os.path.basename(path),
        "hash": text_hash,
    })
    
    with open("./contents/metadata.json", "w") as f:
        json.dump(paper_contents, f, indent=4)