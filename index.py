from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from PyPDF2 import PdfReader
from utils import update_metadata, load_paper_contents
import json

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class PaperInfo(BaseModel):
    title: str
    year: int
    author: list[str]

def get_metadata(text):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a metadata extractor. Extract the title, year, and author from the text"},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format=PaperInfo
    )
    message = completion.choices[0].message
    return message.parsed

def extract_first_page_text(pdf_path):
    reader = PdfReader(pdf_path)
    first_page = reader.pages[0]
    text = first_page.extract_text()
    return text

def extract_paper_metadata(path):
    text = extract_first_page_text(path)
    metadata = get_metadata(text)
    update_metadata(metadata, path, text)

    return {
        "title": metadata.title,
        "year": metadata.year,
        "author": metadata.author,
        "file_path": path,
        "file_name": os.path.basename(path),
    }

def update_index(index, path):
    global paper_contents
    # Update the index with the new metadata
    reader = SimpleDirectoryReader(
        input_files=[path],
        file_metadata=extract_paper_metadata, # also updates metadata.json
    )
    print(f"Processing {path}...")
    docs = reader.load_data()
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)

def remove_nodes(index, paper_name):
    node_ids_to_remove = []
    for node in index.docstore.docs.values():
        if node.metadata["title"] == paper_name:
            node_ids_to_remove.append(node.node_id)
            # print(node.node_id)
    index.delete_nodes(node_ids_to_remove, delete_from_docstore=True)