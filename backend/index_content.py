from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from PyPDF2 import PdfReader

from preprocess import extract_text

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
    
    return {
        "title": metadata.title,
        "year": metadata.year,
        "author": metadata.author,
        "file_path": path,
        "file_name": os.path.basename(path),
    }

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
reader = SimpleDirectoryReader(
    input_dir="../contents/files/",
    file_metadata=extract_paper_metadata,
)

docs = reader.load_data()
print(f"Count of Techcrunch articles: {len(docs)}")
print(docs[0])

node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
nodes = node_parser.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

index.storage_context.persist(
    persist_dir="../contents/indexes/",
)

if __name__ == "__main__":
    retriever = index.as_retriever(similarity_top_k=3)
    query = "Tell me about ViT"

    for node in retriever.retrieve(query):
        print(node)
        print(node.get_content())
        print("===")