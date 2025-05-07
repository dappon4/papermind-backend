from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
import json

from index import extract_paper_metadata

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
