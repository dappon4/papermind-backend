from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from index import extract_paper_metadata

def index_similarity():
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    reader = SimpleDirectoryReader(
        input_dir="./contents/files/",
        file_metadata=extract_paper_metadata,
    )

    docs = reader.load_data()
    print(f"Count of Techcrunch articles: {len(docs)}")
    print(docs[0])

    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    index.storage_context.persist(
        persist_dir="./contents/indexes/",
    )

def index_qdrant():
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    print("Reading files")
    
    reader = SimpleDirectoryReader(
    input_dir="./contents/files/",
    file_metadata=extract_paper_metadata,
    )

    docs = reader.load_data()
    client = QdrantClient(path="./contents/qdrant_storage")
    
    if client.collection_exists("paper_collection"):
        print("Collection already exists, deleting it...")
        client.delete_collection("paper_collection")
    
    vector_store = QdrantVectorStore(
    collection_name="paper_collection",
    client=client,
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex.from_documents(
        docs,
        # our dense embedding model
        embed_model=embed_model,
        storage_context=storage_context,
    )

if __name__ == "__main__":
    print("which index to use?\n[1] similarity\n[2] qdrant")
    index_type = input("index type: ")
    if index_type == "1":
        index_similarity()
    elif index_type == "2":
        index_qdrant()
    else:
        print("Invalid index type")