"""
Starts a Flask server that handles API requests from the frontend.
"""
import warnings
import json
import os

from rag_response import generate_completion_rag, generate_completion_zero_shot, post_analysis
from index import update_index, remove_nodes
from utils import load_paper_contents
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from llama_index.core import StorageContext, load_index_from_storage
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex

warnings.filterwarnings("ignore", category=FutureWarning)  # noqa

SESSIONS = dict()
app = Flask(__name__)
CORS(app)  # For Access-Control-Allow-Origin

SUCCESS = True
FAILURE = False

# load index
# storage_context = StorageContext.from_defaults(persist_dir="./contents/indexes")
# index = load_index_from_storage(storage_context)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
client = QdrantClient(path="./contents/qdrant_storage")
vector_store = QdrantVectorStore(
    collection_name="paper_collection",
    client=client,
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
)
retriever = index.as_retriever(similarity_top_k=5)

@app.route('/api/suggestions', methods=['POST'])
@cross_origin(origin='*')
def suggestions():
    content = request.json
    
    text = content['text']
    max_suggestions = content['maximumSuggestions']
    # suggestions = generate_completion(df, vectors, prompt)
    if content["useRAG"]:
        thoughts, suggestions, references = generate_completion_rag(text, max_suggestions, retriever)
        scores = post_analysis(suggestions, text)
        
        suggestions = [{"content": suggestion, "thoughts": thought, "references": reference, "scores": score} for (thought, suggestion, reference, score) in zip(thoughts, suggestions, references, scores)]
        # print(suggestions)
    else:
        suggestions = generate_completion_zero_shot(text, max_suggestions)
        suggestions = [{"content": suggestion} for suggestion in suggestions]
    
    print("generated:", len(suggestions), "suggestions")
    return jsonify({
        'success': SUCCESS,
        'data': suggestions
    })

@app.route('/api/upload', methods=['POST'])
@cross_origin(origin='*')
def upload():
    """
    Endpoint to handle PDF file uploads from the frontend
    """
    if 'file' not in request.files:
        return jsonify({
            'success': FAILURE,
            'error': 'No file part in the request'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': FAILURE,
            'error': 'No file selected'
        }), 400
    
    # Check if file is a PDF
    if not file.filename.endswith('.pdf'):
        return jsonify({
            'success': FAILURE,
            'error': 'File must be a PDF'
        }), 400
    
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "./contents/files"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        print(f"File saved to {file_path}")
        # Process the PDF file (extract information)
        update_index(index, file_path)
        
        return jsonify({
            'success': SUCCESS,
            'message': 'File uploaded successfully',
            'filename': file.filename
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'success': FAILURE,
            'error': str(e)
        }), 500
    
    

@app.route('/api/paper_list', methods=['GET'])
@cross_origin(origin='*')
def paper_list():
    """
    Endpoint to get the list of uploaded papers
    """
    metadata_path = "./contents/metadata.json"
    # read the metadata.json file
    if not os.path.exists(metadata_path):
        return jsonify({
            'success': FAILURE,
            'error': 'Metadata file not found'
        }), 404
    with open(metadata_path, 'r') as f:
        metadata = f.read()
    metadata = json.loads(metadata)
    papers = []
    for paper in metadata:
        papers.append({
            'title': paper['title'],
            'year': paper['year'],
            'author': paper['author'],
        })
    
    return jsonify({
        'success': SUCCESS,
        'data': papers
    })

@app.route('/api/paper', methods=['DELETE'])
@cross_origin(origin='*')
def remove_paper():
    """
    Endpoint to remove a paper from the system
    """
    try:
        content = request.json
        title = content.get('title')
        
        if not title:
            return jsonify({
                'success': FAILURE,
                'error': 'Paper title is required'
            }), 400
            
        metadata_path = "./contents/metadata.json"
        # Check if metadata file exists
        if not os.path.exists(metadata_path):
            return jsonify({
                'success': FAILURE,
                'error': 'Metadata file not found'
            }), 404
            
        # Read metadata
        metadata = load_paper_contents(metadata_path)
        
        # Find paper to remove
        paper_found = False
        for i, paper in enumerate(metadata):
            if paper['title'] == title:
                print(f"Removing paper: {paper['title']}")
                # Remove corresponding file if it exists
                file_path = paper['file_path']
                print(f"Removing file path: {file_path}")
                if os.path.exists(file_path):
                    print(f"File exists, removing: {file_path}")
                    os.remove(file_path)
                
                # Remove from metadata
                metadata.pop(i)
                paper_found = True
                break
        
        if not paper_found:
            return jsonify({
                'success': FAILURE,
                'error': 'Paper not found'
            }), 404
        
        # Update metadata file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # remove corresponding nodes from index
        remove_nodes(index, title)
        
        return jsonify({
            'success': SUCCESS,
            'message': f'Paper "{title}" removed successfully'
        })
    
    except Exception as e:
        print(f"Error removing paper: {e}")
        return jsonify({
            'success': FAILURE,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run()
