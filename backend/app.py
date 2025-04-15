"""
Starts a Flask server that handles API requests from the frontend.
"""

import os
import gc
import shutil
import random
import openai
import warnings
import numpy as np
from time import time
from argparse import ArgumentParser

from response import generate_completion
from rag_response import generate_completion_rag
from preprocess import extract_info

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

warnings.filterwarnings("ignore", category=FutureWarning)  # noqa

SESSIONS = dict()
app = Flask(__name__)
CORS(app)  # For Access-Control-Allow-Origin

SUCCESS = True
FAILURE = False

@app.route('/api/query', methods=['POST'])
@cross_origin(origin='*')
def query():
    content = request.json
    
    text = content['text']
    
    # suggestions = generate_completion(df, vectors, prompt)
    suggestions = generate_completion_rag(text)
    
    return jsonify({
        'success': SUCCESS,
        'data': suggestions
    })

if __name__ == '__main__':
    app.run()
