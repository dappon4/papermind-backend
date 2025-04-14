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
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_embeddings(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding