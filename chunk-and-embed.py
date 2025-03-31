from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
from typing import List, Tuple

def extract_text_from_files(directory_path: str) -> List[Tuple[str, str]]:
    """
    Extract text from all TXT files in the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing TXT files
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (filename, text_content)
    """
    documents = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if text_content.strip():  # Only add non-empty documents
                documents.append((filename, text_content))
    
    return documents

model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Create chunks
all_chunks = []
metadata = []

for filename, doc_text in extract_text_from_files("path/to/your/docs"):
    chunks = text_splitter.split_text(doc_text)
    all_chunks.extend(chunks)
    metadata.extend([{"source": filename}] * len(chunks))

# Embed
embeddings = model.encode(all_chunks)

# Index with FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))