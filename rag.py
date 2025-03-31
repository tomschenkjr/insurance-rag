"""
This module implements a RAG (Retrieval Augmented Generation) pipeline that:
1. Extracts text from PDF documents
2. Chunks the text into smaller segments
3. Creates embeddings for the chunks
4. Indexes the embeddings for similarity search
5. Uses Ollama with DeepSeek-1:14B for question answering
"""

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
from typing import List, Tuple, Dict
import requests
import json

def extract_text_from_pdf(folder_path: str) -> list[tuple[str, str]]:
    """
    Extract text content from all PDF files in the specified directory.
    
    Args:
        folder_path (str): Path to the directory containing PDF files
        
    Returns:
        list[tuple[str, str]]: List of tuples, where each tuple contains:
            - filename (str): The name of the PDF file
            - full_text (str): The extracted text content from the file
            
    Example:
        >>> docs = extract_text_from_pdf("path/to/docs")
        >>> for filename, text in docs:
        ...     print(f"File: {filename}")
        ...     print(f"Content: {text[:100]}...")
    """
    documents = []
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(folder_path):
        # Only process files with .pdf extension
        if filename.endswith(".pdf"):
            # Construct full file path and open the PDF
            file_path = os.path.join(folder_path, filename)
            reader = PdfReader(file_path)
            
            # Extract text from all pages and join with newlines
            # This preserves the document structure
            full_text = "\n".join([page.extract_text() for page in reader.pages])
            
            # Add the filename and extracted text as a tuple to our results
            documents.append((filename, full_text))
            
    return documents

def create_embeddings_index(documents: List[Tuple[str, str]], 
                          chunk_size: int = 500,
                          chunk_overlap: int = 100) -> Tuple[faiss.Index, List[dict]]:
    """
    Create embeddings and FAISS index from the document chunks.
    
    Args:
        documents: List of (filename, text) tuples from PDF processing
        chunk_size: Size of text chunks for embedding
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        Tuple containing:
        - FAISS index for similarity search
        - List of metadata dictionaries for each chunk
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create chunks from all documents
    all_chunks = []
    metadata = []
    
    for filename, doc_text in documents:
        chunks = text_splitter.split_text(doc_text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": filename}] * len(chunks))
    
    # Initialize the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast
    
    # Create embeddings for all chunks
    embeddings = model.encode(all_chunks)
    
    # Create and populate FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    return index, metadata

def query_ollama(prompt: str, model: str = "deepseek:1.14b") -> str:
    """
    Query the Ollama API with DeepSeek model.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use (default: deepseek:1.14b)
        
    Returns:
        str: The model's response
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Error querying Ollama: {response.text}")

def search_and_answer(query: str, 
                     index: faiss.Index, 
                     metadata: List[Dict], 
                     chunks: List[str],
                     model: str = "deepseek:1.14b",
                     k: int = 3) -> str:
    """
    Search for relevant chunks and generate an answer using the LLM.
    
    Args:
        query: The user's question
        index: FAISS index containing embeddings
        metadata: List of metadata for each chunk
        chunks: List of text chunks
        model: Ollama model to use
        k: Number of chunks to retrieve
        
    Returns:
        str: The model's answer
    """
    # Create embedding for the query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]
    
    # Search for similar chunks
    distances, indices = index.search(np.array([query_embedding]), k)
    
    # Prepare context from retrieved chunks
    context = "\n\n".join([chunks[i] for i in indices[0]])
    
    # Create prompt with context and query
    prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    # Get answer from Ollama
    return query_ollama(prompt, model)

def main():
    """
    Main function to demonstrate the RAG pipeline.
    """
    # Set the path to your PDF documents
    docs_path = "docs"
    
    # Check if directory exists
    if not os.path.exists(docs_path):
        print(f"Error: Directory '{docs_path}' not found. Please create it and add your PDF files.")
        return
        
    # Check if directory has PDF files
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in '{docs_path}'. Please add some PDF files.")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {docs_path}")
    
    # Process PDF documents
    docs = extract_text_from_pdf(docs_path)
    
    # Create embeddings and index
    index, metadata = create_embeddings_index(docs)
    
    print(f"Processed {len(docs)} documents")
    print(f"Created {len(metadata)} chunks")
    print(f"Index shape: {index.d}")
    
    # Example query
    query = "What are the key points from the documents?"
    answer = search_and_answer(query, index, metadata, chunks)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main() 