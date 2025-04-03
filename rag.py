"""
This module implements a RAG (Retrieval Augmented Generation) pipeline that:
1. Extracts text from PDF documents
2. Chunks the text into smaller segments
3. Creates embeddings for the chunks
4. Indexes the embeddings for similarity search
5. Uses Ollama with DeepSeek-R1:14B for question answering
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
    Extract text content from all PDF files in directories named "Current" and their subdirectories.
    
    Args:
        folder_path (str): Path to the root directory to search for "Current" directories
        
    Returns:
        list[tuple[str, str]]: List of tuples, where each tuple contains:
            - filename (str): The name of the PDF file
            - full_text (str): The extracted text content from the file
    """
    documents = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        # Check if Current directory is in the path
        if "Current" in root:
            # Process each file in the Current directory
            for filename in files:
                # Only process files with .pdf extension
                if filename.endswith(".pdf"):
                    try:
                        # Construct full file path and open the PDF
                        file_path = os.path.join(root, filename)
                        print(f"Processing file: {file_path}")
                        reader = PdfReader(file_path)
                        
                        # Extract text from all pages and join with newlines
                        # This preserves the document structure
                        full_text = "\n".join([page.extract_text() for page in reader.pages])
                        
                        if not full_text.strip():
                            print(f"Warning: No text extracted from {file_path}")
                            continue
                            
                        # Add the filename and extracted text as a tuple to our results
                        # Use relative path from the Current directory for the filename
                        rel_path = os.path.relpath(file_path, folder_path)
                        documents.append((rel_path, full_text))
                        print(f"Successfully processed {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        continue
    
    return documents

def create_embeddings_index(documents: List[Tuple[str, str]], 
                          chunk_size: int = 500,
                          chunk_overlap: int = 100) -> Tuple[faiss.Index, List[dict], List[str]]:
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
        - List of text chunks
    """
    if not documents:
        raise ValueError("No documents provided to create embeddings from. Please check if PDF files were found in 'Current' directories.")
    
    print(f"\nProcessing {len(documents)} documents for chunking...")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create chunks from all documents
    all_chunks = []
    metadata = []
    
    for filename, doc_text in documents:
        try:
            chunks = text_splitter.split_text(doc_text)
            print(f"Created {len(chunks)} chunks from {filename}")
            all_chunks.extend(chunks)
            metadata.extend([{"source": filename}] * len(chunks))
        except Exception as e:
            print(f"Error chunking {filename}: {str(e)}")
            continue
    
    if not all_chunks:
        raise ValueError("No text chunks were created from the documents. Please check if the PDF files contain extractable text.")
    
    print(f"\nCreating embeddings for {len(all_chunks)} chunks...")
    
    # Initialize the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast
    
    # Create embeddings for all chunks
    try:
        # Print first chunk for debugging
        if all_chunks:
            print(f"First chunk preview: {all_chunks[0][:100]}...")
        
        embeddings = model.encode(all_chunks)
        print(f"Successfully created embeddings with shape {embeddings.shape}")
        
        if embeddings.size == 0:
            raise ValueError("No embeddings were created. Check if the chunks contain valid text.")
            
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        print(f"Number of chunks: {len(all_chunks)}")
        if all_chunks:
            print(f"First chunk length: {len(all_chunks[0])}")
        raise ValueError(f"Error creating embeddings: {str(e)}")
    
    # Create and populate FAISS index
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        print("Successfully created and populated FAISS index")
    except Exception as e:
        raise ValueError(f"Error creating FAISS index: {str(e)}")
    
    return index, metadata, all_chunks

def query_ollama(prompt: str, model: str = "deepseek-r1:14B") -> str:
    """
    Query the Ollama API with DeepSeek model.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use (default: deepseek-r1:14B)
        
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
        # Get the response and filter out <think> and </think> tags
        response_text = response.json()["response"]
        filtered_response = response_text.replace("<think>", "").replace("</think>", "").strip()
        return filtered_response
    else:
        raise Exception(f"Error querying Ollama: {response.text}")

def search_and_answer(query: str, 
                     index: faiss.Index, 
                     metadata: List[Dict], 
                     chunks: List[str],
                     model: str = "deepseek-r1:14B",
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
    # Create embedding for the query using the same model instance
    query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query])[0]
    
    # Search for similar chunks
    distances, indices = index.search(np.array([query_embedding]), k)
    
    # Prepare context from retrieved chunks with source information
    context_parts = []
    for i in indices[0]:
        source = metadata[i]["source"]
        chunk_text = chunks[i]
        context_parts.append(f"[From {source}]:\n{chunk_text}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt with context and query
    prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so. You are very knowledgeable. An expert. Think and respond with confidence.

Context:
{context}

Question: {query}

Answer:"""
    
    # Get answer from Ollama
    return query_ollama(prompt, model)

def chat_loop(index: faiss.Index, metadata: List[Dict], chunks: List[str]) -> None:
    """
    Interactive chat loop for asking questions about the documents.
    
    Args:
        index: FAISS index containing embeddings
        metadata: List of metadata for each chunk
        chunks: List of text chunks
    """
    print("\nChat started! Type 'exit' to end the conversation.")
    print("Ask any questions about your documents.\n")
    
    while True:
        try:
            # Get user input
            query = input("\nYour question: ").strip()
            
            # Check for exit command
            if query.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
                
            # Skip empty queries
            if not query:
                continue
                
            # Get and display answer
            answer = search_and_answer(query, index, metadata, chunks)
            print("\nAnswer:", answer)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'exit' to quit.")

def main():
    """
    Main function to demonstrate the RAG pipeline.
    """
    # Set the path to your root directory
    root_path = "."
    
    # Check if directory exists
    if not os.path.exists(root_path):
        print(f"Error: Directory '{root_path}' not found.")
        return
        
    # Check if there are any "Current" directories with PDF files
    pdf_files = []
    current_dirs = []
    for root, dirs, files in os.walk(root_path):
        if "Current" in root:
            current_dirs.append(root)
            pdf_files.extend([f for f in files if f.endswith('.pdf')])
    
    if not current_dirs:
        print(f"Error: No 'Current' directories found in the directory structure.")
        return
    
    if not pdf_files:
        print(f"Error: No PDF files found in any 'Current' directories.")
        print("Found 'Current' directories at:")
        for dir_path in current_dirs:
            print(f"  - {dir_path}")
        print("\nPlease add some PDF files to these directories.")
        return
    
    print(f"Found {len(pdf_files)} PDF files in 'Current' directories:")
    for dir_path in current_dirs:
        dir_pdfs = [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
        if dir_pdfs:
            print(f"  - {dir_path}:")
            for pdf in dir_pdfs:
                print(f"    * {pdf}")
    
    # Process PDF documents
    print("\nProcessing documents...")
    docs = extract_text_from_pdf(root_path)
    
    if not docs:
        print("Error: No documents were successfully processed from the PDF files.")
        return
    
    # Create embeddings and index
    print("Creating embeddings and search index...")
    try:
        index, metadata, chunks = create_embeddings_index(docs)
        
        print(f"\nProcessing complete!")
        print(f"- Processed {len(docs)} documents")
        print(f"- Created {len(metadata)} chunks")
        print(f"- Index shape: {index.d}")
        
        # Start interactive chat
        chat_loop(index, metadata, chunks)
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("Please check if your PDF files contain extractable text and are not corrupted.")

if __name__ == "__main__":
    main() 