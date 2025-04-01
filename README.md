# PDF RAG System with DeepSeek

This project implements a RAG (Retrieval Augmented Generation) system that processes PDF documents and uses DeepSeek-1:14B via Ollama to answer questions about their contents.

## Prerequisites

- Python 3.8 or higher
- Ollama installed on your system
- Sufficient disk space and RAM (DeepSeek-1:14B is a large model)
- At least 16GB RAM recommended for optimal performance

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd insurance-rag
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
   - Follow the installation instructions at [Ollama's website](https://ollama.ai/)
   - Start the Ollama service
   - Pull the DeepSeek model:
```bash
ollama pull deepseek:1.14b
```

## Setup

1. Create a directory for your PDF documents:
```bash
mkdir docs
```

2. Place your PDF files in the `docs` directory.

## Usage

1. Run the RAG system:
```bash
python rag.py
```

The system will:
- Process all PDF files in the `docs` directory
- Create embeddings and a search index
- Start an interactive chat session

### Interactive Chat Interface

Once the system is running, you can:
- Ask questions about your documents
- Get immediate answers based on the document content
- Type 'exit', 'quit', or 'bye' to end the session
- Press Ctrl+C at any time to exit

Example interaction:
```
Found 3 PDF files in docs

Processing documents...
Creating embeddings and search index...

Processing complete!
- Processed 3 documents
- Created 45 chunks
- Index shape: 384

Chat started! Type 'exit' to end the conversation.
Ask any questions about your documents.

Your question: What are the key points from the documents?

Answer: Based on the provided documents, the key points include...

Your question: Can you summarize the main topics?

Answer: The documents cover several key areas including...

Your question: exit

Goodbye!
```

### Tips for Better Questions
- Be specific in your questions
- Ask about particular sections or topics
- Request summaries or key points
- Ask for comparisons or relationships between topics
- Request clarification on specific terms or concepts

## Project Structure

- `rag.py`: Main script containing the RAG pipeline
- `requirements.txt`: Python package dependencies
- `docs/`: Directory containing PDF files to process
- `index/`: Directory containing saved FAISS indices and metadata (created after first run)

## How it Works

1. **Document Processing**:
   - Extracts text from PDF files
   - Splits text into manageable chunks
   - Preserves document structure

2. **Embedding Creation**:
   - Uses SentenceTransformer to create embeddings
   - Builds a FAISS index for efficient similarity search

3. **Question Answering**:
   - Finds relevant text chunks using similarity search
   - Uses DeepSeek-1:14B via Ollama to generate answers
   - Provides interactive chat interface for continuous questioning

## Customization

You can customize various aspects of the system by modifying parameters in `rag.py`:

1. **Text Chunking**:
```python
# In create_embeddings_index function
chunk_size = 500      # Size of each text chunk
chunk_overlap = 100   # Number of characters to overlap between chunks
```

2. **Search Parameters**:
```python
# In search_and_answer function
k = 3  # Number of chunks to retrieve for each query
```

3. **Model Selection**:
```python
# In query_ollama function
model = "deepseek:1.14b"  # Change to other Ollama models if needed
```

## Index Persistence

The system supports saving and loading the FAISS index and metadata:

1. **Saving the Index**:
```python
# After creating the index
faiss.write_index(index, "index/faiss.index")
with open("index/metadata.json", "w") as f:
    json.dump(metadata, f)
```

2. **Loading the Index**:
```python
# Before querying
index = faiss.read_index("index/faiss.index")
with open("index/metadata.json", "r") as f:
    metadata = json.load(f)
```

This allows you to:
- Process documents once and reuse the index
- Share the index with others
- Save processing time on subsequent runs

## Troubleshooting

1. **Ollama Connection Issues**:
   - Ensure Ollama is running (`ollama serve`)
   - Check if the service is accessible at `http://localhost:11434`
   - Verify the DeepSeek model is pulled (`ollama list`)
   - Check system logs for any Ollama errors
   - Ensure firewall isn't blocking local connections

2. **PDF Processing Issues**:
   - Verify PDF files are in the `docs` directory
   - Check file extensions are `.pdf`
   - Ensure PDFs are not corrupted or password-protected
   - Check PDFs are readable (not scanned images)
   - Verify file permissions allow reading

3. **Memory and Performance Issues**:
   - Reduce chunk size in `create_embeddings_index`
   - Process fewer documents at once
   - Ensure sufficient RAM is available
   - Monitor system resources during processing
   - Consider using a swap file if RAM is limited
   - Process documents in batches for large collections

4. **Embedding and Index Issues**:
   - Check if the index directory exists and has write permissions
   - Verify enough disk space for index storage
   - Monitor embedding model download progress
   - Check for CUDA/GPU errors if using GPU acceleration
   - Verify FAISS index integrity

5. **Common Error Messages and Solutions**:
   - "Connection refused": Ollama service not running
   - "No PDF files found": Check docs directory
   - "MemoryError": Reduce chunk size or process fewer documents
   - "CUDA out of memory": Reduce batch size or use CPU only
   - "Index not found": Rebuild the index

## License

MIT License

Copyright (c) 2025 Tom Schenk Jr.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 