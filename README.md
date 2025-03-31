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
- Run an example query

### Example Output
```
Found 3 PDF files in docs
Processed 3 documents
Created 45 chunks
Index shape: 384

Question: What are the key points from the documents?
Answer: Based on the provided documents, the key points include...
```

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

[Your chosen license] 