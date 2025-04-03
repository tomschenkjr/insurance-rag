# Insurance RAG System with DeepSeek

This project implements a RAG (Retrieval Augmented Generation) system that processes PDF documents and uses DeepSeek-R1:14B via Ollama to answer questions about their contents. The system is specifically designed to work with insurance-related documents.

## Prerequisites

- Python 3.8 or higher
- Ollama installed on your system
- Sufficient disk space and RAM (DeepSeek-R1:14B is a large model)
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
ollama pull deepseek-r1:14B
```

## Setup

1. Create a directory named "Current" in your project root:
```bash
mkdir Current
```

2. Place your PDF files in the `Current` directory or any subdirectory of `Current`.

## Usage

1. Run the RAG system:
```bash
python rag.py
```

The system will:
- Process all PDF files in the `Current` directory and its subdirectories
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
Found 12 PDF files in Current directories

Processing documents...
Creating embeddings and search index...

Processing complete!
- Processed 12 documents
- Created 180 chunks
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
- `Current/`: Directory containing PDF files to process
- `index/`: Directory containing saved FAISS indices and metadata (created after first run)

## How it Works

1. **Document Processing**:
   - Extracts text from PDF files in the `Current` directory and its subdirectories
   - Splits text into manageable chunks
   - Preserves document structure and source information

2. **Embedding Creation**:
   - Uses SentenceTransformer to create embeddings
   - Builds a FAISS index for efficient similarity search

3. **Question Answering**:
   - Finds relevant text chunks using similarity search
   - Uses DeepSeek-R1:14B via Ollama to generate answers
   - Provides interactive chat interface for continuous questioning
   - Includes source document information in responses

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
model = "deepseek-r1:14B"  # Change to other Ollama models if needed
```

## License

MIT License

Copyright (c) 2024 Tom Schenk Jr

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