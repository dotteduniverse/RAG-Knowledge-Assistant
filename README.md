# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions over financial documents (e.g., SEC 10-K filings) using open-source LLMs and vector search.

## Architecture

![Architecture](docs/architecture.png) *(Placeholder)*

The system consists of:
- **Ingestion Pipeline**: Load PDFs, chunk text, generate embeddings with Ollama (nomic-embed-text), store in ChromaDB.
- **Query Interface**: FastAPI backend and Streamlit UI that accepts a question, retrieves relevant chunks, and generates an answer using Ollama LLM.
- **Evaluation**: RAGAS metrics (faithfulness, answer relevance) to assess performance.

## Tech Stack
- **Python 3.10+**
- **LangChain** – orchestration
- **ChromaDB** – vector store
- **Ollama** – embeddings (`nomic-embed-text`) and LLM (`llama3.2` or `mistral`)
- **FastAPI** – API
- **Streamlit** – UI
- **RAGAS** – evaluation

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/rag-knowledge-assistant.git
   cd rag-knowledge-assistant