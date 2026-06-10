"""Ingest documents: load, chunk, embed, and store in ChromaDB."""
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DATA_FOLDER,
)

from utils import load_pdfs_from_folder, get_text_splitter

def main():
    # 1. Load documents
    print(f"Loading PDFs from {DATA_FOLDER}...")
    docs = load_pdfs_from_folder(DATA_FOLDER)
    if not docs:
        print("No PDFs found. Please add files to the 'data/' folder.")
        return

    # 2. Split into chunks
    print(f"Splitting {len(docs)} documents into chunks...")
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create embeddings and persist to Chroma
    print("Creating embeddings and storing in Chroma...")
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBEDDING_MODEL,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()
    print(f"Successfully stored {len(chunks)} chunks in {CHROMA_PERSIST_DIR}.")

if __name__ == "__main__":
    main()