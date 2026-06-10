import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdfs_from_folder(folder_path):
    """Load all PDFs in a folder and return a list of documents."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def get_text_splitter():
    """Return a text splitter configured from env variables."""
    from config import CHUNK_SIZE, CHUNK_OVERLAP
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )