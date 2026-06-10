import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)
# --- 1. Connect to your persistent database ---
# This is the most crucial step. You must use the SAME path and settings
# that you used when you originally created the database in your RAG app.

# Define the path where your Chroma data is stored.
# Common default paths:
# - If you just used `chromadb.PersistentClient()`, it's usually "./chroma_db" or the path you set.
# - If you used a library like LlamaIndex, it may have its own default.
# REPLACE THIS PATH with the one from your project.
PERSIST_DIRECTORY = CHROMA_PERSIST_DIR  # <<<--- CHANGE THIS

print(f"🔍 Connecting to Chroma DB at: {PERSIST_DIRECTORY}")

# Create the client using the same embedding function used to build the DB.
# If you don't have the same function, omit it to inspect only IDs and metadata.
try:
    # Example: if you used the 'all-MiniLM-L6-v2' model from sentence-transformers
    sent_embeddings_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY, embedding_function=sent_embeddings_fn)
    print("✅ Connected with SentenceTransformer embedding function.")
except Exception as e:
    print(f"⚠️ Could not load the embedding function: {e}")
    print("📌 Connecting without an embedding function. You can still inspect IDs and metadata, but may not see the 'documents' field.")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# --- 2. List all Collections in your database ---
print("\n--- All Collections ---")
collections = client.list_collections()  # <-- Gets a list of all Collection objects[reference:8]
if not collections:
    print("No collections found in the database.")
    exit()
    
for i, coll in enumerate(collections):
    # Each 'coll' is a Collection object with attributes like .name, .metadata, .count[reference:9]
    print(f"{i+1}. Name: '{coll.name}', Metadata: {coll.metadata}, Count: {coll.count()}")

# --- 3. Pick a specific collection to inspect ---
# Replace 'your_collection_name' with the actual name from the list above
COLLECTION_NAME = COLLECTION_NAME  # <<<--- CHANGE THIS

print(f"\n--- Deep Dive into Collection: '{COLLECTION_NAME}' ---")
collection = client.get_collection(COLLECTION_NAME)
if collection is None:
    print(f"Could not find collection named '{COLLECTION_NAME}'.")
    exit()

# Get the total number of items (documents) in the collection
total_items = collection.count()  # <-- Simple and fast way to get the count[reference:10][reference:11]
print(f"\n📊 Total documents in collection: {total_items}")

# Get details for the first N items. For large collections, use limit.
# This method retrieves stored IDs, metadatas, and documents from the collection[reference:12].
# 'limit=10' means we fetch the first 10 items.
inspect_result = collection.get(limit=10)

print(f"\n📄 Retrieved {len(inspect_result.get('ids', []))} items.\n")

# --- 4. Print the retrieved data ---
# Collection.get() returns a dictionary with keys like 'ids', 'metadatas', 'documents'
ids = inspect_result.get('ids', [])
metadatas = inspect_result.get('metadatas', [])
documents = inspect_result.get('documents', [])

for i in range(len(ids)):
    print(f"--- Item {i+1} ---")
    print(f"ID: {ids[i]}")
    if metadatas and i < len(metadatas):
        print(f"Metadata: {metadatas[i]}")
    if documents and i < len(documents):
        # Print first 100 chars to avoid huge output
        doc_preview = (documents[i][:100] + '...') if len(documents[i]) > 100 else documents[i]
        print(f"Document Preview: {doc_preview}")
    print("-" * 20)

# --- 5. (Optional) View the collection's configuration ---
print("\n--- Collection Configuration ---")
print(f"Name: {collection.name}")
print(f"Metadata: {collection.metadata}")
print(f"Embedding Function: {collection.embedding_function}")
# Note: More advanced configuration (like HNSW params) can be accessed but is less common.