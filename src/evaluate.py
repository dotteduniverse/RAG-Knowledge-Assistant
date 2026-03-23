"""Evaluate RAG performance using RAGAS metrics."""
import os
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from src.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)

# Sample evaluation questions and ground truth (synthetic)
questions = [
    "What was the total revenue in 2023?",
    "Who is the CEO?",
    "What are the main risks mentioned?",
]
ground_truths = [
    "Total revenue was $XX million.",
    "The CEO is John Doe.",
    "The risks include market competition and regulatory changes.",
]

# Load retriever
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Load LLM for answer generation
llm = Ollama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)

# Generate answers
answers = []
contexts = []
for q in questions:
    docs = retriever.get_relevant_documents(q)
    contexts.append([doc.page_content for doc in docs])
    # Simple prompt for answer (you can use a chain for better results)
    prompt = f"Answer the following question based on the context:\nContext: {docs[0].page_content}\nQuestion: {q}\nAnswer:"
    answer = llm.invoke(prompt)
    answers.append(answer)

# Prepare dataset for RAGAS
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}
dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings,
)

print(result)