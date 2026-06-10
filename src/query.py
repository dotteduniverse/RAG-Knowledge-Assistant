"""FastAPI endpoint for querying the RAG system."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)

app = FastAPI(title="RAG Knowledge Assistant")

# Initialize components once
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = Ollama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = qa_chain.invoke({"query": request.question})
        answer = result["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}