from fastapi import FastAPI
from app.api.routes import search, query

app = FastAPI(
    title="Research Paper Intelligence System",
    description="Hybrid RAG System with BM42 + Dense Search and RRF Fusion",
    version="4.0.0"
)

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(query.router, prefix="/api", tags=["query"])

@app.get("/")
def root():
    return {"message": "Week 3: Multi-Agent Workflow", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "agents": 3, "workflow": "LlamaIndex"}