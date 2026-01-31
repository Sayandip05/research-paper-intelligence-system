from fastapi import FastAPI
from app.api.routes import search, query, upload, image_search

app = FastAPI(
    title="Research Paper Intelligence System",
    description="Multimodal RAG System with Hybrid Text Search (BM42 + Dense + RRF) and CLIP Image Search",
    version="5.0.0"
)

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(image_search.router, prefix="/api", tags=["Image Search"])

@app.get("/")
def root():
    return {"message": "Hybrid RAG System", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "agents": 3, "workflow": "LlamaIndex"}