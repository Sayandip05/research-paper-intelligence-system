from fastapi import FastAPI
from app.api.routes import search, query

app = FastAPI(
    title="Research Paper Intelligence System",
    description="Week 2: Intelligent RAG with LlamaIndex + Groq",
    version="2.0.0"
)

# Include routes
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(query.router, prefix="/api", tags=["query"])


@app.get("/")
def root():
    return {
        "message": "Research Paper Analyzer - Week 1",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}