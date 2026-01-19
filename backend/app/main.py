from fastapi import FastAPI
from app.api.routes import search

app = FastAPI(
    title="Research Paper Analyzer",
    description="Week 1: Document Processing & Search",
    version="1.0.0"
)

# Include routes
app.include_router(search.router, prefix="/api", tags=["search"])


@app.get("/")
def root():
    return {
        "message": "Research Paper Analyzer - Week 1",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}