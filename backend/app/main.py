from fastapi import FastAPI
from app.api.routes import search, query, workflow_query

app = FastAPI(
    title="Research Paper Intelligence System",
    description="Week 3: Event-Driven Multi-Agent Workflow",
    version="3.0.0"
)

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(workflow_query.router, prefix="/api", tags=["workflow"])

@app.get("/")
def root():
    return {"message": "Week 3: Multi-Agent Workflow", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "agents": 3, "workflow": "LlamaIndex"}