from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import search, query, upload, image_search, images, sessions, voice
from app.config import get_settings

settings = get_settings()

# Initialize Langfuse tracing for LlamaIndex
if settings.enable_langfuse and settings.langfuse_public_key:
    try:
        from langfuse.llama_index import LlamaIndexInstrumentor
        
        LlamaIndexInstrumentor().start()
        print("✅ Langfuse LlamaIndex Instrumentor enabled")
    except ImportError as e:
        print(f"⚠️ Langfuse LlamaIndex instrumentor not available: {e}")
    except Exception as e:
        print(f"⚠️ Langfuse LlamaIndex instrumentor failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Langfuse client for @observe decorator
    if settings.enable_langfuse:
        from app.services.langfuse_utils import get_langfuse
        get_langfuse()
    yield
    # Shutdown: Flush pending Langfuse events
    if settings.enable_langfuse:
        from app.services.langfuse_utils import flush_langfuse
        flush_langfuse()
        print("✅ Langfuse flushed on shutdown")


app = FastAPI(
    title="Research Paper Intelligence System",
    description="Multimodal RAG System with Hybrid Text Search (BM42 + Dense + RRF) and CLIP Image Search",
    version="5.0.0",
    lifespan=lifespan
)

# CORS middleware — allows frontend (Streamlit) to call the API
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(image_search.router, prefix="/api", tags=["Image Search"])
app.include_router(images.router, prefix="/api", tags=["Images"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(voice.router, prefix="/api", tags=["Voice"])

@app.get("/")
def root():
    return {"message": "Hybrid RAG System", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "agents": 3, "workflow": "LlamaIndex"}
