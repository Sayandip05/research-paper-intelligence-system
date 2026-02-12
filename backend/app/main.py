from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import search, query, upload, image_search, images, sessions
from app.config import get_settings

settings = get_settings()

# Initialize Langfuse tracing for LlamaIndex
if settings.enable_langfuse and settings.langfuse_public_key:
    try:
        from llama_index.core import Settings as LlamaSettings
        from llama_index.core.callbacks import CallbackManager
        from llama_index.callbacks.langfuse import LlamaIndexCallbackHandler
        
        langfuse_handler = LlamaIndexCallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host
        )
        LlamaSettings.callback_manager = CallbackManager([langfuse_handler])
        print("✅ Langfuse LlamaIndex callback enabled")
    except ImportError as e:
        print(f"⚠️ Langfuse LlamaIndex callback not available: {e}")
    except Exception as e:
        print(f"⚠️ Langfuse LlamaIndex callback failed: {e}")


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

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(image_search.router, prefix="/api", tags=["Image Search"])
app.include_router(images.router, prefix="/api", tags=["Images"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])

@app.get("/")
def root():
    return {"message": "Hybrid RAG System", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "agents": 3, "workflow": "LlamaIndex"}
