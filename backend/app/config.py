from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "research_papers_hybrid"  # Text collection
    qdrant_image_collection_name: str = "research_papers_images"  # 🆕 Image collection
    
    # Dense Embeddings (Text)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    
    # Sparse Embeddings (BM42)
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    enable_hybrid_search: bool = True
    
    # Hybrid Search Parameters
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    
    # 🆕 CLIP Embeddings (Multimodal)
    clip_model_name: str = "ViT-B/32"  # 512-dim
    clip_embedding_dim: int = 512
    enable_multimodal: bool = True  # Toggle image extraction
    
    # 🆕 Image Extraction Settings
    min_image_width: int = 100  # Filter tiny images
    min_image_height: int = 100
    
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_top_k: int = 5
    
    # Workflow
    enable_guardrails: bool = True
    confidence_threshold: float = 0.5
    
    # Langfuse Tracing
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    enable_langfuse: bool = True
    
    corpus_dir: str = "../corpus"
    
    class Config:
        env_file = "../.env"  # .env is at project root, not backend/


@lru_cache()
def get_settings() -> Settings:
    return Settings()