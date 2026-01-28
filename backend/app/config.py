from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "research_papers_hybrid"  # 🆕 New collection for hybrid
    
    # MongoDB (NEW - Week 3)
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "research_papers"
    
    # Dense Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    
    # 🆕 Sparse Embeddings (BM42)
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    enable_hybrid_search: bool = True  # Toggle hybrid on/off
    
    # 🆕 Hybrid Search Parameters
    rrf_k: int = 60  # Reciprocal Rank Fusion constant (standard: 60)
    dense_weight: float = 0.5  # Weight for dense retrieval (0-1)
    sparse_weight: float = 0.5  # Weight for sparse retrieval (0-1)
    
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_top_k: int = 5
    
    # Workflow (NEW - Week 3)
    enable_guardrails: bool = True
    confidence_threshold: float = 0.5
    
    corpus_dir: str = "./corpus"
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()