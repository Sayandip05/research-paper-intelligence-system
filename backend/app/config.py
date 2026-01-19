from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "research_papers"
    
    # Embeddings (Local - FREE)
    # Options:
    # - "sentence-transformers/all-MiniLM-L6-v2": 384 dim, classic
    # - "BAAI/bge-small-en-v1.5": 384 dim, better quality (RECOMMENDED)
    # - "BAAI/bge-base-en-v1.5": 768 dim, best quality
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Paths
    corpus_dir: str = "./corpus"
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()