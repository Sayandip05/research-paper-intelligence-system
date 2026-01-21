from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq API (FREE LLM)
    groq_api_key: str = ""
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "research_papers"
    
    # Embeddings (Local - FREE)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    
    # LLM Settings
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval
    similarity_top_k: int = 5
    
    # Paths
    corpus_dir: str = "./corpus"
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()