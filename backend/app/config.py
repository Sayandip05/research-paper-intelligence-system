from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "research_papers"
    
    # MongoDB (NEW - Week 3)
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "research_papers"
    
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    
    llm_model: str = "openai/gpt-oss-120b"  # Fixed Groq model name
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