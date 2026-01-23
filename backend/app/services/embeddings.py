"""
Embeddings Service with LlamaIndex Integration
"""

from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """
    LlamaIndex-powered embeddings (FREE!)
    """
    
    def __init__(self):
        print(f"📦 Loading embedding model: {settings.embedding_model}")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            device="cpu"
        )
        
        self.dimension = settings.embedding_dim
        print(f"✅ Embeddings loaded! Dimension: {self.dimension}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        embedding = self.embed_model.get_text_embedding(text)
        return embedding
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts (batched)"""
        embeddings = self.embed_model.get_text_embedding_batch(
            texts,
            show_progress=show_progress
        )
        return embeddings
    
    def get_embed_model(self) -> BaseEmbedding:
        """
        Get LlamaIndex embed model
        
        This is used by Query Engine in Week 2!
        """
        return self.embed_model


# Global instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_llamaindex_embed_model() -> BaseEmbedding:
    """
    Get LlamaIndex embed model for Query Engine
    
    Used in Week 2 for RAG pipeline
    """
    service = get_embedding_service()
    return service.get_embed_model()