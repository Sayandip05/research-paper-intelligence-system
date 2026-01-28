"""
Embeddings Service with Dense + Sparse (BM42) Support
"""
from typing import List, Tuple
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """
    Dense embeddings (BGE) - Semantic understanding
    """
    
    def __init__(self):
        print(f"📦 Loading DENSE embedding model: {settings.embedding_model}")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            device="cpu"
        )
        
        self.dimension = settings.embedding_dim
        print(f"✅ Dense embeddings loaded! Dimension: {self.dimension}")
    
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
        """Get LlamaIndex embed model"""
        return self.embed_model


class SparseEmbeddingService:
    """
    🆕 Sparse embeddings (BM42) - Keyword matching
    """
    
    def __init__(self):
        print(f"📦 Loading SPARSE embedding model: {settings.sparse_embedding_model}")
        
        # Initialize FastEmbed BM42 model
        self.sparse_model = SparseTextEmbedding(
            model_name=settings.sparse_embedding_model,
            # Uncomment if using GPU:
            # providers=["CUDAExecutionProvider"],
        )
        
        print(f"✅ Sparse embeddings loaded! (BM42)")
    
    def generate_sparse_embedding(self, text: str) -> SparseVector:
        """Generate sparse embedding for single text"""
        embeddings = list(self.sparse_model.embed([text]))
        
        if not embeddings:
            return SparseVector(indices=[], values=[])
        
        embedding = embeddings[0]
        
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist()
        )
    
    def generate_sparse_embeddings(self, texts: List[str]) -> List[SparseVector]:
        """Generate sparse embeddings for multiple texts (batched)"""
        embeddings = list(self.sparse_model.embed(texts))
        
        sparse_vectors = []
        for embedding in embeddings:
            sparse_vectors.append(
                SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist()
                )
            )
        
        return sparse_vectors


# 🆕 Global instances
_embedding_service = None
_sparse_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create DENSE embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_sparse_embedding_service() -> SparseEmbeddingService:
    """🆕 Get or create SPARSE embedding service"""
    global _sparse_embedding_service
    if _sparse_embedding_service is None:
        _sparse_embedding_service = SparseEmbeddingService()
    return _sparse_embedding_service


def get_llamaindex_embed_model() -> BaseEmbedding:
    """Get LlamaIndex embed model for Query Engine"""
    service = get_embedding_service()
    return service.get_embed_model()