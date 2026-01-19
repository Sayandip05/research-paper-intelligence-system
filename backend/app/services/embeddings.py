"""
FREE Embeddings using LlamaIndex
Much better integration for Week 2!
"""

from typing import List
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config import get_settings

settings = get_settings()


class LlamaIndexEmbeddingService:
    """
    LlamaIndex-powered embeddings (FREE!)
    
    Benefits:
    ✅ Seamless LlamaIndex integration (ready for Week 2)
    ✅ Automatic batching and caching
    ✅ 50+ model options
    ✅ Production-tested
    ✅ No API costs
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize with a FREE local model
        
        Popular models:
        - BAAI/bge-small-en-v1.5: 384 dim, fast, good quality (RECOMMENDED)
        - sentence-transformers/all-MiniLM-L6-v2: 384 dim, classic
        - BAAI/bge-base-en-v1.5: 768 dim, better quality, slower
        - thenlper/gte-small: 384 dim, good for QA
        """
        self.model_name = model_name or settings.embedding_model
        
        print(f"📦 Loading LlamaIndex embedding model: {self.model_name}")
        
        # Initialize LlamaIndex HuggingFace embeddings
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            device="cpu"  # Use "cuda" if you have GPU
        )
        
        self.dimension = settings.embedding_dim
        print(f"✅ LlamaIndex embeddings loaded!")
        print(f"   Dimension: {self.dimension}")
        print(f"   Device: cpu")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Input text
        
        Returns:
            List of floats (embedding vector)
        
        Cost: $0.00 (FREE!)
        """
        # LlamaIndex get_text_embedding handles everything
        embedding = self.embed_model.get_text_embedding(text)
        return embedding
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (BATCHED)
        
        LlamaIndex automatically batches and optimizes!
        
        Args:
            texts: List of texts
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        
        Cost: $0.00 (FREE!)
        """
        # LlamaIndex get_text_embedding_batch handles batching
        embeddings = self.embed_model.get_text_embedding_batch(
            texts,
            show_progress=show_progress
        )
        return embeddings
    
    def get_embed_model(self) -> BaseEmbedding:
        """
        Get the underlying LlamaIndex embed model
        
        This is useful for Week 2 when we build the full RAG pipeline!
        LlamaIndex agents/query engines need this format
        """
        return self.embed_model


# Alternative: Optimized BGE Model (Better quality!)
class BGEEmbeddingService:
    """
    BGE (Beijing Academy of AI) embeddings
    
    Benefits:
    ✅ State-of-the-art quality
    ✅ Better than all-MiniLM
    ✅ Still FREE!
    ✅ Used by many production systems
    
    Model options:
    - BAAI/bge-small-en-v1.5: 384 dim, fast
    - BAAI/bge-base-en-v1.5: 768 dim, better
    - BAAI/bge-large-en-v1.5: 1024 dim, best (slower)
    """
    
    def __init__(self, model_size: str = "small"):
        """
        Initialize BGE embeddings
        
        Args:
            model_size: 'small' (384d), 'base' (768d), or 'large' (1024d)
        """
        model_map = {
            "small": "BAAI/bge-small-en-v1.5",
            "base": "BAAI/bge-base-en-v1.5",
            "large": "BAAI/bge-large-en-v1.5"
        }
        
        self.model_name = model_map.get(model_size, model_map["small"])
        
        print(f"📦 Loading BGE model: {self.model_name}")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            device="cpu"
        )
        
        dim_map = {"small": 384, "base": 768, "large": 1024}
        self.dimension = dim_map.get(model_size, 384)
        
        print(f"✅ BGE embeddings loaded!")
        print(f"   Dimension: {self.dimension}")
    
    def generate_embedding(self, text: str) -> List[float]:
        return self.embed_model.get_text_embedding(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embed_model.get_text_embedding_batch(texts, show_progress=True)
    
    def get_embed_model(self) -> BaseEmbedding:
        return self.embed_model


# Global instance (singleton pattern)
_embedding_service = None

def get_embedding_service(use_bge: bool = False) -> LlamaIndexEmbeddingService:
    """
    Get or create embedding service
    
    Args:
        use_bge: Use BGE model (better quality) vs default model
    
    Returns:
        Embedding service instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        if use_bge:
            _embedding_service = BGEEmbeddingService(model_size="base")
        else:
            _embedding_service = LlamaIndexEmbeddingService()
    
    return _embedding_service


# For Week 2: Get LlamaIndex-native embed model
def get_llamaindex_embed_model() -> BaseEmbedding:
    """
    Get LlamaIndex embed model for RAG pipeline
    
    This is what you'll use in Week 2 for:
    - VectorStoreIndex
    - Query engines
    - Agents
    """
    service = get_embedding_service()
    return service.get_embed_model()


# Comparison function for testing
def compare_embedding_models():
    """
    Compare different embedding models
    Useful for deciding which to use
    """
    import time
    
    test_text = "The Transformer is a model architecture based on attention mechanisms."
    
    models = [
        ("all-MiniLM-L6-v2", 384),
        ("BAAI/bge-small-en-v1.5", 384),
        ("BAAI/bge-base-en-v1.5", 768),
    ]
    
    print("\n" + "="*60)
    print("  EMBEDDING MODEL COMPARISON")
    print("="*60)
    
    for model_name, dim in models:
        print(f"\n📊 Testing: {model_name}")
        
        try:
            start = time.time()
            embed = HuggingFaceEmbedding(model_name=model_name)
            load_time = time.time() - start
            
            start = time.time()
            embedding = embed.get_text_embedding(test_text)
            embed_time = time.time() - start
            
            print(f"   Load time: {load_time:.2f}s")
            print(f"   Embed time: {embed_time*1000:.0f}ms")
            print(f"   Dimension: {len(embedding)}")
            print(f"   ✅ Works!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test the embedding service
    print("\n" + "="*60)
    print("  Testing LlamaIndex Embeddings")
    print("="*60)
    
    # Test single embedding
    service = get_embedding_service()
    text = "The Transformer model uses self-attention."
    embedding = service.generate_embedding(text)
    print(f"\n✅ Generated embedding: {len(embedding)} dimensions")
    
    # Test batch embeddings
    texts = [
        "We use ImageNet for training.",
        "The model achieves 95% accuracy.",
        "Future work includes multimodal learning."
    ]
    embeddings = service.generate_embeddings(texts, show_progress=False)
    print(f"✅ Generated {len(embeddings)} embeddings in batch")
    
    # Test LlamaIndex integration
    embed_model = get_llamaindex_embed_model()
    print(f"✅ LlamaIndex embed model ready for Week 2!")
    print(f"   Type: {type(embed_model)}")