from typing import List
from openai import AsyncOpenAI
from app.config import get_settings
import asyncio

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)


class EmbeddingService:
    def __init__(self, model: str = None):
        self.model = model or settings.embedding_model
        self.dimension = settings.embedding_dim
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batched)"""
        # OpenAI supports batch embedding
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    async def generate_embeddings_with_retry(
        self,
        texts: List[str],
        batch_size: int = 100,
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        Generate embeddings with batching and retry logic
        Useful for large number of chunks
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    embeddings = await self.generate_embeddings(batch)
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Retry {attempt + 1}/{max_retries} for batch {i}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return all_embeddings


# Alternative: Local embedding model (faster, no API costs)
from sentence_transformers import SentenceTransformer


class LocalEmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Local embedding model - runs on CPU/GPU
        Model: all-MiniLM-L6-v2
        - Dimension: 384
        - Fast and lightweight
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # For all-MiniLM-L6-v2
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding locally"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()


# Factory function to choose embedding service
def get_embedding_service(use_local: bool = False) -> EmbeddingService | LocalEmbeddingService:
    """
    Get embedding service
    
    Args:
        use_local: If True, use local model (no API costs)
                   If False, use OpenAI API (better quality)
    """
    if use_local:
        return LocalEmbeddingService()
    else:
        return EmbeddingService()