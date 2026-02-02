from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import get_settings
from app.services.llm_service import get_llm
from app.services.embeddings import get_llamaindex_embed_model

settings = get_settings()

class IntelligentQueryEngine:
    def __init__(self):
        # 1. Initialize Qdrant Client
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        
        # 2. Setup Vector Store with named vector for hybrid collection
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.qdrant_collection_name,
            vector_name="text-dense"  # Match LlamaIndex naming convention
        )
        
        # 3. Setup LLM and Embedding Model
        self.llm = get_llm()
        self.embed_model = get_llamaindex_embed_model()
        
        # 4. Initialize Index from existing Vector Store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        
        # 5. Create Query Engine
        self.engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=settings.similarity_top_k
        )
        
        # 6. 🆕 CLIP for image retrieval
        self.clip_service = None
        self.qdrant_service = None
        if settings.enable_multimodal:
            try:
                from app.services.clip_embedding import get_clip_embedding_service
                from app.db.qdrant_client import QdrantService
                self.clip_service = get_clip_embedding_service()
                self.qdrant_service = QdrantService()
                print("✅ Intelligent Query Engine initialized (multimodal)")
            except Exception as e:
                print(f"⚠️ CLIP not available: {e}")
                print("✅ Intelligent Query Engine initialized (text-only)")
        else:
            print("✅ Intelligent Query Engine initialized")

    def query(self, question: str, similarity_top_k: int = 5, response_mode: str = "compact") -> Dict[str, Any]:
        """
        Execute a query against the research papers
        
        🆕 Returns BOTH text sources AND related images
        """
        # Update top_k if specified
        if similarity_top_k != settings.similarity_top_k:
            self.engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=similarity_top_k,
                response_mode=response_mode
            )
            
        # Run query
        response = self.engine.query(question)
        
        # Parse sources
        sources = []
        for node in response.source_nodes:
            metadata = node.node.metadata
            sources.append({
                "paper_id": metadata.get("paper_id", "unknown"),
                "paper_title": metadata.get("paper_title", metadata.get("title", "Unknown Paper")),
                "section_title": metadata.get("section_title", "Full Text"),
                "page_start": metadata.get("page_start", 1),
                "page_end": metadata.get("page_end", 1),
                "score": node.score or 0.0,
                "text": node.node.get_content()
            })
        
        # 🆕 Retrieve related images
        images = []
        if self.clip_service and self.qdrant_service:
            try:
                clip_query = self.clip_service.generate_text_embedding(question)
                image_results = self.qdrant_service.search_images(
                    query_vector=clip_query,
                    limit=3,
                    min_score=0.15
                )
                images = [
                    {
                        "image_id": img.image_id,
                        "paper_title": img.paper_title,
                        "page_number": img.page_number,
                        "caption": img.caption,
                        "image_type": img.metadata.image_type,
                        "score": img.score
                    }
                    for img in image_results
                ]
            except Exception as e:
                print(f"⚠️ Image retrieval failed: {e}")
            
        return {
            "question": question,
            "answer": str(response),
            "sources": sources,
            "images": images,  # 🆕 Include images
            "num_sources": len(sources),
            "response_mode": response_mode
        }

_query_engine = None

def get_query_engine() -> IntelligentQueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = IntelligentQueryEngine()
    return _query_engine

