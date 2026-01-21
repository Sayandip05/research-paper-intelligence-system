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
        
        # 2. Setup Vector Store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.qdrant_collection_name
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
        
        print("✅ Intelligent Query Engine initialized")

    def query(self, question: str, similarity_top_k: int = 5, response_mode: str = "compact") -> Dict[str, Any]:
        """
        Execute a query against the research papers
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
            
        return {
            "question": question,
            "answer": str(response),
            "sources": sources,
            "num_sources": len(sources),
            "response_mode": response_mode
        }

_query_engine = None

def get_query_engine() -> IntelligentQueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = IntelligentQueryEngine()
    return _query_engine
