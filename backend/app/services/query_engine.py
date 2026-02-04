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

    def query(self, question: str, similarity_top_k: int = 5, response_mode: str = "compact", search_mode: str = "hybrid") -> Dict[str, Any]:
        """
        Execute a query against the research papers
        
        Args:
            search_mode: "dense" (BGE only), "sparse" (BM42 only), or "hybrid" (both with RRF)
        
        🆕 Returns BOTH text sources AND related images
        """
        # For hybrid/sparse, we need to query Qdrant directly
        if search_mode in ["sparse", "hybrid"]:
            return self._query_with_mode(question, similarity_top_k, response_mode, search_mode)
        
        # Dense-only uses LlamaIndex
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
        images = self._get_related_images(question)
            
        return {
            "question": question,
            "answer": str(response),
            "sources": sources,
            "images": images,
            "num_sources": len(sources),
            "response_mode": response_mode,
            "search_mode": search_mode
        }
    
    def _query_with_mode(self, question: str, top_k: int, response_mode: str, search_mode: str) -> Dict[str, Any]:
        """Query Qdrant directly with specified search mode"""
        from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
        from qdrant_client.models import SearchParams, Prefetch, Query
        
        # Generate embeddings based on mode
        dense_embedding = None
        sparse_embedding = None
        
        if search_mode in ["dense", "hybrid"]:
            dense_service = get_embedding_service()
            dense_embedding = dense_service.generate_embeddings([question])[0]
        
        if search_mode in ["sparse", "hybrid"]:
            sparse_service = get_sparse_embedding_service()
            sparse_result = sparse_service.generate_sparse_embeddings([question])[0]
            sparse_embedding = sparse_result
        
        # Query Qdrant based on mode
        if search_mode == "dense":
            results = self.client.query_points(
                collection_name=settings.qdrant_collection_name,
                query=dense_embedding,
                using="text-dense",
                limit=top_k,
                with_payload=True
            ).points
        elif search_mode == "sparse":
            from qdrant_client.models import SparseVector
            results = self.client.query_points(
                collection_name=settings.qdrant_collection_name,
                query=SparseVector(indices=sparse_embedding.indices, values=sparse_embedding.values),
                using="sparse",
                limit=top_k,
                with_payload=True
            ).points
        else:  # hybrid
            from qdrant_client.models import SparseVector
            results = self.client.query_points(
                collection_name=settings.qdrant_collection_name,
                prefetch=[
                    Prefetch(query=dense_embedding, using="text-dense", limit=top_k * 2),
                    Prefetch(
                        query=SparseVector(indices=sparse_embedding.indices, values=sparse_embedding.values),
                        using="sparse", 
                        limit=top_k * 2
                    )
                ],
                query=Query.fusion("rrf"),
                limit=top_k,
                with_payload=True
            ).points
        
        # Build context from results
        sources = []
        context_texts = []
        for point in results:
            payload = point.payload
            sources.append({
                "paper_id": payload.get("paper_id", "unknown"),
                "paper_title": payload.get("paper_title", "Unknown"),
                "section_title": payload.get("section_title", "Full Text"),
                "page_start": payload.get("page_start", 1),
                "page_end": payload.get("page_end", 1),
                "score": point.score or 0.0,
                "text": payload.get("text", "")[:500]
            })
            context_texts.append(payload.get("text", ""))
        
        # Generate answer with LLM
        context = "\n\n".join(context_texts[:top_k])
        prompt = f"""Based on the following research paper excerpts, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        from app.services.llm_service import get_llm
        llm = get_llm()
        answer = str(llm.complete(prompt))
        
        # Get images
        images = self._get_related_images(question)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "images": images,
            "num_sources": len(sources),
            "response_mode": response_mode,
            "search_mode": search_mode
        }
    
    def _get_related_images(self, question: str) -> list:
        """Retrieve related images using CLIP"""
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
        return images

_query_engine = None

def get_query_engine() -> IntelligentQueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = IntelligentQueryEngine()
    return _query_engine

