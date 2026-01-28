"""
Search API Routes - Hybrid Search Support

Provides vector search endpoints with BM42 hybrid search.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.models.chunk import SearchRequest, SearchResponse
from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings

settings = get_settings()
router = APIRouter()


class HybridSearchRequest(BaseModel):
    """Request for hybrid search"""
    query: str
    top_k: int = 5
    sections: Optional[List[str]] = None


class HybridSearchResponse(BaseModel):
    """Response from hybrid search"""
    query: str
    mode: str
    results: list
    total_found: int
    paper_coverage: int


@router.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest):
    """
    Search across all papers (dense-only for backward compatibility)
    
    Example:
    {
        "query": "What is the transformer architecture?",
        "top_k": 5
    }
    """
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.generate_embedding(request.query)
    
    qdrant_service = QdrantService()
    results = qdrant_service.search(
        query_vector=query_embedding,
        limit=request.top_k
    )
    
    return SearchResponse(
        query=request.query,
        results=results,
        total_found=len(results)
    )


@router.post("/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search(request: HybridSearchRequest):
    """
    🆕 Hybrid Search - Dense + BM42 Sparse with RRF Fusion
    
    Uses both semantic understanding (BGE embeddings) and 
    keyword matching (BM42 sparse embeddings) combined with
    Reciprocal Rank Fusion for better retrieval.
    
    Example:
    {
        "query": "What is LoRA rank?",
        "top_k": 5,
        "sections": ["Methods", "Results"]
    }
    """
    # Get embedding services
    dense_service = get_embedding_service()
    qdrant_service = QdrantService()
    
    # Generate dense embedding
    dense_vector = dense_service.generate_embedding(request.query)
    
    # Generate sparse embedding if hybrid enabled
    sparse_vector = None
    mode = "dense"
    
    if settings.enable_hybrid_search:
        sparse_service = get_sparse_embedding_service()
        sparse_vector = sparse_service.generate_sparse_embedding(request.query)
        mode = "hybrid"
    
    # Execute search
    results = qdrant_service.search_with_filter(
        query_vector=dense_vector,
        limit=request.top_k,
        allowed_sections=request.sections,
        query_sparse_vector=sparse_vector
    )
    
    # Calculate paper coverage
    paper_ids = {r.metadata.paper_id for r in results if r.metadata.paper_id}
    
    return HybridSearchResponse(
        query=request.query,
        mode=mode,
        results=[
            {
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "score": r.score,
                "paper_title": r.metadata.paper_title,
                "section": r.metadata.section_title,
                "pages": f"{r.metadata.page_start}-{r.metadata.page_end}"
            }
            for r in results
        ],
        total_found=len(results),
        paper_coverage=len(paper_ids)
    )


@router.get("/corpus/stats")
def corpus_stats():
    """Get corpus statistics"""
    qdrant_service = QdrantService()
    count = qdrant_service.count()
    
    return {
        "total_chunks": count,
        "collection": settings.qdrant_collection_name,
        "hybrid_enabled": settings.enable_hybrid_search,
        "dense_model": settings.embedding_model,
        "sparse_model": settings.sparse_embedding_model if settings.enable_hybrid_search else None,
        "status": "ready"
    }