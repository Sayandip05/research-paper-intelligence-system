from fastapi import APIRouter
from app.models.chunk import SearchRequest, SearchResponse
from app.services.embeddings import get_embedding_service
from app.db.qdrant_client import QdrantService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest):
    """
    Search across all papers in the corpus
    
    Example:
    {
        "query": "What is the transformer architecture?",
        "top_k": 5
    }
    """
    # Generate query embedding
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.generate_embedding(request.query)
    
    # Search in Qdrant
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


@router.get("/corpus/stats")
def corpus_stats():
    """Get corpus statistics"""
    qdrant_service = QdrantService()
    count = qdrant_service.count()
    
    return {
        "total_chunks": count,
        "status": "ready"
    }