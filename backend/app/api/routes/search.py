from fastapi import APIRouter, HTTPException
from app.models.chunk import SearchRequest, SearchResponse, SearchResult
from app.services.embeddings import get_embedding_service
from app.db.qdrant_client import QdrantService
import time

router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Search across all uploaded papers
    
    Args:
        query: Natural language query
        top_k: Number of results to return
        filter_by: Optional filters (year, paper_id, section, etc.)
    
    Returns:
        Ranked list of relevant chunks with citations
    """
    start_time = time.time()
    
    try:
        # Generate query embedding
        embedding_service = get_embedding_service(use_local=False)
        query_embedding = await embedding_service.generate_embedding(request.query)
        
        # Search in Qdrant
        qdrant_service = QdrantService()
        results = qdrant_service.search(
            query_vector=query_embedding,
            limit=request.top_k,
            filters=request.filter_by
        )
        
        # Calculate search time
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            search_time_ms=round(search_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


@router.get("/papers")
async def list_papers():
    """List all uploaded papers (from Qdrant metadata)"""
    try:
        qdrant_service = QdrantService()
        info = qdrant_service.get_collection_info()
        
        return {
            "total_chunks": info.get("points_count", 0),
            "message": "Paper listing to be implemented with MongoDB"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing papers: {str(e)}"
        )


@router.get("/health")
async def search_health():
    """Check search service health"""
    try:
        qdrant_service = QdrantService()
        info = qdrant_service.get_collection_info()
        
        return {
            "status": "healthy",
            "qdrant_collection": info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }