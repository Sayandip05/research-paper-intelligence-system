"""
🆕 Image Search API Endpoint
"""
from fastapi import APIRouter, HTTPException
from langfuse.decorators import observe
from app.models.image import ImageSearchRequest, ImageSearchResponse
from app.services.clip_embedding import get_clip_embedding_service
from app.db.qdrant_client import QdrantService

router = APIRouter()


@router.post("/image-search", response_model=ImageSearchResponse)
@observe(name="CLIP_Image_Search")
async def search_images(request: ImageSearchRequest):
    """
    Search for images using text query
    
    Example:
```
    POST /api/image-search
    {
      "query": "show me LoRA architecture diagram",
      "top_k": 3,
      "min_score": 0.3
    }
```
    """
    try:
        # Get CLIP service
        clip_service = get_clip_embedding_service()
        
        # Generate text embedding
        query_embedding = clip_service.generate_text_embedding(request.query)
        
        # Search Qdrant image collection
        qdrant = QdrantService()
        results = qdrant.search_images(
            query_vector=query_embedding,
            limit=request.top_k,
            min_score=request.min_score
        )
        
        return ImageSearchResponse(
            query=request.query,
            results=results,
            total_found=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image-stats")
async def get_image_stats():
    """Get statistics about indexed images"""
    try:
        qdrant = QdrantService()
        total_images = qdrant.count_images()
        
        return {
            "total_images": total_images,
            "collection_name": "research_papers_images"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))