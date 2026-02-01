
"""
Image metadata models for multimodal RAG
"""
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image as PILImage


class ImageMetadata(BaseModel):
    """Metadata for extracted image"""
    image_id: str
    paper_id: str
    paper_title: str
    page_number: int
    caption: Optional[str] = None
    image_type: str = "figure"  # figure, chart, diagram, table
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1] if available


class ExtractedImage(BaseModel):
    """Image extracted from PDF (in-memory only)"""
    metadata: ImageMetadata
    # Note: PIL Image object not stored here, only during processing
    
    class Config:
        arbitrary_types_allowed = True


class ImageSearchResult(BaseModel):
    """Result from image search"""
    image_id: str
    paper_title: str
    page_number: int
    caption: Optional[str]
    score: float
    metadata: ImageMetadata


class ImageSearchRequest(BaseModel):
    """Request for image search"""
    query: str  # Text query like "show me LoRA architecture"
    top_k: int = 3
    min_score: float = 0.15  # Lower threshold for CLIP similarity


class ImageSearchResponse(BaseModel):
    """Response from image search"""
    query: str
    results: List[ImageSearchResult]
    total_found: int