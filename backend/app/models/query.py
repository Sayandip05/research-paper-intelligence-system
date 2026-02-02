from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question you want to ask")
    similarity_top_k: int = Field(5, description="Number of source chunks to retrieve")
    response_mode: str = Field("compact", description="LlamaIndex response mode")

class SourceInfo(BaseModel):
    paper_id: str
    paper_title: str
    section_title: str
    page_start: int
    page_end: int
    score: float
    text: Optional[str] = None

# 🆕 Image info for unified search
class ImageInfo(BaseModel):
    image_id: str
    paper_title: str
    page_number: int
    caption: Optional[str] = None
    image_type: str = "figure"
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    images: List[ImageInfo] = Field(default_factory=list)  # 🆕 Related images
    num_sources: int
    response_mode: str
