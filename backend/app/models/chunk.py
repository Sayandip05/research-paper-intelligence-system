from pydantic import BaseModel
from typing import List, Optional, Any


class ChunkMetadata(BaseModel):
    paper_id: str
    paper_title: str
    section_title: str
    page_start: int
    page_end: int


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Any] = None  # BM42 SparseVector


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int