from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ChunkMetadata(BaseModel):
    paper_id: str
    paper_title: str
    authors: List[str]
    year: Optional[int]
    section_title: str
    section_id: str
    page_start: int
    page_end: int
    chunk_index: int  # Position in the paper
    total_chunks: int
    has_table: bool = False
    has_equation: bool = False
    has_figure: bool = False


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None  # Will be generated


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata
    highlight: Optional[str] = None  # Highlighted matching text


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filter_by: Optional[Dict[str, Any]] = None  # Filter by year, authors, etc.


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float