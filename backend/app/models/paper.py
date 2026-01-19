from pydantic import BaseModel
from typing import List, Optional


class PaperMetadata(BaseModel):
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    num_pages: int = 0


class Section(BaseModel):
    section_id: str
    title: str
    content: str
    page_start: int
    page_end: int


class ParsedPaper(BaseModel):
    paper_id: str
    filename: str
    metadata: PaperMetadata
    sections: List[Section] = []
    raw_text: str