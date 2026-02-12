"""
Session & Message Models
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SessionCreate(BaseModel):
    title: Optional[str] = None


class SessionRename(BaseModel):
    title: str


class SessionMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[dict]] = None
    images: Optional[List[dict]] = None
    search_mode: Optional[str] = None
    timestamp: datetime


class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionDetail(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[SessionMessage] = []


class SessionQueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    similarity_top_k: int = Field(5, description="Number of source chunks")
    search_mode: str = Field("hybrid", description="dense, sparse, or hybrid")
