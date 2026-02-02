"""
Event Models for Workflow System

🆕 Now supports MULTIMODAL retrieval (text + images)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from llama_index.core.workflow.events import Event


class IntentType(str, Enum):
    """Query intent classification - Canonical intent names only"""
    SUMMARY = "summary"
    COMPARISON = "comparison"
    RESEARCH_GAPS = "research_gaps"


class StartEvent(Event):
    """Initial event from user question"""
    question: str
    session_id: Optional[str] = None
    human_constraints: Optional[Dict[str, Any]] = None


class RetrievalEvent(Event):
    """Event to trigger evidence retrieval"""
    intent_type: IntentType
    target_sections: List[str] = Field(default_factory=list)
    confidence_threshold: float = 0.5
    human_review_hint: bool = False
    similarity_top_k: int = 5
    original_question: str = ""


class EvidenceChunk(BaseModel):
    """Retrieved text evidence with metadata"""
    text: str
    paper_title: str
    section_title: str
    page_start: int
    page_end: int
    score: float


# 🆕 NEW: Image evidence model
class ImageEvidence(BaseModel):
    """Retrieved image evidence"""
    image_id: str
    paper_title: str
    page_number: int
    caption: Optional[str] = None
    image_type: str = "figure"
    score: float


class AnalysisEvent(Event):
    """
    Event to trigger analysis and synthesis
    
    🆕 Now includes images alongside text chunks
    """
    intent_type: IntentType
    chunks: List[EvidenceChunk]
    images: List[ImageEvidence] = Field(default_factory=list)  # 🆕 Image results
    coverage_stats: Dict[str, Any]
    confidence_threshold: float
    original_question: str


class HumanReviewEvent(Event):
    """Event to request human intervention"""
    reason: str
    chunks: Optional[List[EvidenceChunk]] = None
    images: Optional[List[ImageEvidence]] = None  # 🆕 Include images in review
    missing_papers: List[str] = Field(default_factory=list)
    conflicting_claims: Optional[str] = None
    suggested_actions: List[str] = Field(default_factory=list)


class StopEvent(Event):
    """
    Final event with answer or refusal
    
    🆕 Now includes related images
    """
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)  # 🆕 Related images
    confidence_score: float = 0.0
    refused: bool = False
    refusal_reason: Optional[str] = None
    intent_type: Optional[IntentType] = None