"""
Event Models for Workflow System

Only 5 events allowed:
- start_event
- retrieval_event
- analysis_event
- human_review_event
- stop_event
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from llama_index.core.workflow.events import Event


class IntentType(str, Enum):
    """Query intent classification"""
    FACTUAL_EXTRACTION = "factual_extraction"
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
    """Retrieved evidence with metadata"""
    text: str
    paper_title: str
    section_title: str
    page_start: int
    page_end: int
    score: float


class AnalysisEvent(Event):
    """Event to trigger analysis and synthesis"""
    intent_type: IntentType
    chunks: List[EvidenceChunk]
    coverage_stats: Dict[str, Any]
    confidence_threshold: float
    original_question: str


class HumanReviewEvent(Event):
    """Event to request human intervention"""
    reason: str
    chunks: Optional[List[EvidenceChunk]] = None
    missing_papers: List[str] = Field(default_factory=list)
    conflicting_claims: Optional[str] = None
    suggested_actions: List[str] = Field(default_factory=list)


class StopEvent(Event):
    """Final event with answer or refusal"""
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    refused: bool = False
    refusal_reason: Optional[str] = None
    intent_type: Optional[IntentType] = None