"""
Human-in-the-Loop (HITL) Gate

Deterministic confidence check that blocks answer generation
when retrieval quality is insufficient.

NO ML, NO LLM - pure rule-based logic.
"""

from dataclasses import dataclass
from typing import List, Optional
from app.models.chunk import SearchResult


# HITL Trigger Thresholds (MANDATORY)
MIN_CHUNKS_REQUIRED = 2
MIN_INTENT_CONFIDENCE = 0.6


@dataclass
class HITLDecision:
    """Result of HITL gate evaluation"""
    should_proceed: bool
    requires_human_review: bool
    reason: Optional[str]
    intent: str
    intent_confidence: float
    retrieved_chunks_count: int
    paper_coverage: int


def evaluate_hitl_gate(
    intent: str,
    intent_confidence: float,
    retrieved_chunks: List[SearchResult]
) -> HITLDecision:
    """
    Evaluate whether to proceed to answer generation or require human review.
    
    HITL is triggered if ANY of:
    - retrieved_chunks_count < 2
    - intent_confidence < 0.6
    - paper_coverage == 0
    
    Args:
        intent: Detected intent name
        intent_confidence: Confidence score from intent classifier
        retrieved_chunks: List of SearchResult from Qdrant
        
    Returns:
        HITLDecision with proceed/block decision and reason
    """
    chunks_count = len(retrieved_chunks)
    paper_coverage = len(
        {chunk.metadata.paper_id for chunk in retrieved_chunks if chunk.metadata.paper_id}
    )
    
    reasons = []
    
    # Check trigger conditions
    if chunks_count < MIN_CHUNKS_REQUIRED:
        reasons.append(f"Insufficient evidence: only {chunks_count} chunks retrieved (minimum: {MIN_CHUNKS_REQUIRED})")
    
    if intent_confidence < MIN_INTENT_CONFIDENCE:
        reasons.append(f"Low intent confidence: {intent_confidence:.2f} (minimum: {MIN_INTENT_CONFIDENCE})")
    
    if paper_coverage == 0:
        reasons.append("No paper coverage in retrieved evidence")
    
    requires_review = len(reasons) > 0
    
    return HITLDecision(
        should_proceed=not requires_review,
        requires_human_review=requires_review,
        reason="; ".join(reasons) if reasons else None,
        intent=intent,
        intent_confidence=intent_confidence,
        retrieved_chunks_count=chunks_count,
        paper_coverage=paper_coverage
    )


def format_hitl_response(decision: HITLDecision) -> dict:
    """
    Format HITL decision as structured response for API.
    
    Returns:
        Dict with status, reason, and suggestion
    """
    if decision.should_proceed:
        return {
            "status": "proceed",
            "intent": decision.intent,
            "retrieved_chunks": decision.retrieved_chunks_count,
            "paper_coverage": decision.paper_coverage
        }
    
    return {
        "status": "human_review_required",
        "reason": decision.reason,
        "intent": decision.intent,
        "intent_confidence": decision.intent_confidence,
        "retrieved_chunks": decision.retrieved_chunks_count,
        "paper_coverage": decision.paper_coverage,
        "suggestion": "Please rephrase your question or confirm if you want to proceed with limited evidence."
    }


if __name__ == "__main__":
    # Test the HITL gate
    from app.models.chunk import ChunkMetadata
    
    print("\n" + "="*70)
    print("  HITL GATE TEST")
    print("="*70)
    
    # Mock search results
    def make_result(paper_id: str, score: float) -> SearchResult:
        return SearchResult(
            text="Sample chunk",
            score=score,
            metadata=ChunkMetadata(
                paper_id=paper_id,
                paper_title="Test Paper",
                section_title="Methods",
                page_start=1,
                page_end=2
            )
        )
    
    # Test cases
    test_cases = [
        ("Good coverage (RRF scores)", "methodology", 1.0, [make_result("p1", 0.016), make_result("p1", 0.015), make_result("p2", 0.008)]),
        ("Low chunks", "methodology", 1.0, [make_result("p1", 0.8)]),
        ("Low intent conf", "general", 0.5, [make_result("p1", 0.8), make_result("p2", 0.7)]),
        ("No paper coverage", "methodology", 1.0, [make_result("", 0.8), make_result("", 0.7)]),
        ("No chunks", "methodology", 1.0, []),
    ]
    
    for name, intent, conf, chunks in test_cases:
        decision = evaluate_hitl_gate(intent, conf, chunks)
        
        status = "✅ PROCEED" if decision.should_proceed else "⚠️ REVIEW"
        print(f"\n  {name}: {status}")
        print(f"    Intent: {intent}, Conf: {conf}, Chunks: {len(chunks)}, Papers: {decision.paper_coverage}")
        if decision.reason:
            print(f"    Reason: {decision.reason}")
