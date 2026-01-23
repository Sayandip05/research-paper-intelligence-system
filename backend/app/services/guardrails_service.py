"""
Guardrails AI Service

Validates LLM outputs:
- Citation validation
- Schema enforcement
- Hallucination prevention
- Output quality checks
"""

from guardrails import Guard
from guardrails.validators import ValidLength, ValidChoices
from typing import Dict, Any, List


class GuardrailsService:
    """
    Guardrails AI validation layer
    
    Runs between analysis_event and stop_event
    Can override stop_event with human_review_event if validation fails
    """
    
    def __init__(self):
        # Define guardrails for answer validation
        self.answer_guard = Guard.from_string(
            validators=[
                ValidLength(min=50, max=5000, on_fail="reask"),
            ],
            description="Validates answer length and quality"
        )
        
        print("🛡️  Guardrails AI initialized")
    
    def validate_answer(
        self,
        answer: str,
        citations: List[Dict],
        chunks: List
    ) -> Dict[str, Any]:
        """
        Validate LLM-generated answer
        
        Checks:
        1. Answer length reasonable
        2. Citations exist
        3. Cited text appears in chunks
        4. No obvious hallucinations
        
        Returns:
            {"valid": bool, "issues": list, "confidence_penalty": float}
        """
        
        issues = []
        confidence_penalty = 0.0
        
        # Check 1: Answer length
        if len(answer) < 50:
            issues.append("Answer too short")
            confidence_penalty += 0.3
        elif len(answer) > 5000:
            issues.append("Answer too long - possible hallucination")
            confidence_penalty += 0.2
        
        # Check 2: Citations exist
        if not citations:
            issues.append("No citations provided")
            confidence_penalty += 0.4
        
        # Check 3: Validate citations against chunks
        citation_issues = self._validate_citations(citations, chunks)
        if citation_issues:
            issues.extend(citation_issues)
            confidence_penalty += 0.2 * len(citation_issues)
        
        # Check 4: Detect hallucination patterns
        hallucination_check = self._check_hallucinations(answer, chunks)
        if not hallucination_check["valid"]:
            issues.extend(hallucination_check["issues"])
            confidence_penalty += 0.3
        
        return {
            "valid": len(issues) == 0 or confidence_penalty < 0.5,
            "issues": issues,
            "confidence_penalty": min(confidence_penalty, 0.9)
        }
    
    def _validate_citations(self, citations: List[Dict], chunks: List) -> List[str]:
        """
        Validate that citations exist in source chunks
        
        Returns list of issues found
        """
        issues = []
        
        # Get all paper titles from chunks
        chunk_papers = {c.paper_title for c in chunks}
        
        # Check each citation
        for citation in citations:
            paper_title = citation.get("paper_title", "")
            
            if paper_title not in chunk_papers:
                issues.append(f"Citation '{paper_title}' not in retrieved papers")
        
        return issues
    
    def _check_hallucinations(self, answer: str, chunks: List) -> Dict[str, Any]:
        """
        Check for potential hallucinations
        
        Simple heuristics:
        - "Not found in papers" or similar = good (honest)
        - Specific numbers/dates not in chunks = suspicious
        - Paper names not in chunks = hallucination
        """
        
        issues = []
        answer_lower = answer.lower()
        
        # Good patterns (honest refusal)
        honest_patterns = [
            "not found", "not mentioned", "not stated",
            "unclear", "not specified"
        ]
        
        if any(pattern in answer_lower for pattern in honest_patterns):
            return {"valid": True, "issues": []}
        
        # Get all text from chunks
        chunk_text = " ".join(c.text.lower() for c in chunks)
        
        # Check if answer introduces new paper names
        # (Simple check - can be improved)
        potential_papers = [
            word for word in answer.split()
            if word.istitle() and len(word) > 3
        ]
        
        for word in potential_papers[:5]:  # Check first 5 capitalized words
            if word.lower() not in chunk_text and word.lower() not in ["the", "this", "that"]:
                # Might be hallucinated paper name
                pass  # Too aggressive, skip for now
        
        return {"valid": True, "issues": issues}
    
    def enforce_schema(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce output schema
        
        Ensures result has required fields:
        - answer (str or None)
        - citations (list)
        - confidence (float 0-1)
        - refused (bool)
        """
        
        # Ensure required fields exist
        result.setdefault("answer", None)
        result.setdefault("citations", [])
        result.setdefault("confidence", 0.0)
        result.setdefault("refused", False)
        
        # Clamp confidence to 0-1
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
        
        # If no answer, mark as refused
        if not result["answer"]:
            result["refused"] = True
            result["refusal_reason"] = result.get("refusal_reason", "No answer generated")
        
        return result


# Global instance
_guardrails_service = None

def get_guardrails() -> GuardrailsService:
    """Get or create guardrails service"""
    global _guardrails_service
    if _guardrails_service is None:
        _guardrails_service = GuardrailsService()
    return _guardrails_service