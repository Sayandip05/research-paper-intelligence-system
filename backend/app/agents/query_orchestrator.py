"""
Agent 1: Query Orchestrator Agent

The Brain - Not a worker

Consumes: start_event
Emits: retrieval_event

NEVER retrieves documents
NEVER answers questions
"""

from app.models.events import StartEvent, RetrievalEvent, IntentType
from app.services.llm_service import get_llm
from langfuse.decorators import observe
from typing import Optional


class QueryOrchestratorAgent:
    """
    Understands user intent and plans retrieval strategy
    
    Does NOT:
    - Retrieve documents
    - Answer questions
    - Make citations
    """
    
    def __init__(self):
        self.llm = get_llm()
        print("🧠 Query Orchestrator Agent initialized")
    
    @observe(name="Agent_QueryOrchestrator")
    def process(self, event: StartEvent) -> RetrievalEvent:
        """
        Classify intent and emit retrieval instructions
        
        Args:
            event: StartEvent with user question
        
        Returns:
            RetrievalEvent with retrieval strategy
        """
        question = event.question
        
        # Classify intent using LLM
        intent_type = self._classify_intent(question)
        
        # Determine target sections based on intent
        target_sections = self._determine_sections(intent_type, question)
        
        # Set confidence threshold
        confidence_threshold = self._get_confidence_threshold(intent_type)
        
        # Decide if human review might be needed
        human_review_hint = self._predict_human_review_needed(question)
        
        print(f"\n🧠 Query Orchestrator Analysis:")
        print(f"   Intent: {intent_type.value}")
        print(f"   Target sections: {target_sections}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Human review hint: {human_review_hint}")
        
        return RetrievalEvent(
            intent_type=intent_type,
            target_sections=target_sections,
            confidence_threshold=confidence_threshold,
            human_review_hint=human_review_hint,
            similarity_top_k=5,
            original_question=question
        )
    
    @observe(name="Intent_Classification")
    def _classify_intent(self, question: str) -> IntentType:
        """
        Classify question intent
        
        Types:
        - summary: "What is the paper about?", "Explain the methodology"
        - comparison: "Compare LoRA vs QLoRA"
        - research_gaps: "What are the limitations?"
        """
        
        # Use LLM for classification
        prompt = f"""Classify this research question into ONE category:

Question: "{question}"

Categories:
1. summary - asking for explanations, overviews, facts, datasets, methods, results
2. comparison - comparing multiple papers or approaches
3. research_gaps - asking about limitations, future work, open problems

Respond with ONLY the category name, nothing else."""
        
        response = self.llm.complete(prompt).text.strip().lower()
        
        # Map to enum
        if "comparison" in response:
            return IntentType.COMPARISON
        elif "gap" in response or "limitation" in response:
            return IntentType.RESEARCH_GAPS
        else:
            return IntentType.SUMMARY
    
    def _determine_sections(self, intent: IntentType, question: str) -> list:
        """Determine which paper sections to target"""
        
        # Default sections by intent
        section_map = {
            IntentType.SUMMARY: ["Abstract", "Introduction", "Methods", "Results"],
            IntentType.COMPARISON: ["Results", "Methods", "Experiments"],
            IntentType.RESEARCH_GAPS: ["Discussion", "Conclusion", "Limitations", "Future Work"]
        }
        
        return section_map.get(intent, ["Abstract", "Introduction"])
    
    def _get_confidence_threshold(self, intent: IntentType) -> float:
        """Set confidence threshold by intent"""
        
        thresholds = {
            IntentType.SUMMARY: 0.5,           # Standard threshold for summaries
            IntentType.COMPARISON: 0.5,        # Medium confidence OK
            IntentType.RESEARCH_GAPS: 0.4      # Lower threshold for exploratory
        }
        
        return thresholds.get(intent, 0.5)
    
    def _predict_human_review_needed(self, question: str) -> bool:
        """
        Predict if human review might be needed
        
        Heuristics:
        - Very specific questions might need review
        - Broad questions usually don't
        """
        
        # Simple heuristic: questions with "compare" or "all" might need review
        keywords = ["compare", "all", "every", "comprehensive"]
        return any(kw in question.lower() for kw in keywords)