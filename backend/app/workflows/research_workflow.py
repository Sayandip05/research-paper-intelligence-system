"""
Research Workflow - Native LlamaIndex Workflow

Uses LlamaIndex Workflow class with @step decorators
Event-driven, non-sequential, human-controllable
"""

from llama_index.core.workflow import (
    Workflow, StartEvent, StopEvent, step
)
from llama_index.core.workflow.events import Event

from app.models.events import (
    RetrievalEvent, AnalysisEvent, HumanReviewEvent, IntentType,
    StopEvent as InternalStopEvent
)
from app.agents.query_orchestrator import QueryOrchestratorAgent
from app.agents.evidence_retrieval import EvidenceRetrievalAgent
from app.agents.analysis_synthesis import AnalysisSynthesisAgent
from langfuse.decorators import observe
from typing import Optional
import uuid


class ResearchWorkflow(Workflow):
    """
    Native LlamaIndex Workflow for research paper analysis
    
    Uses @step decorators for event-driven execution
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize agents
        self.orchestrator = QueryOrchestratorAgent()
        self.retriever = EvidenceRetrievalAgent()
        self.analyzer = AnalysisSynthesisAgent()
        
        print("\n" + "="*70)
        print("  🔬 LLAMAINDEX WORKFLOW INITIALIZED")
        print("="*70)
        print("  Framework: Native LlamaIndex Workflow")
        print("  Agents: Query Orchestrator, Evidence Retrieval, Analysis")
        print("  Pattern: Event-driven with @step decorators")
        print("="*70 + "\n")
    
    @step
    @observe(name="Workflow_Step1_Orchestrate")
    async def orchestrate_query(self, ev: StartEvent) -> RetrievalEvent:
        """
        Step 1: Query Orchestrator
        
        Consumes: StartEvent (from user)
        Emits: RetrievalEvent
        """
        print("━"*70)
        print("STEP 1: QUERY ORCHESTRATOR (LlamaIndex @step)")
        print("━"*70)
        
        session_id = ev.get("session_id") or str(uuid.uuid4())
        
        # Create internal start event
        from app.models.events import StartEvent as InternalStart
        start = InternalStart(
            question=ev.get("question"),
            session_id=session_id,
            human_constraints=ev.get("human_constraints")
        )
        
        # Process with orchestrator agent
        retrieval_event = self.orchestrator.process(start)
        
        return retrieval_event
    
    @step
    @observe(name="Workflow_Step2_Retrieve")
    async def retrieve_evidence(self, ev: RetrievalEvent) -> AnalysisEvent | HumanReviewEvent:
        """
        Step 2: Evidence Retrieval
        
        Consumes: RetrievalEvent
        Emits: AnalysisEvent OR HumanReviewEvent
        """
        print("\n" + "━"*70)
        print("STEP 2: EVIDENCE RETRIEVAL (LlamaIndex @step)")
        print("━"*70)
        
        # Process with retriever agent
        result = self.retriever.process(ev)
        
        return result
    
    @step
    @observe(name="Workflow_Step3_Analyze")
    async def analyze_and_synthesize(self, ev: AnalysisEvent) -> StopEvent | HumanReviewEvent:
        """
        Step 3: Analysis & Synthesis
        
        Consumes: AnalysisEvent
        Emits: StopEvent OR HumanReviewEvent
        """
        print("\n" + "━"*70)
        print("STEP 3: ANALYSIS & SYNTHESIS (LlamaIndex @step)")
        print("━"*70)
        
        # Process with analyzer agent
        result = self.analyzer.process(ev)
        
        # Convert to LlamaIndex StopEvent if final
        if isinstance(result, InternalStopEvent):
            return StopEvent(result={
                "answer": result.answer,
                "citations": result.citations,
                "confidence": result.confidence_score,
                "refused": result.refused,
                "refusal_reason": result.refusal_reason,
                "intent_type": result.intent_type.value if result.intent_type else None
            })
        
        return result
    
    @step
    @observe(name="Workflow_Step4_HumanReview")
    async def handle_human_review(self, ev: HumanReviewEvent) -> StopEvent:
        """
        Handle human review requests
        
        In production: pause and wait for human
        For demo: auto-approve
        """
        print(f"\n⚠️  HUMAN REVIEW REQUESTED")
        print(f"   Reason: {ev.reason}")
        print(f"   Auto-approving for demo...")
        
        # Auto-approve with limited evidence
        if ev.chunks:
            return StopEvent(result={
                "answer": f"[Limited evidence] {ev.chunks[0].text[:500]}...",
                "citations": [
                    {"paper_title": c.paper_title, "pages": f"{c.page_start}-{c.page_end}"}
                    for c in ev.chunks[:3]
                ],
                "confidence": 0.4,
                "refused": False,
                "human_review_note": ev.reason
            })
        else:
            return StopEvent(result={
                "answer": None,
                "citations": [],
                "confidence": 0.0,
                "refused": True,
                "refusal_reason": ev.reason
            })


# Global instance
_workflow = None

def get_workflow() -> ResearchWorkflow:
    """Get or create workflow instance"""
    global _workflow
    if _workflow is None:
        _workflow = ResearchWorkflow()
    return _workflow


# Convenience function for simple execution
@observe(name="Research_Workflow_Execute")
async def execute_workflow(question: str, session_id: Optional[str] = None) -> dict:
    """Execute workflow with simple interface"""
    workflow = get_workflow()
    
    result = await workflow.run(
        question=question,
        session_id=session_id
    )
    
    return result