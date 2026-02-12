"""
Agent 3: Analysis & Synthesis Agent

The Reasoning Layer

Consumes: analysis_event
Emits: stop_event OR human_review_event

Adapts behavior based on intent:
- summary (with verbosity control)
- comparison
- research_gaps
"""

from app.models.events import (
    AnalysisEvent, StopEvent, HumanReviewEvent, IntentType
)
from app.services.llm_service import get_llm
from langfuse.decorators import observe


class AnalysisSynthesisAgent:
    """
    Performs all reasoning and synthesis
    
    Does NOT:
    - Retrieve new documents
    - Use external knowledge
    - Make uncited claims
    """
    
    def __init__(self):
        self.llm = get_llm()
        print("🧪 Analysis & Synthesis Agent initialized")
    
    @observe(name="Agent_AnalysisSynthesis")
    def process(self, event: AnalysisEvent) -> StopEvent | HumanReviewEvent:
        """
        Analyze evidence and synthesize answer
        
        Returns:
            StopEvent if confident answer
            HumanReviewEvent if uncertain/conflicting
        """
        
        print(f"\n🧪 Analysis & Synthesis:")
        print(f"   Intent: {event.intent_type.value}")
        print(f"   Chunks: {len(event.chunks)}")
        
        # Detect verbosity hint for summary intent
        brief_mode = self._is_brief_summary_requested(event.original_question)
        
        # Route to appropriate synthesis method based on intent
        if event.intent_type == IntentType.SUMMARY:
            result = self._synthesize_summary(event, brief_mode=brief_mode)
        elif event.intent_type == IntentType.COMPARISON:
            result = self._compare_papers(event)
        elif event.intent_type == IntentType.RESEARCH_GAPS:
            result = self._identify_gaps(event)
        else:
            result = self._synthesize_summary(event, brief_mode=False)  # Default
        
        # Check confidence
        if result["confidence"] >= event.confidence_threshold:
            print(f"   ✅ Confident answer (confidence: {result['confidence']:.2f})")
            
            return StopEvent(
                answer=result["answer"],
                citations=result["citations"],
                confidence_score=result["confidence"],
                refused=False,
                intent_type=event.intent_type
            )
        else:
            print(f"   ⚠️  Low confidence ({result['confidence']:.2f}) - requesting review")
            
            return HumanReviewEvent(
                reason=f"Low confidence answer (score: {result['confidence']:.2f})",
                chunks=event.chunks,
                conflicting_claims=result.get("conflicts"),
                suggested_actions=[
                    "Approve answer with disclaimer",
                    "Request more evidence",
                    "Refine question"
                ]
            )
    
    def _is_brief_summary_requested(self, question: str) -> bool:
        """
        Detect if user requested a brief/short summary.
        
        Rule-based, NO LLM - just keyword detection.
        """
        if not question:
            return False
        
        question_lower = question.lower()
        
        # Brevity hint keywords
        brevity_hints = [
            "small", "short", "brief", "quick", "concise",
            "simple", "tldr", "tl;dr", "in short", "briefly"
        ]
        
        return any(hint in question_lower for hint in brevity_hints)
    
    @observe(name="Synthesize_Summary")
    def _synthesize_summary(self, event: AnalysisEvent, brief_mode: bool = False) -> dict:
        """
        Synthesize summary with verbosity control.
        
        Args:
            event: AnalysisEvent with chunks and question
            brief_mode: If True, produce concise output (bullet points)
        """
        # Build context from chunks
        context = self._build_context(event.chunks)
        
        if brief_mode:
            # Brief mode - concise output
            prompt = f"""You are summarizing research papers CONCISELY.

CONTEXT FROM PAPERS:
{context}

QUESTION: {event.original_question}

RULES:
1. Keep your answer SHORT - 3-5 bullet points maximum
2. Use simple, direct language
3. Cite the paper title for key claims
4. Focus only on the most important points
5. No lengthy explanations

Provide a BRIEF summary:"""
        else:
            # Standard mode - full summary
            prompt = f"""You are analyzing research papers to answer a question.

CONTEXT FROM PAPERS:
{context}

QUESTION: {event.original_question}

RULES:
1. Extract information ONLY from the provided context
2. Cite the paper title for every claim
3. If information is not in context, say "Not found in provided papers"
4. Be specific and thorough
5. Format citations as [Paper Title, Page X]

Provide your answer:"""
        
        answer = self.llm.complete(prompt).text
        
        # Extract citations
        citations = self._extract_citations(event.chunks)
        
        # Estimate confidence
        confidence = self._estimate_confidence(answer, event.chunks)
        
        if brief_mode:
            print(f"   📝 Brief mode: concise summary generated")
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }
    
    def _extract_facts(self, event: AnalysisEvent) -> dict:
        """Extract factual information from evidence (legacy method)"""
        return self._synthesize_summary(event, brief_mode=False)
    
    @observe(name="Compare_Papers")
    def _compare_papers(self, event: AnalysisEvent) -> dict:
        """Compare approaches across papers"""
        
        context = self._build_context(event.chunks)
        
        prompt = f"""You are comparing research papers.

CONTEXT FROM PAPERS:
{context}

QUESTION: {event.original_question}

RULES:
1. Compare ONLY what is stated in the context
2. Create a structured comparison
3. Cite papers for each point
4. Highlight differences AND similarities
5. If papers don't address the comparison point, state that

Provide your comparison:"""
        
        answer = self.llm.complete(prompt).text
        citations = self._extract_citations(event.chunks)
        confidence = self._estimate_confidence(answer, event.chunks)
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }
    
    @observe(name="Identify_Gaps")
    def _identify_gaps(self, event: AnalysisEvent) -> dict:
        """Identify research gaps and limitations"""
        
        context = self._build_context(event.chunks)
        
        prompt = f"""You are identifying research gaps and limitations.

CONTEXT FROM PAPERS:
{context}

QUESTION: {event.original_question}

RULES:
1. Identify gaps ONLY from explicit statements in papers
2. Look for: limitations sections, future work, challenges mentioned
3. Cite which paper mentions each gap
4. Do NOT invent gaps - only report what papers state
5. Organize by theme if multiple gaps found

Provide your analysis:"""
        
        answer = self.llm.complete(prompt).text
        citations = self._extract_citations(event.chunks)
        confidence = self._estimate_confidence(answer, event.chunks)
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }
    
    def _build_context(self, chunks) -> str:
        """Build context string from evidence chunks"""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] From '{chunk.paper_title}' (Section: {chunk.section_title}, "
                f"Pages {chunk.page_start}-{chunk.page_end}):\n{chunk.text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_citations(self, chunks) -> list:
        """Extract citation metadata from chunks"""
        
        citations = []
        seen = set()
        
        for chunk in chunks:
            key = f"{chunk.paper_title}_{chunk.page_start}"
            if key not in seen:
                citations.append({
                    "paper_title": chunk.paper_title,
                    "section": chunk.section_title,
                    "pages": f"{chunk.page_start}-{chunk.page_end}",
                    "score": chunk.score
                })
                seen.add(key)
        
        return citations
    
    def _estimate_confidence(self, answer: str, chunks) -> float:
        """
        Estimate confidence in answer
        
        Simple heuristic:
        - High if answer is detailed and references papers
        - Medium if answer is brief
        - Low if answer says "not found" or is uncertain
        """
        
        answer_lower = answer.lower()
        
        # Low confidence indicators
        if any(phrase in answer_lower for phrase in [
            "not found", "unclear", "uncertain", "cannot determine"
        ]):
            return 0.3
        
        # Medium confidence
        if len(answer) < 200:
            return 0.6
        
        # High confidence - detailed answer with likely citations
        if any(paper.paper_title.lower() in answer_lower for paper in chunks[:3]):
            return 0.85
        
        return 0.7  # Default