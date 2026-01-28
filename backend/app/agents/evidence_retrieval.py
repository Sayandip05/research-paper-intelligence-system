"""
🆕 Agent 2: Evidence Retrieval Agent with HYBRID SEARCH

Combines dense (semantic) + sparse (keyword) retrieval

Consumes: retrieval_event
Emits: analysis_event OR human_review_event
"""

from app.models.events import (
    RetrievalEvent, AnalysisEvent, HumanReviewEvent, EvidenceChunk
)
from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings
from typing import Union

settings = get_settings()


class EvidenceRetrievalAgent:
    """
    🆕 Retrieves evidence using HYBRID search (dense + sparse)
    
    Does NOT:
    - Summarize
    - Interpret
    - Hallucinate
    - Access memory
    """
    
    def __init__(self):
        self.qdrant = QdrantService()
        self.dense_embeddings = get_embedding_service()
        
        # 🆕 Initialize sparse embeddings if hybrid enabled
        self.sparse_embeddings = None
        if settings.enable_hybrid_search:
            self.sparse_embeddings = get_sparse_embedding_service()
            print("🔍 Evidence Retrieval Agent initialized (HYBRID mode)")
        else:
            print("🔍 Evidence Retrieval Agent initialized (DENSE-only mode)")
    
    def process(self, event: RetrievalEvent) -> Union[AnalysisEvent, HumanReviewEvent]:
        """
        🆕 Retrieve evidence using hybrid search
        
        Returns:
            AnalysisEvent if evidence sufficient
            HumanReviewEvent if evidence weak/incomplete
        """
        
        print(f"\n🔍 Evidence Retrieval:")
        print(f"   Query: {event.original_question}")
        print(f"   Intent: {event.intent_type.value}")
        print(f"   Top-K: {event.similarity_top_k}")
        
        # Generate DENSE query embedding
        dense_query = self.dense_embeddings.generate_embedding(event.original_question)
        
        # 🆕 Generate SPARSE query embedding
        sparse_query = None
        if settings.enable_hybrid_search and self.sparse_embeddings:
            sparse_query = self.sparse_embeddings.generate_sparse_embedding(
                event.original_question
            )
        
        # 🆕 Search Qdrant with HYBRID (or dense-only)
        results = self.qdrant.search_with_filter(
            query_vector=dense_query,
            limit=event.similarity_top_k,
            allowed_sections=event.target_sections if event.target_sections else None,
            query_sparse_vector=sparse_query  # 🆕 Pass sparse vector
        )
        
        # Convert to EvidenceChunk
        chunks = [
            EvidenceChunk(
                text=r.text,
                paper_title=r.metadata.paper_title,
                section_title=r.metadata.section_title,
                page_start=r.metadata.page_start,
                page_end=r.metadata.page_end,
                score=r.score
            )
            for r in results
        ]
        
        # Calculate coverage stats
        coverage_stats = self._calculate_coverage(chunks)
        
        print(f"   Retrieved: {len(chunks)} chunks")
        print(f"   Coverage: {coverage_stats['unique_papers']} papers")
        print(f"   Avg score: {coverage_stats['avg_score']:.3f}")
        
        # Decision: Is evidence sufficient?
        if self._is_evidence_sufficient(chunks, event.confidence_threshold, coverage_stats):
            # Branch A: Proceed to analysis
            print("   ✅ Evidence sufficient - proceeding to analysis")
            
            return AnalysisEvent(
                intent_type=event.intent_type,
                chunks=chunks,
                coverage_stats=coverage_stats,
                confidence_threshold=event.confidence_threshold,
                original_question=event.original_question
            )
        else:
            # Branch B: Request human review
            print("   ⚠️  Evidence insufficient - requesting human review")
            
            return HumanReviewEvent(
                reason="Insufficient evidence coverage or low confidence scores",
                chunks=chunks,
                missing_papers=[],
                suggested_actions=[
                    "Refine query for better results",
                    "Add more papers to corpus",
                    "Approve proceeding with available evidence"
                ]
            )
    
    def _calculate_coverage(self, chunks: list[EvidenceChunk]) -> dict:
        """Calculate evidence coverage statistics"""
        
        if not chunks:
            return {
                "unique_papers": 0,
                "unique_sections": 0,
                "avg_score": 0.0,
                "score_variance": 0.0
            }
        
        papers = set(c.paper_title for c in chunks)
        sections = set(c.section_title for c in chunks)
        scores = [c.score for c in chunks]
        
        return {
            "unique_papers": len(papers),
            "unique_sections": len(sections),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores)
        }
    
    def _is_evidence_sufficient(
        self,
        chunks: list[EvidenceChunk],
        threshold: float,
        coverage: dict
    ) -> bool:
        """
        Decide if evidence is sufficient to proceed
        
        Criteria:
        - At least 3 chunks retrieved
        - Average score above threshold
        - At least 1 paper found
        """
        
        if len(chunks) < 3:
            return False
        
        if coverage["avg_score"] < threshold:
            return False
        
        if coverage["unique_papers"] < 1:
            return False
        
        return True