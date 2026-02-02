"""
🆕 Agent 2: Multimodal Evidence Retrieval Agent

Combines:
- Text: Dense (BGE) + Sparse (BM42) → Hybrid search
- Images: CLIP embeddings → Visual search

Returns BOTH text chunks + related images
"""

from app.models.events import (
    RetrievalEvent, AnalysisEvent, HumanReviewEvent, 
    EvidenceChunk, ImageEvidence
)
from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
from app.services.clip_embedding import get_clip_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings
from typing import Union

settings = get_settings()


class EvidenceRetrievalAgent:
    """
    🆕 Multimodal retrieval: Text (hybrid) + Images (CLIP)
    
    Does NOT:
    - Summarize
    - Interpret
    - Hallucinate
    """
    
    def __init__(self):
        self.qdrant = QdrantService()
        self.dense_embeddings = get_embedding_service()
        
        # Sparse embeddings
        self.sparse_embeddings = None
        if settings.enable_hybrid_search:
            self.sparse_embeddings = get_sparse_embedding_service()
        
        # 🆕 CLIP embeddings for images
        self.clip_embeddings = None
        if settings.enable_multimodal:
            try:
                self.clip_embeddings = get_clip_embedding_service()
                print("🔍 Evidence Retrieval Agent initialized (HYBRID + MULTIMODAL)")
            except Exception as e:
                print(f"⚠️  CLIP not available: {e}")
                print("🔍 Evidence Retrieval Agent initialized (HYBRID only)")
        else:
            print("🔍 Evidence Retrieval Agent initialized (HYBRID only)")
    
    def process(self, event: RetrievalEvent) -> Union[AnalysisEvent, HumanReviewEvent]:
        """
        🆕 Retrieve BOTH text and images
        
        Returns:
            AnalysisEvent with text chunks + images
            HumanReviewEvent if evidence weak
        """
        
        print(f"\n🔍 Multimodal Evidence Retrieval:")
        print(f"   Query: {event.original_question}")
        print(f"   Intent: {event.intent_type.value}")
        print(f"   Top-K: {event.similarity_top_k}")
        
        # ========== TEXT RETRIEVAL (HYBRID) ==========
        # Generate DENSE query embedding
        dense_query = self.dense_embeddings.generate_embedding(event.original_question)
        
        # Generate SPARSE query embedding
        sparse_query = None
        if settings.enable_hybrid_search and self.sparse_embeddings:
            sparse_query = self.sparse_embeddings.generate_sparse_embedding(
                event.original_question
            )
        
        # Search Qdrant with HYBRID
        text_results = self.qdrant.search_with_filter(
            query_vector=dense_query,
            limit=event.similarity_top_k,
            allowed_sections=event.target_sections if event.target_sections else None,
            query_sparse_vector=sparse_query
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
            for r in text_results
        ]
        
        print(f"   📝 Retrieved: {len(chunks)} text chunks")
        
        # ========== IMAGE RETRIEVAL (CLIP) ==========
        images = []
        if settings.enable_multimodal and self.clip_embeddings:
            try:
                # Generate CLIP text embedding
                clip_query = self.clip_embeddings.generate_text_embedding(
                    event.original_question
                )
                
                # Search image collection (top-3 images)
                image_results = self.qdrant.search_images(
                    query_vector=clip_query,
                    limit=3,  # Always fetch top-3 images
                    min_score=0.15  # Lower threshold for better recall
                )
                
                # Convert to ImageEvidence
                images = [
                    ImageEvidence(
                        image_id=img.image_id,
                        paper_title=img.paper_title,
                        page_number=img.page_number,
                        caption=img.caption,
                        image_type=img.metadata.image_type,
                        score=img.score
                    )
                    for img in image_results
                ]
                
                print(f"   🖼️  Retrieved: {len(images)} related images")
                
            except Exception as e:
                print(f"   ⚠️  Image search failed: {e}")
                images = []
        
        # ========== COVERAGE STATS ==========
        coverage_stats = self._calculate_coverage(chunks, images)
        
        print(f"   Coverage: {coverage_stats['unique_papers']} papers")
        print(f"   Avg text score: {coverage_stats['avg_text_score']:.3f}")
        if images:
            print(f"   Avg image score: {coverage_stats.get('avg_image_score', 0):.3f}")
        
        # ========== DECISION: SUFFICIENT? ==========
        if self._is_evidence_sufficient(chunks, event.confidence_threshold, coverage_stats):
            # Branch A: Proceed to analysis
            print("   ✅ Evidence sufficient - proceeding to analysis")
            
            return AnalysisEvent(
                intent_type=event.intent_type,
                chunks=chunks,
                images=images,  # 🆕 Include images
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
                images=images,  # 🆕 Include images in review
                missing_papers=[],
                suggested_actions=[
                    "Refine query for better results",
                    "Add more papers to corpus",
                    "Approve proceeding with available evidence"
                ]
            )
    
    def _calculate_coverage(
        self,
        chunks: list[EvidenceChunk],
        images: list[ImageEvidence]
    ) -> dict:
        """
        🆕 Calculate coverage for BOTH text and images
        """
        
        if not chunks:
            return {
                "unique_papers": 0,
                "unique_sections": 0,
                "avg_text_score": 0.0,
                "avg_image_score": 0.0,
                "total_evidence": 0
            }
        
        # Text stats
        papers = set(c.paper_title for c in chunks)
        sections = set(c.section_title for c in chunks)
        text_scores = [c.score for c in chunks]
        
        # Image stats
        image_scores = [img.score for img in images] if images else []
        
        return {
            "unique_papers": len(papers),
            "unique_sections": len(sections),
            "avg_text_score": sum(text_scores) / len(text_scores),
            "avg_image_score": sum(image_scores) / len(image_scores) if image_scores else 0.0,
            "min_text_score": min(text_scores),
            "max_text_score": max(text_scores),
            "total_evidence": len(chunks) + len(images)
        }
    
    def _is_evidence_sufficient(
        self,
        chunks: list[EvidenceChunk],
        threshold: float,
        coverage: dict
    ) -> bool:
        """
        Decide if evidence is sufficient
        
        Criteria:
        - At least 3 text chunks OR 1 text + 1 image
        - Average text score above threshold
        - At least 1 paper found
        """
        
        # Need at least some text evidence
        if len(chunks) < 2:
            return False
        
        # Check text quality
        if coverage["avg_text_score"] < threshold:
            return False
        
        # Need at least one paper
        if coverage["unique_papers"] < 1:
            return False
        
        return True