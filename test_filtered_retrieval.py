"""
Test Intent-Based Filtered Retrieval

Demonstrates the full pipeline:
1. User question → Intent detection
2. Intent → Allowed sections
3. Sections → Qdrant filter
4. Filtered vector search
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.intent_classifier import get_intent_classifier
from app.services.embeddings import get_embedding_service
from app.db.qdrant_client import QdrantService


def test_filtered_retrieval(query: str):
    """Test the full intent-aware retrieval pipeline"""
    print(f"\n{'='*70}")
    print(f"  QUERY: {query}")
    print('='*70)
    
    # Step 1: Intent classification
    classifier = get_intent_classifier()
    intent_result = classifier.classify(query)
    
    print(f"\n📌 INTENT DETECTION")
    print(f"   Intent: {intent_result.intent}")
    print(f"   Allowed: {intent_result.allowed_sections}")
    print(f"   Confidence: {intent_result.confidence}")
    
    # Step 2: Get query embedding
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.generate_embeddings([query])[0]
    
    # Step 3: Filtered search
    qdrant = QdrantService()
    results = qdrant.search_with_filter(
        query_vector=query_embedding,
        limit=5,
        allowed_sections=intent_result.allowed_sections
    )
    
    print(f"\n📚 RETRIEVED CHUNKS")
    for i, result in enumerate(results, 1):
        print(f"\n   [{i}] Section: {result.metadata.section_title}")
        print(f"       Paper: {result.metadata.paper_title}")
        print(f"       Score: {result.score:.3f}")
        print(f"       Text: {result.text[:150]}...")


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What are the limitations of LoRA?",
        "How does the method work?",
        "What experiments were conducted?",
    ]
    
    print("\n" + "="*70)
    print("  INTENT-BASED FILTERED RETRIEVAL TEST")
    print("="*70)
    
    for query in test_queries:
        test_filtered_retrieval(query)
    
    print("\n" + "="*70)
    print("  TEST COMPLETE")
    print("="*70 + "\n")
