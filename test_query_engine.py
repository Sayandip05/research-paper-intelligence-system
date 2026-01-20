#!/usr/bin/env python3
"""
Test Week 2 Query Engine

Run this to test the intelligent query system
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.query_engine import get_query_engine


def test_query_engine():
    """Test the query engine with sample questions"""
    
    print("\n" + "="*70)
    print("  WEEK 2: TESTING INTELLIGENT QUERY ENGINE")
    print("="*70)
    
    # Initialize query engine
    print("\n🔧 Initializing query engine...")
    engine = get_query_engine()
    print("✅ Query engine ready!")
    
    # Test questions
    questions = [
        "What is the main contribution of the paper?",
        "What datasets were used for evaluation?",
        "What are the key findings?",
    ]
    
    for i, question in enumerate(questions, 1):
        print("\n" + "-"*70)
        print(f"📝 Question {i}: {question}")
        print("-"*70)
        
        try:
            # Query
            result = engine.query(question, similarity_top_k=3)
            
            # Display answer
            print(f"\n💡 Answer:")
            print(result["answer"])
            
            # Display sources
            print(f"\n📚 Sources Used ({result['num_sources']}):")
            for j, source in enumerate(result["sources"], 1):
                print(f"\n  {j}. {source['paper_title']}")
                print(f"     Section: {source['section_title']}")
                print(f"     Pages: {source['page_start']}-{source['page_end']}")
                print(f"     Relevance: {source['score']:.3f}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*70)
    print("  ✅ WEEK 2 TESTING COMPLETE!")
    print("="*70)
    print("\n🚀 Next: Start API and try via Swagger UI")
    print("   1. cd backend")
    print("   2. uvicorn app.main:app --reload")
    print("   3. Visit: http://localhost:8000/docs")
    print("   4. Try POST /api/query/query")
    print()


if __name__ == "__main__":
    try:
        test_query_engine()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()