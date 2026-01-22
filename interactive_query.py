#!/usr/bin/env python3
"""
Interactive Query - Ask your own questions!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.query_engine import get_query_engine


def main():
    print("\n" + "="*70)
    print("  🤖 RESEARCH PAPER Q&A - Interactive Mode")
    print("="*70)
    print("\n🔧 Initializing query engine...")
    
    engine = get_query_engine()
    
    print("✅ Ready! Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        question = input("💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Searching...")
        
        try:
            # Query
            result = engine.query(question, similarity_top_k=5)
            
            # Display answer
            print("\n" + "─"*70)
            print("💡 ANSWER:")
            print("─"*70)
            print(result["answer"])
            
            # Display sources
            print(f"\n📚 SOURCES ({result['num_sources']}):")
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. {source['paper_title']}")
                print(f"     Pages {source['page_start']}-{source['page_end']}, Score: {source['score']:.3f}")
            
            print("\n" + "="*70 + "\n")
        
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")