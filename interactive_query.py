#!/usr/bin/env python3
"""
Interactive Query - Ask your own questions!
Uses Hybrid RAG Workflow System
"""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.workflows.research_workflow import get_workflow


async def main():
    print("\n" + "="*70)
    print("  🤖 RESEARCH PAPER Q&A - Interactive Mode (Hybrid RAG)")
    print("="*70)
    print("\n🔧 Initializing 3-agent workflow system...")
    
    workflow = get_workflow()
    
    print("✅ Ready! Type 'quit' to exit.\n")
    
    session_id = None
    
    while True:
        # Get user input
        question = input("💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Executing workflow (3 agents)...")
        
        try:
            # Execute workflow
            result = await workflow.run(
                question=question,
                session_id=session_id
            )
            
            # Display answer
            print("\n" + "─"*70)
            print("💡 ANSWER:")
            print("─"*70)
            
            if result.get("refused"):
                print(f"❌ Cannot answer: {result.get('refusal_reason')}")
            else:
                print(result.get("answer", "No answer generated"))
            
            # Display metadata
            print(f"\n📊 METADATA:")
            print(f"   Intent: {result.get('intent_type', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            
            # Display citations
            citations = result.get("citations", [])
            if citations:
                print(f"\n📚 CITATIONS ({len(citations)}):")
                for i, cite in enumerate(citations[:5], 1):
                    print(f"   {i}. {cite.get('paper_title', 'Unknown')}")
                    print(f"      Pages: {cite.get('pages', 'N/A')}")
            
            print("\n" + "="*70 + "\n")
        
        except Exception as e:
            print(f"❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")