
import sys
import os
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.getcwd())

from app.services.query_engine import get_query_engine

def debug_query():
    print("🚀 Initializing Query Engine for Debugging...")
    try:
        engine = get_query_engine()
        print("✅ Engine initialized")
        
        question = "what is lora"
        print(f"❓ Asking: {question}")
        
        # Enable verbose logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        response = engine.query(question, similarity_top_k=2)
        print(f"✅ Success! Response: {response}")
        
    except Exception as e:
        print("\n❌ ERROR DETECTED:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_query()
