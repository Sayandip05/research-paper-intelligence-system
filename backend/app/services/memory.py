"""
Memory System - NOT an Agent

Stores:
- Session state
- Last intent
- Papers used
- Confidence history

Read: Only at start_event
Write: Only after stop_event (with guardrails approval)
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


class MemorySystem:
    """
    Deterministic, auditable memory storage
    
    NOT an agent - just a database wrapper
    """
    
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017"):
        self.client = MongoClient(mongodb_url)
        self.db = self.client["research_papers"]
        self.sessions = self.db["sessions"]
        self.history = self.db["query_history"]
        
        print("💾 Memory system initialized")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create new session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session = {
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "last_intent": None,
            "last_topic": None,
            "papers_used": [],
            "confidence_history": [],
            "queries": []
        }
        
        self.sessions.insert_one(session)
        return session_id
    
    def read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read session state (only at start_event)"""
        return self.sessions.find_one({"session_id": session_id})
    
    def write_session(
        self,
        session_id: str,
        intent: str,
        topic: str,
        papers: List[str],
        confidence: float,
        question: str,
        answer: str
    ):
        """Write session state (only after stop_event)"""
        
        # Update session
        self.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "last_intent": intent,
                    "last_topic": topic,
                    "updated_at": datetime.utcnow()
                },
                "$push": {
                    "papers_used": {"$each": papers},
                    "confidence_history": confidence,
                    "queries": {
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.utcnow()
                    }
                }
            }
        )
        
        # Also log to history
        self.history.insert_one({
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "intent": intent,
            "confidence": confidence,
            "papers": papers,
            "timestamp": datetime.utcnow()
        })
    
    def get_query_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent queries for context"""
        return list(
            self.history.find({"session_id": session_id})
            .sort("timestamp", -1)
            .limit(limit)
        )


# Global instance
_memory_system = None

def get_memory_system() -> MemorySystem:
    """Get or create memory system"""
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem()
    return _memory_system