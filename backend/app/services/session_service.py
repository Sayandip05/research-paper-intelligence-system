"""
Session Service — ChatGPT-style session memory using LlamaIndex MongoChatStore
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from llama_index.storage.chat_store.mongo import MongoChatStore
from llama_index.core.llms import ChatMessage, MessageRole

from app.config import get_settings
from app.db.mongo_client import get_mongo_db

settings = get_settings()


class SessionService:
    def __init__(self):
        # LlamaIndex MongoChatStore for chat history
        self.chat_store = MongoChatStore.from_uri(
            uri=settings.mongodb_uri,
            db_name=settings.mongodb_db_name
        )
        # Direct pymongo for session metadata
        self.db = get_mongo_db()
        self.sessions_collection = self.db["sessions"]
        
        # Create index for faster lookups
        self.sessions_collection.create_index("session_id", unique=True)
        print("✅ SessionService initialized")

    # ── Session CRUD ──────────────────────────────────────────────

    def create_session(self, title: Optional[str] = None) -> dict:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        session = {
            "session_id": session_id,
            "title": title or "New Chat",
            "created_at": now,
            "updated_at": now
        }
        self.sessions_collection.insert_one(session)
        return {"session_id": session_id, "title": session["title"], "created_at": now, "updated_at": now}

    def list_sessions(self) -> List[dict]:
        """List all sessions, most recent first"""
        sessions = list(
            self.sessions_collection.find(
                {}, {"_id": 0, "session_id": 1, "title": 1, "created_at": 1, "updated_at": 1}
            ).sort("updated_at", -1)
        )
        # Add message count
        for s in sessions:
            messages = self.chat_store.get_messages(s["session_id"])
            s["message_count"] = len(messages)
        return sessions

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session metadata + full message history"""
        session = self.sessions_collection.find_one(
            {"session_id": session_id}, {"_id": 0}
        )
        if not session:
            return None
        
        # Get chat messages from LlamaIndex store
        messages = self.chat_store.get_messages(session_id)
        session["messages"] = [
            {
                "role": str(msg.role.value) if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content,
                "sources": msg.additional_kwargs.get("sources"),
                "images": msg.additional_kwargs.get("images"),
                "search_mode": msg.additional_kwargs.get("search_mode"),
                "timestamp": msg.additional_kwargs.get("timestamp", session["created_at"].isoformat())
            }
            for msg in messages
        ]
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages"""
        result = self.sessions_collection.delete_one({"session_id": session_id})
        # Clear messages from chat store
        self.chat_store.delete_messages(session_id)
        return result.deleted_count > 0

    def rename_session(self, session_id: str, title: str) -> bool:
        """Rename a session"""
        result = self.sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"title": title, "updated_at": datetime.now(timezone.utc)}}
        )
        return result.modified_count > 0

    # ── Chat Memory ───────────────────────────────────────────────

    def add_user_message(self, session_id: str, content: str, search_mode: str = "hybrid"):
        """Save user message to chat store"""
        now = datetime.now(timezone.utc)
        msg = ChatMessage(
            role=MessageRole.USER,
            content=content,
            additional_kwargs={"search_mode": search_mode, "timestamp": now.isoformat()}
        )
        self.chat_store.add_message(session_id, msg)
        
        # Auto-title: if this is the first message, use it as title
        messages = self.chat_store.get_messages(session_id)
        if len(messages) == 1:
            short_title = content[:60] + ("..." if len(content) > 60 else "")
            self.rename_session(session_id, short_title)
        
        # Update session timestamp
        self.sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"updated_at": now}}
        )

    def add_assistant_message(self, session_id: str, content: str,
                               sources: List[dict] = None, images: List[dict] = None,
                               search_mode: str = "hybrid"):
        """Save assistant response to chat store"""
        now = datetime.now(timezone.utc)
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            additional_kwargs={
                "sources": sources or [],
                "images": images or [],
                "search_mode": search_mode,
                "timestamp": now.isoformat()
            }
        )
        self.chat_store.add_message(session_id, msg)
        
        # Update session timestamp
        self.sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"updated_at": now}}
        )


# Singleton
_session_service = None

def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
