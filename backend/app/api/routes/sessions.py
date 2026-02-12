"""
Session API Routes — ChatGPT-style chat sessions
"""
from fastapi import APIRouter, HTTPException
from app.models.session import (
    SessionCreate, SessionRename, SessionQueryRequest,
    SessionInfo, SessionDetail
)
from app.services.session_service import get_session_service
from app.services.query_engine import get_query_engine

router = APIRouter()


@router.get("/sessions")
async def list_sessions():
    """List all chat sessions (most recent first)"""
    service = get_session_service()
    sessions = service.list_sessions()
    return {"sessions": sessions}


@router.post("/sessions")
async def create_session(request: SessionCreate = SessionCreate()):
    """Create a new chat session"""
    service = get_session_service()
    session = service.create_session(title=request.title)
    return session


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a session with its full message history"""
    service = get_session_service()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    service = get_session_service()
    deleted = service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, request: SessionRename):
    """Rename a chat session"""
    service = get_session_service()
    updated = service.rename_session(session_id, request.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "renamed", "session_id": session_id, "title": request.title}


@router.post("/sessions/{session_id}/query")
async def session_query(session_id: str, request: SessionQueryRequest):
    """
    Query within a session — saves both user question and assistant answer.
    This is the main endpoint the frontend uses for chat.
    """
    try:
        service = get_session_service()
        
        # Verify session exists
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 1. Save user message
        service.add_user_message(
            session_id=session_id,
            content=request.question,
            search_mode=request.search_mode
        )
        
        # 2. Run RAG query (existing logic)
        query_engine = get_query_engine()
        result = query_engine.query(
            question=request.question,
            similarity_top_k=request.similarity_top_k,
            search_mode=request.search_mode
        )
        
        # 3. Save assistant response
        service.add_assistant_message(
            session_id=session_id,
            content=result["answer"],
            sources=result.get("sources", []),
            images=result.get("images", []),
            search_mode=request.search_mode
        )
        
        # 4. Return response (same format as /api/query)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "images": result.get("images", []),
            "num_sources": result.get("num_sources", 0),
            "search_mode": request.search_mode,
            "session_id": session_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Session Query Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
