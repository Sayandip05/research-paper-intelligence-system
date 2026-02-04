"""
Intelligent Query API Routes

This is the main endpoint users will use!
"""

from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest, QueryResponse, SourceInfo, ImageInfo
from app.services.query_engine import get_query_engine

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def intelligent_query(request: QueryRequest):
    """
    🧠 Intelligent Query Endpoint
    
    Ask questions about your research papers and get smart answers with citations.
    
    Examples:
    - "What datasets were used in Vision Transformer papers?"
    - "What is the main contribution of the Transformer paper?"
    - "Compare BERT and GPT architectures"
    - "What are the limitations mentioned in the papers?"
    
    The system will:
    1. Retrieve relevant chunks from Qdrant
    2. Use Groq LLM to understand and synthesize
    3. Generate answer with proper citations
    4. Return structured response
    """
    try:
        # Get query engine
        query_engine = get_query_engine()
        
        # Execute query
        result = query_engine.query(
            question=request.question,
            similarity_top_k=request.similarity_top_k,
            response_mode=request.response_mode,
            search_mode=request.search_mode  # 🆕 dense/sparse/hybrid
        )
        
        # Convert to response model
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            images=[ImageInfo(**img) for img in result.get("images", [])],  # 🆕
            num_sources=result["num_sources"],
            response_mode=result["response_mode"]
        )
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Query Error: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/query/examples")
async def get_example_queries():
    """
    Get example queries to try
    
    Returns a list of good questions to ask the system
    """
    return {
        "examples": [
            {
                "category": "Factual",
                "questions": [
                    "What is the main contribution of the paper?",
                    "What datasets were used for evaluation?",
                    "What is the architecture of the proposed model?",
                    "What are the hyperparameters used for training?"
                ]
            },
            {
                "category": "Comparative",
                "questions": [
                    "Compare the performance of different models",
                    "What are the differences between BERT and GPT?",
                    "How does this approach differ from previous work?"
                ]
            },
            {
                "category": "Analytical",
                "questions": [
                    "What are the limitations mentioned in the papers?",
                    "What future work is suggested?",
                    "What are the key findings?",
                    "What challenges are discussed?"
                ]
            },
            {
                "category": "Extraction",
                "questions": [
                    "List all datasets mentioned",
                    "What evaluation metrics were used?",
                    "What baseline models were compared against?"
                ]
            }
        ]
    }


@router.post("/query/simple")
async def simple_query(question: str, top_k: int = 5):
    """
    Simplified query endpoint - just question and top_k
    
    Example:
    POST /api/query/simple?question=What%20is%20the%20Transformer?&top_k=3
    """
    try:
        query_engine = get_query_engine()
        
        result = query_engine.query(
            question=question,
            similarity_top_k=top_k
        )
        
        return {
            "question": question,
            "answer": result["answer"],
            "num_sources": result["num_sources"]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/query/health")
async def query_health():
    """Check if query engine is ready"""
    from app.config import get_settings
    settings = get_settings()
    
    try:
        query_engine = get_query_engine()
        return {
            "status": "healthy",
            "message": "Query engine is ready!",
            "llm": settings.llm_model,
            "dense_embeddings": settings.embedding_model,
            "sparse_embeddings": settings.sparse_embedding_model if settings.enable_hybrid_search else "disabled",
            "hybrid_search": settings.enable_hybrid_search,
            "vector_store": "Qdrant",
            "collection": settings.qdrant_collection_name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }