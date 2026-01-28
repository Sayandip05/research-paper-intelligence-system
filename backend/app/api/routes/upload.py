"""
PDF Upload API Routes - Auto Processing

Upload PDFs and automatically process them with hybrid embeddings.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import shutil

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.services.pdf_parser import SectionAwarePDFParser
from app.services.chunking import Chunker
from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings

settings = get_settings()
router = APIRouter()

# Processing status tracker
processing_status = {}


class UploadResponse(BaseModel):
    """Response after PDF upload"""
    filename: str
    status: str
    message: str


class ProcessingStatus(BaseModel):
    """Status of PDF processing"""
    filename: str
    status: str
    chunks_created: int = 0
    error: str = None


def process_pdf(filepath: Path, filename: str):
    """
    Background task to process uploaded PDF with hybrid embeddings
    """
    global processing_status
    
    try:
        processing_status[filename] = {"status": "processing", "chunks_created": 0}
        
        # Initialize services
        qdrant_service = QdrantService()
        qdrant_service.create_collection()
        
        dense_embeddings = get_embedding_service()
        sparse_embeddings = None
        if settings.enable_hybrid_search:
            sparse_embeddings = get_sparse_embedding_service()
        
        chunker = Chunker()
        
        # Parse PDF
        parser = SectionAwarePDFParser(str(filepath))
        paper = parser.parse()
        
        # Chunk paper
        chunks = chunker.chunk_paper(paper)
        
        # Generate dense embeddings
        texts = [chunk.text for chunk in chunks]
        dense_vecs = dense_embeddings.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, dense_vecs):
            chunk.embedding = embedding
        
        # Generate sparse embeddings
        if settings.enable_hybrid_search and sparse_embeddings:
            sparse_vecs = sparse_embeddings.generate_sparse_embeddings(texts)
            for chunk, sparse_vec in zip(chunks, sparse_vecs):
                chunk.sparse_embedding = sparse_vec
        
        # Insert into Qdrant
        qdrant_service.insert_chunks(chunks)
        
        processing_status[filename] = {
            "status": "completed",
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        processing_status[filename] = {
            "status": "failed",
            "chunks_created": 0,
            "error": str(e)
        }


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    📤 Upload PDF and automatically process with hybrid embeddings
    
    The PDF will be:
    1. Saved to corpus/ folder
    2. Parsed with section detection
    3. Chunked with metadata
    4. Embedded (Dense + BM42 Sparse)
    5. Stored in Qdrant hybrid collection
    
    Processing happens in background - check status with /upload/status/{filename}
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )
    
    # Save to corpus directory
    corpus_dir = Path(settings.corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = corpus_dir / file.filename
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Start background processing
    background_tasks.add_task(process_pdf, filepath, file.filename)
    
    return UploadResponse(
        filename=file.filename,
        status="processing",
        message=f"PDF uploaded and processing started. Check status at /api/upload/status/{file.filename}"
    )


@router.get("/upload/status/{filename}")
async def get_processing_status(filename: str):
    """
    Check the processing status of an uploaded PDF
    """
    if filename not in processing_status:
        return {
            "filename": filename,
            "status": "not_found",
            "message": "No processing record found for this file"
        }
    
    status = processing_status[filename]
    return {
        "filename": filename,
        **status
    }


@router.get("/upload/list")
async def list_corpus_files():
    """
    List all PDFs in the corpus folder
    """
    corpus_dir = Path(settings.corpus_dir)
    
    if not corpus_dir.exists():
        return {"files": [], "count": 0}
    
    pdf_files = list(corpus_dir.glob("*.pdf"))
    
    return {
        "files": [f.name for f in pdf_files],
        "count": len(pdf_files),
        "corpus_path": str(corpus_dir.absolute())
    }
