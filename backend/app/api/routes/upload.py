from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.paper import PaperUploadResponse
from app.services.pdf_parser import PDFParser
from app.services.chunking import SectionAwareChunker
from app.services.embeddings import get_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings
import os
import shutil
import time

router = APIRouter()
settings = get_settings()


@router.post("/pdf", response_model=PaperUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF research paper
    
    Steps:
    1. Save PDF to disk
    2. Parse PDF (extract text, sections, metadata)
    3. Chunk the paper (section-aware)
    4. Generate embeddings
    5. Store in Qdrant
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Validate file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    temp_file_path = None
    
    try:
        # Save uploaded file
        os.makedirs(settings.upload_dir, exist_ok=True)
        file_path = os.path.join(settings.upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > settings.max_file_size * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Max size: {settings.max_file_size}MB"
                    )
                buffer.write(chunk)
        
        temp_file_path = file_path
        
        # Parse PDF
        print(f"Parsing PDF: {file.filename}")
        parser = PDFParser(file_path)
        parsed_paper = parser.parse()
        parser.close()
        
        # Chunk paper
        print(f"Chunking paper: {parsed_paper.paper_id}")
        chunker = SectionAwareChunker()
        chunks = chunker.chunk_paper(parsed_paper)
        
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        embedding_service = get_embedding_service(use_local=False)  # Use OpenAI
        
        texts = [chunk.text for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings_with_retry(texts)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Store in Qdrant
        print("Storing in Qdrant...")
        qdrant_service = QdrantService()
        
        # Create collection if it doesn't exist
        qdrant_service.create_collection(dimension=len(embeddings[0]))
        
        # Insert chunks
        qdrant_service.insert_chunks(chunks)
        
        return PaperUploadResponse(
            paper_id=parsed_paper.paper_id,
            filename=file.filename,
            status="success",
            message=f"Successfully processed {len(chunks)} chunks",
            metadata=parsed_paper.metadata,
            num_chunks=len(chunks)
        )
    
    except Exception as e:
        # Clean up file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@router.get("/status/{paper_id}")
async def get_paper_status(paper_id: str):
    """Get processing status of a paper"""
    # TODO: Implement status tracking in MongoDB
    return {
        "paper_id": paper_id,
        "status": "completed"
    }


@router.delete("/{paper_id}")
async def delete_paper(paper_id: str):
    """Delete a paper and its chunks"""
    try:
        qdrant_service = QdrantService()
        qdrant_service.delete_paper_chunks(paper_id)
        
        return {
            "paper_id": paper_id,
            "status": "deleted",
            "message": "Paper and all chunks deleted successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting paper: {str(e)}"
        )