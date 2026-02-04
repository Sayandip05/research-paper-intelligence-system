"""
Image Serving API - Extract and serve images from PDFs on-demand
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import fitz  # PyMuPDF
from pathlib import Path
import io

from app.config import get_settings

settings = get_settings()
router = APIRouter()


@router.get("/image/{paper_title}/{page_number}/{image_index}")
async def get_image(paper_title: str, page_number: int, image_index: int = 0):
    """
    Serve an image from a PDF on-demand
    
    Args:
        paper_title: Name of the PDF (without extension)
        page_number: Page number (1-indexed)
        image_index: Index of image on that page (0-indexed)
    
    Returns:
        Image as PNG bytes
    """
    corpus_dir = Path(settings.corpus_dir)
    
    # Find the PDF file (try different naming patterns)
    pdf_path = None
    for pattern in [f"{paper_title}.pdf", f"{paper_title.upper()}.pdf", f"*{paper_title[:20]}*.pdf"]:
        matches = list(corpus_dir.glob(pattern))
        if matches:
            pdf_path = matches[0]
            break
    
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {paper_title}")
    
    try:
        doc = fitz.open(str(pdf_path))
        
        if page_number < 1 or page_number > len(doc):
            raise HTTPException(status_code=404, detail=f"Invalid page number: {page_number}")
        
        page = doc[page_number - 1]  # Convert to 0-indexed
        image_list = page.get_images(full=True)
        
        if image_index >= len(image_list):
            raise HTTPException(status_code=404, detail=f"Image index {image_index} not found on page {page_number}")
        
        # Extract the image
        xref = image_list[image_index][0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        doc.close()
        
        # Determine content type
        content_type = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
        }.get(image_ext.lower(), "image/png")
        
        return Response(content=image_bytes, media_type=content_type)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting image: {str(e)}")


@router.get("/image-by-id/{image_id}")
async def get_image_by_id(image_id: str):
    """
    Serve an image by looking up its metadata in Qdrant and extracting from PDF
    """
    from app.db.qdrant_client import QdrantService
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    qdrant = QdrantService()
    
    # Find image metadata by image_id
    try:
        results = qdrant.client.scroll(
            collection_name=settings.qdrant_image_collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))]
            ),
            limit=1,
            with_payload=True
        )
        
        if not results[0]:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
        
        payload = results[0][0].payload
        paper_title = payload.get("paper_title", "")
        page_number = payload.get("page_number", 1)
        
        # Now extract from PDF
        corpus_dir = Path(settings.corpus_dir)
        
        # Find matching PDF
        pdf_path = None
        for pdf in corpus_dir.glob("*.pdf"):
            if paper_title.lower()[:20] in pdf.stem.lower() or pdf.stem.lower()[:20] in paper_title.lower():
                pdf_path = pdf
                break
        
        if not pdf_path:
            # Try first PDF as fallback
            pdfs = list(corpus_dir.glob("*.pdf"))
            if pdfs:
                pdf_path = pdfs[0]
        
        if not pdf_path:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        doc = fitz.open(str(pdf_path))
        
        if page_number < 1 or page_number > len(doc):
            page_number = 1
        
        page = doc[page_number - 1]
        image_list = page.get_images(full=True)
        
        if not image_list:
            raise HTTPException(status_code=404, detail="No images on page")
        
        # Get first image on page (or could store image_index in metadata)
        xref = image_list[0][0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        doc.close()
        
        content_type = "image/png" if image_ext.lower() == "png" else "image/jpeg"
        return Response(content=image_bytes, media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
