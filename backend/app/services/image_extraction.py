"""
Image Extraction from PDFs (in-memory only, stateless)
"""
import fitz  # PyMuPDF
from PIL import Image as PILImage
import io
from typing import List, Tuple
import uuid
from app.models.image import ImageMetadata, ExtractedImage


class PDFImageExtractor:
    """
    Extract images from PDFs using PyMuPDF
    
    Strategy: IN-MEMORY ONLY
    - Extract images as PIL objects
    - Generate CLIP embeddings immediately
    - Store only metadata + embeddings in Qdrant
    - No disk persistence
    """
    
    def __init__(self):
        self.min_width = 100   # Filter tiny images
        self.min_height = 100
    
    def extract_images_from_pdf(
        self,
        pdf_path: str,
        paper_id: str,
        paper_title: str
    ) -> List[Tuple[PILImage.Image, ImageMetadata]]:
        """
        Extract images from PDF
        
        Returns:
            List of (PIL_Image, ImageMetadata) tuples
            Images are kept in memory only
        """
        images = []
        
        # Open PDF
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get images on this page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # Extract image
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = PILImage.open(io.BytesIO(image_bytes))
                    
                    # Filter small images (likely icons/logos)
                    if pil_image.width < self.min_width or pil_image.height < self.min_height:
                        continue
                    
                    # Convert CMYK/other to RGB
                    if pil_image.mode not in ('RGB', 'L'):
                        pil_image = pil_image.convert('RGB')
                    
                    # Create metadata
                    metadata = ImageMetadata(
                        image_id=str(uuid.uuid4()),
                        paper_id=paper_id,
                        paper_title=paper_title,
                        page_number=page_num + 1,  # 1-indexed
                        caption=None,  # Could extract from surrounding text
                        image_type=self._classify_image_type(pil_image)
                    )
                    
                    images.append((pil_image, metadata))
                    
                except Exception as e:
                    # Skip problematic images
                    print(f"      ⚠️  Skipped image on page {page_num + 1}: {e}")
                    continue
        
        doc.close()
        
        return images
    
    def _classify_image_type(self, pil_image: PILImage.Image) -> str:
        """
        Simple heuristic to classify image type
        
        Returns: "figure", "chart", "diagram", "table"
        """
        width, height = pil_image.size
        aspect_ratio = width / height
        
        # Wide images are often charts/tables
        if aspect_ratio > 2.0:
            return "chart"
        
        # Very tall images might be diagrams
        if aspect_ratio < 0.5:
            return "diagram"
        
        # Default to figure
        return "figure"


def get_image_extractor() -> PDFImageExtractor:
    """Get image extractor instance"""
    return PDFImageExtractor()