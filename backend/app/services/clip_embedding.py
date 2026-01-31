"""
CLIP Embedding Service for Image-Text Multimodal Embeddings
"""
from typing import List, Union
import torch
import clip
from PIL import Image as PILImage
import io
from app.config import get_settings

settings = get_settings()


class CLIPEmbeddingService:
    """
    CLIP (ViT-B/32) embedding service
    
    Capabilities:
    - Text → 512-dim vector
    - Image → 512-dim vector
    - Same embedding space (cross-modal search!)
    """
    
    def __init__(self):
        print(f"📦 Loading CLIP model: {settings.clip_model_name}")
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            settings.clip_model_name,
            device=self.device
        )
        
        self.dimension = 512  # ViT-B/32 outputs 512-dim
        
        print(f"✅ CLIP loaded on {self.device}! Dimension: {self.dimension}")
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate CLIP embedding for text query
        
        Used for: "show me LoRA architecture diagram"
        """
        # Tokenize text
        text_tokens = clip.tokenize([text]).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize (CLIP uses cosine similarity)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embedding = text_features.cpu().numpy()[0].tolist()
        
        return embedding
    
    def generate_image_embedding(self, pil_image: PILImage.Image) -> List[float]:
        """
        Generate CLIP embedding for image
        
        Args:
            pil_image: PIL Image object (in-memory)
        
        Returns:
            512-dim embedding vector
        """
        # Preprocess image
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embedding = image_features.cpu().numpy()[0].tolist()
        
        return embedding
    
    def generate_image_embeddings_batch(
        self,
        pil_images: List[PILImage.Image]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple images (batched for speed)
        """
        if not pil_images:
            return []
        
        # Preprocess all images
        image_inputs = torch.stack([
            self.preprocess(img) for img in pil_images
        ]).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list of lists
        embeddings = image_features.cpu().numpy().tolist()
        
        return embeddings


# Global instance
_clip_service = None


def get_clip_embedding_service() -> CLIPEmbeddingService:
    """Get or create CLIP embedding service"""
    global _clip_service
    if _clip_service is None:
        _clip_service = CLIPEmbeddingService()
    return _clip_service