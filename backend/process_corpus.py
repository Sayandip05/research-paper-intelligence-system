"""
Process all PDFs in corpus folder with text + image extraction
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.pdf_parser import SectionAwarePDFParser
from app.services.chunking import Chunker
from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
from app.services.image_extraction import PDFImageExtractor
from app.services.clip_embedding import get_clip_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings

settings = get_settings()

print(f"Corpus directory: {settings.corpus_dir}")
print(f"Multimodal enabled: {settings.enable_multimodal}")

# Initialize services
qdrant_service = QdrantService()
qdrant_service.create_collection()

dense_embeddings = get_embedding_service()
sparse_embeddings = get_sparse_embedding_service() if settings.enable_hybrid_search else None
chunker = Chunker()

# Get PDFs from corpus
corpus_path = Path(settings.corpus_dir)
pdfs = list(corpus_path.glob("*.pdf"))
print(f"\nFound {len(pdfs)} PDFs in {corpus_path.absolute()}")

for pdf_path in pdfs:
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")
    
    # Parse
    parser = SectionAwarePDFParser(str(pdf_path))
    paper = parser.parse()
    
    # Chunk
    chunks = chunker.chunk_paper(paper)
    print(f"   📄 Text chunks: {len(chunks)}")
    
    if not chunks:
        print("   ⚠️ No chunks - skipping")
        continue
    
    # Embeddings
    texts = [c.text for c in chunks]
    dense_vecs = dense_embeddings.generate_embeddings(texts)
    for c, e in zip(chunks, dense_vecs):
        c.embedding = e
    
    if sparse_embeddings:
        sparse_vecs = sparse_embeddings.generate_sparse_embeddings(texts)
        for c, s in zip(chunks, sparse_vecs):
            c.sparse_embedding = s
    
    # Insert text
    qdrant_service.insert_chunks(chunks)
    
    # Extract images
    if settings.enable_multimodal:
        extractor = PDFImageExtractor()
        images = extractor.extract_images_from_pdf(
            str(pdf_path), paper.paper_id, paper.metadata.title
        )
        print(f"   🖼️ Images found: {len(images)}")
        
        if images:
            clip_service = get_clip_embedding_service()
            images_data = []
            for pil_img, meta in images:
                try:
                    emb = clip_service.generate_image_embedding(pil_img)
                    images_data.append((meta, emb))
                except Exception as e:
                    print(f"      ⚠️ Failed: {e}")
            
            if images_data:
                qdrant_service.create_image_collection()
                qdrant_service.insert_images(images_data)
                print(f"   ✅ Stored {len(images_data)} images")

print("\n" + "="*60)
print("DONE!")

# Final stats
info = qdrant_service.client.get_collection(settings.qdrant_collection_name)
print(f"Total text chunks: {info.points_count}")

try:
    img_info = qdrant_service.client.get_collection(settings.qdrant_image_collection_name)
    print(f"Total images: {img_info.points_count}")
except:
    print("No image collection yet")
