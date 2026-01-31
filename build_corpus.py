#!/usr/bin/env python3
"""
🆕 Build MULTIMODAL corpus (Text + Images)

Features:
- Text: Dense (BGE) + Sparse (BM42) embeddings
- Images: CLIP embeddings (ViT-B/32)
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.pdf_parser import SectionAwarePDFParser
from app.services.chunking import Chunker
from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
from app.services.clip_embeddings import get_clip_embedding_service
from app.services.image_extraction import get_image_extractor
from app.db.qdrant_client import QdrantService
from app.config import get_settings

settings = get_settings()


def build_corpus():
    """Process all PDFs with TEXT + IMAGE indexing"""
    
    print("\n" + "="*60)
    print("  🆕 BUILDING MULTIMODAL CORPUS (Text + Images)")
    print("="*60)
    
    # Check corpus directory
    corpus_dir = Path(settings.corpus_dir)
    if not corpus_dir.exists():
        print(f"\n❌ Corpus directory not found: {corpus_dir}")
        print("   Creating it now...")
        corpus_dir.mkdir(parents=True)
        print(f"\n📁 Please put your PDF files in: {corpus_dir}")
        print("   Then run this script again.")
        return
    
    # Find PDFs
    pdf_files = list(corpus_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in: {corpus_dir}")
        print("\n📝 Instructions:")
        print(f"   1. Copy your PDFs to: {corpus_dir}")
        print("   2. Run this script again: python build_corpus.py")
        return
    
    print(f"\n📚 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")
    
    # Initialize services
    print("\n🔧 Initializing services...")
    qdrant_service = QdrantService()
    qdrant_service.create_collection()  # Text collection
    if settings.enable_multimodal:
        qdrant_service.create_image_collection()  # 🆕 Image collection
    
    dense_embeddings = get_embedding_service()
    
    # Sparse embeddings
    sparse_embeddings = None
    if settings.enable_hybrid_search:
        print("🆕 Hybrid mode enabled - loading sparse embeddings...")
        sparse_embeddings = get_sparse_embedding_service()
    
    # 🆕 CLIP embeddings
    clip_embeddings = None
    if settings.enable_multimodal:
        print("🆕 Multimodal mode enabled - loading CLIP embeddings...")
        clip_embeddings = get_clip_embedding_service()
    
    # 🆕 Image extractor
    image_extractor = None
    if settings.enable_multimodal:
        image_extractor = get_image_extractor()
    
    chunker = Chunker()
    
    # Process each PDF
    all_text_chunks = []
    all_image_data = []  # List of (ImageMetadata, embedding)
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_path.name}")
        
        try:
            # ========== TEXT PROCESSING ==========
            # Parse PDF
            print("   [1/6] Parsing PDF...")
            parser = SectionAwarePDFParser(str(pdf_path))
            paper = parser.parse()
            print(f"      ✓ Title: {paper.metadata.title}")
            print(f"      ✓ Pages: {paper.metadata.num_pages}")
            
            # Chunk paper
            print("   [2/6] Chunking text...")
            chunks = chunker.chunk_paper(paper)
            print(f"      ✓ Created {len(chunks)} text chunks")
            
            # Generate DENSE embeddings
            print("   [3/6] Generating DENSE embeddings (BGE)...")
            texts = [chunk.text for chunk in chunks]
            dense_vecs = dense_embeddings.generate_embeddings(texts)
            
            # Attach dense embeddings
            for chunk, embedding in zip(chunks, dense_vecs):
                chunk.embedding = embedding
            
            print(f"      ✓ Generated {len(dense_vecs)} dense embeddings")
            
            # Generate SPARSE embeddings
            if settings.enable_hybrid_search and sparse_embeddings:
                print("   [4/6] Generating SPARSE embeddings (BM42)...")
                sparse_vecs = sparse_embeddings.generate_sparse_embeddings(texts)
                
                # Attach sparse embeddings
                for chunk, sparse_vec in zip(chunks, sparse_vecs):
                    chunk.sparse_embedding = sparse_vec
                
                print(f"      ✓ Generated {len(sparse_vecs)} sparse embeddings")
            else:
                print("   [4/6] Skipping sparse embeddings (hybrid disabled)")
            
            all_text_chunks.extend(chunks)
            
            # ========== IMAGE PROCESSING ==========
            if settings.enable_multimodal and clip_embeddings and image_extractor:
                print("   [5/6] Extracting images from PDF...")
                
                # Extract images (in-memory only)
                images_with_metadata = image_extractor.extract_images_from_pdf(
                    str(pdf_path),
                    paper.paper_id,
                    paper.metadata.title
                )
                
                if images_with_metadata:
                    print(f"      ✓ Extracted {len(images_with_metadata)} images")
                    
                    # Generate CLIP embeddings for all images (batched)
                    print("      Generating CLIP embeddings...")
                    pil_images = [img for img, _ in images_with_metadata]
                    metadatas = [meta for _, meta in images_with_metadata]
                    
                    clip_vecs = clip_embeddings.generate_image_embeddings_batch(pil_images)
                    
                    # Pair metadata with embeddings
                    for metadata, clip_vec in zip(metadatas, clip_vecs):
                        all_image_data.append((metadata, clip_vec))
                    
                    print(f"      ✓ Generated {len(clip_vecs)} CLIP embeddings")
                    
                    # Update paper metadata
                    paper.metadata.num_images = len(images_with_metadata)
                else:
                    print("      ⚠️  No images found in this PDF")
            else:
                print("   [5/6] Skipping image extraction (multimodal disabled)")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== INDEXING ==========
    # Insert text chunks
    if all_text_chunks:
        print(f"\n   [6/6] Inserting {len(all_text_chunks)} text chunks into Qdrant...")
        qdrant_service.insert_chunks(all_text_chunks)
    else:
        print("\n   ❌ No text chunks to insert!")
    
    # 🆕 Insert image embeddings
    if all_image_data:
        print(f"   [6/6] Inserting {len(all_image_data)} image embeddings into Qdrant...")
        qdrant_service.insert_images(all_image_data)
    else:
        print("   ⚠️  No images to insert")
    
    # Summary
    print("\n" + "="*60)
    print("  ✅ MULTIMODAL CORPUS BUILD COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Papers processed: {len(pdf_files)}")
    print(f"   Text chunks: {len(all_text_chunks)}")
    if settings.enable_hybrid_search:
        print(f"   Dense vectors: ✅ ({settings.embedding_dim}-dim BGE)")
        print(f"   Sparse vectors: ✅ (BM42)")
    else:
        print(f"   Dense vectors: ✅ ({settings.embedding_dim}-dim BGE)")
        print(f"   Sparse vectors: ❌ (disabled)")
    
    if settings.enable_multimodal:
        print(f"   Image embeddings: ✅ ({len(all_image_data)} images, {settings.clip_embedding_dim}-dim CLIP)")
    else:
        print(f"   Image embeddings: ❌ (disabled)")
    
    print(f"   Vector database: Qdrant")
    print(f"\n🚀 Ready for multimodal search!")
    print(f"\n   Start API: cd backend && uvicorn app.main:app --reload")
    print(f"   Then visit: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    try:
        build_corpus()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()