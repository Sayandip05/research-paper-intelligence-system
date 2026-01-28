#!/usr/bin/env python3
"""
🆕 Build corpus with HYBRID embeddings (dense + sparse)

Usage:
1. Put your PDFs in ./corpus folder
2. Run: python build_corpus.py
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.pdf_parser import SectionAwarePDFParser
from app.services.chunking import Chunker
from app.services.embeddings import get_embedding_service, get_sparse_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings

settings = get_settings()


def build_corpus():
    """Process all PDFs with HYBRID embeddings"""
    
    print("\n" + "="*60)
    print("  🆕 BUILDING HYBRID CORPUS (Dense + Sparse)")
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
    qdrant_service.create_collection()
    
    dense_embeddings = get_embedding_service()
    
    # 🆕 Initialize sparse embeddings if hybrid enabled
    sparse_embeddings = None
    if settings.enable_hybrid_search:
        print("🆕 Hybrid mode enabled - loading sparse embeddings...")
        sparse_embeddings = get_sparse_embedding_service()
    
    chunker = Chunker()
    
    # Process each PDF
    all_chunks = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_path.name}")
        
        try:
            # Parse PDF
            print("   [1/5] Parsing PDF...")
            parser = SectionAwarePDFParser(str(pdf_path))
            paper = parser.parse()
            print(f"      ✓ Title: {paper.metadata.title}")
            print(f"      ✓ Pages: {paper.metadata.num_pages}")
            
            # Chunk paper
            print("   [2/5] Chunking...")
            chunks = chunker.chunk_paper(paper)
            print(f"      ✓ Created {len(chunks)} chunks")
            
            # Generate DENSE embeddings
            print("   [3/5] Generating DENSE embeddings (BGE)...")
            texts = [chunk.text for chunk in chunks]
            dense_vecs = dense_embeddings.generate_embeddings(texts)
            
            # Attach dense embeddings
            for chunk, embedding in zip(chunks, dense_vecs):
                chunk.embedding = embedding
            
            print(f"      ✓ Generated {len(dense_vecs)} dense embeddings")
            
            # 🆕 Generate SPARSE embeddings
            if settings.enable_hybrid_search and sparse_embeddings:
                print("   [4/5] Generating SPARSE embeddings (BM42)...")
                sparse_vecs = sparse_embeddings.generate_sparse_embeddings(texts)
                
                # Attach sparse embeddings
                for chunk, sparse_vec in zip(chunks, sparse_vecs):
                    chunk.sparse_embedding = sparse_vec
                
                print(f"      ✓ Generated {len(sparse_vecs)} sparse embeddings")
            else:
                print("   [4/5] Skipping sparse embeddings (hybrid disabled)")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_chunks:
        print("\n❌ No chunks to insert!")
        return
    
    # Insert into Qdrant
    print(f"\n   [5/5] Inserting {len(all_chunks)} chunks into Qdrant...")
    qdrant_service.insert_chunks(all_chunks)
    
    # Summary
    print("\n" + "="*60)
    print("  ✅ HYBRID CORPUS BUILD COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Papers processed: {len(pdf_files)}")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Dense vectors: ✅ ({settings.embedding_dim}-dim)")
    if settings.enable_hybrid_search:
        print(f"   Sparse vectors: ✅ (BM42)")
    else:
        print(f"   Sparse vectors: ❌ (disabled)")
    print(f"   Vector database: Qdrant")
    print(f"\n🚀 Ready to search!")
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