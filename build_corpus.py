#!/usr/bin/env python3
"""
Build corpus from PDFs in ./corpus folder

Usage:
1. Put your 2 PDFs in ./corpus folder
2. Run: python build_corpus.py
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.pdf_parser import PDFParser
from app.services.chunking import Chunker
from app.services.embeddings import get_embedding_service
from app.db.qdrant_client import QdrantService
from app.config import get_settings

settings = get_settings()


def build_corpus():
    """Process all PDFs in corpus directory"""
    
    print("\n" + "="*60)
    print("  BUILDING CORPUS FROM LOCAL PDFs")
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
        print(f"   1. Copy your 2 PDF files to: {corpus_dir}")
        print("   2. Run this script again: python build_corpus.py")
        return
    
    print(f"\n📚 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")
    
    # Initialize services
    print("\n🔧 Initializing services...")
    qdrant_service = QdrantService()
    qdrant_service.create_collection()
    
    embedding_service = get_embedding_service()
    chunker = Chunker()
    
    # Process each PDF
    all_chunks = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_path.name}")
        
        try:
            # Parse PDF
            print("   [1/4] Parsing PDF...")
            parser = PDFParser(str(pdf_path))
            paper = parser.parse()
            # Note: LlamaIndex SimpleDirectoryReader handles cleanup automatically
            print(f"      ✓ Title: {paper.metadata.title}")
            print(f"      ✓ Pages: {paper.metadata.num_pages}")
            
            # Chunk paper
            print("   [2/4] Chunking...")
            chunks = chunker.chunk_paper(paper)
            print(f"      ✓ Created {len(chunks)} chunks")
            
            # Generate embeddings
            print("   [3/4] Generating embeddings...")
            texts = [chunk.text for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(texts)
            
            # Attach embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            print(f"      ✓ Generated {len(embeddings)} embeddings")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            continue
    
    if not all_chunks:
        print("\n❌ No chunks to insert!")
        return
    
    # Insert into Qdrant
    print(f"\n   [4/4] Inserting {len(all_chunks)} chunks into Qdrant...")
    qdrant_service.insert_chunks(all_chunks)
    
    # Summary
    print("\n" + "="*60)
    print("  ✅ CORPUS BUILD COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Papers processed: {len(pdf_files)}")
    print(f"   Total chunks: {len(all_chunks)}")
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