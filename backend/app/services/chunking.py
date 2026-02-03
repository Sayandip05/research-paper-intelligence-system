"""
Smart Chunking using LlamaIndex
Much better than custom chunking!
"""

import uuid
from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from app.models.paper import ParsedPaper
from app.models.chunk import Chunk, ChunkMetadata
from app.config import get_settings

settings = get_settings()


class LlamaIndexChunker:
    """
    LlamaIndex-powered chunking
    
    Benefits over custom chunking:
    ✅ Respects sentence boundaries
    ✅ Better semantic preservation
    ✅ Configurable overlap
    ✅ Production-tested by LlamaIndex
    """
    
    def __init__(self):
        # Initialize LlamaIndex's SentenceSplitter
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            paragraph_separator="\n\n",
            separator=" "
        )
        
        print(f"✅ LlamaIndex Chunker initialized")
        print(f"   Chunk size: {settings.chunk_size}")
        print(f"   Overlap: {settings.chunk_overlap}")
    
    def chunk_paper(self, paper: ParsedPaper) -> List[Chunk]:
        """
        Chunk paper using LlamaIndex
        
        Args:
            paper: Parsed paper object
        
        Returns:
            List of chunks with embeddings ready to generate
        """
        # Use section-aware chunking by default for proper metadata
        return self.chunk_with_metadata(paper, section_aware=True)
    
    def _chunk_full_text(self, paper: ParsedPaper) -> List[Chunk]:
        """
        Fallback: Chunk the entire paper as one document
        Used when no sections are detected
        """
        # Combine all text from the paper
        full_text = ""
        if paper.sections:
            full_text = "\n\n".join([s.content for s in paper.sections if s.content])
        
        # If still empty, try full_text attribute
        if not full_text and hasattr(paper, 'full_text'):
            full_text = paper.full_text
        
        if not full_text:
            print("⚠️ Warning: Paper has no extractable text!")
            return []
        
        # Create document for full text
        doc = Document(
            text=full_text,
            metadata={
                "paper_id": paper.paper_id,
                "title": paper.metadata.title,
                "section_id": "full",
                "section_title": "Full Text"
            }
        )
        
        # Chunk it
        nodes = self.splitter.get_nodes_from_documents([doc])
        
        chunks = []
        for node in nodes:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                text=node.get_content(),
                metadata=ChunkMetadata(
                    paper_id=paper.paper_id,
                    paper_title=paper.metadata.title,
                    section_title="Full Text",
                    page_start=1,
                    page_end=paper.metadata.total_pages or 1
                )
            )
            chunks.append(chunk)
        
        print(f"   📄 Created {len(chunks)} chunks from full text")
        return chunks
    
    def chunk_with_metadata(
        self,
        paper: ParsedPaper,
        section_aware: bool = True
    ) -> List[Chunk]:
        """
        Advanced: Chunk with section awareness
        
        If paper has sections, chunk each section separately
        This preserves section boundaries better
        """
        # Fallback to full-text chunking if no sections
        if not section_aware or not paper.sections:
            return self._chunk_full_text(paper)
        
        all_chunks = []
        
        for section in paper.sections:
            # Skip empty sections
            if not section.content or not section.content.strip():
                continue
                
            # Create document for this section
            section_doc = Document(
                text=section.content,
                metadata={
                    "paper_id": paper.paper_id,
                    "title": paper.metadata.title,
                    "section_id": section.section_id,
                    "section_title": section.title
                }
            )
            
            # Chunk this section
            nodes = self.splitter.get_nodes_from_documents([section_doc])
            
            # Convert to our Chunk format
            for node in nodes:
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=node.get_content(),
                    metadata=ChunkMetadata(
                        paper_id=paper.paper_id,
                        paper_title=paper.metadata.title,
                        section_title=section.title,
                        page_start=section.page_start,
                        page_end=section.page_end
                    )
                )
                all_chunks.append(chunk)
        
        # If section-aware chunking produced nothing, fallback to full-text
        if not all_chunks:
            print("   ⚠️ Section chunking produced 0 chunks, falling back to full-text")
            return self._chunk_full_text(paper)
        
        print(f"   ✅ Created {len(all_chunks)} chunks from {len(paper.sections)} sections")
        return all_chunks


# For backward compatibility, keep the same name
class Chunker(LlamaIndexChunker):
    """Alias for LlamaIndexChunker"""
    pass


# You can also use other LlamaIndex chunkers:

class SemanticChunker:
    """
    Semantic-based chunking (more advanced)
    Requires embeddings during chunking
    """
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    
    def __init__(self, embed_model=None):
        """
        Note: This requires embeddings during chunking
        More expensive but better quality chunks
        """
        # TODO: Implement with LlamaIndex fully integrated
        pass


class SentenceWindowChunker:
    """
    Creates chunks with sentence windows
    Good for maintaining context
    """
    from llama_index.core.node_parser import SentenceWindowNodeParser
    
    def __init__(self):
        """
        This creates overlapping sentence windows
        Better context preservation
        """
        # TODO: Implement
        pass


# Export the main chunker
def get_chunker() -> Chunker:
    """Get the default chunker (LlamaIndex-based)"""
    return Chunker()

    