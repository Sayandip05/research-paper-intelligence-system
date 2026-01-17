from typing import List
from app.models.paper import ParsedPaper, Section
from app.models.chunk import Chunk, ChunkMetadata
from app.config import get_settings
import uuid
import re

settings = get_settings()


class SectionAwareChunker:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk_paper(self, paper: ParsedPaper) -> List[Chunk]:
        """
        Create section-aware chunks from a parsed paper
        
        Strategy:
        1. Keep sections together when possible
        2. Split large sections with overlap
        3. Attach rich metadata to each chunk
        """
        chunks = []
        chunk_index = 0
        
        # If paper has sections, chunk by section
        if paper.sections:
            for section in paper.sections:
                section_chunks = self._chunk_section(
                    section=section,
                    paper=paper,
                    start_index=chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        else:
            # Fallback: chunk raw text
            chunks = self._chunk_raw_text(paper)
        
        # Update total_chunks in metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata.total_chunks = total_chunks
        
        return chunks
    
    def _chunk_section(
        self,
        section: Section,
        paper: ParsedPaper,
        start_index: int
    ) -> List[Chunk]:
        """Chunk a single section"""
        chunks = []
        
        # If section is small enough, keep it as one chunk
        if len(section.content) <= self.chunk_size:
            chunk = self._create_chunk(
                text=section.content,
                paper=paper,
                section=section,
                chunk_index=start_index,
                page_start=section.page_start,
                page_end=section.page_end
            )
            chunks.append(chunk)
        else:
            # Split large section with overlap
            section_chunks = self._split_with_overlap(section.content)
            
            for i, chunk_text in enumerate(section_chunks):
                chunk = self._create_chunk(
                    text=chunk_text,
                    paper=paper,
                    section=section,
                    chunk_index=start_index + i,
                    page_start=section.page_start,
                    page_end=section.page_end
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Get chunk
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence ending near the boundary
                sentence_ends = [
                    m.end() for m in re.finditer(r'[.!?]\s+', text[end-100:end])
                ]
                if sentence_ends:
                    end = end - 100 + sentence_ends[-1]
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start forward with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        paper: ParsedPaper,
        section: Section,
        chunk_index: int,
        page_start: int,
        page_end: int
    ) -> Chunk:
        """Create a chunk with metadata"""
        chunk_id = str(uuid.uuid4())
        
        # Check if chunk contains tables, equations, figures
        has_table = any(
            table.section_id == section.section_id
            for table in paper.tables
        )
        has_equation = any(
            eq.section_id == section.section_id
            for eq in paper.equations
        )
        has_figure = any(
            fig.section_id == section.section_id
            for fig in paper.figures
        )
        
        metadata = ChunkMetadata(
            paper_id=paper.paper_id,
            paper_title=paper.metadata.title,
            authors=[author.name for author in paper.metadata.authors],
            year=paper.metadata.year,
            section_title=section.title,
            section_id=section.section_id,
            page_start=page_start,
            page_end=page_end,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            has_table=has_table,
            has_equation=has_equation,
            has_figure=has_figure
        )
        
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata
        )
    
    def _chunk_raw_text(self, paper: ParsedPaper) -> List[Chunk]:
        """Fallback: chunk raw text without section structure"""
        chunks = []
        text_chunks = self._split_with_overlap(paper.raw_text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())
            
            metadata = ChunkMetadata(
                paper_id=paper.paper_id,
                paper_title=paper.metadata.title,
                authors=[author.name for author in paper.metadata.authors],
                year=paper.metadata.year,
                section_title="Full Text",
                section_id="raw",
                page_start=1,
                page_end=paper.metadata.num_pages or 1,
                chunk_index=i,
                total_chunks=len(text_chunks)
            )
            
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks


def smart_chunk_by_sentences(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Alternative chunking strategy: by sentences
    Ensures no sentence is split across chunks
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks