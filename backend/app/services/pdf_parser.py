"""
PDF Parsing using LlamaIndex
Much better than custom PyMuPDF!
"""

import uuid
from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from app.models.paper import ParsedPaper, Section, PaperMetadata


class LlamaIndexPDFParser:
    """
    LlamaIndex-powered PDF parsing
    
    Benefits over custom PyMuPDF:
    ✅ Better text extraction
    ✅ Handles complex layouts
    ✅ Automatic metadata extraction
    ✅ Built-in chunking support
    ✅ Production-tested
    ✅ Integrates with Week 2 RAG pipeline
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.paper_id = str(uuid.uuid4())
    
    def parse(self) -> ParsedPaper:
        """
        Parse PDF using LlamaIndex
        
        Returns:
            ParsedPaper object with all extracted content
        """
        print(f"   Parsing with LlamaIndex: {self.file_path.name}")
        
        # Use LlamaIndex SimpleDirectoryReader
        # This automatically handles PDF parsing
        documents = SimpleDirectoryReader(
            input_files=[str(self.file_path)]
        ).load_data()
        
        # Extract metadata and content
        metadata = self._extract_metadata(documents)
        raw_text = self._extract_text(documents)
        sections = self._extract_sections(documents)
        
        return ParsedPaper(
            paper_id=self.paper_id,
            filename=self.file_path.name,
            metadata=metadata,
            sections=sections,
            raw_text=raw_text
        )
    
    def _extract_metadata(self, documents: List[Document]) -> PaperMetadata:
        """Extract metadata from LlamaIndex documents"""
        
        if not documents:
            return PaperMetadata(
                title="Unknown",
                authors=[],
                year=None,
                num_pages=0
            )
        
        # Get metadata from first document
        first_doc = documents[0]
        doc_metadata = first_doc.metadata
        
        # Extract title (from filename or metadata)
        title = doc_metadata.get('file_name', self.file_path.stem)
        if title.endswith('.pdf'):
            title = title[:-4]
        
        # Extract from first page text
        first_page = documents[0].text if documents else ""
        
        # Try to find year
        import re
        years = re.findall(r'\b(20[0-2][0-9])\b', first_page[:1000])
        year = int(years[0]) if years else None
        
        # Count pages
        num_pages = doc_metadata.get('total_pages', len(documents))
        
        return PaperMetadata(
            title=title,
            authors=[],  # Could enhance with NER later
            year=year,
            num_pages=num_pages
        )
    
    def _extract_text(self, documents: List[Document]) -> str:
        """Combine all document text"""
        return "\n\n".join(doc.text for doc in documents)
    
    def _extract_sections(self, documents: List[Document]) -> List[Section]:
        """
        Extract sections from documents
        
        For now, treats each page as a section
        Can be enhanced with section detection later
        """
        sections = []
        
        for i, doc in enumerate(documents, 1):
            section = Section(
                section_id=str(uuid.uuid4()),
                title=f"Page {i}",
                content=doc.text,
                page_start=i,
                page_end=i
            )
            sections.append(section)
        
        return sections
    
    def get_llamaindex_documents(self) -> List[Document]:
        """
        Get raw LlamaIndex documents
        
        Useful for Week 2 when building RAG pipeline!
        You can directly feed these to VectorStoreIndex
        """
        documents = SimpleDirectoryReader(
            input_files=[str(self.file_path)]
        ).load_data()
        
        return documents


class AdvancedPDFParser:
    """
    Advanced PDF parsing with better layout understanding
    Uses pypdf for more control
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.paper_id = str(uuid.uuid4())
    
    def parse(self) -> ParsedPaper:
        """
        Parse PDF with advanced features
        
        Features:
        - Better metadata extraction
        - Table detection (basic)
        - Section detection (basic)
        """
        from llama_index.readers.file import PyMuPDFReader
        
        # Use PyMuPDFReader for better control
        reader = PyMuPDFReader()
        documents = reader.load(file_path=str(self.file_path))
        
        # Extract components
        metadata = self._extract_metadata_advanced(documents)
        raw_text = "\n\n".join(doc.text for doc in documents)
        sections = self._detect_sections(documents, raw_text)
        
        return ParsedPaper(
            paper_id=self.paper_id,
            filename=self.file_path.name,
            metadata=metadata,
            sections=sections,
            raw_text=raw_text
        )
    
    def _extract_metadata_advanced(self, documents: List[Document]) -> PaperMetadata:
        """Advanced metadata extraction"""
        import re
        
        if not documents:
            return PaperMetadata(title="Unknown", authors=[], year=None, num_pages=0)
        
        first_page = documents[0].text
        
        # Extract title (first significant line)
        lines = [l.strip() for l in first_page.split('\n') if len(l.strip()) > 10]
        title = lines[0] if lines else self.file_path.stem
        
        # Extract year
        years = re.findall(r'\b(20[0-2][0-9])\b', first_page)
        year = int(years[0]) if years else None
        
        # Extract authors (simple heuristic)
        # Look for lines with names (capitals followed by lowercase)
        author_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        potential_authors = re.findall(author_pattern, first_page[:500])
        authors = list(set(potential_authors[:5]))  # Max 5 authors
        
        return PaperMetadata(
            title=title,
            authors=authors,
            year=year,
            num_pages=len(documents)
        )
    
    def _detect_sections(self, documents: List[Document], full_text: str) -> List[Section]:
        """
        Detect sections in paper
        
        Looks for common section headers:
        - Abstract
        - Introduction
        - Methods/Methodology
        - Results
        - Discussion
        - Conclusion
        - References
        """
        import re
        
        # Common section headers
        section_patterns = [
            r'\n\s*(Abstract|ABSTRACT)\s*\n',
            r'\n\s*(\d+\.?\s+)?Introduction\s*\n',
            r'\n\s*(\d+\.?\s+)?(Related Work|Background)\s*\n',
            r'\n\s*(\d+\.?\s+)?(Methods?|Methodology)\s*\n',
            r'\n\s*(\d+\.?\s+)?(Experiments?|Results?)\s*\n',
            r'\n\s*(\d+\.?\s+)?Discussion\s*\n',
            r'\n\s*(\d+\.?\s+)?Conclusion\s*\n',
            r'\n\s*References?\s*\n',
        ]
        
        sections = []
        matches = []
        
        # Find all section headers
        for pattern in section_patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                matches.append((match.start(), match.group(0).strip()))
        
        # Sort by position
        matches.sort(key=lambda x: x[0])
        
        # Create sections
        for i, (start_pos, header) in enumerate(matches):
            end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
            
            content = full_text[start_pos:end_pos].strip()
            
            section = Section(
                section_id=str(uuid.uuid4()),
                title=header,
                content=content,
                page_start=1,  # Calculate from position
                page_end=1     # Calculate from position
            )
            sections.append(section)
        
        # If no sections found, create one big section
        if not sections:
            sections.append(Section(
                section_id=str(uuid.uuid4()),
                title="Full Text",
                content=full_text,
                page_start=1,
                page_end=len(documents)
            ))
        
        return sections


# For backward compatibility
class PDFParser(LlamaIndexPDFParser):
    """Alias to maintain compatibility"""
    pass


# Factory function
def get_pdf_parser(file_path: str, advanced: bool = False):
    """
    Get PDF parser
    
    Args:
        file_path: Path to PDF file
        advanced: Use advanced parser (slower but better)
    
    Returns:
        Parser instance
    """
    if advanced:
        return AdvancedPDFParser(file_path)
    else:
        return LlamaIndexPDFParser(file_path)


if __name__ == "__main__":
    # Test PDF parsing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print(f"\n{'='*60}")
    print("  Testing LlamaIndex PDF Parser")
    print('='*60)
    
    # Parse PDF
    parser = LlamaIndexPDFParser(pdf_path)
    paper = parser.parse()
    
    print(f"\n✅ Parsed: {paper.filename}")
    print(f"   Title: {paper.metadata.title}")
    print(f"   Year: {paper.metadata.year}")
    print(f"   Pages: {paper.metadata.num_pages}")
    print(f"   Sections: {len(paper.sections)}")
    print(f"   Text length: {len(paper.raw_text)} chars")
    
    # Show first section
    if paper.sections:
        first_section = paper.sections[0]
        print(f"\n📄 First section: {first_section.title}")
        print(f"   Content preview: {first_section.content[:200]}...")


