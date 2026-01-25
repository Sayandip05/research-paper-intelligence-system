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
                title="Unknown",
                content=full_text,
                page_start=1,
                page_end=len(documents)
            ))
        
        return sections
    
    def _match_section_header(self, line: str) -> Optional[str]:
        """
        Match a line against section header patterns
        
        Returns normalized section title or None
        """
        for pattern, normalized in self.SECTION_PATTERNS:
            match = self.re.match(pattern, line, self.re.IGNORECASE)
            if match:
                if normalized:
                    return normalized
                else:
                    # Generic numbered section - extract title part
                    groups = match.groups()
                    if len(groups) >= 2:
                        raw_title = groups[1].strip()
                        # Try to normalize common variations
                        return self._normalize_section_title(raw_title)
        return None
    
    def _normalize_section_title(self, title: str) -> str:
        """Normalize section title to standard names"""
        title_lower = title.lower()
        
        # Map variations to standard names
        if 'abstract' in title_lower:
            return 'Abstract'
        elif 'introduc' in title_lower:
            return 'Introduction'
        elif 'related' in title_lower or 'background' in title_lower or 'literature' in title_lower:
            return 'Related Work'
        elif 'method' in title_lower or 'approach' in title_lower:
            return 'Methods'
        elif 'experiment' in title_lower or 'setup' in title_lower:
            return 'Experiments'
        elif 'result' in title_lower:
            return 'Results'
        elif 'discussion' in title_lower:
            return 'Discussion'
        elif 'limitation' in title_lower:
            return 'Limitations'
        elif 'future' in title_lower:
            return 'Future Work'
        elif 'conclu' in title_lower:
            return 'Conclusion'
        elif 'reference' in title_lower or 'bibliograph' in title_lower:
            return 'References'
        elif 'appendix' in title_lower or 'appendic' in title_lower:
            return 'Appendix'
        else:
            # Capitalize first letter
            return title.title() if title else 'Unknown'
    
    def _extract_section_content(
        self,
        page_texts: List[dict],
        start_page: int,
        start_line: int,
        next_match: Optional[dict]
    ) -> str:
        """
        Extract content between section header and next section
        """
        content_parts = []
        
        for page_info in page_texts:
            page_num = page_info['page_num']
            
            # Skip pages before section start
            if page_num < start_page:
                continue
            
            # Check if we've reached the next section
            if next_match and page_num > next_match['page_num']:
                break
            
            lines = page_info['text'].split('\n')
            
            # Handle start page
            if page_num == start_page:
                # Skip lines before section header
                lines = lines[start_line + 1:]
            
            # Handle page with next section
            if next_match and page_num == next_match['page_num']:
                # Only take lines before next section header
                lines = lines[:next_match['line_idx']]
            
            content_parts.append('\n'.join(lines))
        
        return '\n\n'.join(content_parts)


class SectionAwarePDFParser:
    """
    Section-aware PDF parser for research papers
    
    Detects real section headers and creates proper section boundaries.
    This is the RECOMMENDED parser for building section-aware RAG.
    
    Detected sections:
    - Abstract
    - Introduction
    - Related Work / Background
    - Methods / Methodology
    - Experiments / Experimental Setup
    - Results
    - Discussion
    - Limitations / Future Work
    - Conclusion
    - References / Bibliography
    - Appendix
    """
    
    # Section header patterns (order matters for normalization)
    SECTION_PATTERNS = [
        (r'^(?:Abstract|ABSTRACT)\s*$', 'Abstract'),
        (r'^(?:\d+\.?\s+)?(?:Introduction|INTRODUCTION)\s*$', 'Introduction'),
        (r'^(?:\d+\.?\s+)?(?:Related\s+Work|RELATED\s+WORK|Background|BACKGROUND|Literature\s+Review)\s*$', 'Related Work'),
        (r'^(?:\d+\.?\s+)?(?:Methods?|METHODS?|Methodology|METHODOLOGY)\s*$', 'Methods'),
        (r'^(?:\d+\.?\s+)?(?:Experiments?|EXPERIMENTS?|Experimental\s+Setup|Experimental\s+Settings?)\s*$', 'Experiments'),
        (r'^(?:\d+\.?\s+)?(?:Results?|RESULTS?)\s*$', 'Results'),
        (r'^(?:\d+\.?\s+)?(?:Discussion|DISCUSSION)\s*$', 'Discussion'),
        (r'^(?:\d+\.?\s+)?(?:Limitations?|LIMITATIONS?)\s*$', 'Limitations'),
        (r'^(?:\d+\.?\s+)?(?:Future\s+Work|FUTURE\s+WORK)\s*$', 'Future Work'),
        (r'^(?:\d+\.?\s+)?(?:Conclusions?|CONCLUSIONS?|Concluding\s+Remarks?)\s*$', 'Conclusion'),
        (r'^(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*$', 'References'),
        (r'^(?:Appendix|APPENDIX|Appendices|APPENDICES)(?:\s*[A-Z])?\.?\s*$', 'Appendix'),
        # Catch numbered sections like "3 Method" or "5.2 Experiments"
        (r'^(\d+\.?\d*)\s+(\w[\w\s]*?)\s*$', None),  # Generic numbered section
    ]
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.paper_id = str(uuid.uuid4())
        import re
        self.re = re
    
    def parse(self) -> ParsedPaper:
        """
        Parse PDF with section detection
        
        Returns:
            ParsedPaper with properly detected sections
        """
        print(f"   Parsing with Section Detection: {self.file_path.name}")
        
        # Load documents (one per page)
        documents = SimpleDirectoryReader(
            input_files=[str(self.file_path)]
        ).load_data()
        
        # Build page text with page numbers
        page_texts = []
        for i, doc in enumerate(documents, 1):
            page_texts.append({
                'page_num': i,
                'text': doc.text
            })
        
        # Detect sections across all pages
        sections = self._detect_sections(page_texts)
        
        # Extract metadata
        metadata = self._extract_metadata(documents)
        
        # Combine raw text
        raw_text = "\n\n".join(doc.text for doc in documents)
        
        print(f"      ✓ Detected {len(sections)} sections")
        for section in sections:
            print(f"        - {section.title} (pages {section.page_start}-{section.page_end})")
        
        return ParsedPaper(
            paper_id=self.paper_id,
            filename=self.file_path.name,
            metadata=metadata,
            sections=sections,
            raw_text=raw_text
        )
    
    def _extract_metadata(self, documents: List[Document]) -> PaperMetadata:
        """Extract metadata from documents"""
        if not documents:
            return PaperMetadata(title="Unknown", authors=[], year=None, num_pages=0)
        
        first_page = documents[0].text
        
        # Extract title from first significant line
        lines = [l.strip() for l in first_page.split('\n') if len(l.strip()) > 10]
        title = lines[0] if lines else self.file_path.stem
        
        # Clean title
        if title.endswith('.pdf'):
            title = title[:-4]
        # Limit length
        if len(title) > 100:
            title = title[:100] + "..."
        
        # Extract year
        years = self.re.findall(r'\b(20[0-2][0-9])\b', first_page[:1000])
        year = int(years[0]) if years else None
        
        return PaperMetadata(
            title=title,
            authors=[],
            year=year,
            num_pages=len(documents)
        )
    
    def _detect_sections(self, page_texts: List[dict]) -> List[Section]:
        """
        Detect sections by finding headers in page text
        
        Returns list of Section objects with proper page ranges
        """
        # Collect all potential section headers with positions
        section_matches = []
        
        for page_info in page_texts:
            page_num = page_info['page_num']
            text = page_info['text']
            
            # Split into lines
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line_clean = line.strip()
                
                # Skip empty or very short lines
                if len(line_clean) < 3:
                    continue
                
                # Skip very long lines (not headers)
                if len(line_clean) > 80:
                    continue
                
                # Try to match against section patterns
                normalized_title = self._match_section_header(line_clean)
                
                if normalized_title:
                    section_matches.append({
                        'title': normalized_title,
                        'page_num': page_num,
                        'line_idx': line_idx,
                        'original': line_clean
                    })
        
        # If no sections detected, return single "Unknown" section
        if not section_matches:
            full_text = "\n\n".join(p['text'] for p in page_texts)
            return [Section(
                section_id=str(uuid.uuid4()),
                title="Unknown",
                content=full_text,
                page_start=1,
                page_end=len(page_texts)
            )]
        
        # Build section content from matches
        sections = []
        
        for i, match in enumerate(section_matches):
            # Determine page range
            page_start = match['page_num']
            
            if i + 1 < len(section_matches):
                page_end = section_matches[i + 1]['page_num']
            else:
                page_end = len(page_texts)
            
            # Extract content between this section and the next
            content = self._extract_section_content(
                page_texts,
                match['page_num'],
                match['line_idx'],
                section_matches[i + 1] if i + 1 < len(section_matches) else None
            )
            
            # Skip empty sections
            if len(content.strip()) < 50:
                continue
            
            section = Section(
                section_id=str(uuid.uuid4()),
                title=match['title'],
                content=content,
                page_start=page_start,
                page_end=page_end
            )
            sections.append(section)
        
        return sections if sections else [Section(
            section_id=str(uuid.uuid4()),
            title="Unknown",
            content="\n\n".join(p['text'] for p in page_texts),
            page_start=1,
            page_end=len(page_texts)
        )]
    
    def _match_section_header(self, line: str) -> Optional[str]:
        """
        Match a line against section header patterns
        
        Returns normalized section title or None
        """
        for pattern, normalized in self.SECTION_PATTERNS:
            match = self.re.match(pattern, line, self.re.IGNORECASE)
            if match:
                if normalized:
                    return normalized
                else:
                    # Generic numbered section - extract title part
                    groups = match.groups()
                    if len(groups) >= 2:
                        raw_title = groups[1].strip()
                        # Try to normalize common variations
                        return self._normalize_section_title(raw_title)
        return None
    
    def _normalize_section_title(self, title: str) -> str:
        """
        Normalize raw section title to STRICT CANONICAL taxonomy.
        
        ONLY these 13 values are allowed:
        - Abstract, Introduction, Related Work, Methods, Experiments
        - Results, Discussion, Limitations, Future Work, Conclusion
        - References, Appendix, Unknown
        
        All noisy/invalid headers map to "Unknown".
        """
        if not title:
            return 'Unknown'
        
        # Step 1: Clean and normalize whitespace
        cleaned = ' '.join(title.split()).strip()
        
        # Step 2: Reject noise patterns BEFORE keyword matching
        if self._is_noise(cleaned):
            return 'Unknown'
        
        # Step 3: Keyword-based canonical mapping (case-insensitive)
        title_lower = cleaned.lower()
        
        # Exact/keyword matches in priority order
        if 'abstract' in title_lower:
            return 'Abstract'
        elif 'introduc' in title_lower or 'problem statement' in title_lower:
            return 'Introduction'
        elif any(kw in title_lower for kw in ['related work', 'background', 'literature', 'prior work']):
            return 'Related Work'
        elif any(kw in title_lower for kw in ['method', 'approach', 'technique', 'algorithm']):
            return 'Methods'
        elif any(kw in title_lower for kw in ['experiment', 'evaluation', 'setup', 'setting', 'benchmark']):
            return 'Experiments'
        elif 'result' in title_lower:
            return 'Results'
        elif 'discussion' in title_lower:
            return 'Discussion'
        elif 'limitation' in title_lower:
            return 'Limitations'
        elif 'future' in title_lower:
            return 'Future Work'
        elif any(kw in title_lower for kw in ['conclu', 'summary', 'closing']):
            return 'Conclusion'
        elif any(kw in title_lower for kw in ['reference', 'bibliograph', 'citation']):
            return 'References'
        elif any(kw in title_lower for kw in ['appendix', 'appendic', 'supplement']):
            return 'Appendix'
        else:
            # No match - return Unknown (NEVER return raw title)
            return 'Unknown'
    
    def _is_noise(self, text: str) -> bool:
        """
        Detect noisy headers that should NOT be treated as sections.
        
        Returns True if the text is noise (should be rejected).
        """
        if not text:
            return True
        
        # Very short strings (1-2 chars) are noise
        if len(text) <= 2:
            return True
        
        # Mostly numbers (e.g., "2 4 8 16 32")
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.3:  # Less than 30% letters
            return True
        
        # Common noise patterns
        noise_patterns = [
            r'^[IVXLCDM]+$',  # Roman numerals only
            r'^[A-Z]\.?\d*$',  # Single letter labels like "A", "B.1"
            r'^(Figure|Fig|Table|Tab|Equation|Eq)\.?\s*\d*',  # Figure/Table labels
            r'^\d+(\.\d+)*$',  # Pure numbers like "3.2.1"
            r'^[a-z]{1,3}\d*$',  # Variable names like "x1", "bf16"
            r'^\W+$',  # Only special characters
        ]
        
        for pattern in noise_patterns:
            if self.re.match(pattern, text, self.re.IGNORECASE):
                return True
        
        return False
    
    def _extract_section_content(
        self,
        page_texts: List[dict],
        start_page: int,
        start_line: int,
        next_match: Optional[dict]
    ) -> str:
        """
        Extract content between section header and next section
        """
        content_parts = []
        
        for page_info in page_texts:
            page_num = page_info['page_num']
            
            # Skip pages before section start
            if page_num < start_page:
                continue
            
            # Check if we've reached the next section
            if next_match and page_num > next_match['page_num']:
                break
            
            lines = page_info['text'].split('\n')
            
            # Handle start page
            if page_num == start_page:
                # Skip lines before section header
                lines = lines[start_line + 1:]
            
            # Handle page with next section
            if next_match and page_num == next_match['page_num']:
                # Only take lines before next section header
                lines = lines[:next_match['line_idx']]
            
            content_parts.append('\n'.join(lines))
        
        return '\n\n'.join(content_parts)


# For backward compatibility
class PDFParser(LlamaIndexPDFParser):
    """Alias to maintain compatibility"""
    pass


# Factory function
def get_pdf_parser(file_path: str, advanced: bool = False, section_aware: bool = False):
    """
    Get PDF parser
    
    Args:
        file_path: Path to PDF file
        advanced: Use advanced parser (slower but better)
        section_aware: Use section-aware parser (RECOMMENDED for RAG)
    
    Returns:
        Parser instance
    """
    if section_aware:
        return SectionAwarePDFParser(file_path)
    elif advanced:
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


