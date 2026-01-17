import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional
from app.models.paper import (
    ParsedPaper, Section, Table, Figure, Equation,
    PaperMetadata, Author
)
import uuid
from datetime import datetime


class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = fitz.open(file_path)
        self.num_pages = len(self.doc)
        
    def parse(self) -> ParsedPaper:
        """Main parsing function"""
        paper_id = str(uuid.uuid4())
        
        # Extract metadata
        metadata = self._extract_metadata()
        
        # Extract text with structure
        raw_text = self._extract_raw_text()
        sections = self._extract_sections()
        
        # Extract tables, figures, equations
        tables = self._extract_tables()
        figures = self._extract_figures()
        equations = self._extract_equations()
        
        return ParsedPaper(
            paper_id=paper_id,
            filename=self.file_path.split('/')[-1],
            file_path=self.file_path,
            metadata=metadata,
            sections=sections,
            tables=tables,
            figures=figures,
            equations=equations,
            raw_text=raw_text,
            upload_date=datetime.utcnow(),
            processing_status="completed"
        )
    
    def _extract_metadata(self) -> PaperMetadata:
        """Extract paper metadata from PDF"""
        # Try to get metadata from PDF info
        pdf_metadata = self.doc.metadata
        
        # Extract from first page (common pattern)
        first_page = self.doc[0]
        text = first_page.get_text()
        
        # Extract title (usually in large font at top)
        title = self._extract_title(text)
        
        # Extract authors
        authors = self._extract_authors(text)
        
        # Extract year
        year = self._extract_year(text)
        
        # Extract abstract
        abstract = self._extract_abstract()
        
        return PaperMetadata(
            title=title or pdf_metadata.get('title', 'Unknown'),
            authors=authors,
            year=year,
            abstract=abstract,
            num_pages=self.num_pages
        )
    
    def _extract_title(self, first_page_text: str) -> Optional[str]:
        """Extract title from first page"""
        lines = first_page_text.split('\n')
        # Title is usually one of the first non-empty lines
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and not line.startswith('http'):
                return line
        return None
    
    def _extract_authors(self, first_page_text: str) -> List[Author]:
        """Extract authors from first page"""
        # Simple heuristic: look for names after title
        # In real implementation, use more sophisticated NLP
        authors = []
        
        # Look for email patterns to find author section
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, first_page_text)
        
        # For now, return empty list
        # TODO: Implement proper author extraction
        return authors
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year"""
        # Look for 4-digit years (2000-2099)
        years = re.findall(r'\b(20[0-2][0-9])\b', text)
        if years:
            return int(years[0])
        return None
    
    def _extract_abstract(self) -> Optional[str]:
        """Extract abstract section"""
        for page_num in range(min(3, self.num_pages)):  # Check first 3 pages
            page = self.doc[page_num]
            text = page.get_text()
            
            # Look for "Abstract" section
            match = re.search(
                r'Abstract\s*(.+?)(?=\n\s*\n|\n\s*1\.|Introduction)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_raw_text(self) -> str:
        """Extract all text from PDF"""
        text = ""
        for page in self.doc:
            text += page.get_text()
        return text
    
    def _extract_sections(self) -> List[Section]:
        """Extract sections with hierarchy"""
        sections = []
        current_section_id = None
        
        for page_num, page in enumerate(self.doc):
            text = page.get_text("dict")  # Get structured text with fonts
            blocks = text.get("blocks", [])
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    lines = block.get("lines", [])
                    for line in lines:
                        spans = line.get("spans", [])
                        for span in spans:
                            text_content = span.get("text", "").strip()
                            font_size = span.get("size", 0)
                            
                            # Heuristic: larger font = section header
                            if font_size > 11 and self._is_section_header(text_content):
                                section_id = str(uuid.uuid4())
                                level = self._get_section_level(text_content)
                                
                                sections.append(Section(
                                    section_id=section_id,
                                    title=text_content,
                                    level=level,
                                    content="",  # Will be filled later
                                    page_start=page_num + 1,
                                    page_end=page_num + 1,
                                    parent_section_id=current_section_id
                                ))
                                
                                if level == 1:
                                    current_section_id = section_id
        
        return sections
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header"""
        # Common section patterns
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][a-z]+\s*$',  # "Introduction"
            r'^\d+\.\d+',  # "1.1 Background"
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        # Check if it's a common section name
        common_sections = [
            'abstract', 'introduction', 'background', 'related work',
            'methodology', 'methods', 'experiments', 'results',
            'discussion', 'conclusion', 'references', 'appendix'
        ]
        
        return text.lower() in common_sections
    
    def _get_section_level(self, text: str) -> int:
        """Determine section hierarchy level"""
        # "1. Introduction" = level 1
        # "1.1 Background" = level 2
        # "1.1.1 Details" = level 3
        
        match = re.match(r'^(\d+\.)+', text)
        if match:
            return match.group(0).count('.')
        return 1
    
    def _extract_tables(self) -> List[Table]:
        """Extract tables from PDF"""
        tables = []
        
        for page_num, page in enumerate(self.doc):
            # Look for table-like structures
            # This is a simplified version
            text = page.get_text()
            
            # Look for "Table X:" captions
            table_matches = re.finditer(
                r'Table\s+(\d+)[:\.]?\s*(.+?)(?=\n)',
                text,
                re.IGNORECASE
            )
            
            for match in table_matches:
                table_id = f"table_{match.group(1)}"
                caption = match.group(2).strip()
                
                tables.append(Table(
                    table_id=table_id,
                    caption=caption,
                    content="",  # TODO: Extract actual table content
                    page=page_num + 1
                ))
        
        return tables
    
    def _extract_figures(self) -> List[Figure]:
        """Extract figure references"""
        figures = []
        
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            
            # Look for "Figure X:" captions
            figure_matches = re.finditer(
                r'Figure\s+(\d+)[:\.]?\s*(.+?)(?=\n)',
                text,
                re.IGNORECASE
            )
            
            for match in figure_matches:
                figure_id = f"figure_{match.group(1)}"
                caption = match.group(2).strip()
                
                figures.append(Figure(
                    figure_id=figure_id,
                    caption=caption,
                    page=page_num + 1
                ))
        
        return figures
    
    def _extract_equations(self) -> List[Equation]:
        """Extract equations (basic implementation)"""
        equations = []
        
        # TODO: Implement proper LaTeX equation extraction
        # This requires more sophisticated parsing
        
        return equations
    
    def close(self):
        """Close PDF document"""
        self.doc.close()