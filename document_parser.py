# src/preprocessing/document_parser.py

import PyPDF2
from tika import parser
import pytesseract
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional
import re

class DocumentParser:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def parse_pdf(self, file_path: str) -> Dict[str, str]:
        """
        Parse PDF documents using multiple methods and return the best result
        """
        try:
            # Try PyPDF2 first
            content = self._parse_with_pypdf2(file_path)
            if self._is_valid_extraction(content):
                return {'content': content, 'method': 'pypdf2'}
            
            # Try Tika if PyPDF2 fails
            content = self._parse_with_tika(file_path)
            if self._is_valid_extraction(content):
                return {'content': content, 'method': 'tika'}
            
            # Fall back to OCR if needed
            content = self._parse_with_ocr(file_path)
            return {'content': content, 'method': 'ocr'}
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise
            
    def _parse_with_pypdf2(self, file_path: str) -> str:
        """Parse PDF using PyPDF2"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _parse_with_tika(self, file_path: str) -> str:
        """Parse PDF using Tika"""
        parsed = parser.from_file(file_path)
        return parsed["content"]

    def _parse_with_ocr(self, file_path: str) -> str:
        """Parse PDF using Tesseract OCR"""
        # Note: This is a simplified version. In practice, you'd need to convert
        # PDF pages to images first
        return pytesseract.image_to_string(file_path)

    def parse_html(self, content: str) -> str:
        """Parse HTML content"""
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text()

    def segment_paragraphs(self, text: str) -> List[Dict[str, str]]:
        """
        Segment text into paragraphs based on rules defined in config
        """
        paragraphs = []
        current_text = ""
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Check for section breaks
            if self._is_section_break(line):
                if current_text:
                    para = self._create_paragraph(current_text)
                    if para:
                        paragraphs.append(para)
                    current_text = ""
                continue
                
            current_text += line + " "
            
        # Add final paragraph
        if current_text:
            para = self._create_paragraph(current_text)
            if para:
                paragraphs.append(para)
                
        return paragraphs
        
    def _is_section_break(self, line: str) -> bool:
        """Check if line represents a section break"""
        if not line:
            return True
        if re.match(r'^[A-Z\s]{10,}$', line):  # All caps header
            return True
        if re.match(r'^\d+\.\s', line):  # Numbered section
            return True
        return False
        
    def _create_paragraph(self, text: str) -> Optional[Dict[str, str]]:
        """Create paragraph dict if text meets criteria"""
        text = text.strip()
        if len(text) < 50 or len(text) > 2000:
            return None
            
        sentences = len(re.findall(r'[.!?]+', text))
        if sentences < 2 or sentences > 15:
            return None
            
        return {
            'text': text,
            'char_count': len(text),
            'sentence_count': sentences
        }

    def _is_valid_extraction(self, content: str) -> bool:
        """
        Check if the extracted content meets quality criteria
        """
        if not content or len(content.strip()) < 100:
            return False
        
        # Check for common PDF extraction issues
        if '%PDF' in content or '/Type' in content:
            return False
            
        # Check for reasonable text-to-garbage ratio
        text_chars = len(re.findall(r'[a-zA-Z\s]', content))
        total_chars = len(content)
        if total_chars == 0 or text_chars / total_chars < 0.7:
            return False
            
        return True