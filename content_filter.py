# src/preprocessing/content_filter.py

from typing import Dict, List, Optional
import re

class ContentFilter:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_bullet_words = 10
        self.excluded_sections = self.config.get('excluded_sections', [
            'financial statements',
            'notes to financial statements',
            'independent auditor\'s report'
        ])
        
    def should_include(self, text: str, metadata: Dict) -> bool:
        """Determine if content should be included based on filtering rules"""
        if self._is_excluded_section(metadata.get('section', '')):
            return False
        if self._is_table(text):
            return False
        if self._is_financial_statement(text):
            return False
        if self._is_legal_disclaimer(text):
            return False
        if self._is_short_paragraph(text):
            return False
        if self._is_short_bullet_list(text):
            return False
        if self._is_footnote(text):
            return False
        if self._is_image_caption(text):
            return False
            
        return True
        
    def _is_excluded_section(self, section: str) -> bool:
        """Check if section is in excluded list"""
        section = section.lower()
        return any(excl.lower() in section for excl in self.excluded_sections)
        
    def _is_table(self, text: str) -> bool:
        """Check if text appears to be a table"""
        lines = text.split('\n')
        num_columns = 0
        for line in lines:
            numbers = re.findall(r'\b\d+\.?\d*\b', line)
            if len(numbers) > 3:
                num_columns += 1
        return num_columns > len(lines) / 2

    def _is_financial_statement(self, text: str) -> bool:
        """Check if text appears to be a financial statement"""
        financial_keywords = [
            'balance sheet', 'income statement', 'cash flow',
            'statement of operations', 'financial position'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in financial_keywords)

    def _is_legal_disclaimer(self, text: str) -> bool:
        """Check if text is a legal disclaimer"""
        disclaimer_patterns = [
            r'forward-looking statements?',
            r'safe harbor',
            r'legal notice',
            r'terms and conditions'
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in disclaimer_patterns)

    def _is_short_paragraph(self, text: str) -> bool:
        """Check if text is a short single-sentence paragraph"""
        if len(text) < 100 and len(re.findall(r'[.!?]+', text)) <= 1:
            return True
        return False

    def _is_short_bullet_list(self, text: str) -> bool:
        """Check if text is a bullet list with short items"""
        bullet_items = re.split(r'[-•]', text)
        if len(bullet_items) > 1:
            words_per_item = [len(item.split()) for item in bullet_items]
            return any(count < self.min_bullet_words for count in words_per_item)
        return False

    def _is_footnote(self, text: str) -> bool:
        """Check if text appears to be a footnote"""
        footnote_patterns = [
            r'^\d+\s+[A-Z]',  # Numbered footnote
            r'^\*\s+',        # Asterisk footnote
            r'^Note\s+\d+:'   # Note reference
        ]
        return any(re.match(pattern, text) for pattern in footnote_patterns)

    def _is_image_caption(self, text: str) -> bool:
        """Check if text appears to be an image caption"""
        caption_patterns = [
            r'^(Figure|Fig\.|Table|Exhibit)\s+\d+',
            r'^Image:',
            r'^\(Source:'
        ]
        return any(re.match(pattern, text) for pattern in caption_patterns)