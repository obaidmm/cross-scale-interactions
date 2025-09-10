# src/preprocessing/text_cleaner.py

import re
from typing import Dict, List
import unicodedata
from datetime import datetime

class TextCleaner:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.currency_symbols = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY'
        }

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning rules to input text
        """
        text = self._remove_special_chars(text)
        text = self._standardize_quotes(text)
        text = self._standardize_numbers(text)
        text = self._convert_currencies(text)
        text = self._standardize_dates(text)
        text = self._normalize_whitespace(text)
        text = self._standardize_bullets(text)
        return text

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters and symbols"""
        special_chars = ['§', '©', '®', '™']
        for char in special_chars:
            text = text.replace(char, '')
        return text

    def _standardize_quotes(self, text: str) -> str:
        """Standardize different types of quotes"""
        # Replace curly quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text

    def _standardize_numbers(self, text: str) -> str:
        """Convert numbers to standard format"""
        def replace_number(match):
            number = match.group(0).replace(',', '')
            return number
        return re.sub(r'\d{1,3}(,\d{3})+', replace_number, text)

    def _convert_currencies(self, text: str) -> str:
        """Replace currency symbols with ISO codes"""
        for symbol, code in self.currency_symbols.items():
            text = text.replace(symbol, code + ' ')
        return text

    def _standardize_dates(self, text: str) -> str:
        """Convert dates to ISO format"""
        # This is a simplified version - in practice you'd need more patterns
        def replace_date(match):
            try:
                date = datetime.strptime(match.group(0), '%B %d, %Y')
                return date.strftime('%Y-%m-%d')
            except ValueError:
                return match.group(0)
        
        text = re.sub(r'[A-Z][a-z]+ \d{1,2}, \d{4}', replace_date, text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Remove excessive whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        return text.strip()

    def _standardize_bullets(self, text: str) -> str:
        """Standardize different bullet point markers"""
        bullet_patterns = [r'•', r'∙', r'◦', r'⦿', r'⚫']
        for pattern in bullet_patterns:
            text = text.replace(pattern, '-')
        return text

# src/preprocessing/content_filter.py

class ContentFilter:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_bullet_words = 10
        
    def should_include(self, text: str, metadata: Dict) -> bool:
        """
        Determine if content should be included based on filtering rules
        """
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
        
    def _is_table(self, text: str) -> bool:
        """Check if text appears to be a table"""
        # Count number of numerical columns
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