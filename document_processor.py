# src/preprocessing/document_processor.py

import os
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from .document_parser import DocumentParser
from .content_filter import ContentFilter
from .text_cleaner import TextCleaner
from ..models.text_embedder import TextEmbedder
from ..models.high_level_classifier import SystemsThinkingClassifier
from ..models.subdimension_classifier import SubdimensionClassifier
from ..rag.vector_store import VectorStore
from ..rag.retriever import SemanticRetriever

class DocumentProcessor:
    """
    Main document processing pipeline implementing Steps 1-5 from the
    Systems Thinking research approach.
    
    This class orchestrates:
    1. Document parsing and cleaning
    2. Text segmentation
    3. Content filtering
    4. Embedding generation
    5. Systems thinking classification
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the document processor with all required components
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.parser = DocumentParser(config.get('parser_config', {}))
        self.cleaner = TextCleaner(config.get('cleaner_config', {}))
        self.content_filter = ContentFilter(config.get('filter_config', {}))
        
        # Initialize embedder
        embedding_model = config.get('embedding_model', 'text-embedding-3-large')
        self.embedder = TextEmbedder(embedding_model, config.get('embedder_config', {}))
        
        # Initialize classifiers if paths provided
        st_model_path = config.get('st_classifier_path')
        subdim_model_path = config.get('subdimension_classifier_path')
        
        if st_model_path:
            self.st_classifier = SystemsThinkingClassifier(
                st_model_path, 
                config.get('st_classifier_config', {})
            )
        else:
            self.st_classifier = None
            self.logger.warning("No Systems Thinking classifier path provided")
        
        if subdim_model_path:
            self.subdim_classifier = SubdimensionClassifier(
                subdim_model_path,
                config.get('subdim_classifier_config', {})
            )
        else:
            self.subdim_classifier = None
            self.logger.warning("No Subdimension classifier path provided")
        
        # Initialize vector store and retriever if provided
        if 'vector_store_config' in config:
            self.vector_store = VectorStore(config['vector_store_config'])
            self.retriever = SemanticRetriever(
                self.vector_store,
                self.embedder,
                config.get('retriever_config', {})
            )
        else:
            self.vector_store = None
            self.retriever = None
            
        # Set parameters from config
        self.confidence_threshold = config.get('confidence_threshold', 0.85)
        self.batch_size = config.get('batch_size', 32)
        self.min_character_length = config.get('min_character_length', 50)
        self.max_character_length = config.get('max_character_length', 2000)
        self.max_sentence_count = config.get('max_sentence_count', 99)
            
    def process_document(self, file_path: str) -> Dict:
        """
        Process a document through the entire pipeline
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing document: {file_path}")
        
        # Extract metadata
        file_name = os.path.basename(file_path)
        report_year = self._extract_year(file_name)
        company_name = self._extract_company(file_name)
        
        metadata = {
            'file_name': file_name,
            'company': company_name,
            'year': report_year,
            'processed_date': datetime.now().isoformat()
        }
        
        # Step 1: Parse document
        result = self.parser.parse_pdf(file_path)
        raw_text = result['content']
        metadata['parse_method'] = result['method']
        
        # Step 2: Clean text
        cleaned_text = self.cleaner.clean(raw_text)
        
        # Step 3: Segment into paragraphs
        paragraphs = self.parser.segment_paragraphs(cleaned_text)
        
        # Step 4: Filter irrelevant content
        filtered_paragraphs = []
        for para in paragraphs:
            if self.content_filter.should_include(para['text'], {}):
                para['metadata'] = metadata.copy()
                filtered_paragraphs.append(para)
                
        self.logger.info(f"Extracted {len(filtered_paragraphs)} paragraphs after filtering")
        
        # Step 5: Generate embeddings
        texts = [p['text'] for p in filtered_paragraphs]
        
        if texts:  # Only proceed if we have paragraphs
            embeddings = self.embedder.embed_batch(texts)
            
            # Step 6: Classify systems thinking
            if self.st_classifier:
                classifications = self.st_classifier.batch_predict(texts)
                
                # Add classification results to paragraphs
                for i, (is_st, confidence) in enumerate(classifications):
                    filtered_paragraphs[i]['is_systems_thinking'] = is_st
                    filtered_paragraphs[i]['st_confidence'] = confidence
                
                # Step 7: Classify subdimensions for positive ST paragraphs
                st_paragraphs = [p for i, p in enumerate(filtered_paragraphs) 
                               if classifications[i][0] and classifications[i][1] >= self.confidence_threshold]
                
                if self.subdim_classifier and st_paragraphs:
                    st_texts = [p['text'] for p in st_paragraphs]
                    
                    # Process in batches
                    for i in range(0, len(st_texts), self.batch_size):
                        batch_texts = st_texts[i:i+self.batch_size]
                        
                        # Use RAG if available to improve classification
                        if self.retriever:
                            # Get similar examples for context
                            contexts = [self.retriever.retrieve(text, k=3) for text in batch_texts]
                            
                            # Classify with context - implement this method in SubdimensionClassifier
                            if hasattr(self.subdim_classifier, 'predict_with_context'):
                                subdim_results = [
                                    self.subdim_classifier.predict_with_context(text, ctx) 
                                    for text, ctx in zip(batch_texts, contexts)
                                ]
                            else:
                                # Fall back to regular classification
                                subdim_results = [self.subdim_classifier.predict(text) for text in batch_texts]
                        else:
                            # Regular classification without RAG
                            subdim_results = [self.subdim_classifier.predict(text) for text in batch_texts]
                        
                        # Add subdimension results to paragraphs
                        for j, result in enumerate(subdim_results):
                            idx = i + j
                            if idx < len(st_paragraphs):
                                st_paragraphs[idx]['subdimensions'] = result
            else:
                self.logger.warning("No Systems Thinking classifier available, skipping classification")
        
        result = {
            'metadata': metadata,
            'paragraphs': filtered_paragraphs,
            'stats': {
                'total_paragraphs': len(paragraphs),
                'filtered_paragraphs': len(filtered_paragraphs),
                'systems_thinking_paragraphs': sum(1 for p in filtered_paragraphs 
                                                 if p.get('is_systems_thinking', False))
            }
        }
        
        return result
    
    def index_document(self, result: Dict) -> None:
        """
        Index document results to vector store for RAG
        
        Args:
            result: Document processing result
        """
        if not self.vector_store:
            self.logger.warning("No vector store available for indexing")
            return
            
        paragraphs = result['paragraphs']
        texts = [p['text'] for p in paragraphs]
        metadata_list = [p.get('metadata', {}) for p in paragraphs]
        
        # Add classification results to metadata
        for i, p in enumerate(paragraphs):
            if 'is_systems_thinking' in p:
                metadata_list[i]['is_systems_thinking'] = p['is_systems_thinking']
                metadata_list[i]['st_confidence'] = p['st_confidence']
                
            if 'subdimensions' in p:
                metadata_list[i]['subdimensions'] = p['subdimensions']
                
        # Generate embeddings if needed
        if not hasattr(paragraphs[0], 'embedding'):
            embeddings = self.embedder.embed_batch(texts)
        else:
            embeddings = np.array([p['embedding'] for p in paragraphs])
            
        # Store in vector database
        self.vector_store.store(embeddings, texts, metadata_list)
        self.logger.info(f"Indexed {len(texts)} paragraphs to vector store")
    
    def _extract_year(self, filename: str) -> Optional[int]:
        """Extract report year from filename"""
        # Simple extraction - look for 4-digit year
        import re
        match = re.search(r'(19|20)\d{2}', filename)
        if match:
            return int(match.group(0))
        return None
    
    def _extract_company(self, filename: str) -> str:
        """Extract company name from filename"""
        # Simple extraction - remove extension and year
        base = os.path.splitext(filename)[0]
        # Remove year if present
        company = re.sub(r'(19|20)\d{2}', '', base)
        # Clean up
        company = re.sub(r'[_\-]', ' ', company).strip()
        return company
        
    def export_results(self, result: Dict, output_path: str) -> str:
        """
        Export processing results to CSV
        
        Args:
            result: Document processing result
            output_path: Directory to save output
            
        Returns:
            Path to output file
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Create dataframe from paragraphs
        rows = []
        for p in result['paragraphs']:
            row = {
                'company': result['metadata']['company'],
                'year': result['metadata']['year'],
                'text': p['text'],
                'char_count': p.get('char_count', len(p['text'])),
                'sentence_count': p.get('sentence_count', 0),
                'is_systems_thinking': p.get('is_systems_thinking', False),
                'st_confidence': p.get('st_confidence', 0.0)
            }
            
            # Add subdimensions if available
            if 'subdimensions' in p:
                for dim, value in p['subdimensions'].items():
                    row[f'subdim_{dim}'] = value
                    
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Generate output filename
        company = result['metadata']['company'].replace(' ', '_')
        year = result['metadata']['year'] or 'unknown'
        filename = f"{company}_{year}_systems_thinking_analysis.csv"
        output_file = os.path.join(output_path, filename)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        self.logger.info(f"Results exported to {output_file}")
        
        return output_file
