"""
Document loader for the tender evaluation system
"""

import os
import io
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

from src.utils.file_utils import get_files_in_directory, get_file_extension

# Try to import Docling, fall back to existing methods if not available
try:
    from docling.document_converter import DocumentConverter
    DOCSTRING_AVAILABLE = True
except ImportError:
    DOCSTRING_AVAILABLE = False
    from src.ocr.ocr_processor import OCRProcessor
    import pdfplumber


class DocumentLoader:
    """Load documents from various sources"""
    
    def __init__(self, ocr_config: Dict[str, Any] = None):
        self.supported_formats = {'.pdf', '.txt'}
        # Initialize OCR processor only when Docling is not available
        # With Docling as primary method, OCR is rarely needed
        if not DOCSTRING_AVAILABLE:
            from src.ocr.ocr_processor import OCRProcessor
            self.ocr_processor = OCRProcessor(ocr_config or {})
        else:
            self.ocr_processor = None  # Docling handles all PDF processing
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        documents = []
        files = get_files_in_directory(directory_path)
        
        for file_path in files:
            extension = get_file_extension(file_path)
            if extension in self.supported_formats:
                try:
                    docs = self._load_document(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
        
        return documents
    
    def _load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on its extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        extension = get_file_extension(file_path)
        
        if extension == '.pdf':
            return self._load_pdf_document(file_path)
        elif extension == '.txt':
            return self._load_text_document(file_path)
        else:
            raise ValueError(f"Unsupported document format: {extension}")
    
    def _load_pdf_document(self, file_path: str) -> List[Document]:
        """
        Load a PDF document using Docling as the primary method for all PDFs including scanned ones
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        text = ""
        
        if DOCSTRING_AVAILABLE:
            # Use Docling as the primary and preferred method for ALL PDFs
            try:
                converter = DocumentConverter()
                result = converter.convert(file_path)
                text = result.document.export_to_markdown()
                
                # If Docling returns empty text, try to extract with pdfplumber as backup
                if not text.strip():
                    print(f"Warning: Docling returned empty text for {file_path}, using pdfplumber as backup...")
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
            except Exception as e:
                print(f"Docling failed for {file_path}: {e}. Using pdfplumber as backup.")
                # Use pdfplumber as backup
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
        else:
            # Use original method when Docling is not available
            text = self._load_pdf_with_fallback(file_path)
        
        # Create document with extracted text
        doc = Document(page_content=text.strip(), metadata={"source": file_path})
        
        # Add document size information to metadata for large document handling
        doc.metadata['file_size_chars'] = len(text.strip())
        doc.metadata['is_large_document'] = len(text.strip()) > 10000  # Mark if document is large
        
        return [doc]
    
    def _load_pdf_with_fallback(self, file_path: str) -> str:
        """
        Original PDF loading method with fallback chain for backward compatibility
        """
        text = ""
        try:
            # Try to extract text using pdfplumber
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If no text was extracted, and OCR processor is available, fallback to OCR
            if not text.strip() and self.ocr_processor:
                print(f"No text found in {file_path}, using OCR fallback...")
                try:
                    text = self.ocr_processor.process_scanned_pdf(file_path)
                except RuntimeError as ocr_error:
                    print(f"OCR fallback failed for {file_path}: {ocr_error}")
                    # Still try to determine if it's a scanned PDF and provide more helpful error
                    if hasattr(self.ocr_processor, 'is_scanned_pdf') and self.ocr_processor.is_scanned_pdf(file_path):
                        raise RuntimeError(f"Scanned PDF detected but OCR failed: {ocr_error}")
                    else:
                        raise RuntimeError(f"Could not extract text from {file_path} and it doesn't appear to be a scanned PDF: {ocr_error}")
            elif not text.strip() and not self.ocr_processor:
                # If OCR processor is not available, just return the text as is
                print(f"No text found in {file_path} and OCR not available (using Docling)")
                
        except Exception as e:
            # If pdfplumber fails and OCR processor is available, try OCR as fallback
            if self.ocr_processor:
                print(f"pdfplumber failed for {file_path}: {e}. Using OCR fallback...")
                try:
                    text = self.ocr_processor.process_scanned_pdf(file_path)
                except Exception as ocr_error:
                    # Try to determine if it's a scanned PDF and provide more context
                    try:
                        if hasattr(self.ocr_processor, 'is_scanned_pdf') and self.ocr_processor.is_scanned_pdf(file_path):
                            raise RuntimeError(f"PDF appears to be scanned but OCR processing failed. Install Tesseract OCR to process scanned documents: {ocr_error}")
                        else:
                            raise RuntimeError(f"Failed to extract text from {file_path}: {ocr_error}")
                    except:
                        # If is_scanned_pdf fails, just raise the original error
                        raise RuntimeError(f"Failed to extract text from {file_path}: {ocr_error}")
            else:
                # If OCR is not available, just raise the original error
                raise RuntimeError(f"Failed to extract text from {file_path}: {e}")
        
        return text
    
    def _load_text_document(self, file_path: str) -> List[Document]:
        """
        Load a text document
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create document with extracted text
        doc = Document(page_content=text, metadata={"source": file_path})
        
        # Add document size information to metadata for large document handling
        doc.metadata['file_size_chars'] = len(text)
        doc.metadata['is_large_document'] = len(text) > 10000  # Mark if document is large
        
        return [doc]