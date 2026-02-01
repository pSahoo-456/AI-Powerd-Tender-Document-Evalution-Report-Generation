"""
OCR processor for extracting text from scanned PDFs
"""

import pytesseract
from PIL import Image
import pdf2image
import io
from pathlib import Path
from typing import List, Dict, Any


class OCRProcessor:
    """Process scanned PDFs using OCR to extract text"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize OCR processor
        
        Args:
            config: Configuration dictionary with OCR settings
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.language = self.config.get('language', 'eng')
        self.tesseract_available = self._check_tesseract()
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def process_scanned_pdf(self, pdf_path: str) -> str:
        """
        Process a scanned PDF and extract text using OCR
        
        Args:
            pdf_path: Path to the scanned PDF file
            
        Returns:
            Extracted text from the PDF
        """
        if not self.enabled:
            raise RuntimeError("OCR is not enabled in configuration")
        
        if not self.tesseract_available:
            raise RuntimeError("Tesseract OCR is not installed. Please install Tesseract to use OCR functionality.")
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            # Extract text from each image
            extracted_text = ""
            for i, image in enumerate(images):
                # Preprocess image for better OCR results
                # Convert to grayscale
                image = image.convert('L')
                
                # Apply threshold to get image with only black and white
                # image = image.point(lambda x: 0 if x < 128 else 255, '1')
                
                text = pytesseract.image_to_string(image, lang=self.language)
                extracted_text += f"\n--- Page {i+1} ---\n{text}\n"
            
            return extracted_text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF with OCR: {str(e)}")
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Heuristic check to determine if a PDF is scanned (image-based)
        This is a simplified check - in practice, more sophisticated methods might be needed
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF appears to be scanned, False otherwise
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Check first few pages for text content
            pages_to_check = min(3, len(doc))
            text_content = ""
            
            for i in range(pages_to_check):
                page = doc.load_page(i)
                text_content += page.get_text()
            
            doc.close()
            
            # If very little text is found, it's likely a scanned PDF
            # We'll use a threshold of 100 characters as an indicator
            return len(text_content.strip()) < 100
        except Exception as e:
            # If we can't determine, assume it's not scanned
            print(f"Warning: Could not determine if PDF is scanned: {e}")
            return False