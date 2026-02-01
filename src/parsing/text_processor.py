"""
Text processor for parsing and chunking documents
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any


class TextProcessor:
    """Process and chunk text documents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize text processor
        
        Args:
            config: Configuration dictionary with processing settings
        """
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # For large documents, use hierarchical chunking
        self.large_doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 3,  # Larger chunks for initial split
            chunk_overlap=self.chunk_overlap * 2,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", " ", "", ". ", ".\n"]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks, with special handling for large documents
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunked_documents = []
        
        for doc in documents:
            # Determine if document is large based on character count
            doc_length = len(doc.page_content)
            if doc_length > 10000:  # Consider document large if >10k characters
                # For large documents, use hierarchical chunking
                chunks = self._hierarchical_chunk_document(doc)
            else:
                # For regular documents, use standard chunking
                chunks = self.text_splitter.split_documents([doc])
            chunked_documents.extend(chunks)
        
        return chunked_documents
    
    def _hierarchical_chunk_document(self, document: Document) -> List[Document]:
        """
        Hierarchically chunk a large document
        
        Args:
            document: Large Document object to chunk hierarchically
            
        Returns:
            List of hierarchically chunked Document objects
        """
        # First level: Split into larger sections
        large_chunks = self.large_doc_splitter.split_documents([document])
        
        final_chunks = []
        for chunk in large_chunks:
            # For each large chunk, further subdivide if still too large
            if len(chunk.page_content) > 2000:
                sub_chunks = self.text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        # Add metadata to indicate this came from a large document
        for chunk in final_chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['large_document_chunk'] = True
            chunk.metadata['original_doc_length'] = len(document.page_content)
        
        return final_chunks
    
    def extract_metadata(self, document: Document, source_type: str = "organization") -> Dict[str, Any]:
        """
        Extract metadata from a document
        
        Args:
            document: Document to extract metadata from
            source_type: Type of document source ("organization" or "applicant")
            
        Returns:
            Dictionary of metadata
        """
        metadata = document.metadata.copy() if document.metadata else {}
        metadata['source_type'] = source_type
        
        # Add any additional metadata extraction logic here
        # For example, extract document name, creation date, etc.
        
        return metadata