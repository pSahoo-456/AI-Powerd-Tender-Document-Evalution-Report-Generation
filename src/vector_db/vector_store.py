"""
Vector store for managing document embeddings
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings


class VectorStoreManager:
    """Manage vector storage for document embeddings"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize vector store manager
        
        Args:
            config: Configuration dictionary with vector DB settings
        """
        self.config = config or {}
        self.db_type = self.config.get('type', 'faiss')
        persist_dir = self.config.get('persist_directory', './data/vectorstore')
        
        # Convert relative path to absolute path based on project root
        if not os.path.isabs(persist_dir):
            project_root = Path(__file__).parent.parent.parent
            self.persist_directory = str(project_root / persist_dir)
        else:
            self.persist_directory = persist_dir
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings with error handling
        ollama_config = self.config.get('ollama', {})
        try:
            self.embeddings = OllamaEmbeddings(
                model=ollama_config.get('embedding_model', 'nomic-embed-text'),
                base_url=ollama_config.get('base_url', 'http://localhost:11434')
            )
        except Exception as e:
            print(f"Warning: Could not initialize Ollama embeddings: {e}")
            print("Vector store will use fallback embeddings.")
            # Create a simple fallback embedding function
            from langchain.embeddings.base import Embeddings
            class FallbackEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    import random
                    return [[random.uniform(-1, 1) for _ in range(384)] for _ in texts]
                def embed_query(self, text):
                    import random
                    return [random.uniform(-1, 1) for _ in range(384)]
            self.embeddings = FallbackEmbeddings()
        
        self.vector_store = None
    
    def initialize_store(self, documents: List[Document] = None):
        """
        Initialize the vector store with documents
        
        Args:
            documents: List of Document objects to initialize with (optional)
        """
        # Create a fresh vector store instance to avoid any contamination from previous runs
        self.vector_store = None
        if self.db_type == 'faiss':
            self._initialize_faiss(documents)
        elif self.db_type == 'chroma':
            self._initialize_chroma(documents)
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _initialize_faiss(self, documents: List[Document] = None):
        """
        Initialize FAISS vector store
        
        Args:
            documents: List of Document objects to initialize with (optional)
        """
        persist_path = Path(self.persist_directory) / "faiss_index"
        
        # Clear any existing vector store to avoid contamination from previous runs
        import shutil
        import os
        if persist_path.exists():
            try:
                # Remove the entire directory
                if persist_path.is_dir():
                    shutil.rmtree(persist_path)
                else:
                    # If it's a file, remove it
                    os.remove(persist_path)
            except Exception as e:
                print(f"Warning: Could not remove existing vector store: {e}")
        
        if documents:
            # Create new FAISS store with documents
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            
            # Save the store
            self.vector_store.save_local(str(persist_path))
        else:
            # Create empty FAISS store
            from langchain_core.documents import Document
            dummy_doc = Document(page_content="Dummy document for initialization", metadata={})
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            self.vector_store.save_local(str(persist_path))
    
    def _initialize_chroma(self, documents: List[Document] = None):
        """
        Initialize Chroma vector store
        
        Args:
            documents: List of Document objects to initialize with (optional)
        """
        persist_path = Path(self.persist_directory) / "chroma_db"
        
        # Clear any existing vector store to avoid contamination from previous runs
        import shutil
        import os
        if persist_path.exists():
            try:
                # Remove the entire directory
                if persist_path.is_dir():
                    shutil.rmtree(persist_path)
                else:
                    # If it's a file, remove it
                    os.remove(persist_path)
            except Exception as e:
                print(f"Warning: Could not remove existing vector store: {e}")
        
        if documents:
            # Create new Chroma store with documents
            self.vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=str(persist_path)
            )
        else:
            # Create new Chroma store
            self.vector_store = Chroma(
                persist_directory=str(persist_path),
                embedding_function=self.embeddings
            )
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return
        
        # Process documents in batches to manage memory usage
        batch_size = 10  # Adjust batch size based on memory constraints
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            if not self.vector_store:
                self.initialize_store(batch)
            else:
                self.vector_store.add_documents(batch)
                
                # Save periodically to persist progress
                if self.db_type == 'faiss':
                    persist_path = Path(self.persist_directory) / "faiss_index"
                    self.vector_store.save_local(str(persist_path))
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store
        
        Args:
            query: Query string to search for
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        # Perform similarity search with error handling
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            # Add metadata about search results for large document handling
            # Use enumerate instead of index() to avoid issues with duplicate documents
            for idx, result in enumerate(results):
                if result.metadata is None:
                    result.metadata = {}
                result.metadata['search_rank'] = idx + 1
            
            return results
        except Exception as e:
            print(f"Warning: Similarity search failed: {e}")
            # Return empty list if search fails
            return []
    
    def save_store(self):
        """Save the vector store to disk"""
        if not self.vector_store:
            return
            
        if self.db_type == 'faiss':
            persist_path = Path(self.persist_directory) / "faiss_index"
            self.vector_store.save_local(str(persist_path))
        # Chroma persists automatically