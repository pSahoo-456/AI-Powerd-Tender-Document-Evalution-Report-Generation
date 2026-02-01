"""
Similarity search functionality for document comparison
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from src.vector_db.vector_store import VectorStoreManager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilaritySearcher:
    """Perform similarity search between organization requirements and applicant documents"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initialize similarity searcher
        
        Args:
            vector_store_manager: Vector store manager instance
        """
        self.vector_store = vector_store_manager
    
    def search_applicants_by_requirements(self, requirements: List[Document], 
                                        applicants: List[Document], 
                                        top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for the most similar applicant documents to the organization requirements
        
        Args:
            requirements: List of requirement documents
            applicants: List of applicant documents
            top_k: Number of top matches to return
            
        Returns:
            List of tuples (applicant_document, similarity_score)
        """
        if not applicants:
            return []
        
        # Try to use vector store search first
        try:
            # Combine all requirement texts into a single query
            requirement_texts = [req.page_content for req in requirements]
            combined_query = " ".join(requirement_texts)
            
            # For large document collections, process in batches
            if len(applicants) > 50:
                # Process applicants in batches to manage memory
                batch_size = 20
                all_results = []
                
                for i in range(0, len(applicants), batch_size):
                    batch = applicants[i:i + batch_size]
                    
                    # Use vector store search on batch
                    similar_docs = self.vector_store.similarity_search(combined_query, k=min(top_k, len(batch)))
                    
                    # Filter out organization documents - only keep applicant documents from this batch
                    batch_sources = {applicant.metadata.get('source') for applicant in batch if applicant.metadata and applicant.metadata.get('source')}
                    
                    # Create a mapping to track original document content and metadata
                    # Use unique_id if available, otherwise fall back to source
                    original_docs_map = {}
                    for doc in batch:
                        if doc.metadata:
                            # Prefer unique_id if available, otherwise use source
                            key = doc.metadata.get('unique_id') or doc.metadata.get('source')
                            if key:
                                original_docs_map[key] = doc
                    
                    # Filter docs but preserve original metadata where possible
                    filtered_docs = []
                    for doc in similar_docs:
                        if doc.metadata and (doc.metadata.get('source') in batch_sources or doc.metadata.get('unique_id')):
                            # Try to preserve original document with its original metadata
                            # Use unique_id if available, otherwise fall back to source
                            original_key = doc.metadata.get('unique_id') or doc.metadata.get('source')
                            if original_key in original_docs_map:
                                # Use original document to preserve original metadata
                                original_doc = original_docs_map[original_key]
                                # Create new doc with original content/metadata but similarity search results
                                preserved_doc = Document(
                                    page_content=doc.page_content,  # Use content from similarity search
                                    metadata=original_doc.metadata.copy()  # Preserve original metadata
                                )
                                filtered_docs.append(preserved_doc)
                            else:
                                filtered_docs.append(doc)
                        else:
                            # This might be an organization document, skip it
                            continue
                    
                    # Calculate similarity scores for filtered docs
                    batch_results = self._calculate_batch_similarities(requirements, filtered_docs, combined_query)
                    all_results.extend(batch_results)
                    
                # Sort all results and return top_k
                all_results.sort(key=lambda x: x[1], reverse=True)
                return all_results[:top_k]
            else:
                # Use vector store search
                similar_docs = self.vector_store.similarity_search(combined_query, k=min(top_k * 2, len(applicants)))
                
                # Filter out organization documents - only keep applicant documents
                applicant_sources = {applicant.metadata.get('source') for applicant in applicants if applicant.metadata and applicant.metadata.get('source')}
                
                # Create a mapping to track original document content and metadata
                # Use unique_id if available, otherwise fall back to source
                original_docs_map = {}
                for doc in applicants:
                    if doc.metadata:
                        # Prefer unique_id if available, otherwise use source
                        key = doc.metadata.get('unique_id') or doc.metadata.get('source')
                        if key:
                            original_docs_map[key] = doc
                
                # Filter docs but preserve original metadata where possible
                filtered_docs = []
                for doc in similar_docs:
                    if doc.metadata and (doc.metadata.get('source') in applicant_sources or doc.metadata.get('unique_id')):
                        # Try to preserve original document with its original metadata
                        # Use unique_id if available, otherwise fall back to source
                        original_key = doc.metadata.get('unique_id') or doc.metadata.get('source')
                        if original_key in original_docs_map:
                            # Use original document to preserve original metadata
                            original_doc = original_docs_map[original_key]
                            # Create new doc with original content/metadata but similarity search results
                            preserved_doc = Document(
                                page_content=doc.page_content,  # Use content from similarity search
                                metadata=original_doc.metadata.copy()  # Preserve original metadata
                            )
                            filtered_docs.append(preserved_doc)
                        else:
                            filtered_docs.append(doc)
                    else:
                        # This might be an organization document, skip it
                        continue
                
                # Calculate similarity scores for filtered docs
                results = self._calculate_batch_similarities(requirements, filtered_docs, combined_query)
                
                # Sort by similarity score (descending) and return top_k
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
            
        except Exception as e:
            # If similarity search fails, fall back to simple ranking
            print(f"Similarity search failed, using fast ranking: {e}")
            return self.rank_applicants(requirements, applicants[:top_k])
    
    def _calculate_batch_similarities(self, requirements: List[Document], filtered_docs: List[Document], combined_query: str) -> List[Tuple[Document, float]]:
        """
        Calculate similarity scores for a batch of documents
        
        Args:
            requirements: List of requirement documents
            filtered_docs: List of documents to calculate similarities for
            combined_query: Combined query string
            
        Returns:
            List of tuples (document, similarity_score)
        """
        # Get embeddings for requirements and filtered documents
        req_embeddings = [req.metadata.get('embedding') for req in requirements if req.metadata.get('embedding')]
        doc_embeddings = [doc.metadata.get('embedding') for doc in filtered_docs if doc.metadata.get('embedding')]
        
        # If we have embeddings, use cosine similarity
        if req_embeddings and doc_embeddings:
            # Average requirement embeddings
            avg_req_embedding = np.mean(req_embeddings, axis=0)
            
            # Calculate cosine similarity for each document
            results = []
            for doc in filtered_docs:
                doc_embedding = doc.metadata.get('embedding')
                if doc_embedding:
                    # Calculate cosine similarity
                    sim_score = cosine_similarity([avg_req_embedding], [doc_embedding])[0][0]
                    
                    # Add similarity score to document metadata
                    doc_with_score = Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata.copy() if doc.metadata else {}
                    )
                    doc_with_score.metadata['similarity_score'] = float(sim_score)
                    results.append((doc_with_score, sim_score))
                else:
                    # Fallback to text-based similarity
                    score = self._calculate_similarity(combined_query, doc.page_content)
                    doc_with_score = Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata.copy() if doc.metadata else {}
                    )
                    doc_with_score.metadata['similarity_score'] = float(score)
                    results.append((doc_with_score, score))
        else:
            # Fallback to text-based similarity
            results = []
            for doc in filtered_docs:
                score = self._calculate_similarity(combined_query, doc.page_content)
                doc_with_score = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                doc_with_score.metadata['similarity_score'] = float(score)
                results.append((doc_with_score, score))
        
        return results
    
    def rank_applicants(self, requirements: List[Document], 
                       applicants: List[Document]) -> List[Tuple[Document, float]]:
        """
        Rank applicants based on similarity to requirements using embeddings when available
        
        Args:
            requirements: List of requirement documents
            applicants: List of applicant documents
            
        Returns:
            List of tuples (applicant_document, similarity_score) sorted by score
        """
        # Check if we have embeddings for both requirements and applicants
        req_embeddings = [req.metadata.get('embedding') for req in requirements if req.metadata.get('embedding')]
        applicant_embeddings = [applicant.metadata.get('embedding') for applicant in applicants if applicant.metadata.get('embedding')]
        
        # If we have embeddings for both, use cosine similarity
        if req_embeddings and applicant_embeddings:
            try:
                # Average requirement embeddings
                avg_req_embedding = np.mean(req_embeddings, axis=0)
                
                # Calculate cosine similarity for each applicant
                ranked_applicants = []
                for applicant in applicants:
                    doc_embedding = applicant.metadata.get('embedding')
                    if doc_embedding:
                        # Calculate cosine similarity
                        sim_score = cosine_similarity([avg_req_embedding], [doc_embedding])[0][0]
                        
                        # Add similarity score to document metadata
                        applicant_with_score = Document(
                            page_content=applicant.page_content,
                            metadata=applicant.metadata.copy() if applicant.metadata else {}
                        )
                        applicant_with_score.metadata['similarity_score'] = float(sim_score)
                        ranked_applicants.append((applicant_with_score, sim_score))
                    else:
                        # Fallback to text-based similarity
                        requirement_texts = [req.page_content for req in requirements]
                        combined_query = " ".join(requirement_texts)
                        score = self._calculate_similarity(combined_query, applicant.page_content)
                        applicant_with_score = Document(
                            page_content=applicant.page_content,
                            metadata=applicant.metadata.copy() if applicant.metadata else {}
                        )
                        applicant_with_score.metadata['similarity_score'] = float(score)
                        ranked_applicants.append((applicant_with_score, score))
                
                # Sort by similarity score (descending)
                ranked_applicants.sort(key=lambda x: x[1], reverse=True)
                return ranked_applicants
                
            except Exception as e:
                print(f"Embedding-based similarity failed, falling back to text-based: {e}")
        
        # Fallback to text-based similarity
        # For each applicant, calculate similarity to requirements
        ranked_applicants = []
        
        # Combine all requirement texts into a single query
        requirement_texts = [req.page_content for req in requirements]
        combined_query = " ".join(requirement_texts)
        
        # Quick text-based similarity for all applicants
        for applicant in applicants:
            # Use fast text-based similarity (much faster than embeddings)
            score = self._calculate_similarity(combined_query, applicant.page_content)
            
            # Add similarity score to document metadata
            applicant_with_score = Document(
                page_content=applicant.page_content,
                metadata=applicant.metadata.copy() if applicant.metadata else {}
            )
            applicant_with_score.metadata['similarity_score'] = float(score)
            
            ranked_applicants.append((applicant_with_score, score))
        
        # Sort by similarity score (descending)
        ranked_applicants.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_applicants
    
    def _get_document_embedding(self, document: Document) -> List[float]:
        """
        Extract embedding from document metadata
        
        Args:
            document: Document object
            
        Returns:
            Embedding vector or None if not found
        """
        if hasattr(document, 'metadata') and document.metadata:
            return document.metadata.get('embedding')
        return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using TF-IDF and cosine similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            # Fallback to Jaccard similarity if TF-IDF fails
            # Convert to lowercase and split into words
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Calculate Jaccard similarity
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if len(union) == 0:
                return 0.0
            
            return len(intersection) / len(union)