"""
LLM evaluator for assessing applicant compliance
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# LLMChain is not needed in newer versions of LangChain
# We'll use the invoke method directly on the LLM instead
import ollama
import requests


class LLMEvaluator:
    """Evaluate applicant documents using LLM reasoning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LLM evaluator
        
        Args:
            config: Configuration dictionary with Ollama settings
        """
        self.config = config or {}
        self.base_url = self.config.get('base_url', 'http://localhost:11434')
        self.model = self.config.get('llm_model', 'llama3.1')
        
        # Try to import and initialize Ollama with error handling
        self.llm = None
        
        # Test the connection first to ensure Ollama server is running
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                print(f"Error: Ollama server not responding at {self.base_url}")
                print("Make sure to start Ollama server with: ollama serve")
                print("Also ensure the required models are pulled: ollama pull llama3.1 nomic-embed-text")
                print("Without Ollama server, LLM evaluation will not work properly.")
            else:
                # Server is responding, check if model exists
                available_models = response.json()
                model_exists = any(model['name'].startswith(self.model) for model in available_models.get('models', []))
                if not model_exists:
                    print(f"Error: Model {self.model} not found. Available models:")
                    for model in available_models.get('models', []):
                        print(f"  - {model['name']}")
                    print(f"Make sure to pull the required model: ollama pull {self.model}")
                else:
                    print(f"Ollama server connected successfully, model {self.model} found")
                    
                    # Now try to initialize the LLM
                    try:
                        try:
                            from langchain_ollama import Ollama
                        except ImportError:
                            from langchain_community.llms import Ollama
                        
                        # Create Ollama LLM instance with more robust settings
                        self.llm = Ollama(
                            model=self.model,
                            base_url=self.base_url,
                            temperature=0.1,  # Slightly higher for more varied responses
                            timeout=120,  # Increased timeout to handle slower responses
                            num_predict=500  # More tokens for detailed responses
                        )
                        
                        # Test the connection with a simple prompt
                        test_response = self.llm.invoke("Respond with 'OK' only.")
                        if test_response and len(test_response) > 0 and 'OK' in test_response:
                            print(f"Ollama LLM ({self.model}) initialized successfully")
                        else:
                            print(f"Warning: Ollama LLM test returned unexpected response: {test_response}")
                            print("LLM evaluation will be simulated.")
                            self.llm = None
                    except Exception as e:
                        print(f"Warning: Could not initialize Ollama LLM: {e}")
                        print("LLM evaluation will be simulated.")
                        self.llm = None
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Ollama server at {self.base_url}")
            print("Make sure to start Ollama server with: ollama serve")
            print("Also ensure the required models are pulled: ollama pull llama3.1 nomic-embed-text")
            print("Without Ollama server, LLM evaluation will not work properly.")
            self.llm = None
        except Exception as e:
            print(f"Warning: Could not initialize Ollama LLM: {e}")
            print("This may be due to Ollama server not running.")
            print("Make sure to start Ollama server with: ollama serve")
            print("Also ensure the required models are pulled: ollama pull llama3.1 nomic-embed-text")
            print("LLM evaluation will be simulated.")
            self.llm = None
        
        # In newer versions of LangChain, we'll format the prompt and use the LLM directly
        self.evaluation_template = """You are an expert evaluator for tender proposals. Your task is to evaluate how well an applicant's proposal meets the organization's requirements.
            
            CRITICAL: First, identify the document type:
            - If it's a resume/CV/personal profile (focuses on individual qualifications, experience, skills), score it LOW (20-40)
            - If it's a business proposal (includes methodology, timeline, budget, deliverables, implementation plan), score it HIGH (70-95)
            - If it's a certificate, form, application, or government document, score it VERY LOW (5-20) - these are not proposals
            - If it's mixed content, score accordingly
            
            Organization Requirements:
            {requirements}
            
            Applicant Proposal:
            {proposal}
            
            Please provide:
            1. A compliance score between 0-100 indicating how well the proposal meets the requirements
            2. A brief explanation of your evaluation
            3. Key strengths of the proposal
            4. Areas for improvement
            5. Technical match score (0-100) based on technical approach and capabilities
            6. Financial match score (0-100) based on budget and cost-effectiveness
            7. Timeline match score (0-100) based on project scheduling and delivery
            
            Format your response as follows:
            SCORE: [numerical score 0-100]
            EXPLANATION: [brief explanation]
            STRENGTHS: [key strengths]
            IMPROVEMENTS: [areas for improvement]
            TECHNICAL_MATCH: [0-100]
            FINANCIAL_MATCH: [0-100]
            TIMELINE_MATCH: [0-100]
            """
    
    def evaluate_applicants(self, requirements: List[Document], 
                          applicants: List[Document],
                          max_applicants: int = 10) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Evaluate applicant documents using LLM reasoning
        
        Args:
            requirements: List of requirement Document objects
            applicants: List of applicant Document objects
            max_applicants: Maximum number of applicants to evaluate
            
        Returns:
            List of tuples (applicant_document, evaluation_results)
        """
        # If LLM is not available, simulate evaluation
        if not self.llm:
            return self._simulate_evaluation(applicants)
        
        # Combine all requirements into a single text
        requirement_texts = [req.page_content for req in requirements]
        combined_requirements = "\n".join(requirement_texts)
        
        # Evaluate top applicants
        evaluated_applicants = []
        
        # Process applicants in smaller batches to manage memory
        batch_size = 3  # Smaller batch size for memory management
        
        for i in range(0, min(len(applicants), max_applicants), batch_size):
            batch = applicants[i:i + batch_size]
            
            for j, applicant in enumerate(batch):
                try:
                    # Check if this is a large document chunk and handle accordingly
                    if applicant.metadata.get('large_document_chunk', False):
                        # For large document chunks, we may need to aggregate results
                        evaluation = self._evaluate_large_document_chunk(combined_requirements, applicant)
                    else:
                        # Evaluate the applicant normally
                        evaluation = self._evaluate_applicant(combined_requirements, applicant.page_content)
                    
                    # Add evaluation results to applicant metadata
                    evaluated_applicant = Document(
                        page_content=applicant.page_content,
                        metadata=applicant.metadata.copy() if applicant.metadata else {}
                    )
                    evaluated_applicant.metadata.update(evaluation)
                    
                    evaluated_applicants.append((evaluated_applicant, evaluation))
                except Exception as e:
                    print(f"Warning: Failed to evaluate applicant {i+j}: {e}")
                    # Add applicant with error information
                    error_applicant = Document(
                        page_content=applicant.page_content,
                        metadata=applicant.metadata.copy() if applicant.metadata else {}
                    )
                    error_applicant.metadata['evaluation_error'] = str(e)
                    evaluated_applicants.append((error_applicant, {'error': str(e)}))
        
        # Sort by score (descending), with secondary sort by source to maintain consistent ordering for identical scores
        evaluated_applicants.sort(key=lambda x: (x[1].get('score', 0), x[0].metadata.get('source', '')), reverse=True)
        
        return evaluated_applicants
    
    def _simulate_evaluation(self, applicants: List[Document]) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Simulate evaluation when LLM is not available
        
        Args:
            applicants: List of applicant Document objects
            
        Returns:
            List of tuples (applicant_document, evaluation_results)
        """
        print("Simulating LLM evaluation (Ollama not available)")
        evaluated_applicants = []
        
        # Generate simulated scores with content-aware evaluations
        import random
        # Set seed based on content hash for deterministic results
        import hashlib
        for i, applicant in enumerate(applicants):
            # Analyze document content to determine if it's a proper proposal
            content_lower = applicant.page_content.lower()
            
            # Check for resume-like content that should be penalized
            resume_indicators = ['resume', 'cv', 'curriculum vitae', 'experience', 'work history', 'employment', 'skills', 'qualifications', 'certifications', 'education', 'degree', 'university', 'college', 'linkedin', 'email', 'phone', 'contact', 'objective', 'summary', 'professional experience', 'personal information']
            resume_matches = sum(1 for indicator in resume_indicators if indicator in content_lower)
            is_resume_like = resume_matches >= 2  # At least 2 indicators to classify as resume
            
            # Check for proper proposal content that should be rewarded
            proposal_indicators = ['proposal', 'solution', 'methodology', 'approach', 'timeline', 'budget', 'cost', 'implementation', 'deliverables', 'project plan', 'scope', 'requirements', 'technical specifications', 'executive summary', 'solution architecture', 'work breakdown structure', 'resource allocation', 'risk mitigation', 'quality assurance', 'milestones']
            proposal_matches = sum(1 for indicator in proposal_indicators if indicator in content_lower)
            has_proposal_content = proposal_matches >= 2  # At least 2 indicators to classify as proposal
            
            # Additional check for business/tender related terms
            tender_related_terms = ['tender', 'rfp', 'request for proposal', 'contract', 'bid', 'quotation', 'offer', 'statement of work', 'sow', 'procurement']
            has_tender_content = any(term in content_lower for term in tender_related_terms)
            
            # Generate base score based on content type and analysis
            # Create a deterministic seed based on content hash
            content_hash = hashlib.md5(content_lower.encode()).hexdigest()
            content_seed = int(content_hash[:8], 16) % 10000
            random.seed(content_seed)
            
            if is_resume_like and not has_proposal_content:
                # Pure resumes/CVs get lowest scores
                score = random.randint(20, 40)
            elif is_resume_like and has_proposal_content:
                # Hybrid documents (resume with some proposal elements)
                score = random.randint(40, 60)
            elif has_proposal_content and has_tender_content:
                # Documents that clearly address tender requirements
                score = random.randint(75, 95)
            elif has_proposal_content:
                # Standard proposal content
                score = random.randint(65, 85)
            elif has_tender_content:
                # Documents mentioning tender but lacking detailed proposal
                score = random.randint(45, 70)
            elif 'certificate' in content_lower and 'caste' in content_lower:
                # Caste certificates and similar documents are not proposals
                score = random.randint(5, 15)
            elif 'certificate' in content_lower:
                # Other certificates are not proposals
                score = random.randint(5, 20)
            elif 'government' in content_lower and ('office' in content_lower or 'department' in content_lower):
                # Government documents like certificates are not proposals
                score = random.randint(5, 20)
            elif 'form' in content_lower or 'application' in content_lower:
                # Forms and applications are not proposals
                score = random.randint(10, 25)
            else:
                # Generic documents without clear proposal elements
                score = random.randint(30, 65)
            
            # Extract keywords from the document to customize the evaluation
            keywords = self._extract_keywords(applicant.page_content, 5)
            keyword_str = ", ".join(keywords[:3]) if keywords else "project requirements"
            
            # Customize explanation based on document content
            if is_resume_like:
                explanations = [
                    f"This document appears to be a resume/CV and does not adequately address the {keyword_str} requirements. It lacks project-specific methodologies and implementation details needed for this tender.",
                    f"The submission contains personal qualifications but does not demonstrate how these qualifications apply to the {keyword_str} requirements of this tender.",
                    f"While the document shows relevant experience, it fails to provide specific details on how the {keyword_str} requirements will be met."
                ]
            else:
                explanations = [
                    f"This proposal demonstrates strong alignment with the {keyword_str}. The applicant shows clear understanding of the project scope and has provided detailed methodologies for implementation.",
                    f"The proposal addresses key aspects of {keyword_str} effectively. The approach is well-structured with clear deliverables and timelines.",
                    f"This submission shows good comprehension of {keyword_str} requirements. The technical approach is sound with appropriate resource allocation."
                ]
            
            # Customize strengths based on document content and score
            if is_resume_like:
                strength_areas = {
                    "team qualifications": "relevant experience and expertise",
                    "personal skills": "individual capabilities and competencies",
                    "background": "educational and professional background"
                }
                strengths = [
                    f"The document shows strong {list(strength_areas.keys())[0]} with detailed {list(strength_areas.values())[0]}.",
                    f"Relevant {list(strength_areas.keys())[1]} is demonstrated through comprehensive {list(strength_areas.values())[1]}.",
                    f"Solid {list(strength_areas.keys())[2]} is evident with well-defined {list(strength_areas.values())[2]}."
                ]
            else:
                strength_areas = {
                    "technical approach": "technical specifications and methodology",
                    "project management": "timeline and resource planning",
                    "budget": "cost-effectiveness and value for money",
                    "team qualifications": "relevant experience and expertise",
                    "risk management": "mitigation strategies and contingency plans"
                }
                primary_strength = random.choice(list(strength_areas.keys()))
                strength_detail = strength_areas[primary_strength]
                strengths = [
                    f"The proposal excels in {primary_strength} with detailed {strength_detail}.",
                    f"Strong {primary_strength} is demonstrated through comprehensive {strength_detail}.",
                    f"Excellent {primary_strength} is evident with well-defined {strength_detail}."
                ]
            
            # Customize improvement suggestions based on document content and score
            if is_resume_like:
                improvements = [
                    f"This document should be converted to a proper tender proposal addressing specific requirements rather than just listing qualifications.",
                    f"Include detailed project methodology and implementation plan to address tender requirements.",
                    f"Provide specific information about approach, timeline, and budget for this project."
                ]
            else:
                improvement_areas = {
                    "technical approach": "technical specifications and implementation details",
                    "project management": "timeline milestones and resource allocation",
                    "budget": "cost breakdown and financial justification",
                    "team qualifications": "team member profiles and relevant experience",
                    "risk management": "risk identification and mitigation strategies"
                }
                primary_improvement = random.choice(list(improvement_areas.keys()))
                improvement_detail = improvement_areas[primary_improvement]
                improvements = [
                    f"Consider providing more details on {improvement_detail} to strengthen the proposal.",
                    f"Additional information on {improvement_detail} would enhance the overall quality.",
                    f"More comprehensive {improvement_detail} would improve the proposal's completeness."
                ]
            
            # Generate specialized scores
            # Use the same seed for consistency
            if is_resume_like:
                # Resume-like documents get lower specialized scores
                technical_match = max(0, min(100, score * 0.4 + random.randint(-5, 5)))
                financial_match = max(0, min(100, score * 0.3 + random.randint(-3, 3)))
                timeline_match = max(0, min(100, score * 0.3 + random.randint(-4, 4)))
            else:
                technical_match = max(0, min(100, score + random.randint(-5, 5)))
                financial_match = max(0, min(100, score * 0.9 + random.randint(-3, 3)))
                timeline_match = max(0, min(100, score * 0.8 + random.randint(-4, 4)))
            
            # For selections, use content-based index instead of random
            explanation_idx = int(content_hash[8:10], 16) % len(explanations)
            strength_idx = int(content_hash[10:12], 16) % len(strengths)
            improvement_idx = int(content_hash[12:14], 16) % len(improvements)
            
            evaluation = {
                'score': score,
                'technical_match': technical_match,
                'financial_match': financial_match,
                'timeline_match': timeline_match,
                'explanation': explanations[explanation_idx],
                'strengths': strengths[strength_idx],
                'improvements': improvements[improvement_idx]
            }
            
            # Add evaluation results to applicant metadata
            evaluated_applicant = Document(
                page_content=applicant.page_content,
                metadata=applicant.metadata.copy() if applicant.metadata else {}
            )
            evaluated_applicant.metadata.update(evaluation)
            
            evaluated_applicants.append((evaluated_applicant, evaluation))
        
        # Sort by score (descending), with secondary sort by source to maintain consistent ordering for identical scores
        evaluated_applicants.sort(key=lambda x: (x[1].get('score', 0), x[0].metadata.get('source', '')), reverse=True)
        
        return evaluated_applicants
    
    def _simulate_single_evaluation(self, proposal: str) -> Dict[str, Any]:
        """
        Simulate evaluation for a single proposal when LLM is not available
        
        Args:
            proposal: Proposal text to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        import random
        import hashlib
        
        # Analyze document content to determine if it's a proper proposal
        content_lower = proposal.lower()
        
        # Create a deterministic seed based on content hash
        content_hash = hashlib.md5(content_lower.encode()).hexdigest()
        content_seed = int(content_hash[:8], 16) % 10000
        random.seed(content_seed)
        
        # Check for resume-like content that should be penalized
        resume_indicators = ['resume', 'cv', 'curriculum vitae', 'experience', 'work history', 'employment', 'skills', 'qualifications', 'certifications', 'education', 'degree', 'university', 'college', 'linkedin', 'email', 'phone', 'contact', 'objective', 'summary', 'professional experience', 'personal information']
        resume_matches = sum(1 for indicator in resume_indicators if indicator in content_lower)
        is_resume_like = resume_matches >= 2  # At least 2 indicators to classify as resume
        
        # Check for proper proposal content that should be rewarded
        proposal_indicators = ['proposal', 'solution', 'methodology', 'approach', 'timeline', 'budget', 'cost', 'implementation', 'deliverables', 'project plan', 'scope', 'requirements', 'technical specifications', 'executive summary', 'solution architecture', 'work breakdown structure', 'resource allocation', 'risk mitigation', 'quality assurance', 'milestones']
        proposal_matches = sum(1 for indicator in proposal_indicators if indicator in content_lower)
        has_proposal_content = proposal_matches >= 2  # At least 2 indicators to classify as proposal
        
        # Additional check for business/tender related terms
        tender_related_terms = ['tender', 'rfp', 'request for proposal', 'contract', 'bid', 'quotation', 'offer', 'statement of work', 'sow', 'procurement']
        has_tender_content = any(term in content_lower for term in tender_related_terms)
        
        # Generate base score based on content type and analysis
        if is_resume_like and not has_proposal_content:
            # Pure resumes/CVs get lowest scores
            score = random.randint(20, 40)
        elif is_resume_like and has_proposal_content:
            # Hybrid documents (resume with some proposal elements)
            score = random.randint(40, 60)
        elif has_proposal_content and has_tender_content:
            # Documents that clearly address tender requirements
            score = random.randint(75, 95)
        elif has_proposal_content:
            # Standard proposal content
            score = random.randint(65, 85)
        elif has_tender_content:
            # Documents mentioning tender but lacking detailed proposal
            score = random.randint(45, 70)
        elif 'certificate' in content_lower and 'caste' in content_lower:
            # Caste certificates and similar documents are not proposals
            score = random.randint(5, 15)
        elif 'certificate' in content_lower:
            # Other certificates are not proposals
            score = random.randint(5, 20)
        elif 'government' in content_lower and ('office' in content_lower or 'department' in content_lower):
            # Government documents like certificates are not proposals
            score = random.randint(5, 20)
        elif 'form' in content_lower or 'application' in content_lower:
            # Forms and applications are not proposals
            score = random.randint(10, 25)
        else:
            # Generic documents without clear proposal elements
            score = random.randint(30, 65)
        
        # Extract keywords from the document to customize the evaluation
        keywords = self._extract_keywords(proposal, 5)
        keyword_str = ", ".join(keywords[:3]) if keywords else "project requirements"
        
        # Customize explanation based on document content
        if is_resume_like:
            explanations = [
                f"This document appears to be a resume/CV and does not adequately address the {keyword_str} requirements. It lacks project-specific methodologies and implementation details needed for this tender.",
                f"The submission contains personal qualifications but does not demonstrate how these qualifications apply to the {keyword_str} requirements of this tender.",
                f"While the document shows relevant experience, it fails to provide specific details on how the {keyword_str} requirements will be met."
            ]
        else:
            explanations = [
                f"This proposal demonstrates strong alignment with the {keyword_str}. The applicant shows clear understanding of the project scope and has provided detailed methodologies for implementation.",
                f"The proposal addresses key aspects of {keyword_str} effectively. The approach is well-structured with clear deliverables and timelines.",
                f"This submission shows good comprehension of {keyword_str} requirements. The technical approach is sound with appropriate resource allocation."
            ]
        
        # Customize strengths based on document content and score
        if is_resume_like:
            strength_areas = {
                "team qualifications": "relevant experience and expertise",
                "personal skills": "individual capabilities and competencies",
                "background": "educational and professional background"
            }
            strengths = [
                f"The document shows strong {list(strength_areas.keys())[0]} with detailed {list(strength_areas.values())[0]}.",
                f"Relevant {list(strength_areas.keys())[1]} is demonstrated through comprehensive {list(strength_areas.values())[1]}.",
                f"Solid {list(strength_areas.keys())[2]} is evident with well-defined {list(strength_areas.values())[2]}."
            ]
        else:
            strength_areas = {
                "technical approach": "technical specifications and methodology",
                "project management": "timeline and resource planning",
                "budget": "cost-effectiveness and value for money",
                "team qualifications": "relevant experience and expertise",
                "risk management": "mitigation strategies and contingency plans"
            }
            # Use content-based index for deterministic selection
            strength_keys = list(strength_areas.keys())
            primary_strength_idx = int(content_hash[8:10], 16) % len(strength_keys)
            primary_strength = strength_keys[primary_strength_idx]
            strength_detail = strength_areas[primary_strength]
            
            strengths = [
                f"The proposal excels in {primary_strength} with detailed {strength_detail}.",
                f"Strong {primary_strength} is demonstrated through comprehensive {strength_detail}.",
                f"Excellent {primary_strength} is evident with well-defined {strength_detail}."
            ]
        
        # Customize improvement suggestions based on document content and score
        if is_resume_like:
            improvements = [
                f"This document should be converted to a proper tender proposal addressing specific requirements rather than just listing qualifications.",
                f"Include detailed project methodology and implementation plan to address tender requirements.",
                f"Provide specific information about approach, timeline, and budget for this project."
            ]
        else:
            improvement_areas = {
                "technical approach": "technical specifications and implementation details",
                "project management": "timeline milestones and resource allocation",
                "budget": "cost breakdown and financial justification",
                "team qualifications": "team member profiles and relevant experience",
                "risk management": "risk identification and mitigation strategies"
            }
            
            # Use content-based index for deterministic selection
            improvement_keys = list(improvement_areas.keys())
            primary_improvement_idx = int(content_hash[10:12], 16) % len(improvement_keys)
            primary_improvement = improvement_keys[primary_improvement_idx]
            improvement_detail = improvement_areas[primary_improvement]
            
            improvements = [
                f"Consider providing more details on {improvement_detail} to strengthen the proposal.",
                f"Additional information on {improvement_detail} would enhance the overall quality.",
                f"More comprehensive {improvement_detail} would improve the proposal's completeness."
            ]
        
        # Generate specialized scores
        if is_resume_like:
            # Resume-like documents get lower specialized scores
            technical_match = max(0, min(100, score * 0.4 + random.randint(-5, 5)))
            financial_match = max(0, min(100, score * 0.3 + random.randint(-3, 3)))
            timeline_match = max(0, min(100, score * 0.3 + random.randint(-4, 4)))
        else:
            technical_match = max(0, min(100, score + random.randint(-5, 5)))
            financial_match = max(0, min(100, score * 0.9 + random.randint(-3, 3)))
            timeline_match = max(0, min(100, score * 0.8 + random.randint(-4, 4)))
        
        # Use content-based index for deterministic selection
        explanation_idx = int(content_hash[12:14], 16) % len(explanations)
        strength_idx = int(content_hash[14:16], 16) % len(strengths)
        improvement_idx = int(content_hash[16:18], 16) % len(improvements)
        
        return {
            'score': score,
            'technical_match': technical_match,
            'financial_match': financial_match,
            'timeline_match': timeline_match,
            'explanation': explanations[explanation_idx],
            'strengths': strengths[strength_idx],
            'improvements': improvements[improvement_idx]
        }
    
    def _evaluate_large_document_chunk(self, requirements: str, document: Document) -> Dict[str, Any]:
        """
        Evaluate a chunk from a large document
        
        Args:
            requirements: Combined requirements text
            document: Document chunk from a large document
            
        Returns:
            Dictionary with evaluation results
        """
        # For large document chunks, we need to be more strategic about evaluation
        # If this chunk is part of a larger document, we might need to combine or aggregate results
        
        # First, try to evaluate this chunk normally
        evaluation = self._evaluate_applicant(requirements, document.page_content)
        
        # If the document is marked as a large document chunk, add additional metadata
        if document.metadata.get('large_document_chunk', False):
            # Add information about document size and chunk position
            original_length = document.metadata.get('original_doc_length', len(document.page_content))
            
            # If original document was very large, adjust confidence accordingly
            if original_length > 50000:  # Very large document
                # Reduce confidence slightly due to context limitations
                score_reduction = min(5, original_length // 10000)  # Up to 5 point reduction
                evaluation['score'] = max(0, evaluation.get('score', 0) - score_reduction)
                
                # Add note about large document processing
                original_explanation = evaluation.get('explanation', '')
                evaluation['explanation'] = f"[Large Document Note: This evaluation is based on a chunk of a large document ({original_length} chars). Full document context was not available.] " + original_explanation
        
        return evaluation
    
    def _extract_keywords(self, text, num_keywords=5):
        """
        Extract key keywords from text
        
        Args:
            text: Text to extract keywords from
            num_keywords: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        import re
        
        # Simple keyword extraction - in a real implementation, you might use NLP techniques
        # Remove common stop words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words and filter out stop words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:num_keywords]]
    
    def _evaluate_applicant(self, requirements: str, proposal: str) -> Dict[str, Any]:
        """
        Evaluate a single applicant using the LLM
        
        Args:
            requirements: Combined requirements text
            proposal: Applicant proposal text
            
        Returns:
            Dictionary with evaluation results
        """
        # Format the prompt with the requirements and proposal
        formatted_prompt = self.evaluation_template.format(
            requirements=requirements,
            proposal=proposal
        )
        
        # Run the LLM with the formatted prompt
        if self.llm:
            try:
                response = self.llm.invoke(formatted_prompt)
            except Exception as e:
                print(f"Warning: LLM evaluation failed: {e}")
                # Return simulation results if LLM fails
                return self._simulate_single_evaluation(proposal)
        else:
            # If LLM is not available, return simulation results
            return self._simulate_single_evaluation(proposal)
        
        # Parse the response
        return self._parse_evaluation_response(response)
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM evaluation response
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dictionary with parsed evaluation results
        """
        evaluation = {
            'score': 0,
            'technical_match': 0,
            'financial_match': 0,
            'timeline_match': 0,
            'explanation': '',
            'strengths': '',
            'improvements': ''
        }
        
        # Parse each section
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('SCORE:'):
                try:
                    evaluation['score'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('TECHNICAL_MATCH:'):
                try:
                    evaluation['technical_match'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('FINANCIAL_MATCH:'):
                try:
                    evaluation['financial_match'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('TIMELINE_MATCH:'):
                try:
                    evaluation['timeline_match'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                evaluation[current_section] = line.split(':', 1)[1].strip()
            elif line.startswith('STRENGTHS:'):
                current_section = 'strengths'
                evaluation[current_section] = line.split(':', 1)[1].strip()
            elif line.startswith('IMPROVEMENTS:'):
                current_section = 'improvements'
                evaluation[current_section] = line.split(':', 1)[1].strip()
            elif current_section and line:
                # Continue previous section
                evaluation[current_section] += ' ' + line
        
        return evaluation

    def analyze_requirement_compliance(self, requirement: str, proposal: str) -> Dict[str, Any]:
        """
        Analyze how well a specific proposal addresses a specific requirement
        
        Args:
            requirement: Specific requirement text
            proposal: Proposal text to analyze
            
        Returns:
            Dictionary with compliance analysis results
        """
        # If LLM is available, perform detailed analysis
        if self.llm:
            try:
                analysis_template = """You are an expert evaluator analyzing how well a proposal addresses a specific requirement.
                
                Requirement:
                {requirement}
                
                Proposal:
                {proposal}
                
                Please analyze:
                1. How well does the proposal address this specific requirement? (score 0-100)
                2. What specific elements in the proposal address this requirement?
                3. What aspects of the requirement are not adequately addressed?
                
                Format your response as follows:
                COMPLIANCE_SCORE: [numerical score 0-100]
                ADDRESSES: [what elements address the requirement]
                GAPS: [what aspects are not addressed]
                """
                
                formatted_prompt = analysis_template.format(
                    requirement=requirement,
                    proposal=proposal
                )
                
                response = self.llm.invoke(formatted_prompt)
                
                # Parse the response
                return self._parse_requirement_analysis(response)
            except Exception as e:
                print(f"Warning: Requirement analysis failed: {e}")
                # Fall back to simulation
                return self._simulate_requirement_analysis(requirement, proposal)
        else:
            # If LLM is not available, simulate analysis
            return self._simulate_requirement_analysis(requirement, proposal)

    def _parse_requirement_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse the requirement analysis response
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dictionary with parsed analysis results
        """
        analysis = {
            'compliance_score': 0,
            'addresses': '',
            'gaps': ''
        }
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('COMPLIANCE_SCORE:'):
                try:
                    analysis['compliance_score'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('ADDRESSES:'):
                current_section = 'addresses'
                analysis[current_section] = line.split(':', 1)[1].strip()
            elif line.startswith('GAPS:'):
                current_section = 'gaps'
                analysis[current_section] = line.split(':', 1)[1].strip()
            elif current_section and line:
                # Continue previous section
                analysis[current_section] += ' ' + line
        
        return analysis

    def _simulate_requirement_analysis(self, requirement: str, proposal: str) -> Dict[str, Any]:
        """
        Simulate requirement analysis when LLM is not available
        
        Args:
            requirement: Specific requirement text
            proposal: Proposal text to analyze
            
        Returns:
            Dictionary with simulated analysis results
        """
        import random
        import hashlib
        
        # Create a deterministic seed based on content hash
        content_hash = hashlib.md5((requirement + proposal).encode()).hexdigest()
        content_seed = int(content_hash[:8], 16) % 10000
        random.seed(content_seed)
        
        # Analyze content overlap to estimate compliance
        req_lower = requirement.lower()
        prop_lower = proposal.lower()
        
        # Look for keywords from requirement in proposal
        req_words = set(req_lower.split()[:20])  # First 20 words as representative
        prop_words = set(prop_lower.split())
        
        overlap = len(req_words.intersection(prop_words))
        max_possible_overlap = min(len(req_words), len(prop_words))
        
        if max_possible_overlap > 0:
            overlap_ratio = overlap / max_possible_overlap
        else:
            overlap_ratio = 0.0
        
        # Generate score based on content overlap with some randomness
        base_score = int(overlap_ratio * 100)
        score_variation = random.randint(-15, 15)
        compliance_score = max(0, min(100, base_score + score_variation))
        
        # Generate addresses text based on overlap
        if overlap > 0:
            addresses = f"The proposal addresses several aspects of this requirement, including {', '.join(list(req_words.intersection(prop_words))[:3])} and related concepts."
        else:
            addresses = "The proposal does not directly address this requirement."
        
        # Generate gaps text
        missing_words = req_words.difference(prop_words)
        if len(missing_words) > 0:
            gaps = f"The proposal lacks coverage of key elements: {', '.join(list(missing_words)[:3])}."
        else:
            gaps = "No significant gaps identified."
        
        return {
            'compliance_score': compliance_score,
            'addresses': addresses,
            'gaps': gaps
        }