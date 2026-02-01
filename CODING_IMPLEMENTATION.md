# 9. Coding & Implementation

This section provides a deep dive into the modular source code of the AI-Powered Tender Evaluation System. The codebase is organized into distinct packages to ensure a clear separation of concerns, making the system maintainable and scalable.

## 9.1 System Architecture Overview

The system follows a modular architecture with six core components that work together in a pipeline:

```
Document Ingestion → Embedding Generation → Similarity Search → 
Rule Filtering → LLM Evaluation → Report Generation
```

Each module is designed with specific responsibilities and clear interfaces, enabling independent development and testing.

## 9.2 Core Module Implementations

### 9.2.1 Document Ingestion (`src/ingestion/`)

**Purpose**: Load and preprocess PDF documents with automatic OCR fallback for scanned documents.

**Key Implementation**:

```python
class DocumentLoader:
    """Load documents from various sources with intelligent PDF processing"""
    
    def __init__(self, ocr_config: Dict[str, Any] = None):
        self.supported_formats = {'.pdf', '.txt'}
        # Initialize OCR processor only when Docling is not available
        if not DOCSTRING_AVAILABLE:
            from src.ocr.ocr_processor import OCRProcessor
            self.ocr_processor = OCRProcessor(ocr_config or {})
        else:
            self.ocr_processor = None  # Docling handles all PDF processing
    
    def _load_pdf_document(self, file_path: str) -> List[Document]:
        """Primary PDF loading method using Docling as preferred approach"""
        text = ""
        
        if DOCSTRING_AVAILABLE:
            # Use Docling as the primary method for ALL PDFs including scanned ones
            try:
                converter = DocumentConverter()
                result = converter.convert(file_path)
                text = result.document.export_to_markdown()
                
                # Fallback to pdfplumber if Docling returns empty text
                if not text.strip():
                    text = self._extract_with_pdfplumber(file_path)
            except Exception as e:
                # Use pdfplumber as backup when Docling fails
                text = self._extract_with_pdfplumber(file_path)
        else:
            # Use original method when Docling is not available
            text = self._load_pdf_with_fallback(file_path)
        
        return [Document(page_content=text.strip(), metadata={"source": file_path})]
    
    def _load_pdf_with_fallback(self, file_path: str) -> str:
        """Fallback chain: pdfplumber → OCR processing"""
        text = ""
        try:
            # Try to extract text using pdfplumber
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If no text was extracted, fallback to OCR
            if not text.strip() and self.ocr_processor:
                text = self.ocr_processor.process_scanned_pdf(file_path)
                
        except Exception as e:
            # If pdfplumber fails, try OCR as final fallback
            if self.ocr_processor:
                text = self.ocr_processor.process_scanned_pdf(file_path)
            else:
                raise RuntimeError(f"Failed to extract text from {file_path}: {e}")
        
        return text
```

**Key Features**:
- **Multi-engine approach**: Docling (primary) → pdfplumber → Tesseract OCR
- **Intelligent detection**: Automatically identifies scanned PDFs
- **Robust error handling**: Multiple fallback mechanisms
- **Metadata preservation**: Maintains source information and processing context

### 9.2.2 Embedding Generation (`src/embeddings/`)

**Purpose**: Convert text to high-dimensional vector representations using the nomic-embed-text model.

**Key Implementation**:

```python
class EmbeddingGenerator:
    """Generate embeddings for text documents using Ollama"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_name = self.config.get('embedding_model', 'nomic-embed-text')
        self.ollama_base_url = self.config.get('ollama_base_url', 'http://localhost:11434')
        self.ollama_available = self._check_ollama_connection()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.ollama_available:
            return self._simulate_embeddings(texts)
        
        try:
            embeddings = []
            for text in texts:
                # Call Ollama API to generate embedding
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                if response.status_code == 200:
                    embedding = response.json()['embedding']
                    embeddings.append(embedding)
                else:
                    raise Exception(f"Ollama API error: {response.status_code}")
            
            return embeddings
        except Exception as e:
            print(f"Warning: Ollama embedding generation failed: {e}")
            return self._simulate_embeddings(texts)
    
    def _simulate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simulated embeddings for testing/fallback"""
        # Generate random embeddings with consistent dimensions (384)
        import numpy as np
        embeddings = []
        for text in texts:
            # Create deterministic embeddings based on text content
            seed = hash(text) % 10000
            np.random.seed(seed)
            embedding = np.random.rand(384).tolist()
            embeddings.append(embedding)
        return embeddings
```

**Key Features**:
- **Model integration**: Uses nomic-embed-text via Ollama API
- **Fallback mechanism**: Simulated embeddings when Ollama unavailable
- **Batch processing**: Efficient handling of multiple texts
- **Error resilience**: Graceful degradation to simulation mode

### 9.2.3 Similarity Search (`src/search/`)

**Purpose**: Match requirements with proposals using cosine similarity algorithm with FAISS vector store.

**Key Implementation**:

```python
class SimilaritySearcher:
    """Perform semantic similarity search between documents"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
    
    def search_applicants_by_requirements(
        self, 
        requirements: List[Document], 
        applicants: List[Document], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search applicants based on requirements similarity"""
        
        results = []
        for req_doc in requirements:
            # Search for similar applicant documents
            similar_docs = self.vector_store.similarity_search(
                req_doc.page_content, 
                k=top_k
            )
            
            # Calculate and rank similarities
            ranked_applicants = self._rank_applicants(
                req_doc.page_content, 
                similar_docs
            )
            results.extend(ranked_applicants)
        
        # Aggregate and sort results
        return self._aggregate_results(results)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            # Use vector store for efficient similarity calculation
            embedding1 = self.vector_store.get_embedding(text1)
            embedding2 = self.vector_store.get_embedding(text2)
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            # Fallback to TF-IDF based similarity
            return self._tfidf_similarity(text1, text2)
    
    def _rank_applicants(
        self, 
        requirement: str, 
        applicant_docs: List[Document]
    ) -> List[Dict[str, Any]]:
        """Rank applicants based on similarity to requirement"""
        ranked = []
        for doc in applicant_docs:
            similarity = self._calculate_similarity(requirement, doc.page_content)
            ranked.append({
                'applicant': doc.metadata.get('source', 'Unknown'),
                'similarity_score': similarity,
                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })
        
        return sorted(ranked, key=lambda x: x['similarity_score'], reverse=True)
```

**Key Features**:
- **Vector-based search**: Efficient FAISS implementation
- **Multiple algorithms**: Cosine similarity with TF-IDF fallback
- **Ranking system**: Comprehensive scoring and sorting
- **Scalable design**: Handles large document collections efficiently

### 9.2.4 Rule Filtering (`src/filtering/`)

**Purpose**: Apply business constraints including budget compliance, timeline feasibility, and certification requirements.

**Key Implementation**:

```python
class RuleFilter:
    """Apply business rules and constraints to filter proposals"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules = self.config.get('filtering_rules', {})
    
    def apply_filters(self, applicants: List[Document]) -> List[Dict[str, Any]]:
        """Apply all configured filters to applicant documents"""
        filtered_results = []
        
        for applicant in applicants:
            result = {
                'document': applicant,
                'passed_filters': True,
                'filter_results': {},
                'score': 100
            }
            
            # Apply budget filter
            budget_result = self._apply_budget_filter(applicant)
            result['filter_results']['budget'] = budget_result
            if not budget_result['passed']:
                result['passed_filters'] = False
                result['score'] -= 20
            
            # Apply timeline filter
            timeline_result = self._apply_timeline_filter(applicant)
            result['filter_results']['timeline'] = timeline_result
            if not timeline_result['passed']:
                result['passed_filters'] = False
                result['score'] -= 20
            
            # Apply certification filter
            cert_result = self._apply_certification_filter(applicant)
            result['filter_results']['certifications'] = cert_result
            if not cert_result['passed']:
                result['passed_filters'] = False
                result['score'] -= 30
            
            # Apply technical prerequisites filter
            tech_result = self._apply_technical_filter(applicant)
            result['filter_results']['technical'] = tech_result
            if not tech_result['passed']:
                result['passed_filters'] = False
                result['score'] -= 30
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _apply_budget_filter(self, applicant: Document) -> Dict[str, Any]:
        """Check budget compliance"""
        content = applicant.page_content.lower()
        budget_rules = self.rules.get('budget', {})
        
        # Extract monetary values using regex
        import re
        monetary_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        matches = re.findall(monetary_pattern, content)
        
        amounts = [float(match.replace(',', '')) for match in matches if match]
        
        if not amounts:
            return {'passed': False, 'reason': 'No budget information found'}
        
        min_budget = budget_rules.get('min_amount', 0)
        max_budget = budget_rules.get('max_amount', float('inf'))
        
        valid_amounts = [amt for amt in amounts if min_budget <= amt <= max_budget]
        
        return {
            'passed': len(valid_amounts) > 0,
            'valid_amounts': valid_amounts,
            'min_found': min(amounts) if amounts else 0,
            'max_found': max(amounts) if amounts else 0
        }
    
    def _apply_timeline_filter(self, applicant: Document) -> Dict[str, Any]:
        """Check timeline feasibility"""
        content = applicant.page_content.lower()
        
        # Look for timeline indicators
        timeline_indicators = [
            'month', 'months', 'week', 'weeks', 'day', 'days',
            'delivery', 'completion', 'implementation', 'deployment'
        ]
        
        timeline_matches = [word for word in timeline_indicators if word in content]
        
        # Extract duration mentions
        import re
        duration_pattern = r'(\d+)\s*(month|months|week|weeks|day|days)'
        duration_matches = re.findall(duration_pattern, content, re.IGNORECASE)
        
        return {
            'passed': len(duration_matches) > 0,
            'timeline_mentions': timeline_matches,
            'durations_found': duration_matches,
            'reason': 'Timeline information found' if duration_matches else 'No timeline information'
        }
```

**Key Features**:
- **Configurable rules**: Flexible rule definitions via configuration
- **Comprehensive validation**: Multiple constraint types (budget, timeline, certifications)
- **Detailed feedback**: Specific reasons for filter results
- **Scoring system**: Weighted compliance scoring

### 9.2.5 LLM Evaluation (`src/evaluation/`)

**Purpose**: Detailed scoring and analysis using TinyLlama via Ollama with multi-dimensional evaluation.

**Key Implementation**:

```python
class LLMEvaluator:
    """Evaluate proposals using Large Language Models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_name = self.config.get('evaluation_model', 'tinydolphin:latest')
        self.ollama_base_url = self.config.get('ollama_base_url', 'http://localhost:11434')
        self.ollama_available = self._check_ollama_connection()
    
    def evaluate_applicants(
        self, 
        requirements: List[Document], 
        applicants: List[Document], 
        max_applicants: int = 10
    ) -> List[Dict[str, Any]]:
        """Evaluate applicant proposals using LLM"""
        
        if not self.ollama_available:
            return self._simulate_evaluation(applicants[:max_applicants])
        
        evaluations = []
        for applicant in applicants[:max_applicants]:
            try:
                evaluation = self._evaluate_applicant(requirements, applicant)
                evaluations.append(evaluation)
            except Exception as e:
                print(f"Warning: LLM evaluation failed for {applicant.metadata.get('source')}: {e}")
                # Fallback to simulated evaluation
                simulated = self._simulate_single_evaluation(applicant)
                evaluations.append(simulated)
        
        return sorted(evaluations, key=lambda x: x['score'], reverse=True)
    
    def _evaluate_applicant(
        self, 
        requirements: List[Document], 
        applicant: Document
    ) -> Dict[str, Any]:
        """Evaluate a single applicant proposal"""
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(requirements, applicant)
        
        # Call Ollama API
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            llm_response = response.json()['response']
            return self._parse_evaluation_response(llm_response, applicant)
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def _create_evaluation_prompt(
        self, 
        requirements: List[Document], 
        applicant: Document
    ) -> str:
        """Create structured prompt for LLM evaluation"""
        
        req_text = "\n".join([req.page_content for req in requirements])
        applicant_text = applicant.page_content
        
        prompt = f"""
        Evaluate the following proposal against the requirements.
        
        REQUIREMENTS:
        {req_text}
        
        PROPOSAL:
        {applicant_text}
        
        Please provide a detailed evaluation including:
        1. Technical Match Score (0-100)
        2. Financial Viability Score (0-100)
        3. Timeline Alignment Score (0-100)
        4. Overall Compliance Score (0-100)
        5. Key Strengths
        6. Areas for Improvement
        7. Detailed Explanation
        
        Format your response as JSON:
        {{
            "technical_match": 85,
            "financial_match": 90,
            "timeline_match": 75,
            "overall_score": 83,
            "strengths": "Strong technical approach with proven methodology",
            "improvements": "Consider more detailed cost breakdown",
            "explanation": "The proposal demonstrates excellent technical capabilities..."
        }}
        """
        
        return prompt
    
    def _parse_evaluation_response(
        self, 
        response: str, 
        applicant: Document
    ) -> Dict[str, Any]:
        """Parse LLM response into structured evaluation"""
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    'applicant': applicant.metadata.get('source', 'Unknown'),
                    'score': parsed.get('overall_score', 0),
                    'technical_match': parsed.get('technical_match', 0),
                    'financial_match': parsed.get('financial_match', 0),
                    'timeline_match': parsed.get('timeline_match', 0),
                    'strengths': parsed.get('strengths', ''),
                    'improvements': parsed.get('improvements', ''),
                    'explanation': parsed.get('explanation', response)
                }
        except Exception as e:
            # Fallback to regex parsing if JSON fails
            return self._parse_with_regex(response, applicant)
        
        # If all parsing fails, return basic evaluation
        return self._basic_evaluation(applicant)
```

**Key Features**:
- **Structured prompting**: Consistent evaluation format
- **Multi-dimensional scoring**: Technical, financial, and timeline aspects
- **Natural language processing**: Detailed explanations and recommendations
- **Robust parsing**: Multiple parsing strategies with fallbacks

### 9.2.6 Report Generation (`src/reporting/`)

**Purpose**: Create professional evaluation reports in LaTeX format with PDF compilation.

**Key Implementation**:

```python
class ReportGenerator:
    """Generate professional LaTeX reports from evaluation results"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.template_name = self.config.get('template', 'comprehensive_tec_template.tex')
        self.disable_pdf_compilation = self.config.get('disable_pdf_compilation', False)
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent.parent.parent / 'templates'
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Setup LaTeX environment
        self._setup_latex_environment()
    
    def generate_evaluation_report(
        self,
        requirements: List[Document],
        evaluation_results: List[Tuple[Document, Dict[str, Any]]]
    ) -> str:
        """Generate comprehensive evaluation report"""
        
        # Prepare data for template
        template_data = self._prepare_template_data(requirements, evaluation_results)
        
        # Render LaTeX template
        template = self.env.get_template(self.template_name)
        latex_content = template.render(**template_data)
        
        # Save LaTeX file
        output_dir = Path(self.config.get('output_dir', './data/reports'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Tender_Evaluation_Report_{timestamp}_{self._generate_hash()}.tex"
        latex_path = output_dir / filename
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        # Compile to PDF if enabled
        if not self.disable_pdf_compilation:
            pdf_path = self._compile_latex_to_pdf(latex_path)
            if pdf_path:
                return str(pdf_path)
        
        return str(latex_path)
    
    def _prepare_template_data(
        self,
        requirements: List[Document],
        evaluation_results: List[Tuple[Document, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Prepare data structure for LaTeX template"""
        
        # Extract and clean requirement texts
        requirement_texts = [
            self._clean_text_for_latex(req.page_content) 
            for req in requirements
        ]
        
        # Process evaluation results
        applicants_data = []
        for i, (applicant_doc, eval_result) in enumerate(evaluation_results):
            applicant_data = {
                'rank': i + 1,
                'name': applicant_doc.metadata.get('source', f'Applicant {i+1}'),
                'score': eval_result.get('score', 0),
                'technical_match': eval_result.get('technical_match', 0),
                'financial_match': eval_result.get('financial_match', 0),
                'timeline_match': eval_result.get('timeline_match', 0),
                'similarity_score': eval_result.get('similarity_score', 0),
                'explanation': self._clean_text_for_latex(eval_result.get('explanation', '')),
                'strengths': self._clean_text_for_latex(eval_result.get('strengths', '')),
                'improvements': self._clean_text_for_latex(eval_result.get('improvements', ''))
            }
            applicants_data.append(applicant_data)
        
        # Calculate summary statistics
        scores = [app['score'] for app in applicants_data]
        summary_stats = {
            'total_applicants': len(applicants_data),
            'average_score': sum(scores) / len(scores) if scores else 0,
            'highest_score': max(scores) if scores else 0,
            'lowest_score': min(scores) if scores else 0
        }
        
        return {
            'title': 'Tender Evaluation Report',
            'date': datetime.now().strftime("%B %d, %Y"),
            'requirements': requirement_texts,
            'applicants': applicants_data,
            'summary_stats': summary_stats,
            'comparison_table': self._generate_comparison_table(applicants_data)
        }
    
    def _clean_text_for_latex(self, text: str) -> str:
        """Clean and escape text for LaTeX output"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove problematic characters
        text = text.replace('\x00', '').replace('\x02', '')
        text = text.replace('\n', ' ')
        
        # Escape LaTeX special characters
        text = text.replace('%', r'\%')
        text = text.replace('#', r'\#')
        text = text.replace('&', r'\&')
        text = text.replace('_', r'\_')
        text = text.replace('{', r'\{')
        text = text.replace('}', r'\}')
        text = text.replace('$', r'\$')
        text = text.replace('~', r'\ensuremath{\sim}')
        text = text.replace('^', r'\^{}')
        
        # Handle common unicode issues
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace('…', '...')
        
        # Limit consecutive whitespaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate very long text
        if len(text) > 1000:
            text = text[:1000] + '... (truncated)'
        
        return text
    
    def _compile_latex_to_pdf(self, latex_path: Path) -> Optional[Path]:
        """Compile LaTeX file to PDF using pdflatex"""
        try:
            # Run pdflatex command
            result = subprocess.run([
                'pdflatex',
                '-interaction=batchmode',
                '-halt-on-error',
                str(latex_path.name)
            ], cwd=latex_path.parent, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                pdf_path = latex_path.with_suffix('.pdf')
                if pdf_path.exists():
                    print(f"PDF report generated successfully: {pdf_path}")
                    return pdf_path
                else:
                    print("Warning: pdflatex succeeded but PDF file not found")
                    return None
            else:
                print(f"Warning: pdflatex compilation failed with return code {result.returncode}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Warning: pdflatex compilation timed out")
            return None
        except Exception as e:
            print(f"Warning: PDF compilation error: {e}")
            return None
```

**Key Features**:
- **Professional formatting**: Comprehensive LaTeX templates
- **Robust escaping**: Proper handling of special characters
- **Flexible compilation**: Configurable PDF generation
- **Error handling**: Graceful fallback to LaTeX files when PDF compilation fails

## 9.3 Integration and Data Flow

### 9.3.1 Pipeline Coordination

The modules work together through a well-defined pipeline:

```python
# Main evaluation pipeline
def evaluate_tender_proposals(
    requirement_files: List[str],
    proposal_files: List[str],
    config: Dict[str, Any]
) -> str:
    """Main pipeline coordinating all modules"""
    
    # 1. Document Ingestion
    document_loader = DocumentLoader(config.get('ocr', {}))
    requirements = document_loader.load_documents(requirement_files)
    proposals = document_loader.load_documents(proposal_files)
    
    # 2. Text Processing
    text_processor = TextProcessor(config.get('processing', {}))
    processed_requirements = text_processor.chunk_documents(requirements)
    processed_proposals = text_processor.chunk_documents(proposals)
    
    # 3. Embedding Generation
    embedding_generator = EmbeddingGenerator(config.get('ollama', {}))
    requirement_embeddings = embedding_generator.generate_document_embeddings(processed_requirements)
    proposal_embeddings = embedding_generator.generate_document_embeddings(processed_proposals)
    
    # 4. Vector Storage
    vector_store = VectorStoreManager(config.get('vector_db', {}))
    vector_store.initialize_store(requirement_embeddings + proposal_embeddings)
    
    # 5. Similarity Search
    similarity_searcher = SimilaritySearcher(vector_store)
    ranked_proposals = similarity_searcher.search_applicants_by_requirements(
        processed_requirements, processed_proposals
    )
    
    # 6. Rule Filtering
    rule_filter = RuleFilter(config.get('filtering', {}))
    filtered_proposals = rule_filter.apply_filters(ranked_proposals)
    
    # 7. LLM Evaluation
    llm_evaluator = LLMEvaluator(config.get('ollama', {}))
    final_evaluations = llm_evaluator.evaluate_applicants(
        processed_requirements, filtered_proposals
    )
    
    # 8. Report Generation
    report_generator = ReportGenerator(config.get('reporting', {}))
    report_path = report_generator.generate_evaluation_report(
        processed_requirements, final_evaluations
    )
    
    return report_path
```

### 9.3.2 Error Handling and Fallbacks

Each module implements robust error handling:

```python
class ModuleCoordinator:
    """Coordinate module execution with proper error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.modules = {}
    
    def execute_with_fallbacks(self, input_data: Any) -> Any:
        """Execute pipeline with comprehensive fallback mechanisms"""
        
        # Module execution with progressive fallbacks
        stages = [
            ('document_ingestion', self._execute_document_ingestion),
            ('embedding_generation', self._execute_embedding_generation),
            ('similarity_search', self._execute_similarity_search),
            ('rule_filtering', self._execute_rule_filtering),
            ('llm_evaluation', self._execute_llm_evaluation),
            ('report_generation', self._execute_report_generation)
        ]
        
        results = input_data
        for stage_name, stage_function in stages:
            try:
                results = stage_function(results)
                print(f"✓ {stage_name} completed successfully")
            except Exception as e:
                print(f"✗ {stage_name} failed: {e}")
                if self._should_continue_with_fallback(stage_name):
                    results = self._apply_fallback(stage_name, results, e)
                    print(f"→ Using fallback for {stage_name}")
                else:
                    raise RuntimeError(f"Critical failure in {stage_name}: {e}")
        
        return results
```

## 9.4 Performance Optimization

### 9.4.1 Memory Management

```python
class MemoryOptimizer:
    """Optimize memory usage during document processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get('chunk_size', 1000)
        self.max_workers = config.get('max_workers', 4)
    
    def process_in_batches(self, documents: List[Document]) -> List[Document]:
        """Process large document collections in memory-efficient batches"""
        batches = self._create_batches(documents)
        processed_batches = []
        
        for batch in batches:
            processed_batch = self._process_batch(batch)
            processed_batches.extend(processed_batch)
            # Clear memory between batches
            gc.collect()
        
        return processed_batches
```

### 9.4.2 Caching Mechanisms

```python
class ResultCache:
    """Cache expensive operations to improve performance"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}
    
    def get_or_compute(self, key: str, compute_func: Callable) -> Any:
        """Get cached result or compute and cache"""
        if key in self.cache:
            return self.cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                self.cache[key] = result
                return result
        
        # Compute and cache
        result = compute_func()
        self.cache[key] = result
        
        # Save to disk
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
```

## 9.5 Testing and Quality Assurance

### 9.5.1 Unit Testing Framework

```python
class ModuleTester:
    """Comprehensive testing for all system modules"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_module_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests for all modules"""
        tests = [
            ('document_ingestion', self.test_document_ingestion),
            ('embedding_generation', self.test_embedding_generation),
            ('similarity_search', self.test_similarity_search),
            ('rule_filtering', self.test_rule_filtering),
            ('llm_evaluation', self.test_llm_evaluation),
            ('report_generation', self.test_report_generation)
        ]
        
        for module_name, test_function in tests:
            try:
                result = test_function()
                self.test_results[module_name] = {
                    'status': 'PASSED',
                    'details': result
                }
            except Exception as e:
                self.test_results[module_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        return self.test_results
```

This comprehensive coding implementation section demonstrates the modular, well-structured approach of the AI-Powered Tender Evaluation System, highlighting the key components, their interactions, and the robust engineering practices employed throughout the codebase.