# AI Tender Evaluation System - Quick Reference

## Main Components and Their Key Functions

### 1. DocumentLoader ([src/ingestion/document_loader.py](file:///d:/Project%202025%20Ollama/src/ingestion/document_loader.py))
**Purpose**: Load documents from files

**Key Methods**:
- `load_documents(directory_path)`: Load all documents from a directory
- `_load_pdf_document(file_path)`: Extract text from PDF files
- `_load_text_document(file_path)`: Read text files

### 2. TextProcessor ([src/parsing/text_processor.py](file:///d:/Project%202025%20Ollama/src/parsing/text_processor.py))
**Purpose**: Process and chunk text

**Key Methods**:
- `chunk_documents(documents)`: Split large documents into smaller chunks
- `extract_metadata(document, source_type)`: Extract document metadata

### 3. OCRProcessor ([src/ocr/ocr_processor.py](file:///d:/Project%202025%20Ollama/src/ocr/ocr_processor.py))
**Purpose**: Handle scanned documents

**Key Methods**:
- `process_scanned_pdf(pdf_path)`: Extract text from scanned PDFs
- `_check_tesseract()`: Verify OCR availability

### 4. EmbeddingGenerator ([src/embeddings/embedding_generator.py](file:///d:/Project%202025%20Ollama/src/embeddings/embedding_generator.py))
**Purpose**: Convert text to vectors

**Key Methods**:
- `generate_embeddings(texts)`: Create embeddings for text list
- `generate_document_embeddings(documents)`: Add embeddings to documents
- `_simulate_embeddings(texts)`: Fallback when Ollama unavailable

### 5. VectorStoreManager ([src/vector_db/vector_store.py](file:///d:/Project%202025%20Ollama/src/vector_db/vector_store.py))
**Purpose**: Manage document embeddings storage

**Key Methods**:
- `initialize_store(documents)`: Set up vector database
- `add_documents(documents)`: Add documents to store
- `similarity_search(query, k)`: Find similar documents

### 6. SimilaritySearcher ([src/search/similarity_search.py](file:///d:/Project%202025%20Ollama/src/search/similarity_search.py))
**Purpose**: Find document similarities

**Key Methods**:
- `search_applicants_by_requirements(requirements, applicants, top_k)`: Rank applicants
- `rank_applicants(requirements, applicants)`: Alternative ranking method
- `_calculate_similarity(text1, text2)`: Compute text similarity

### 7. RuleFilter ([src/filtering/rule_filter.py](file:///d:/Project%202025%20Ollama/src/filtering/rule_filter.py))
**Purpose**: Apply business rules

**Key Methods**:
- `apply_filters(applicants)`: Apply all configured filters
- `_apply_budget_filter(applicants)`: Filter by budget constraints
- `_apply_timeline_filter(applicants)`: Filter by timeline requirements
- `_apply_certification_filter(applicants)`: Filter by certifications

### 8. LLMEvaluator ([src/evaluation/llm_evaluator.py](file:///d:/Project%202025%20Ollama/src/evaluation/llm_evaluator.py))
**Purpose**: AI-powered evaluation

**Key Methods**:
- `evaluate_applicants(requirements, applicants, max_applicants)`: Evaluate proposals
- `_simulate_evaluation(applicants)`: Fallback when Ollama unavailable
- `_evaluate_applicant(requirements, proposal)`: Evaluate single proposal
- `_parse_evaluation_response(response)`: Parse LLM output

### 9. ReportGenerator ([src/reporting/report_generator.py](file:///d:/Project%202025%20Ollama/src/reporting/report_generator.py))
**Purpose**: Create evaluation reports

**Key Methods**:
- `generate_evaluation_report(requirements, evaluated_applicants, title)`: Generate report
- `_convert_to_pdf(latex_file)`: Compile LaTeX to PDF
- `_prepare_applicant_data(evaluated_applicants)`: Format data for template
- `_clean_text_for_latex(text)`: Prepare text for LaTeX

### 10. ConfigLoader ([src/utils/config_loader.py](file:///d:/Project%202025%20Ollama/src/utils/config_loader.py))
**Purpose**: Load system configuration

**Key Methods**:
- `__init__(config_path)`: Initialize with config file
- `_load_config()`: Load YAML configuration
- `get(key_path, default)`: Get config value with dot notation
- `get_ollama_config()`: Get Ollama settings

## Main Entry Points

### Web Interface
**File**: [src/interfaces/professional_streamlit_app.py](file:///d:/Project%202025%20Ollama/src/interfaces/professional_streamlit_app.py)
**Function**: `run_professional_app(config_path)`
**Usage**: `streamlit run src/interfaces/professional_streamlit_app.py`

### CLI Interface
**File**: [src/interfaces/cli_app.py](file:///d:/Project%202025%20Ollama/src/interfaces/cli_app.py)
**Function**: `run_cli_app(config_path)`
**Usage**: `python main.py --mode cli`

### Main Launcher
**File**: [main.py](file:///d:/Project%202025%20Ollama/main.py)
**Function**: `main()`
**Usage**: `python main.py --mode [cli|web]`

## Configuration Files

### Main Config
**File**: [config/config.yaml](file:///d:/Project%202025%20Ollama/config/config.yaml)
**Sections**:
- `ollama`: Ollama service settings
- `vector_db`: Vector database configuration
- `ocr`: OCR settings
- `paths`: File path configurations
- `processing`: Text processing parameters
- `evaluation`: Evaluation settings

### Rules Config
**File**: [config/tender_rules.yaml](file:///d:/Project%202025%20Ollama/config/tender_rules.yaml)
**Sections**:
- `budget`: Budget constraints
- `timeline`: Timeline requirements
- `certifications`: Required certifications
- `technical_requirements`: Technical matching criteria
- `scoring_weights`: Evaluation weight distribution
- `evaluation_thresholds`: Minimum scores and counts

## Sample Data Files

### Tender Requirements
**File**: [data/tender_requirement_IT_project.txt](file:///d:/Project%202025%20Ollama/data/tender_requirement_IT_project.txt)
**Content**: IT project requirements with budget ($2.5M) and timeline (18 months)

### Vendor Proposals
1. **CloudTech Solutions**: [data/proposal_cloudtech_solutions.txt](file:///d:/Project%202025%20Ollama/data/proposal_cloudtech_solutions.txt)
   - Budget: $2.35M
   - Timeline: 16 months
   
2. **DigitalPro Enterprises**: [data/proposal_digitalpro_enterprises.txt](file:///d:/Project%202025%20Ollama/data/proposal_digitalpro_enterprises.txt)
   - Budget: $2.28M
   - Timeline: 17 months

## Key Dependencies

### Core Libraries
- **langchain**: AI and LLM framework
- **pdfplumber**: PDF text extraction
- **faiss/chromadb**: Vector databases
- **ollama**: Local AI model serving
- **streamlit**: Web interface framework
- **jinja2**: Template engine for reports

### Optional Libraries
- **pytesseract**: OCR processing
- **pymupdf**: Alternative PDF processing

## Execution Commands

### Running the System
```bash
# Web interface
streamlit run src/interfaces/professional_streamlit_app.py

# CLI mode
python main.py --mode cli

# Test/demo
python test_system.py
```

### Ollama Setup
```bash
# Install Ollama from https://ollama.com/

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.1
```

### LaTeX Setup (for PDF reports)
**Windows**: Install MiKTeX
**Linux**: `sudo apt-get install texlive-full`
**Mac**: Install MacTeX

## Error Handling Fallbacks

1. **Ollama Unavailable**:
   - EmbeddingGenerator uses random vectors
   - LLMEvaluator uses simulated scores

2. **LaTeX Missing**:
   - ReportGenerator creates .tex files
   - Manual compilation option provided

3. **OCR Unavailable**:
   - Skips OCR processing
   - Continues with available text

4. **PDF Processing Failures**:
   - Multiple extraction attempts
   - Graceful degradation to basic processing

## Testing

**Test Script**: [test_system.py](file:///d:/Project%202025%20Ollama/test_system.py)
**Function**: `test_system()`
**Usage**: `python test_system.py`

Tests the complete workflow with sample data:
1. Load requirements and proposals
2. Generate embeddings
3. Evaluate with LLM
4. Generate report