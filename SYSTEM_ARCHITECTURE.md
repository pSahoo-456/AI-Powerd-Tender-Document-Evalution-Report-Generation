# AI-Powered Tender Evaluation System - Architecture

## System Overview

The AI-Powered Tender Evaluation System is a comprehensive document analysis platform that leverages artificial intelligence to automate and enhance the tender evaluation process. The system processes tender requirements and proposal documents, performs intelligent matching using semantic analysis, and generates professional evaluation reports.

## Architecture Components

### 1. User Interface Layer
- **Professional Web Interface** (`src/interfaces/professional_streamlit_app.py`)
  - Built with Streamlit framework
  - Features: Document upload, configuration, real-time progress monitoring, results display
  - Responsive design with ITR logo integration
  - Interactive dashboards and comparison tables

### 2. Document Processing Layer
- **Document Ingestion** (`src/ingestion/document_loader.py`)
  - Handles PDF document loading and preprocessing
  - Supports multiple document formats
  - Integrates OCR capabilities for scanned documents
  - Text cleaning and normalization

- **Text Processing** (`src/parsing/text_processor.py`)
  - Extracts and cleans text content from documents
  - Segments content into processable chunks
  - Applies preprocessing transformations

- **OCR Processing** (`src/ocr/ocr_processor.py`)
  - Optical Character Recognition for scanned documents
  - Uses advanced OCR engines for accuracy
  - Fallback mechanism for document processing

### 3. AI & Machine Learning Layer
- **Embedding Generation** (`src/embeddings/embedding_generator.py`)
  - Converts text to vector representations using `nomic-embed-text` model
  - Semantic preservation for accurate matching
  - High-dimensional vector space mapping

- **Vector Database** (`src/vector_db/vector_store.py`)
  - Stores and manages document embeddings
  - Implements FAISS for efficient similarity search
  - Optimized for cosine similarity calculations

- **Similarity Search** (`src/search/similarity_search.py`)
  - Performs semantic matching between requirements and proposals
  - Uses cosine similarity algorithm
  - Returns ranked results based on content alignment

### 4. Evaluation Engine
- **Rule Filtering** (`src/filtering/rule_filter.py`)
  - Applies business constraints (budget, timeline, certifications)
  - Validates compliance with tender requirements
  - Filters proposals based on predefined criteria

- **LLM Evaluation** (`src/evaluation/llm_evaluator.py`)
  - Uses TinyLlama via Ollama for detailed analysis
  - Multi-dimensional scoring (technical, financial, timeline)
  - Natural language explanations and recommendations
  - Risk assessment and mitigation suggestions

### 5. Reporting Layer
- **Report Generation** (`src/reporting/report_generator.py`)
  - Creates professional LaTeX reports
  - Generates PDF documents with comprehensive analysis
  - Includes comparison tables, dashboards, and recommendations
  - Properly escapes special characters for LaTeX compilation

### 6. Configuration & Utilities
- **Configuration Loader** (`src/utils/config_loader.py`)
  - Manages YAML configuration files
  - Handles system settings and parameters
  - Supports environment-specific configurations

- **File Utilities** (`src/utils/file_utils.py`)
  - File handling and path management
  - Document validation and processing
  - System utility functions

## Technology Stack

### Frontend
- **Streamlit**: Web interface framework
- **HTML/CSS**: UI customization and styling
- **JavaScript**: Client-side interactions

### Backend
- **Python 3.12+**: Primary programming language
- **LangChain**: LLM integration framework
- **LangChain-Ollama**: Ollama-specific integrations

### AI & ML
- **Ollama**: Local LLM serving
- **TinyLlama**: Lightweight language model
- **nomic-embed-text**: Embedding model for semantic search
- **FAISS**: Vector database for similarity search

### Document Processing
- **Docling**: Advanced PDF processing
- **pdfplumber**: PDF parsing library
- **PyMuPDF**: PDF manipulation
- **Pillow/Pytesseract**: OCR capabilities

### Data & Storage
- **YAML**: Configuration management
- **JSON**: Data serialization
- **ChromaDB**: Vector database (alternative)

### Reporting
- **LaTeX**: Professional document generation
- **Jinja2**: Template engine
- **MiKTeX**: LaTeX distribution

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   User Input    │────│  Document Loader │────│ Text Processor   │
│ (Requirements   │    │   & Ingestion    │    │  & Cleaning      │
│  & Proposals)   │    └──────────────────┘    └──────────────────┘
└─────────────────┘                                     │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Configuration   │────│ Embedding Gen.   │────│ Similarity       │
│   Manager       │    │  & Vector Store  │    │   Search         │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Rule Filter     │────│ LLM Evaluator    │────│ Report Generator │
│ (Constraints)   │    │  & Scoring       │    │  & PDF Output    │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                                                        ▼
                                             ┌──────────────────┐
                                             │   Final Report   │
                                             │ (PDF/LaTeX)      │
                                             └──────────────────┘
```

## System Dependencies

### Core Dependencies
- langchain, langchain-community, langchain-ollama
- python-docling, pdfplumber, PyMuPDF
- faiss-cpu, ollama
- streamlit, jinja2
- PyYAML, numpy, pandas

### Optional Dependencies
- pytesseract, Pillow (for OCR)
- latex (for enhanced LaTeX processing)

## Security Considerations

- Local processing of sensitive documents
- No external data transmission
- Secure configuration management
- Isolated model execution environment

## Scalability Features

- Modular architecture supporting horizontal scaling
- Efficient vector database indexing
- Batch processing capabilities
- Memory-efficient processing for large documents