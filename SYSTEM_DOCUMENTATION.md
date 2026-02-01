# AI-Powered Tender Evaluation System - Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Components](#system-components)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

---

## Introduction

The AI-Powered Tender Evaluation System is an advanced document analysis platform designed to automate and streamline the tender evaluation process. The system combines artificial intelligence, semantic analysis, and professional reporting to provide objective, comprehensive evaluation of tender proposals against organizational requirements.

### Key Features
- **AI-Powered Analysis**: Leverages Large Language Models for intelligent document evaluation
- **Semantic Matching**: Uses advanced embedding techniques for content alignment
- **Rule-Based Filtering**: Applies business constraints and requirements validation
- **Professional Reporting**: Generates comprehensive PDF reports with detailed analysis
- **Web Interface**: User-friendly Streamlit-based interface for ease of use
- **Multi-Dimensional Scoring**: Evaluates proposals across technical, financial, and timeline dimensions

---

## System Components

### Core Modules

#### Document Ingestion (`src/ingestion/`)
- **Purpose**: Load and preprocess documents
- **Capabilities**:
  - PDF document processing
  - OCR for scanned documents
  - Text extraction and cleaning
  - Document validation and format checking

#### Embedding Generation (`src/embeddings/`)
- **Purpose**: Convert text to vector representations
- **Model**: nomic-embed-text
- **Function**: Creates semantic-preserving vector embeddings
- **Output**: High-dimensional vectors for similarity matching

#### Vector Database (`src/vector_db/`)
- **Purpose**: Store and manage document embeddings
- **Implementation**: FAISS (Facebook AI Similarity Search)
- **Features**: Fast similarity search, cosine distance calculation
- **Optimization**: Memory-efficient for large document collections

#### Similarity Search (`src/search/`)
- **Purpose**: Match requirements with proposals
- **Algorithm**: Cosine similarity
- **Output**: Ranked results based on content alignment
- **Features**: Semantic matching, relevance scoring

#### Rule Filtering (`src/filtering/`)
- **Purpose**: Apply business constraints
- **Filters**:
  - Budget compliance validation
  - Timeline feasibility checks
  - Certification requirements
  - Technical prerequisites
- **Validation**: Ensures proposals meet minimum criteria

#### LLM Evaluation (`src/evaluation/`)
- **Purpose**: Detailed scoring and analysis
- **Model**: TinyLlama via Ollama
- **Scoring Dimensions**:
  - Technical match percentage (0-100%)
  - Financial viability (0-100%)
  - Timeline alignment (0-100%)
  - Overall compliance score
- **Output**: Natural language explanations and recommendations

#### Report Generation (`src/reporting/`)
- **Purpose**: Create professional evaluation reports
- **Format**: LaTeX → PDF
- **Features**:
  - Comparative analysis tables
  - Visual dashboards
  - Detailed recommendations
  - Executive summaries
  - Proper LaTeX special character escaping

#### User Interface (`src/interfaces/`)
- **Framework**: Streamlit
- **Features**:
  - Document upload interface
  - Configuration controls
  - Real-time progress monitoring
  - Results display and download
  - Interactive comparison tools

---

## Installation & Setup

### Prerequisites

#### System Requirements
- **Operating System**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: 2GB free space
- **Python**: Version 3.8 or higher

#### Required Software
1. **Python 3.8+**: Programming language runtime
2. **Ollama**: Local LLM serving platform
3. **MiKTeX**: LaTeX distribution for PDF generation
4. **Git**: Version control (for cloning repository)

### Installation Steps

#### 1. Clone Repository
```bash
git clone <repository-url>
cd "Project 2025 Ollama"
```

#### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Install Ollama
- Download from: https://ollama.ai/
- Start Ollama server: `ollama serve`

#### 4. Pull Required Models
```bash
ollama pull tinydolphin:latest
ollama pull nomic-embed-text
```

#### 5. Install LaTeX (for PDF generation)
**Windows**: Install MiKTeX from https://miktex.org/
**Linux**: `sudo apt-get install texlive-full`
**macOS**: Install MacTeX from https://www.tug.org/mactex/

---

## Configuration

### Configuration Files

#### Main Configuration (`config/config.yaml`)
```yaml
ollama:
  base_url: "http://localhost:11434"
  embedding_model: "nomic-embed-text"
  evaluation_model: "tinydolphin:latest"

processing:
  chunk_size: 1000
  overlap: 100
  max_workers: 4

evaluation:
  max_applicants: 10
  min_compliance_score: 70
  similarity_threshold: 0.3

ocr:
  enabled: true
  language: "eng"
  fallback_enabled: true

reporting:
  template: "comprehensive_tec_template.tex"
  disable_pdf_compilation: false
```

#### Tender Rules (`config/tender_rules.yaml`)
```yaml
budget_filters:
  - min_amount: 50000
    max_amount: 200000
    currency: "USD"

timeline_filters:
  - max_duration_months: 12
  - required_start_date: "2024-01-01"

certification_requirements:
  - "ISO 9001"
  - "SOC 2 Type II"
  - "GDPR Compliance"

technical_requirements:
  - "Minimum 5 years experience"
  - "Team of 10+ qualified professionals"
```

### Environment Variables
- `OLLAMA_BASE_URL`: Override default Ollama URL
- `DISABLE_PDF_COMPILATION`: Skip PDF generation (for debugging)
- `MAX_APPlicants_TO_EVALUATE`: Limit number of proposals to evaluate

---

## Usage

### Web Interface (Recommended)

#### Starting the Application
```bash
streamlit run src/interfaces/professional_streamlit_app.py
```

#### Interface Workflow
1. **Upload Documents**:
   - Upload tender requirement document (PDF)
   - Upload multiple proposal documents (PDF)
   - System validates file formats automatically

2. **Configure Evaluation**:
   - Set maximum number of applicants to evaluate
   - Define minimum compliance threshold
   - Select evaluation date

3. **Run Evaluation**:
   - Click "Start Evaluation" button
   - Monitor progress in real-time
   - View intermediate results

4. **Review Results**:
   - Interactive dashboard with scores
   - Detailed comparison tables
   - Downloadable PDF reports
   - Export options for further analysis

### Command Line Interface

#### Basic Usage
```bash
python main.py --mode cli
```

#### Batch Processing
```bash
python main.py --mode cli --batch-mode --input-dir ./data/input --output-dir ./data/output
```

#### Custom Configuration
```bash
python main.py --mode cli --config config/custom.yaml
```

### API Integration (Coming Soon)
The system will support REST API endpoints for integration with other systems.

---

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Problems
**Symptoms**: "Connection refused" or "Model not found"
**Solutions**:
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve

# Pull required models
ollama pull tinydolphin:latest
ollama pull nomic-embed-text
```

#### 2. PDF Processing Failures
**Symptoms**: "Failed to extract text" or empty documents
**Solutions**:
- Ensure PDF files are not password-protected
- Check file permissions
- Verify sufficient disk space
- Try re-saving PDFs using different software

#### 3. LaTeX Compilation Errors
**Symptoms**: PDF generation fails with compilation errors
**Solutions**:
```bash
# Check MiKTeX installation
pdflatex --version

# Update MiKTeX packages
miktex-console --admin --update

# Clear LaTeX cache
miktex-maketfm --quiet --admin
```

#### 4. Memory Issues
**Symptoms**: "Out of memory" or slow performance
**Solutions**:
- Close other applications to free RAM
- Process fewer documents simultaneously
- Increase virtual memory settings
- Use lighter Ollama models

#### 5. OCR Problems
**Symptoms**: Poor text extraction from scanned documents
**Solutions**:
- Verify image quality of scanned documents
- Check OCR language settings
- Ensure Tesseract is properly installed
- Try different scan resolutions (300 DPI recommended)

### Diagnostic Commands

#### System Health Check
```bash
# Check Python environment
python --version
pip list

# Check Ollama status
ollama ps

# Check LaTeX installation
pdflatex --version

# Run system diagnostics
python test_system.py
```

#### Log Analysis
```bash
# View recent logs
tail -f system.log

# Search for specific errors
grep "ERROR" system.log

# Analyze performance
grep "Processing time" system.log
```

---

## Maintenance

### Regular Maintenance Tasks

#### 1. Dependency Updates
```bash
# Update Python packages
pip install -r requirements.txt --upgrade

# Update Ollama models
ollama pull tinydolphin:latest
ollama pull nomic-embed-text
```

#### 2. System Cleanup
```bash
# Remove temporary files
find ./data/temp -type f -mtime +7 -delete

# Clean vector database cache
rm -rf ./data/vector_cache/*

# Archive old reports
mkdir -p ./archive/$(date +%Y-%m-%d)
mv ./data/reports/* ./archive/$(date +%Y-%m-%d)/
```

#### 3. Performance Monitoring
- Monitor system resource usage during processing
- Track processing times for different document volumes
- Review evaluation accuracy and consistency
- Update model configurations based on performance

### Backup Strategy
- Configuration files: Daily backup
- Vector databases: Weekly backup (when updated)
- Generated reports: As needed
- Document archives: Monthly backup

### Security Considerations
- Regular security updates for dependencies
- Access control for sensitive documents
- Secure configuration management
- Data retention policies

---

## Performance Optimization

### Processing Speed
- Optimize chunk sizes for document processing
- Use appropriate number of worker threads
- Implement caching for repeated operations
- Pre-load models when possible

### Memory Management
- Process documents in batches
- Clear memory between operations
- Use generators for large datasets
- Monitor memory usage patterns

### Accuracy Improvements
- Fine-tune embedding model parameters
- Adjust similarity thresholds
- Update evaluation criteria regularly
- Collect feedback for model improvement

---

## Support & Contact

### Getting Help
- Check documentation first
- Review troubleshooting section
- Contact system administrator
- Submit issue through GitHub (if applicable)

### Feedback & Suggestions
- Report bugs through official channels
- Suggest features for future releases
- Share performance observations
- Contribute to documentation improvements

---

*This documentation is maintained as part of the AI-Powered Tender Evaluation System project and reflects the current system capabilities as of January 2026.*