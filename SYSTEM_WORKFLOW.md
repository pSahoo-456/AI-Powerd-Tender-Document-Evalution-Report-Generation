# AI-Powered Tender Evaluation System - Workflow

## Overview

This document describes the complete workflow of the AI-Powered Tender Evaluation System, detailing how documents are processed from initial upload to final report generation. The system follows a systematic, multi-stage process to ensure accurate, comprehensive evaluation of tender proposals.

## System Workflow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Document      │    │  Preprocessing   │    │   Text           │
│   Upload       │───▶│    & Validation  │───▶│   Extraction     │
│ (Requirements   │    │                  │    │                  │
│  & Proposals)   │    │  - Format check  │    │  - PDF parsing   │
└─────────────────┘    │  - File size     │    │  - OCR if needed│
                       │  - Virus scan    │    │  - Cleaning      │
                       └──────────────────┘    └──────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Configuration   │    │ Embedding        │    │ Vector           │
│ Setup          │───▶│   Generation     │───▶│   Storage        │
│                │    │                  │    │                  │
│ - Evaluation    │    │ - nomic-embed-text│   │ - FAISS database│
│   parameters    │    │ - Semantic       │    │ - Cosine         │
│ - Constraints   │    │   vectors        │    │   similarity     │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Rule-Based      │    │ LLM Evaluation   │    │ Report           │
│ Filtering      │───▶│    & Scoring     │───▶│   Generation     │
│                │    │                  │    │                  │
│ - Budget check  │    │ - Multi-dimen.   │    │ - LaTeX template │
│ - Timeline      │    │   scoring        │    │ - PDF compilation│
│ - Certificates  │    │ - Explanations   │    │ - Professional   │
└─────────────────┘    └──────────────────┘    │   formatting     │
                                               └──────────────────┘
```

## Detailed Workflow Stages

### Stage 1: Document Upload & Validation

#### Input Processing
1. **Document Collection**:
   - Upload tender requirement document
   - Upload multiple proposal documents
   - Support for PDF format (with OCR for scanned documents)

2. **Validation Checks**:
   - File format verification (PDF only)
   - File size validation (max 50MB per file)
   - Document integrity check
   - Basic security scanning

3. **Initial Processing**:
   - File metadata extraction
   - Document categorization (requirement vs. proposal)
   - Preparation for ingestion pipeline

#### Expected Duration: 10-30 seconds per document

### Stage 2: Document Preprocessing & Text Extraction

#### Text Extraction Process
1. **Primary Extraction**:
   - Use Docling for direct PDF text extraction
   - Preserve document structure and formatting
   - Extract metadata and document properties

2. **OCR Fallback**:
   - Activate OCR for scanned documents
   - Use RapidOCR engine for accuracy
   - Handle multi-language documents
   - Fallback to Tesseract if needed

3. **Text Cleaning**:
   - Remove extraction artifacts
   - Normalize whitespace and formatting
   - Clean special characters
   - Prepare text for semantic processing

#### Expected Duration: 30-90 seconds per document (depending on size)

### Stage 3: Configuration & Setup

#### System Configuration
1. **Evaluation Parameters**:
   - Set maximum number of applicants to evaluate
   - Define minimum compliance threshold
   - Configure scoring weights
   - Set timeline constraints

2. **Rule Application**:
   - Load budget constraints
   - Apply timeline requirements
   - Validate certification needs
   - Set technical prerequisites

#### Expected Duration: 1-2 seconds

### Stage 4: Embedding Generation

#### Vector Creation Process
1. **Text Chunking**:
   - Split large documents into manageable chunks
   - Maintain context with overlapping segments
   - Optimize chunk size for embedding quality

2. **Embedding Generation**:
   - Use nomic-embed-text model
   - Convert text to high-dimensional vectors
   - Preserve semantic relationships
   - Store vectors with document references

3. **Quality Assurance**:
   - Validate embedding dimensions
   - Check vector integrity
   - Ensure semantic preservation

#### Expected Duration: 2-5 minutes (depending on document count and size)

### Stage 5: Similarity Analysis

#### Semantic Matching
1. **Vector Database Population**:
   - Store requirement vectors in FAISS index
   - Index proposal vectors for fast search
   - Optimize for cosine similarity calculations

2. **Similarity Calculation**:
   - Compute cosine similarity between reqs and proposals
   - Rank proposals by content alignment
   - Identify relevant sections and matches

3. **Initial Ranking**:
   - Create preliminary proposal rankings
   - Flag potential mismatches
   - Prepare for detailed evaluation

#### Expected Duration: 1-3 minutes

### Stage 6: Rule-Based Filtering

#### Constraint Validation
1. **Budget Compliance**:
   - Compare proposed costs with budget limits
   - Validate pricing structures
   - Flag budget overruns

2. **Timeline Feasibility**:
   - Check delivery schedules against requirements
   - Validate milestone timelines
   - Assess project duration appropriateness

3. **Certification Verification**:
   - Confirm required certifications present
   - Validate certificate validity periods
   - Check compliance with standards

4. **Technical Prerequisites**:
   - Verify technical capability claims
   - Check experience requirements
   - Validate resource commitments

#### Expected Duration: 30-60 seconds

### Stage 7: LLM Evaluation & Scoring

#### AI-Powered Analysis
1. **Detailed Analysis**:
   - Use TinyLlama model via Ollama
   - Perform multi-dimensional evaluation
   - Analyze technical, financial, and timeline aspects

2. **Scoring Process**:
   - **Technical Match**: Percentage alignment with technical requirements
   - **Financial Match**: Cost-effectiveness and budget compliance
   - **Timeline Match**: Schedule feasibility and milestone alignment
   - **Overall Score**: Weighted composite of all factors

3. **Natural Language Processing**:
   - Generate explanation for scores
   - Identify strengths and weaknesses
   - Provide risk assessments
   - Create recommendation rationale

#### Expected Duration: 5-15 minutes (depending on proposal count)

### Stage 8: Report Generation

#### Professional Report Creation
1. **Template Population**:
   - Use comprehensive LaTeX template
   - Populate with evaluation results
   - Include comparison tables and dashboards

2. **Content Generation**:
   - Executive summary
   - Detailed comparison matrices
   - Individual proposal analyses
   - Strategic recommendations

3. **Formatting & Compilation**:
   - Properly escape special characters for LaTeX
   - Compile LaTeX to PDF using MiKTeX
   - Apply professional formatting
   - Validate document integrity

#### Expected Duration: 1-3 minutes

## Performance Metrics

### Processing Time Estimates

| Document Type | Count | Avg. Size | Estimated Time |
|---------------|-------|-----------|----------------|
| Small Tender | 1 req + 2-3 props | <10 pages | 15-20 minutes |
| Medium Tender | 1 req + 4-6 props | 10-50 pages | 25-35 minutes |
| Large Tender | 1 req + 7-10 props | 50+ pages | 40-60 minutes |

### Resource Utilization

#### During Processing
- **CPU**: Moderate usage during AI processing
- **RAM**: 2-4GB peak during vector operations
- **Storage**: 50-200MB per complete evaluation
- **Network**: Minimal (local Ollama only)

#### Peak Usage Times
- **Embedding Generation**: High RAM usage
- **LLM Evaluation**: Moderate CPU usage
- **Report Generation**: Moderate storage usage

## Error Handling & Recovery

### Stage-Level Error Handling

#### Document Processing Errors
- **Invalid Format**: Reject with clear error message
- **Corrupted Files**: Skip with notification
- **Security Issues**: Quarantine and alert

#### AI Processing Errors
- **Ollama Unavailable**: Retry with exponential backoff
- **Model Loading Failures**: Fallback to simulation mode
- **Memory Issues**: Process in smaller batches

#### Compilation Errors
- **LaTeX Issues**: Generate LaTeX file as fallback
- **Missing Packages**: Install required packages
- **Permission Issues**: Use alternative directories

### Recovery Procedures
1. **Automatic Recovery**: System attempts recovery for transient errors
2. **Manual Intervention**: User notified for persistent issues
3. **Partial Results**: Save partial results when possible
4. **Rollback**: Revert to previous state if needed

## Quality Assurance

### Validation Points

#### Data Validation
- Document integrity checks
- Text extraction quality assessment
- Vector embedding validation

#### Process Validation
- Similarity threshold verification
- Scoring consistency checks
- Rule application validation

#### Output Validation
- Report completeness verification
- PDF compilation success
- Content accuracy review

## Monitoring & Logging

### System Monitoring
- Processing stage progression
- Resource utilization tracking
- Error rate monitoring
- Performance metric collection

### Log Categories
- **Debug**: Detailed processing information
- **Info**: Stage completion and progress
- **Warning**: Non-critical issues
- **Error**: Critical failures and exceptions

## Customization Options

### Workflow Modifications
- Adjustable processing parameters
- Configurable scoring weights
- Custom rule definitions
- Template modifications

### Performance Tuning
- Parallel processing options
- Memory optimization settings
- Batch size adjustments
- Caching strategies

---

*This workflow document represents the current operational procedures of the AI-Powered Tender Evaluation System as of January 2026. The workflow may be updated as system capabilities evolve.*