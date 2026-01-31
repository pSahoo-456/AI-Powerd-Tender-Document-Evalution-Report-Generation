# AI-Powered Tender Document Evaluation & Report Generation

A Python-based system I built to automate the evaluation of tender proposals using AI and semantic analysis. After spending countless hours manually reviewing tender documents, I decided to build a solution that could objectively assess proposals against organizational requirements.

## What It Does

This system takes organization requirement documents and candidate proposals, then uses AI to:
- Extract and process text from PDFs
- Generate semantic embeddings to understand document content
- Match proposals against requirements using similarity analysis
- Apply configurable business rules for filtering
- Generate detailed evaluation reports

## Tech Stack

- **Python 3.8+**: Core implementation
- **Ollama**: Running local LLMs and embedding models (nomic-embed-text, llama3.1)
- **Streamlit**: Web interface
- **FAISS**: Vector similarity search
- **LaTeX**: Professional report generation
- **pdfplumber/pymupdf**: PDF processing

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install and run Ollama:
   - Download from https://ollama.com/
   - Pull required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1
   ```

3. Install LaTeX for PDF report generation:
   - Windows: Install MiKTeX
   - Linux: `sudo apt-get install texlive-full`
   - Mac: Install MacTeX

4. Run the application:
   ```bash
   streamlit run run_professional_app.py
   ```

## How It Works

The system follows these steps:

1. **Document Ingestion**: Load organization requirements and candidate proposals
2. **Embedding Generation**: Convert documents to semantic vectors
3. **Similarity Matching**: Find relevant sections between requirements and proposals
4. **Rule-Based Filtering**: Apply configurable evaluation criteria
5. **LLM Evaluation**: Detailed analysis and scoring using language models
6. **Report Generation**: Create professional evaluation reports

## Project Structure

```
├── config/                 # Configuration files
│   ├── config.yaml
│   └── tender_rules.yaml
├── src/                    # Source modules
│   ├── embeddings/         # Embedding generation
│   ├── evaluation/         # LLM evaluation
│   ├── filtering/          # Rule-based filtering
│   ├── ingestion/          # Document loading
│   ├── interfaces/         # User interfaces
│   ├── ocr/                # OCR processing
│   ├── parsing/            # Text processing
│   ├── reporting/          # Report generation
│   ├── search/             # Similarity search
│   ├── utils/              # Utilities
│   └── vector_db/          # Vector storage
├── templates/              # Report templates
├── main.py                 # CLI entry point
└── run_professional_app.py # Web app entry point
```

## Configuration

Edit `config/config.yaml` to adjust:
- Ollama settings (API URL, models)
- Processing parameters (chunk size, max proposals)
- Evaluation thresholds

Business rules can be modified in `config/tender_rules.yaml`.

## Usage

The system offers two interfaces:

1. **Web Interface** (recommended):
   ```bash
   streamlit run run_professional_app.py
   ```

2. **CLI Mode**:
   ```bash
   python main.py --mode cli
   ```

## Motivation

This project came from real frustration with the manual, time-consuming process of evaluating tender proposals. Organizations often receive dozens of proposals that need to be evaluated against complex requirements. This system helps ensure consistent, objective evaluation while saving valuable time.

## License

MIT © Prakash Sahoo