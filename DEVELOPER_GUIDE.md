# AI-Powered Tender Evaluation System - Developer Guide

## Overview

This guide provides instructions for developers who want to customize or modify the system, particularly for changing AI models, adding new features, or adapting the system to different environments.

## Table of Contents
1. [Model Configuration Changes](#model-configuration-changes)
2. [Environment Setup Modifications](#environment-setup-modifications)
3. [Code Customization Points](#code-customization-points)
4. [Configuration File Updates](#configuration-file-updates)
5. [Dependency Management](#dependency-management)
6. [Testing and Validation](#testing-and-validation)

---

## Model Configuration Changes

### 1. Changing the Embedding Model

**Files to Modify:**
- `config/config.yaml`
- `src/embeddings/embedding_generator.py`

**Changes Required:**

#### config/config.yaml
```yaml
ollama:
  base_url: "http://localhost:11434"
  embedding_model: "your-new-embedding-model"  # Change this
  evaluation_model: "tinydolphin:latest"
```

#### src/embeddings/embedding_generator.py
```python
class EmbeddingGenerator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # The model name is automatically loaded from config
        self.model_name = self.config.get('embedding_model', 'nomic-embed-text')
```

**Important Notes:**
- Ensure the new model is compatible with Ollama's embedding API
- Vector dimensions may change (nomic-embed-text uses 768 dimensions)
- Update any dimension-dependent code in similarity search

### 2. Changing the LLM Evaluation Model

**Files to Modify:**
- `config/config.yaml`
- `src/evaluation/llm_evaluator.py`

**Changes Required:**

#### config/config.yaml
```yaml
ollama:
  base_url: "http://localhost:11434"
  embedding_model: "nomic-embed-text"
  evaluation_model: "your-new-llm-model"  # Change this
```

#### Code Adaptations
```python
# In src/evaluation/llm_evaluator.py
def _create_evaluation_prompt(self, requirements, applicant):
    # You may need to adjust the prompt format for different models
    prompt = f"""
    # Model-specific prompt formatting here
    # Different models may require different instruction formats
    """
    return prompt

def _parse_evaluation_response(self, response, applicant):
    # Parsing logic may need adjustment for different model outputs
    # Some models may return JSON directly, others may need regex parsing
    pass
```

**Important Considerations:**
- Test prompt engineering with your new model
- Adjust response parsing based on model output format
- Consider model context window limitations
- Update token usage monitoring if needed

### 3. Adding New Ollama Models

**Process:**
1. Pull the new model: `ollama pull model-name`
2. Update configuration files
3. Test integration thoroughly
4. Document model-specific requirements

---

## Environment Setup Modifications

### 1. Different Operating Systems

#### Linux Setup
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3 python3-pip texlive-full tesseract-ocr

# Install Python packages
pip3 install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

#### macOS Setup
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python tesseract
brew install --cask mactex

# Install Python packages
pip3 install -r requirements.txt

# Install Ollama
brew install ollama
```

### 2. Remote Ollama Server Configuration

**Files to Modify:**
- `config/config.yaml`

#### Configuration for Remote Ollama
```yaml
ollama:
  base_url: "http://your-ollama-server:11434"  # Remote server URL
  embedding_model: "nomic-embed-text"
  evaluation_model: "tinydolphin:latest"
  timeout: 300  # Increase timeout for network latency
```

**Network Considerations:**
- Ensure firewall allows connections to Ollama port
- Configure appropriate timeout values
- Consider authentication if required
- Test network reliability

### 3. Memory-Constrained Environments

**Files to Modify:**
- `config/config.yaml`

#### Low-Memory Configuration
```yaml
processing:
  chunk_size: 500          # Smaller chunks
  max_workers: 2           # Fewer parallel processes
  batch_size: 5            # Smaller batch sizes

ollama:
  timeout: 600             # Longer timeouts
  retries: 3               # More retry attempts
```

**Additional Optimizations:**
```python
# In memory-critical sections, add explicit garbage collection
import gc
gc.collect()  # Force garbage collection

# Use context managers for large data processing
with open(file_path, 'r') as f:
    # Process data in chunks
    pass
```

---

## Code Customization Points

### 1. Document Processing Pipeline

**Files to Modify:**
- `src/ingestion/document_loader.py`
- `src/ocr/ocr_processor.py`

#### Adding New Document Types
```python
# In src/ingestion/document_loader.py
def _load_document(self, file_path: str) -> List[Document]:
    extension = get_file_extension(file_path)
    
    if extension == '.docx':
        return self._load_word_document(file_path)
    elif extension == '.xlsx':
        return self._load_excel_document(file_path)
    # Add your new format here
    elif extension == '.your-format':
        return self._load_your_format(file_path)
    else:
        return super()._load_document(file_path)
```

### 2. Similarity Search Algorithms

**Files to Modify:**
- `src/search/similarity_search.py`

#### Adding New Similarity Metrics
```python
class CustomSimilaritySearcher(SimilaritySearcher):
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Add Jaccard similarity as alternative"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_bm25_similarity(self, text1: str, text2: str) -> float:
        """Add BM25 scoring algorithm"""
        # Implementation here
        pass
```

### 3. Report Generation Templates

**Files to Modify:**
- `templates/your-template.tex` (create new)
- `src/reporting/report_generator.py`

#### Creating Custom Report Templates
```python
# In src/reporting/report_generator.py
def generate_custom_report(self, data, template_name):
    """Generate reports using custom templates"""
    template = self.env.get_template(template_name)
    # Add custom data processing
    custom_data = self._process_custom_data(data)
    return template.render(**custom_data)

def _process_custom_data(self, data):
    """Custom data processing logic"""
    # Transform data for your specific template needs
    processed = {
        'custom_field': self._extract_custom_info(data),
        'formatted_data': self._format_data_special_way(data)
    }
    return processed
```

### 4. Evaluation Criteria Modification

**Files to Modify:**
- `src/evaluation/llm_evaluator.py`
- `src/filtering/rule_filter.py`

#### Adding New Evaluation Dimensions
```python
# In LLM evaluator
def _create_evaluation_prompt(self, requirements, applicant):
    prompt = f"""
    Evaluate the proposal with these additional criteria:
    7. Risk Assessment Score (0-100)
    8. Innovation Factor Score (0-100)
    9. Team Experience Score (0-100)
    """
    return prompt

# In Rule Filter
def _apply_custom_filter(self, applicant):
    """Add custom business rules"""
    # Your custom filtering logic
    pass
```

---

## Configuration File Updates

### 1. Main Configuration (`config/config.yaml`)

```yaml
# Core system configuration
system:
  name: "Custom Tender Evaluation System"
  version: "1.0.0"
  environment: "development"  # development/staging/production

# Processing settings
processing:
  chunk_size: 1000
  overlap: 100
  max_workers: 4
  timeout: 300

# AI models configuration
ollama:
  base_url: "http://localhost:11434"
  embedding_model: "your-embedding-model"
  evaluation_model: "your-evaluation-model"
  timeout: 300
  retries: 3

# Filtering rules
filtering:
  budget:
    min_amount: 10000
    max_amount: 500000
    currency: "USD"
  timeline:
    max_duration_months: 24
  certifications:
    - "ISO 9001"
    - "Your Custom Cert"

# Report generation
reporting:
  template: "custom_template.tex"
  output_format: "pdf"  # pdf/latex/html
  company_logo: "path/to/logo.png"
```

### 2. Environment-Specific Configuration

```yaml
# config/development.yaml
ollama:
  base_url: "http://localhost:11434"
  timeout: 120

# config/production.yaml
ollama:
  base_url: "http://production-server:11434"
  timeout: 300
  retries: 5
```

### 3. Model-Specific Configuration

```yaml
# config/model-specific.yaml
models:
  embeddings:
    default: "nomic-embed-text"
    alternatives:
      - "all-minilm"
      - "sentence-transformers"
  llm:
    default: "tinydolphin:latest"
    alternatives:
      - "llama2:7b"
      - "mistral:7b"
    model_specific_prompts:
      llama2: "Specific instructions for Llama2"
      mistral: "Specific instructions for Mistral"
```

---

## Dependency Management

### 1. Adding New Python Dependencies

**Files to Modify:**
- `requirements.txt`
- `setup.py` (if exists)

#### Example Dependency Addition
```txt
# In requirements.txt
# Core dependencies
langchain==0.3.0
langchain-ollama==0.2.0

# Your new dependencies
your-new-package==1.0.0
another-package>=2.0.0
```

**Installation Process:**
```bash
# Install new dependencies
pip install -r requirements.txt

# Or install specific package
pip install your-new-package
```

### 2. Managing Optional Dependencies

#### Feature-Based Dependencies
```txt
# In requirements.txt
# Required dependencies
streamlit==1.28.0

# Optional: OCR capabilities
pytesseract>=0.3.10  # Only if OCR is needed
pillow>=10.0.1

# Optional: Advanced PDF processing
python-docling>=2.0.0  # Only for complex documents

# Optional: LaTeX processing
latex==0.7.0  # Only if generating PDF reports
```

### 3. Version Pinning Strategy

**Recommended Approach:**
- Pin major versions for stability (`langchain==0.3.0`)
- Allow minor version updates (`numpy>=1.24.0,<2.0.0`)
- Use compatible version ranges for ecosystem packages

---

## Testing and Validation

### 1. Unit Testing Framework

```python
# tests/test_model_integration.py
import pytest
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.evaluation.llm_evaluator import LLMEvaluator

class TestModelIntegration:
    def setup_method(self):
        self.embedding_generator = EmbeddingGenerator()
        self.llm_evaluator = LLMEvaluator()
    
    def test_new_embedding_model(self):
        """Test that new embedding model works correctly"""
        test_text = "This is a test document"
        embeddings = self.embedding_generator.generate_embeddings([test_text])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768  # Verify dimension
        
    def test_new_llm_model(self):
        """Test LLM evaluation with new model"""
        result = self.llm_evaluator.evaluate_applicants([], [])
        assert isinstance(result, list)
```

### 2. Integration Testing

```python
# tests/test_full_pipeline.py
def test_complete_evaluation_pipeline():
    """Test the entire evaluation workflow"""
    # Setup test data
    requirements = load_test_requirements()
    proposals = load_test_proposals()
    
    # Run complete pipeline
    result = evaluate_tender_proposals(requirements, proposals)
    
    # Validate results
    assert result is not None
    assert 'report_path' in result
    assert os.path.exists(result['report_path'])
```

### 3. Model Performance Testing

```python
# tests/test_model_performance.py
import time
import psutil

def test_embedding_performance():
    """Test embedding generation performance"""
    generator = EmbeddingGenerator()
    test_texts = ["Test document " + str(i) for i in range(100)]
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    embeddings = generator.generate_embeddings(test_texts)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    # Performance assertions
    assert (end_time - start_time) < 60  # Should complete within 60 seconds
    assert len(embeddings) == 100
    assert (end_memory - start_memory) < 100 * 1024 * 1024  # Less than 100MB increase
```

### 4. Configuration Validation

```python
# tests/test_configuration.py
import yaml
from src.utils.config_loader import ConfigLoader

def test_configuration_validation():
    """Test that configuration files are valid"""
    config_loader = ConfigLoader('config/config.yaml')
    config = config_loader.config
    
    # Validate required fields
    assert 'ollama' in config
    assert 'embedding_model' in config['ollama']
    assert 'evaluation_model' in config['ollama']
    
    # Validate model availability
    # Add custom validation logic here
```

---

## Best Practices for Developers

### 1. Code Organization
- Follow the existing modular structure
- Maintain separation of concerns
- Use dependency injection for flexibility
- Implement proper error handling and logging

### 2. Documentation
- Update docstrings when modifying functions
- Maintain README files for new modules
- Document configuration changes
- Provide examples for new features

### 3. Version Control
- Create feature branches for major changes
- Write descriptive commit messages
- Update version numbers appropriately
- Tag releases consistently

### 4. Backward Compatibility
- Maintain compatibility with existing configurations
- Provide migration paths for breaking changes
- Deprecate features gradually
- Test with existing data sets

### 5. Performance Considerations
- Monitor memory usage during development
- Profile performance bottlenecks
- Implement caching where appropriate
- Consider batch processing for large datasets

---

## Troubleshooting Common Issues

### 1. Model Loading Failures
```python
# Add robust model loading
def load_model_safely(model_name):
    try:
        # Attempt to load model
        model = load_model(model_name)
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        # Try fallback model
        return load_fallback_model()
```

### 2. Memory Issues
```python
# Implement memory monitoring
import psutil
import gc

def monitor_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    if memory_mb > 2000:  # 2GB threshold
        gc.collect()  # Force garbage collection
        print(f"Memory usage: {memory_mb:.2f} MB")
```

### 3. Network Timeouts
```python
# Implement retry logic
import time
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = backoff_factor ** attempt
                    print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
        return wrapper
    return decorator
```

This guide provides comprehensive instructions for developers to customize and extend the AI-Powered Tender Evaluation System according to their specific requirements and environments.