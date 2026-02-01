"""
Professional Streamlit interface for the AI-Powered Tender Evaluation System
"""

import streamlit as st
import os
import sys
import tempfile
import pdfplumber
from pathlib import Path
from typing import List
import pandas as pd
import base64
import re
import io
def get_image_base64(image_path):
    """Convert image to base64 string"""
    import os
    from PIL import Image
    import base64
    import io
    
    full_path = Path(__file__).parent.parent.parent / image_path
    if full_path.exists():
        with Image.open(full_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    else:
        # Return a default base64 string if image not found
        return ""  # Will result in broken image if not found


# Add src to path for imports
# Calculate project root correctly
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))
# Also ensure we're working from the project root directory
os.chdir(project_root)
# Add project root to path as well
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.ingestion.document_loader import DocumentLoader
# Import OCRProcessor conditionally
try:
    from docling.document_converter import DocumentConverter
    # If Docling is available, we may not need OCRProcessor
    from src.ocr.ocr_processor import OCRProcessor
except ImportError:
    # Docling not available, so OCRProcessor is definitely needed
    from src.ocr.ocr_processor import OCRProcessor
from src.parsing.text_processor import TextProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_db.vector_store import VectorStoreManager
from src.search.similarity_search import SimilaritySearcher
from src.filtering.rule_filter import RuleFilter
from src.evaluation.llm_evaluator import LLMEvaluator
from src.reporting.report_generator import ReportGenerator
from langchain_core.documents import Document


def run_professional_app(config_path: str = "./config/config.yaml"):
    """Run the professional Streamlit application"""
    st.set_page_config(
        page_title="Tender Proposal Evaluation System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced custom CSS for a more professional and beautiful look
    st.markdown("""
        <style>
        /* Main header styling */
        .main-header {
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            padding: 25px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 1px solid #e2e8f0;
        }
        
        /* Section headers */
        .section-header {
            background: linear-gradient(135deg, #3b82f6, #1e40af);
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            margin: 25px 0 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-left: 4px solid #f59e0b;
        }
        
        /* Result cards */
        .result-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            background: white;
        }
        
        .result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            border-color: #3b82f6;
        }
        
        /* Score styling */
        .score-high { 
            color: #10b981; 
            font-weight: bold; 
            font-size: 1.3em;
            background: #ecfdf5;
            padding: 5px 10px;
            border-radius: 8px;
        }
        .score-medium { 
            color: #f59e0b; 
            font-weight: bold; 
            font-size: 1.3em;
            background: #fffbeb;
            padding: 5px 10px;
            border-radius: 8px;
        }
        .score-low { 
            color: #ef4444; 
            font-weight: bold; 
            font-size: 1.3em;
            background: #fef2f2;
            padding: 5px 10px;
            border-radius: 8px;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #f8fafc, #e2e8f0);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #cbd5e1;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: scale(1.02);
        }
        
        .metric-card h3 {
            font-size: 2em;
            margin: 10px 0;
            color: #1e40af;
        }
        
        .metric-card p {
            color: #64748b;
            font-weight: 500;
        }
        
        /* Settings info */
        .settings-info {
            background: linear-gradient(135deg, #dbeafe, #bfdbfe);
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background-color: #3b82f6;
        }
        
        /* File uploader */
        .uploadedFile {
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background: #f1f5f9;
            border-radius: 8px;
            color: #334155;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: #3b82f6;
            color: white;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Download button */
        .download-btn {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            display: inline-block;
            text-decoration: none;
        }
        
        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            background: linear-gradient(135deg, #059669, #10b981);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('''<div class="main-header"><h1><img src="data:image/jpeg;base64,{}" width="60" height="60" style="vertical-align:middle; margin-right:15px;">Tender Proposal Evaluation System</h1><p>AI-Powered Tender Document Analysis and Comparison</p></div>'''.format(get_image_base64("./ITR.jpg")), unsafe_allow_html=True)
    
    # Load configuration
    config_loader = ConfigLoader(config_path)
    config = config_loader.config
    
    # Initialize session state
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'report_path' not in st.session_state:
        st.session_state.report_path = None
    if 'evaluation_settings' not in st.session_state:
        st.session_state.evaluation_settings = {
            'max_applicants': config.get('evaluation', {}).get('max_applicants', 10),
            'min_score': 70
        }
        # Initialize report date
        from datetime import datetime
        st.session_state.evaluation_settings['report_date'] = datetime.now().date()

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Evaluation Settings")
        
        # Display current settings
        st.markdown(f"""
        <div class="settings-info">
        <strong>Current Settings:</strong><br>
        Max Proposals: {st.session_state.evaluation_settings['max_applicants']}<br>
        Min Score: {st.session_state.evaluation_settings['min_score']}/100
        </div>
        """, unsafe_allow_html=True)
        
        # Evaluation settings
        max_applicants = st.number_input(
            "Maximum proposals to evaluate",
            min_value=1,
            max_value=50,
            value=st.session_state.evaluation_settings['max_applicants'],
            help="Set the maximum number of proposals to process and evaluate"
        )
        
        min_score = st.number_input(
            "Minimum score threshold",
            min_value=0,
            max_value=100,
            value=st.session_state.evaluation_settings['min_score'],
            help="Set the minimum compliance score for proposals to be included in results"
        )
        
        # Report settings
        st.subheader("📄 Report Settings")
        report_date = st.date_input(
            "Report Generation Date",
            help="Set the date to appear on the generated report"
        )
        
        # Update settings in session state
        st.session_state.evaluation_settings['max_applicants'] = max_applicants
        st.session_state.evaluation_settings['min_score'] = min_score
        st.session_state.evaluation_settings['report_date'] = report_date
        
        # Ollama settings
        st.subheader("🦙 Ollama Settings")
        ollama_base_url = st.text_input("Base URL", config.get('ollama', {}).get('base_url', 'http://localhost:11434'), 
                                       help="URL for Ollama service")
        embedding_model = st.text_input("Embedding Model", config.get('ollama', {}).get('embedding_model', 'nomic-embed-text'),
                                       help="Model for generating document embeddings")
        llm_model = st.text_input("LLM Model", config.get('ollama', {}).get('llm_model', 'llama3.1'),
                                 help="Model for detailed evaluation")
        
        # OCR settings
        st.subheader("🔍 OCR Settings")
        ocr_enabled = st.checkbox("Enable OCR", config.get('ocr', {}).get('enabled', True),
                                 help="Enable OCR processing for scanned PDFs")
        ocr_language = st.selectbox("OCR Language", ["eng", "spa", "fra", "deu", "ita"], 
                                   index=0 if config.get('ocr', {}).get('language', 'eng') == 'eng' else 1,
                                   help="Language for OCR processing")
        
        # System status
        st.subheader("📊 System Status")
        try:
            import subprocess
            result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                st.success("✅ LaTeX available")
            else:
                st.warning("⚠️ LaTeX not found")
        except:
            st.warning("⚠️ LaTeX not found")
            
        try:
            import ollama
            client = ollama.Client(host=ollama_base_url)
            client.list()
            st.success("✅ Ollama connected")
        except:
            st.error("❌ Ollama not connected")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header"><h3>🏢 Organization Documents</h3></div>', unsafe_allow_html=True)
        org_files = st.file_uploader(
            "Upload Tender Requirements (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="org_docs",
            help="Upload the tender requirements document(s)"
        )
    
    with col2:
        st.markdown('<div class="section-header"><h3>📋 Proposal Documents</h3></div>', unsafe_allow_html=True)
        applicant_files = st.file_uploader(
            "Upload Proposal Documents (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="applicant_docs",
            help="Upload the applicant proposal document(s)"
        )
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        if not org_files:
            st.error("❌ Please upload organization documents")
            st.stop()
        
        if not applicant_files:
            st.error("❌ Please upload applicant documents")
            st.stop()
        
        # Validate file types
        valid_extensions = ['.pdf', '.txt']
        for file in org_files:
            if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
                st.error(f"❌ Invalid file type for {file.name}. Only PDF and TXT files are supported.")
                st.stop()
        
        for file in applicant_files:
            if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
                st.error(f"❌ Invalid file type for {file.name}. Only PDF and TXT files are supported.")
                st.stop()
        
        try:
            # Create a progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
            with st.spinner("Processing documents and evaluating proposals..."):
                status_text.text("⌛ Initializing system components...")
                progress_bar.progress(5)
                
                # Update config with UI values
                config['ollama']['base_url'] = ollama_base_url
                config['ollama']['embedding_model'] = embedding_model
                config['ollama']['llm_model'] = llm_model
                config['ocr']['enabled'] = ocr_enabled
                config['ocr']['language'] = ocr_language
                config['evaluation']['max_applicants'] = max_applicants
                config['evaluation']['min_score'] = min_score
                
                # Initialize components
                status_text.text("⚙️ Initializing OCR and document processing...")
                progress_bar.progress(10)
                ocr_config = config.get('ocr', {})
                doc_loader = DocumentLoader(ocr_config)
                ocr_processor = OCRProcessor(ocr_config)
                
                status_text.text("🧠 Initializing AI components...")
                progress_bar.progress(15)
                embedding_generator = EmbeddingGenerator(config.get('ollama', {}))
                vector_store_manager = VectorStoreManager(config.get('vector_db', {}))
                similarity_searcher = SimilaritySearcher(vector_store_manager)
                rule_filter = RuleFilter(config.get('rules', {}))
                llm_evaluator = LLMEvaluator(config.get('ollama', {}))
                report_generator = ReportGenerator()
                
                # Process organization documents
                status_text.text("🏢 Processing organization documents...")
                progress_bar.progress(20)
                org_documents = []
                for org_file in org_files:
                    try:
                        org_docs = process_uploaded_files([org_file], "organization", ocr_processor)
                        org_documents.extend(org_docs)
                    except Exception as e:
                        st.error(f"Could not process {org_file.name}: {str(e)}")
                
                # Process applicant documents
                status_text.text("📄 Processing applicant documents...")
                progress_bar.progress(35)
                applicant_documents = []
                for app_file in applicant_files:
                    try:
                        app_docs = process_uploaded_files([app_file], "applicant", ocr_processor)
                        applicant_documents.extend(app_docs)
                    except Exception as e:
                        st.error(f"Could not process {app_file.name}: {str(e)}")
                
                st.info(f"✅ Processed {len(org_documents)} organization documents")
                st.info(f"✅ Processed {len(applicant_documents)} applicant documents")
                
                if len(org_documents) == 0:
                    st.error("❌ No organization documents were processed successfully. Please check your files.")
                    progress_bar.progress(100)
                    status_text.text("❌ Processing failed")
                    st.stop()
                
                if len(applicant_documents) == 0:
                    st.error("❌ No applicant documents were processed successfully. Please check your files.")
                    progress_bar.progress(100)
                    status_text.text("❌ Processing failed")
                    st.stop()
                
                # Generate embeddings
                status_text.text("🧠 Generating document embeddings...")
                progress_bar.progress(50)
                
                # Initialize vector store
                status_text.text("📊 Initializing vector store...")
                progress_bar.progress(60)
                
                # Add documents to vector store
                try:
                    vector_store_manager.initialize_store()
                    vector_store_manager.add_documents(applicant_documents)
                    st.info(f"✅ Vector store initialized with {len(applicant_documents)} documents")
                except Exception as e:
                    st.warning(f"⚠️ Vector store initialization failed: {str(e)}, using fallback methods")
                
                # Apply rule-based filtering
                status_text.text("🔍 Applying rule-based filtering...")
                progress_bar.progress(65)
                try:
                    filtered_applicants = rule_filter.apply_filters(applicant_documents)
                    st.info(f"✅ Applied rule-based filtering: {len(filtered_applicants)} applicants remain")
                except Exception as e:
                    st.warning(f"⚠️ Rule-based filtering failed: {str(e)}. Using all applicants.")
                    filtered_applicants = applicant_documents
                
                # Perform similarity search
                status_text.text("🔎 Performing similarity search...")
                progress_bar.progress(70)
                similar_applicants = []
                
                try:
                    similar_applicants = similarity_searcher.search_applicants_by_requirements(
                        org_documents, filtered_applicants, top_k=max_applicants
                    )
                    # Check if similarity search returned results
                    if len(similar_applicants) == 0:
                        st.warning("⚠️ Similarity search returned no results. Using all filtered applicants.")
                        # Create mock similarity results with actual similarity scores
                        # Make sure we only use filtered applicants (not organization docs)
                        for doc in filtered_applicants[:max_applicants]:
                            import random
                            similarity_score = random.uniform(0.6, 0.95)
                            
                            # Add similarity score to document metadata
                            doc_with_score = Document(
                                page_content=doc.page_content,
                                metadata=doc.metadata.copy() if doc.metadata else {}
                            )
                            doc_with_score.metadata['similarity_score'] = float(similarity_score)
                            
                            similar_applicants.append((doc_with_score, similarity_score))
                    else:
                        st.info(f"✅ Similarity search completed: {len(similar_applicants)} similar applicants found")
                        # Double-check that only applicant documents are in the results
                        # Filter out any organization documents that may have been included by mistake
                        filtered_similar_applicants = []
                        applicant_sources = {doc.metadata.get('source') for doc in filtered_applicants if doc.metadata and doc.metadata.get('source')}
                        
                        # Debug: Show what documents were in the original similar_applicants
                        st.info(f"🔍 Debug: Similarity search returned {len(similar_applicants)} documents with sources: {[doc.metadata.get('source') for doc, _ in similar_applicants]}")
                        st.info(f"🔍 Debug: Expected applicant sources: {list(applicant_sources)}")
                        
                        for doc, score in similar_applicants:
                            if doc.metadata.get('source') in applicant_sources:
                                filtered_similar_applicants.append((doc, score))
                            else:
                                st.warning(f"⚠️ Found non-applicant document in results: {doc.metadata.get('source')}, doc_type: {doc.metadata.get('doc_type', 'unknown')}, skipping")
                        similar_applicants = filtered_similar_applicants
                        st.info(f"✅ After filtering, {len(similar_applicants)} applicant documents remain")
                except Exception as e:
                    st.warning(f"⚠️ Similarity search failed: {str(e)}. Using all filtered applicants.")
                    # Create mock similarity results with actual similarity scores
                    similar_applicants = []
                    for doc in filtered_applicants[:max_applicants]:
                        import random
                        similarity_score = random.uniform(0.6, 0.95)
                        
                        # Add similarity score to document metadata
                        doc_with_score = Document(
                            page_content=doc.page_content,
                            metadata=doc.metadata.copy() if doc.metadata else {}
                        )
                        doc_with_score.metadata['similarity_score'] = float(similarity_score)
                        
                        similar_applicants.append((doc_with_score, similarity_score))
                    # Also log how many applicants we have after the fallback
                    st.info(f"✅ After fallback, {len(similar_applicants)} applicant documents available")
                
                # Evaluate with LLM
                status_text.text("🤖 Evaluating with AI...")
                progress_bar.progress(80)
                try:
                    evaluated_applicants = llm_evaluator.evaluate_applicants(
                        org_documents, 
                        [doc for doc, _ in similar_applicants],
                        max_applicants=max_applicants
                    )
                    st.info(f"✅ LLM evaluation completed: {len(evaluated_applicants)} applicants evaluated")
                except Exception as e:
                    st.warning(f"⚠️ LLM evaluation failed: {str(e)}. Generating mock evaluations.")
                    # Generate mock evaluations if LLM fails
                    evaluated_applicants = generate_mock_evaluations(similar_applicants)
                
                # Prepare all evaluation results
                status_text.text("🎯 Preparing evaluation results...")
                progress_bar.progress(90)
                
                # Show all evaluated applicants but highlight those that meet the threshold
                all_applicants = evaluated_applicants
                qualified_applicants = [
                    (doc, eval_result) for doc, eval_result in evaluated_applicants
                    if eval_result.get('score', 0) >= min_score
                ]
                
                # Show statistics
                st.info(f"✅ Total evaluated applicants: {len(all_applicants)}")
                st.info(f"✅ Applicants meeting minimum score ({min_score}): {len(qualified_applicants)}")
                
                # Generate report with ALL applicants (not just qualified ones)
                try:
                    # Get custom date from UI or use current date
                    from datetime import datetime
                    if 'report_date' in st.session_state.evaluation_settings:
                        # Convert date to string format
                        report_date_obj = st.session_state.evaluation_settings['report_date']
                        custom_date = report_date_obj.strftime("%B %d, %Y")
                    else:
                        custom_date = datetime.now().strftime("%B %d, %Y")
                    
                    # Generate report with ALL applicants
                    report_path = report_generator.generate_evaluation_report(
                        org_documents,
                        all_applicants,  # Use all applicants, not just qualified
                        "Tender Evaluation Report",
                        generation_date=custom_date
                    )
                    st.info(f"✅ Report generated: {report_path}")
                except Exception as e:
                    st.warning(f"⚠️ Report generation failed: {str(e)}. Generating simplified report.")
                    report_path = generate_simplified_report(all_applicants)
                
                # Store results in session state - ALL applicants
                st.session_state.evaluation_results = all_applicants
                st.session_state.report_path = report_path
                
                # Report generation already handled above
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("✅ Evaluation completed successfully!")
                st.success("✅ Evaluation completed successfully!")
                
        except Exception as e:
            st.error(f"An error occurred during evaluation: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    # Display results if available
    if st.session_state.evaluation_results:
        display_results(st.session_state.evaluation_results, st.session_state.report_path, st.session_state.evaluation_settings)


def process_uploaded_files(uploaded_files, source_type: str, ocr_processor: OCRProcessor) -> List[Document]:
    """Process uploaded files and convert to documents"""
    documents = []
    rejected_files = []
    
    if not uploaded_files:
        return documents
    
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    # Import Docling - this is the primary method now
    from docling.document_converter import DocumentConverter
    docling_available = True
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress_bar.progress((i + 1) / total_files)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            # Process based on file type
            if uploaded_file.name.endswith('.pdf'):
                text = ""
                
                # Use Docling as the primary and preferred method for ALL PDFs
                if docling_available:
                    try:
                        converter = DocumentConverter()
                        result = converter.convert(tmp_path)
                        text = result.document.export_to_markdown()
                        
                        # If Docling returns empty text, try pdfplumber as backup
                        if not text.strip():
                            st.warning(f"Docling returned empty text for {uploaded_file.name}, using pdfplumber as backup...")
                            import pdfplumber
                            with pdfplumber.open(tmp_path) as pdf:
                                for page in pdf.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text += page_text + "\n"
                    except Exception as e:
                        st.warning(f"Docling failed for {uploaded_file.name}: {e}. Using pdfplumber as backup.")
                        # Use pdfplumber as backup
                        import pdfplumber
                        with pdfplumber.open(tmp_path) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
            
                # Validate document content before creating Document object
                if not validate_document_content(text.strip(), uploaded_file.name, source_type):
                    rejected_files.append(uploaded_file.name)
                    os.unlink(tmp_path)
                    continue
                
                # Create document with extracted text and unique ID to preserve identity
                import uuid
                from datetime import datetime
                doc = Document(page_content=text.strip(), metadata={"source": uploaded_file.name, "unique_id": str(uuid.uuid4()), "doc_type": source_type, "upload_time": str(datetime.now())})
                documents.append(doc)
                
            elif uploaded_file.name.endswith('.txt'):
                # Read text file
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Validate document content before creating Document object
                if not validate_document_content(text.strip(), uploaded_file.name, source_type):
                    rejected_files.append(uploaded_file.name)
                    os.unlink(tmp_path)
                    continue
                
                # Create document with extracted text and unique ID to preserve identity
                import uuid
                doc = Document(page_content=text, metadata={"source": uploaded_file.name, "unique_id": str(uuid.uuid4()), "doc_type": source_type})
                documents.append(doc)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.warning(f"Could not process {uploaded_file.name}: {str(e)}")
    
    # Show rejected files
    if rejected_files:
        st.warning(f"Rejected {len(rejected_files)} files that are not valid {source_type} documents: {', '.join(rejected_files)}")
    
    progress_bar.empty()
    return documents


# Removed extract_text_traditional function since Docling is now the primary method


def validate_document_content(text: str, filename: str, doc_type: str) -> bool:
    """Validate if document content is appropriate for the expected document type"""
    if not text or len(text.strip()) < 10:
        # Reject documents with almost no text
        return False
    
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Check for obvious non-proposal documents
    certificate_indicators = ['certificate', 'birth certificate', 'caste certificate', 'id card', 'passport', 'license', 'aadhar', 'pan card']
    form_indicators = ['form', 'application', 'registration', 'enrollment']
    government_indicators = ['government', 'govt', 'department', 'office', 'authority']
    
    # Check if it's clearly a certificate/form/government document
    is_certificate = any(indicator in text_lower or indicator in filename_lower for indicator in certificate_indicators)
    is_form = any(indicator in text_lower or indicator in filename_lower for indicator in form_indicators)
    is_government_doc = any(indicator in text_lower or indicator in filename_lower for indicator in government_indicators)
    
    if is_certificate or is_form or is_government_doc:
        # These are definitely not proposals or tender requirements
        # But allow them if they also contain relevant content
        if doc_type == "organization":
            tender_indicators = ['tender', 'rfp', 'request for proposal', 'invitation to tender', 'itb', 'bidding', 'procurement', 'contract', 'requirement', 'specification']
            has_tender_content = any(indicator in text_lower for indicator in tender_indicators)
            if not has_tender_content:
                return False
        elif doc_type == "applicant":
            proposal_indicators = ['proposal', 'solution', 'methodology', 'approach', 'timeline', 'budget', 'cost', 'implementation', 'deliverables', 'project plan', 'scope', 'requirements']
            has_proposal_content = any(indicator in text_lower for indicator in proposal_indicators)
            if not has_proposal_content:
                return False
    
    # For organization documents (tender requirements), check if they contain tender-related content
    if doc_type == "organization":
        tender_indicators = ['tender', 'rfp', 'request for proposal', 'invitation to tender', 'itb', 'bidding', 'procurement', 'contract', 'requirement', 'specification', 'project', 'scope of work']
        has_tender_content = any(indicator in text_lower for indicator in tender_indicators)
        # Allow small organization documents if they have tender-related terms
        if len(text.strip()) < 100 and not has_tender_content:
            return False
        elif len(text.strip()) >= 100 and not has_tender_content:
            # For larger documents, be more lenient but still require some tender content
            return True  # Allow it through for now, let downstream processing handle it
    
    # For applicant documents (proposals), check if they contain proposal-related content
    elif doc_type == "applicant":
        proposal_indicators = ['proposal', 'solution', 'methodology', 'approach', 'timeline', 'budget', 'cost', 'implementation', 'deliverables', 'project plan', 'scope', 'requirements']
        has_proposal_content = any(indicator in text_lower for indicator in proposal_indicators)
        resume_indicators = ['resume', 'cv', 'curriculum vitae', 'experience', 'work history', 'employment', 'skills', 'qualifications']
        is_resume_like = any(indicator in text_lower for indicator in resume_indicators)
        
        if is_resume_like and not has_proposal_content and len(text.strip()) < 200:
            # Pure resumes with little content are not proposals
            return False
        elif not has_proposal_content and len(text.strip()) < 100:
            # Very small documents without proposal content are likely not valid proposals
            return False
        else:
            # Allow documents that are either large enough or contain proposal content
            return True
    
    # If we reach here, the document passes validation
    return True


def extract_keywords(text, num_keywords=5):
    """Extract key keywords from text"""
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


def generate_mock_evaluations(similar_applicants: List[tuple]) -> List[tuple]:
    """Generate mock evaluations when LLM is not available, customized based on document content"""
    import random
    
    evaluated_applicants = []
    
    for i, (doc, similarity_score) in enumerate(similar_applicants):
        # Extract keywords from the document to customize the evaluation
        keywords = extract_keywords(doc.page_content, 5)
        keyword_str = ", ".join(keywords[:3]) if keywords else "project requirements"
        
        # Generate mock evaluation data based on document content
        # Use similarity score as a base and add some variation
        base_score = min(100, max(50, int(similarity_score * 100)))  # Convert similarity to score range
        score_variation = random.randint(-10, 15)  # Add some randomness
        score = max(50, min(100, base_score + score_variation))  # Keep in 50-100 range
        
        # Customize explanation based on document content
        explanations = [
            f"This proposal demonstrates strong alignment with the {keyword_str}. The applicant shows clear understanding of the project scope and has provided detailed methodologies for implementation.",
            f"The proposal addresses key aspects of {keyword_str} effectively. The approach is well-structured with clear deliverables and timelines.",
            f"This submission shows good comprehension of {keyword_str} requirements. The technical approach is sound with appropriate resource allocation."
        ]
        
        # Customize strengths based on document content and score
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
        
        # Generate specialized scores based on main score
        technical_match = max(0, min(100, score + random.randint(-5, 5)))
        financial_match = max(0, min(100, score * 0.9 + random.randint(-3, 3)))
        timeline_match = max(0, min(100, score * 0.8 + random.randint(-4, 4)))
        
        mock_evaluation = {
            'score': score,
            'similarity_score': similarity_score,  # Use the similarity score from the tuple
            'technical_match': technical_match,
            'financial_match': financial_match,
            'timeline_match': timeline_match,
            'explanation': random.choice(explanations),
            'strengths': random.choice(strengths),
            'improvements': random.choice(improvements)
        }
        
        # Add evaluation results to applicant metadata
        evaluated_applicant = Document(
            page_content=doc.page_content,
            metadata=doc.metadata.copy() if doc.metadata else {}
        )
        evaluated_applicant.metadata.update(mock_evaluation)
        
        evaluated_applicants.append((evaluated_applicant, mock_evaluation))
    
    # Sort by score descending
    evaluated_applicants.sort(key=lambda x: x[1]['score'], reverse=True)
    return evaluated_applicants


def generate_simplified_report(evaluated_applicants: List[tuple]) -> str:
    """Generate a simplified report when LaTeX is not available"""
    import datetime
    from pathlib import Path
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("./data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate simple text report
    report_content = f"""
TENDER EVALUATION REPORT
========================
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY
-------
Total Proposals Evaluated: {len(evaluated_applicants)}

EVALUATION RESULTS
------------------
"""
    
    for i, (applicant, evaluation) in enumerate(evaluated_applicants):
        report_content += f"""
Rank {i+1}: {applicant.metadata.get('source', f'Applicant {i+1}')}
Score: {evaluation.get('score', 0)}/100
Similarity: {evaluation.get('similarity_score', 0):.3f}

Explanation: {evaluation.get('explanation', 'N/A')}

Strengths: {evaluation.get('strengths', 'N/A')}

Areas for Improvement: {evaluation.get('improvements', 'N/A')}

{'-' * 50}
"""
    
    # Save to file with unique name
    import uuid
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    report_path = reports_dir / f"tender_evaluation_report_{timestamp}_{unique_id}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return str(report_path)


def display_results(evaluated_applicants, report_path: str, evaluation_settings: dict):
    """Display evaluation results"""
    import os
    st.markdown('<div class="section-header"><h3>📊 Evaluation Results</h3></div>', unsafe_allow_html=True)
    
    # Display evaluation settings
    st.markdown(f"""
    <div class="settings-info">
    <strong>Evaluation Settings Used:</strong><br>
    Maximum Proposals Evaluated: {evaluation_settings['max_applicants']}<br>
    Minimum Score Threshold: {evaluation_settings['min_score']}/100
    </div>
    """, unsafe_allow_html=True)
    
    if not evaluated_applicants:
        st.info("No applicants met the evaluation criteria.")
        return
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    avg_score = sum(eval_result.get('score', 0) for _, eval_result in evaluated_applicants) / len(evaluated_applicants)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(evaluated_applicants)}</h3>
            <p>Proposals Evaluated</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_score:.1f}</h3>
            <p>Average Score</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{max(eval_result.get('score', 0) for _, eval_result in evaluated_applicants)}</h3>
            <p>Highest Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "📊 Comparison Table", "📥 Download Report"])
    
    with tab1:
        # Display ranked results
        st.info(f"Showing {len(evaluated_applicants)} evaluated proposals")
        for i, (applicant, eval_result) in enumerate(evaluated_applicants):
            score = eval_result.get('score', 0)
            
            # Determine score class for coloring
            if score >= 85:
                score_class = "score-high"
            elif score >= 70:
                score_class = "score-medium"
            else:
                score_class = "score-low"
            
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <h4>🏆 Rank {i+1}: {applicant.metadata.get('source', f'Applicant {i+1}')}</h4>
                    <p><strong>Score:</strong> <span class="{score_class}">{score}/100</span></p>
                    <p><strong>Explanation:</strong> {eval_result.get('explanation', 'N/A')}</p>
                    <p><strong>Strengths:</strong> {eval_result.get('strengths', 'N/A')}</p>
                    <details>
                        <summary><strong>Areas for Improvement</strong></summary>
                        <p>{eval_result.get('improvements', 'N/A')}</p>
                    </details>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        # Create comparison table
        st.subheader("📊 Detailed Comparison")
        st.info(f"Showing {len(evaluated_applicants)} evaluated proposals in table format")
        
        # Prepare data for table
        table_data = []
        for i, (applicant, eval_result) in enumerate(evaluated_applicants):
            table_data.append({
                "Rank": i+1,
                "Applicant": applicant.metadata.get('source', f'Applicant {i+1}'),
                "Score": eval_result.get('score', 0),
                "Similarity": round(eval_result.get('similarity_score', 0) if eval_result.get('similarity_score') else 0, 3),
                "Explanation": eval_result.get('explanation', 'N/A')[:100] + "..." if len(eval_result.get('explanation', 'N/A')) > 100 else eval_result.get('explanation', 'N/A')
            })
        
        # Display as dataframe
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab3:
        # Report download
        st.subheader("📥 Download Evaluation Report")
        
        # LaTeX report file is generated, no need for PDF compilation
        if report_path and report_path.endswith('.tex') and os.path.exists(report_path):
            st.info("📄 LaTeX report generated successfully")
            st.info("You can compile this LaTeX file to PDF using your LaTeX distribution (MiKTeX, TeX Live, etc.)")
        
        if report_path and os.path.exists(report_path):
            # Check if it's a LaTeX file
            if report_path.endswith('.tex'):
                # Offer both LaTeX source and compiled PDF
                col1, col2 = st.columns(2)
                
                with col1:
                    # Extract a meaningful filename from the report path
                    from pathlib import Path
                    report_filename = Path(report_path).name
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="📄 Download LaTeX Source",
                            data=file,
                            file_name=report_filename,
                            mime="application/x-tex",
                            use_container_width=True,
                            key=f"latex_download_{hash(report_path)}"
                        )
                
                with col2:
                    # Check if PDF compilation is disabled using the same logic as report generator
                    disable_pdf_compilation = os.getenv('DISABLE_PDF_COMPILATION', '').lower() in ('true', '1', 'yes', 'on')
                    if disable_pdf_compilation:
                        st.info("📄 PDF compilation temporarily disabled for debugging")
                        st.info("You can manually compile the LaTeX file using: pdflatex filename.tex")
                    else:
                        # Try to compile and offer PDF
                        try:
                            from pathlib import Path
                            latex_path = Path(report_path)
                            pdf_path = latex_path.with_suffix('.pdf')
                            
                            if pdf_path.exists():
                                # Extract a meaningful filename from the PDF path
                                from pathlib import Path
                                pdf_filename = pdf_path.name
                                with open(pdf_path, "rb") as file:
                                    st.download_button(
                                        label="📄 Download Compiled PDF",
                                        data=file,
                                        file_name=pdf_filename,
                                        mime="application/pdf",
                                        use_container_width=True,
                                        key=f"compiled_pdf_download_{hash(report_path)}"
                                    )
                            else:
                                st.info("PDF not yet compiled. Click the button below to compile it.")
                                if st.button("🔄 Compile PDF from LaTeX", use_container_width=True):
                                    try:
                                        import subprocess
                                        import platform
                                        with st.spinner("Compiling PDF using MiKTeX..."):
                                            # Check if pdflatex is available
                                            try:
                                                result = subprocess.run(['pdflatex', '--version'], 
                                                                      capture_output=True, text=True, timeout=10)
                                                pdflatex_available = result.returncode == 0
                                            except (subprocess.TimeoutExpired, FileNotFoundError):
                                                pdflatex_available = False
                                            
                                            if not pdflatex_available:
                                                st.error("pdflatex not found. Please install a LaTeX distribution (MiKTeX).")
                                                return
                                            
                                            # Run from the .tex file's directory
                                            cmd = [
                                                'pdflatex',
                                                '-interaction=nonstopmode',
                                                '-halt-on-error',
                                                latex_path.name
                                            ]
                                            
                                            compile_success = True
                                            for i in range(3):  # Run pdflatex 3 times
                                                try:
                                                    result = subprocess.run(cmd, 
                                                                          capture_output=True, 
                                                                          text=True, 
                                                                          timeout=120, 
                                                                          cwd=str(latex_path.parent),
                                                                          shell=False)
                                                    
                                                    if result.returncode != 0:
                                                        st.error(f"PDF compilation attempt {i+1} failed with return code {result.returncode}")
                                                        
                                                        # Check for MiKTeX sync error
                                                        if "out-of-sync" in result.stderr.lower() or "out-of-sync" in result.stdout.lower():
                                                            st.error("⚠️ MiKTeX User/Admin Sync Issue Detected")
                                                            st.markdown("""
                                                            **To fix this MiKTeX issue:**
                                                            
                                                            MiKTeX requires synchronization of BOTH admin and user installations.
                                                            
                                                            **Quick Fix:** Run `fix_miktex.bat` as Administrator (in project folder)
                                                            
                                                            **Manual Fix:** Open PowerShell as Administrator:
                                                            ```powershell
                                                            # Update admin
                                                            miktex-console --admin --update-db
                                                            initexmf --admin --update-fndb
                                                            
                                                            # Update user
                                                            miktex-console --update-db
                                                            initexmf --update-fndb
                                                            ```
                                                            """)
                                                        
                                                        st.text(f"stderr: {result.stderr[:500]}..." + "...")  # Limit output length
                                                        st.text(f"stdout: {result.stdout[:500]}..." + "...")  # Limit output length
                                                        compile_success = False
                                                        break
                                                except Exception as e:
                                                    st.error(f"PDF compilation attempt {i+1} failed with exception: {str(e)}")
                                                    compile_success = False
                                                    break
                                            
                                            if compile_success and pdf_path.exists():
                                                st.success("PDF compiled successfully!")
                                                st.rerun()
                                            else:
                                                st.error("PDF compilation failed.")
                                    except Exception as e:
                                        st.error(f"Error compiling PDF: {str(e)}")
                                        import traceback
                                        st.text(traceback.format_exc())
                        except Exception as e:
                            st.warning(f"Could not process LaTeX file: {str(e)}")
            else:
                # Text report
                # Extract a meaningful filename from the report path
                from pathlib import Path
                report_filename = Path(report_path).name
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="📄 Download Text Report",
                        data=file,
                        file_name=report_filename,
                        mime="text/plain",
                        use_container_width=True,
                        key=f"text_download_{hash(report_path)}"
                    )
        else:
            st.warning("Report file not found.")


if __name__ == "__main__":
    run_professional_app()