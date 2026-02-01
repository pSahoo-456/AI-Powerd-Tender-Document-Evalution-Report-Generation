#!/usr/bin/env python3
"""
Test script to verify the tender evaluation system fixes
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.ingestion.document_loader import DocumentLoader
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.evaluation.llm_evaluator import LLMEvaluator
from src.reporting.report_generator import ReportGenerator
from langchain_core.documents import Document


def test_system():
    """Test the system with sample documents"""
    print("Testing Tender Evaluation System...")
    
    # Load configuration
    config_loader = ConfigLoader("./config/config.yaml")
    config = config_loader.config
    
    # Initialize components
    print("Initializing components...")
    doc_loader = DocumentLoader(config.get('ocr', {}))
    embedding_generator = EmbeddingGenerator(config.get('ollama', {}))
    llm_evaluator = LLMEvaluator(config.get('ollama', {}))
    report_generator = ReportGenerator()
    
    # Load sample documents
    print("Loading sample documents...")
    
    # Load requirements
    with open("./data/tender_requirement_IT_project.txt", "r", encoding="utf-8") as f:
        requirements_content = f.read()
    
    requirements_doc = Document(
        page_content=requirements_content,
        metadata={"source": "tender_requirement_IT_project.txt"}
    )
    
    # Load proposals
    with open("./data/proposal_cloudtech_solutions.txt", "r", encoding="utf-8") as f:
        proposal1_content = f.read()
        
    with open("./data/proposal_digitalpro_enterprises.txt", "r", encoding="utf-8") as f:
        proposal2_content = f.read()
    
    proposal1_doc = Document(
        page_content=proposal1_content,
        metadata={"source": "proposal_cloudtech_solutions.txt"}
    )
    
    proposal2_doc = Document(
        page_content=proposal2_content,
        metadata={"source": "proposal_digitalpro_enterprises.txt"}
    )
    
    requirements = [requirements_doc]
    applicants = [proposal1_doc, proposal2_doc]
    
    # Generate embeddings
    print("Generating embeddings...")
    requirements = embedding_generator.generate_document_embeddings(requirements)
    applicants = embedding_generator.generate_document_embeddings(applicants)
    
    # Evaluate with LLM
    print("Evaluating with LLM...")
    evaluated_applicants = llm_evaluator.evaluate_applicants(
        requirements, 
        applicants,
        max_applicants=10
    )
    
    # Check if we have actual evaluations
    print(f"Number of evaluated applicants: {len(evaluated_applicants)}")
    
    for i, (applicant, evaluation) in enumerate(evaluated_applicants):
        print(f"\nApplicant {i+1}: {applicant.metadata.get('source', 'Unknown')}")
        print(f"  Score: {evaluation.get('score', 0)}")
        print(f"  Technical Match: {evaluation.get('technical_match', 0)}")
        print(f"  Financial Match: {evaluation.get('financial_match', 0)}")
        print(f"  Timeline Match: {evaluation.get('timeline_match', 0)}")
        print(f"  Explanation: {evaluation.get('explanation', 'N/A')[:100]}...")
    
    # Generate report
    print("\nGenerating report...")
    report_path = report_generator.generate_evaluation_report(
        requirements,
        evaluated_applicants,
        "Test Evaluation Report"
    )
    
    print(f"Report generated: {report_path}")
    
    # Check if it's a PDF or LaTeX file
    if report_path.endswith('.pdf'):
        print("✓ PDF report generated successfully!")
    elif report_path.endswith('.tex'):
        print("✓ LaTeX report generated successfully!")
        print("  (PDF compilation may require LaTeX installation)")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_system()





Search functionality (src/search/) - Finds relevant content using embeddings
Rule filtering (src/filtering/) - Applies business rules
LLM evaluation (src/evaluation/) - Performs AI-based assessment
Report generation (src/reporting/) - Creates final reports
Interfaces (src/interfaces/) - Web and CLI interfaces
Main application files (main.py, run_professional_app.py)
Configuration files (config/)
Documentation files (README, LICENSE, etc.)
Template files (templates/)
Data files (if any sample data is included)