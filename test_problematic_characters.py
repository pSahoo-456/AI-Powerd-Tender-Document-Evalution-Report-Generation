#!/usr/bin/env python3
"""
Test script to verify LaTeX escaping fix with problematic characters
"""

import os
from pathlib import Path
from langchain_core.documents import Document
from src.reporting.report_generator import ReportGenerator

def test_problematic_characters():
    """
    Test the LaTeX escaping with problematic characters like #
    """
    print("Testing LaTeX escaping with problematic characters...")
    
    # Create sample requirements with problematic characters
    requirements = [
        Document(page_content="# DETAILED TENDER DOCUMENT\n## ODISHA STATE CO-OPERATIVE MARKETING FEDERATION LTD. (MARKFED -ODISHA)\n## BHUBANESWAR\nAt- Old Station Road,\nDist: - Khurda (Odisha), Pin-751006\nTel: 2310626,...", metadata={"source": "tender_requirement_1"}),
        Document(page_content="Another requirement with # symbols and % percentages", metadata={"source": "tender_requirement_2"})
    ]
    
    # Create sample applicants with evaluation results
    applicants = [
        (
            Document(
                page_content="Our company CloudTech Solutions offers comprehensive cloud infrastructure services.",
                metadata={"source": "proposal_cloudtech_solutions.pdf"}
            ),
            {
                'score': 92,
                'similarity_score': 0.87,
                'technical_match': 94,
                'financial_match': 88,
                'timeline_match': 95,
                'explanation': 'Strong technical approach.',
                'strengths': 'Excellent capabilities',
                'improvements': 'Minor improvements needed'
            }
        )
    ]
    
    # Initialize report generator
    report_gen = ReportGenerator()
    
    # Generate enhanced evaluation report
    print("\nGenerating report with problematic characters...")
    report_path = report_gen.generate_evaluation_report(
        requirements=requirements,
        evaluated_applicants=applicants,
        report_title="Problematic Characters Test Report"
    )
    
    print(f"\nReport generated successfully!")
    print(f"Report saved to: {report_path}")
    
    # Check if the file was created
    if os.path.exists(report_path):
        print(f"✓ Report file exists at: {report_path}")
        
        # Get file size
        file_size = os.path.getsize(report_path)
        print(f"✓ Report file size: {file_size} bytes")
        
        # Check for problematic characters in the generated LaTeX
        print(f"\nChecking for unescaped problematic characters:")
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for unescaped # characters (not preceded by \)
            import re
            unescaped_hashes = re.findall(r'[^\\]#', content)
            print(f"Unescaped # characters found: {len(unescaped_hashes)}")
            
            # Check for unescaped % characters (not preceded by \)
            unescaped_percents = re.findall(r'[^\\]%', content)
            print(f"Unescaped % characters found: {len(unescaped_percents)}")
            
            if len(unescaped_hashes) == 0 and len(unescaped_percents) == 0:
                print("✓ All problematic characters properly escaped!")
            else:
                print("✗ Found unescaped problematic characters:")
                if unescaped_hashes:
                    print(f"  Unescaped #: {unescaped_hashes[:5]}")  # Show first 5
                if unescaped_percents:
                    print(f"  Unescaped %: {unescaped_percents[:5]}")  # Show first 5
        
        # Show first few lines of the requirements section
        print(f"\nFirst 10 lines of requirements section:")
        lines = content.split('\n')
        req_start = -1
        for i, line in enumerate(lines):
            if 'Requirements Overview' in line or '\\section{Organization Requirements Overview}' in line:
                req_start = i
                break
        
        if req_start != -1:
            for i in range(req_start, min(req_start + 10, len(lines))):
                print(f"{i+1:3d}: {lines[i]}")
        else:
            print("Requirements section not found in expected format")
    else:
        print(f"✗ Report file was not created at: {report_path}")
    
    print("\nProblematic characters test completed!")

if __name__ == "__main__":
    test_problematic_characters()