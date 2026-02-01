#!/usr/bin/env python3
"""
Wrapper script to run the professional Streamlit app from the project root directory
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Add the interfaces directory to the Python path
interfaces_dir = Path(__file__).parent / "src" / "interfaces"
sys.path.insert(0, str(interfaces_dir))

# Import and run the professional Streamlit app
from professional_streamlit_app import run_professional_app

if __name__ == "__main__":
    run_professional_app()