
"""
Main entry point for the AI-Powered Tender Evaluation System
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Tender Evaluation System")
    parser.add_argument("--mode", choices=["cli", "web", "demo"], default="cli", 
                        help="Run mode: cli, web, or demo interface")
    parser.add_argument("--config", default="./config/config.yaml",
                        help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.mode == "web": 
        from src.interfaces.professional_streamlit_app import run_professional_app
        run_professional_app(args.config)
    elif args.mode == "demo":
        # Import and run demo interface
        from streamlit_demo import run_demo_app
        run_demo_app()
    else:
        # Import and run CLI interface
        from src.interfaces.cli_app import run_cli_app
        run_cli_app(args.config)

if __name__ == "__main__":
    main()