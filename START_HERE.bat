@echo off
REM AI-Powered Tender Evaluation System - Quick Start
REM Double-click this file to start the system

echo.
echo =====================================================
echo    AI-Powered Tender Evaluation System - Starting...
echo =====================================================
echo.

REM Change to the project directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM Check if Python is available
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org/
    echo.
    pause
    exit /b 1
)

REM Check if Ollama is running
echo.
echo Checking Ollama server status...
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo Ollama server not running. Starting Ollama in background...
    start cmd /c "title Ollama Server && ollama serve && pause"
    echo Waiting for Ollama to start...
    timeout /t 8 /nobreak >nul
) else (
    echo Ollama server is running.
)

REM Check if required models exist
echo.
echo Checking required models...
python -c "
import subprocess
import json
try:
    result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], 
                           capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        models = [model['name'] for model in data.get('models', [])]
        if 'tinydolphin:latest' in models and 'nomic-embed-text' in models:
            print('All required models are available.')
        else:
            print('Some models are missing. Installing...')
            if 'tinydolphin:latest' not in models:
                print('Installing tinydolphin:latest...')
                subprocess.run(['ollama', 'pull', 'tinydolphin:latest'])
            if 'nomic-embed-text' not in models:
                print('Installing nomic-embed-text...')
                subprocess.run(['ollama', 'pull', 'nomic-embed-text'])
            print('All required models are now available.')
    else:
        print('Could not connect to Ollama server')
except Exception as e:
    print(f'Error checking models: {e}')
"
echo.

REM Install Python dependencies
echo Installing/updating Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo Installing dependencies from scratch...
    pip install streamlit langchain-community langchain-ollama faiss-cpu PyYAML pillow python-docling ollama
)

echo.
echo Starting the AI-Powered Tender Evaluation System...
echo.
echo NOTE: The web interface will open in your browser at:
echo       http://localhost:8501
echo.
echo To access the interface manually, open your browser and go to:
echo       http://localhost:8501
echo.
echo To stop the application, return to this window and press Ctrl+C
echo =====================================================

REM Start the application
streamlit run src/interfaces/professional_streamlit_app.py --server.port=8501 --server.address=localhost

REM If streamlit command fails, try python -m streamlit
if errorlevel 1 (
    echo Streamlit command failed, trying alternate method...
    python -m streamlit run src/interfaces/professional_streamlit_app.py --server.port=8501 --server.address=localhost
)

echo.
echo Application stopped.
echo To restart, run this batch file again.
pause