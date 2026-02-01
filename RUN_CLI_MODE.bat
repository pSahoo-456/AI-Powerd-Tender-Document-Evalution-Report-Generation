@echo off
REM AI-Powered Tender Evaluation System - CLI Mode
REM Command Line Interface version

echo.
echo =====================================================
echo    AI-Powered Tender Evaluation System - CLI Mode
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
            print('Some models are missing. Please install them manually:')
            print('ollama pull tinydolphin:latest')
            print('ollama pull nomic-embed-text')
    else:
        print('Could not connect to Ollama server')
except Exception as e:
    print(f'Error checking models: {e}')
"
echo.

REM Install Python dependencies
echo Installing/updating Python dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting AI-Powered Tender Evaluation System in CLI Mode...
echo.
echo This mode allows you to run evaluations from the command line
echo without the web interface.
echo.
echo =====================================================

REM Run the main application in CLI mode
python main.py --mode cli

echo.
echo CLI mode execution completed.
pause