from setuptools import setup, find_packages

setup(
    name="tender-proposal-evaluation-system",
    version="1.0.0",
    description="AI-powered system for evaluating tender proposals against organizational requirements",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Developers",
    author_email="ai.developers@example.com",
    url="https://github.com/pSahoo-465/tender-proposal-evaluation-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain==0.3.0",
        "langchain-community==0.3.0",
        "langchain-core==0.3.0",
        "pymupdf==1.23.21",
        "pdfplumber==0.10.4",
        "Pillow==10.0.1",
        "faiss-cpu==1.7.4",
        "chromadb==0.4.22",
        "ollama==0.1.7",
        "pydantic==2.7.4",
        "jinja2==3.1.2",
        "streamlit==1.28.0",
        "python-dotenv==1.0.0",
        "numpy==1.24.3",
        "pandas==2.1.1",
        "scikit-learn==1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.2",
        ],
        "ocr": [
            "pytesseract==0.3.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "tender-eval=src.interfaces.professional_streamlit_app:main",
        ],
    },
)