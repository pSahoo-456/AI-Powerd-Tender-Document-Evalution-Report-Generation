"""
File utility functions for the tender evaluation system
"""

import os
import glob
from pathlib import Path
from typing import List, Union


def get_files_in_directory(directory: Union[str, Path], extension: str = None) -> List[Path]:
    """
    Get all files in a directory with optional extension filter
    
    Args:
        directory: Directory path to search
        extension: File extension to filter (e.g., '.pdf', '.txt')
        
    Returns:
        List of Path objects for matching files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    if extension:
        pattern = directory / f"*.{extension.lstrip('.')}"
        return list(Path.glob(directory, f"*.{extension.lstrip('.')}"))
    else:
        return [f for f in directory.iterdir() if f.is_file()]


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the file extension from a file path
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (e.g., '.pdf', '.txt')
    """
    return Path(file_path).suffix.lower()