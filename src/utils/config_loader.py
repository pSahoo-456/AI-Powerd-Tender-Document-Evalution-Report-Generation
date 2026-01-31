"""
Configuration loader for the tender evaluation system
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration settings"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        # Convert relative path to absolute path based on project root
        if not os.path.isabs(config_path):
            # Get the project root (2 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / config_path
        else:
            self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[Any, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'ollama.base_url')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration"""
        return self.config.get('ollama', {})
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration"""
        return self.config.get('vector_db', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.config.get('processing', {})