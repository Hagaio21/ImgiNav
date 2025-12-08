"""
Environment configuration for ImgiNav project.

Provides centralized path management using environment variables and .env files.
This eliminates hardcoded paths and improves portability across different systems.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class ProjectConfig:
    """
    Centralized configuration for project paths and environment settings.
    
    Loads configuration from:
    1. Environment variables (highest priority)
    2. .env file in project root (if exists)
    3. Default values based on project structure
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Load .env file if available
        if DOTENV_AVAILABLE:
            # Try to find .env file in project root (3 levels up from common/)
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
        
        # Get BASE_DIR from environment or default to project root
        base_dir_str = os.getenv("IMGINAV_ROOT")
        if base_dir_str:
            self._base_dir = Path(base_dir_str).resolve()
        else:
            # Default to project root (3 levels up from common/env_config.py)
            self._base_dir = Path(__file__).parent.parent.resolve()
        
        # Get cache directory from environment or default
        cache_dir_str = os.getenv("IMGINAV_CACHE_DIR")
        if cache_dir_str:
            self._cache_dir = Path(cache_dir_str).resolve()
        else:
            # Default to BASE_DIR/.cache
            self._cache_dir = self._base_dir / ".cache"
        
        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
    
    @property
    def BASE_DIR(self) -> Path:
        """Project root directory."""
        return self._base_dir
    
    @property
    def HF_CACHE_DIR(self) -> Path:
        """HuggingFace cache directory."""
        return self._cache_dir
    
    @property
    def CACHE_DIR(self) -> Path:
        """General cache directory (alias for HF_CACHE_DIR)."""
        return self._cache_dir


# Create singleton instance
_config = ProjectConfig()

# Export convenience properties
BASE_DIR = _config.BASE_DIR
HF_CACHE_DIR = _config.HF_CACHE_DIR
CACHE_DIR = _config.CACHE_DIR

