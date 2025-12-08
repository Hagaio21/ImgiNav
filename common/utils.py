#!/usr/bin/env python3
"""
Common utility functions.

Consolidation notes:
- Removed redundant `set_seeds()` function - use `set_deterministic()` instead
- `set_deterministic()` is more comprehensive (handles CUDNN settings)
"""

import json
from pathlib import Path
from typing import Dict
import random
import numpy as np


def safe_mkdir(path: Path, parents: bool = True, exist_ok: bool = True):
    """Create directory safely with error handling."""
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")


def write_json(data: Dict, path: Path, indent: int = 2):
    """Write dictionary to JSON file with automatic directory creation."""
    try:
        safe_mkdir(path.parent)
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON to {path}: {e}")


def set_deterministic(seed: int = 42, strict_determinism: bool = False):
    """
    Set random seeds and deterministic settings for reproducibility.
    
    This is the canonical seed-setting function. It handles:
    - Python random
    - NumPy random  
    - PyTorch CPU and CUDA random seeds
    - CUDNN deterministic settings
    
    Args:
        seed: Random seed value
        strict_determinism: If True, enable strict deterministic algorithms (may warn for some ops)
    """
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CUDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Strict deterministic algorithms (may warn for operations without deterministic impl)
    if strict_determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)


# Alias for backward compatibility
set_seeds = set_deterministic


def numpy_scalar_constructor(loader, node):
    """Convert numpy scalar tags to Python native types for YAML loading."""
    try:
        sequence = loader.construct_sequence(node)
        if len(sequence) >= 2:
            value = sequence[1]
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                return value.item() if hasattr(value, 'item') else float(value)
            return value
        return sequence[0] if sequence else None
    except Exception:
        try:
            sequence = loader.construct_sequence(node)
            return sequence[0] if sequence else None
        except Exception:
            return None


class NumpySafeLoader:
    """YAML loader that safely handles numpy scalar types."""
    pass


def _register_numpy_constructor():
    """Register numpy scalar constructor for YAML loading."""
    import yaml
    global NumpySafeLoader
    NumpySafeLoader = type('NumpySafeLoader', (yaml.SafeLoader,), {})
    NumpySafeLoader.add_constructor(
        'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
        numpy_scalar_constructor
    )


def load_config_with_profile(config_path: str = None, profile: str = None, resolve_checkpoints: bool = True) -> Dict:
    """
    Load configuration file with optional profile support and checkpoint registry resolution.
    
    Args:
        config_path: Path to config file (YAML or JSON)
        profile: Profile name to use (overrides config's profile setting)
        resolve_checkpoints: If True, automatically resolve checkpoint registry references
    
    Returns:
        Loaded and resolved configuration dictionary
    """
    if not config_path:
        return {}
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
            _register_numpy_constructor()
            try:
                with open(path, "r") as f:
                    data = yaml.load(f, Loader=NumpySafeLoader) or {}
            except (yaml.constructor.ConstructorError, yaml.YAMLError) as e:
                print(f"Warning: Custom loader failed, trying FullLoader: {e}")
                with open(path, "r") as f:
                    data = yaml.load(f, Loader=yaml.FullLoader) or {}
        except ImportError as e:
            raise RuntimeError("YAML config requested but 'pyyaml' is not installed") from e
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary")
    
    # Handle profile selection
    profile_name = profile or data.get("profile")
    if profile_name and "profiles" in data:
        if profile_name not in data["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        base_config = {k: v for k, v in data.items() if k not in ("profiles", "profile")}
        base_config.update(data["profiles"][profile_name])
        data = base_config
    
    # Resolve checkpoint registry references
    if resolve_checkpoints:
        try:
            from common.checkpoint_registry import resolve_checkpoint_in_config
            data = resolve_checkpoint_in_config(data, base_dir=None)
        except ImportError:
            pass
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to resolve checkpoint registry references: {e}", UserWarning)
    
    return data
