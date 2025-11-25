#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List
import random
import numpy as np


def safe_mkdir(path: Path, parents: bool = True, exist_ok: bool = True):
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")


def write_json(data: Dict, path: Path, indent: int = 2):
    try:
        safe_mkdir(path.parent)
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON to {path}: {e}")


def create_progress_tracker(total: int, description: str = "Processing"):
    def update_progress(current: int, item_name: str = "", success: bool = True):
        status = "✓" if success else "✗"
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"[{current}/{total}] ({percentage:.1f}%) {status} {description} {item_name}", flush=True)
    return update_progress


def set_deterministic(seed: int = 42, strict_determinism: bool = False):
    """Set random seeds and deterministic settings for reproducibility."""
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDNN deterministic settings (applies to most operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Strict deterministic algorithms (may warn for operations without deterministic impl)
    if strict_determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)


def numpy_scalar_constructor(loader, node):
    """Convert numpy scalar tags to Python native types."""
    # For python/object/apply tags, the node contains a sequence with the function and args
    # numpy.core.multiarray.scalar is called with the value as argument
    try:
        # Construct the sequence which contains [numpy.core.multiarray.scalar, value]
        sequence = loader.construct_sequence(node)
        if len(sequence) >= 2:
            # The value is the second element (first is the function/class)
            value = sequence[1]
            # Convert numpy types to Python native types
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                return value.item() if hasattr(value, 'item') else float(value)
            return value
        return sequence[0] if sequence else None
    except Exception:
        # Fallback: try to construct as sequence and take first value
        try:
            sequence = loader.construct_sequence(node)
            return sequence[0] if sequence else None
        except Exception:
            return None


class NumpySafeLoader:
    """YAML loader that safely handles numpy scalar types."""
    pass


# Register the constructor for numpy scalar types
def _register_numpy_constructor():
    import yaml
    # Create NumpySafeLoader as a subclass of SafeLoader
    global NumpySafeLoader
    NumpySafeLoader = type('NumpySafeLoader', (yaml.SafeLoader,), {})
    NumpySafeLoader.add_constructor(
        'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
        numpy_scalar_constructor
    )


def load_config_with_profile(config_path: str = None, profile: str = None) -> Dict:
    if not config_path:
        return {}
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
            # Register numpy constructor
            _register_numpy_constructor()
            # Try loading with custom loader first, fallback to FullLoader if it fails
            try:
                with open(path, "r") as f:
                    data = yaml.load(f, Loader=NumpySafeLoader) or {}
            except (yaml.constructor.ConstructorError, yaml.YAMLError) as e:
                # If custom loader fails (e.g., other numpy types), try FullLoader
                print(f"Warning: Custom loader failed, trying FullLoader: {e}")
                with open(path, "r") as f:
                    data = yaml.load(f, Loader=yaml.FullLoader) or {}
        except ImportError as e:
            raise RuntimeError("YAML config requested but 'pyyaml' is not installed") from e
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary")
    
    profile_name = profile or data.get("profile")
    if profile_name and "profiles" in data:
        if profile_name not in data["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        base_config = {k: v for k, v in data.items() if k not in ("profiles", "profile")}
        base_config.update(data["profiles"][profile_name])
        return base_config
    
    return data


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)