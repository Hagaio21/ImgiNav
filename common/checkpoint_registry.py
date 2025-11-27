"""
Checkpoint Registry System

Provides centralized checkpoint path management that works across different environments
(local development and HPC clusters).

Usage:
    from common.checkpoint_registry import get_checkpoint_path
    
    # In config files, use registry references:
    # autoencoder:
    #   checkpoint: "@vae_best"  # References registry entry
    
    # Resolve path:
    checkpoint_path = get_checkpoint_path("@vae_best")
    # Returns: "checkpoints/vae_checkpoint_best.pt" (local)
    #       or: "/work3/.../checkpoints/vae_checkpoint_best.pt" (HPC)
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from common.env_config import BASE_DIR


_REGISTRY: Optional[Dict[str, Any]] = None
_ENVIRONMENT: Optional[str] = None


def detect_environment() -> str:
    """
    Detect the current environment (local or HPC).
    
    Returns:
        "local" or "hpc"
    """
    # Check explicit environment variable
    env = os.getenv("IMGINAV_ENV", "").lower()
    if env in ["local", "hpc"]:
        return env
    
    # Check for HPC indicators
    hpc_indicators = [
        "SLURM_JOB_ID",  # SLURM job scheduler
        "PBS_JOBID",     # PBS job scheduler
        "HPC",           # Generic HPC flag
        "CLUSTER",       # Generic cluster flag
    ]
    
    if any(os.getenv(var) for var in hpc_indicators):
        return "hpc"
    
    # Check hostname for common HPC patterns
    hostname = os.getenv("HOSTNAME", "").lower()
    if any(pattern in hostname for pattern in ["compute", "node", "gpu", "hpc"]):
        return "hpc"
    
    # Default to local
    return "local"


def get_environment() -> str:
    """Get current environment (cached)."""
    global _ENVIRONMENT
    if _ENVIRONMENT is None:
        _ENVIRONMENT = detect_environment()
    return _ENVIRONMENT


def load_registry() -> Dict[str, Any]:
    """Load checkpoint registry from YAML file."""
    global _REGISTRY
    
    if _REGISTRY is not None:
        return _REGISTRY
    
    registry_path = BASE_DIR / "config" / "checkpoint_registry.yaml"
    
    if not registry_path.exists():
        # Return empty registry if file doesn't exist
        _REGISTRY = {"checkpoints": {}, "defaults": {}}
        return _REGISTRY
    
    with open(registry_path, "r", encoding="utf-8") as f:
        _REGISTRY = yaml.safe_load(f) or {"checkpoints": {}, "defaults": {}}
    
    return _REGISTRY


def expand_env_vars(path_str: str) -> str:
    """Expand environment variables in path string."""
    return os.path.expandvars(path_str)


def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path string, handling environment variables and relative paths.
    
    Args:
        path_str: Path string (may contain env vars like ${IMGINAV_ROOT})
        base_dir: Base directory for resolving relative paths (default: BASE_DIR)
    
    Returns:
        Resolved Path object
    """
    if base_dir is None:
        base_dir = BASE_DIR
    
    # Expand environment variables
    expanded = expand_env_vars(path_str)
    
    # Resolve path
    path = Path(expanded)
    
    # If relative, resolve relative to base_dir
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    
    return path


def get_checkpoint_path(
    checkpoint_ref: str,
    environment: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Get checkpoint path from registry reference.
    
    Args:
        checkpoint_ref: Checkpoint reference, either:
            - Registry reference: "@checkpoint_name" (e.g., "@vae_best")
            - Direct path: "path/to/checkpoint.pt" (returned as-is)
        environment: Override environment detection ("local" or "hpc")
        base_dir: Base directory for resolving paths (default: BASE_DIR)
    
    Returns:
        Resolved Path to checkpoint
    
    Raises:
        ValueError: If registry reference not found
        FileNotFoundError: If resolved path doesn't exist (warning only)
    """
    # If not a registry reference, return as-is (after resolving)
    if not checkpoint_ref.startswith("@"):
        return resolve_path(checkpoint_ref, base_dir)
    
    # Extract checkpoint name
    checkpoint_name = checkpoint_ref[1:]  # Remove "@"
    
    # Load registry
    registry = load_registry()
    checkpoints = registry.get("checkpoints", {})
    
    if checkpoint_name not in checkpoints:
        available = ", ".join(checkpoints.keys())
        raise ValueError(
            f"Checkpoint '{checkpoint_name}' not found in registry. "
            f"Available checkpoints: {available}"
        )
    
    checkpoint_info = checkpoints[checkpoint_name]
    
    # Determine environment
    env = environment if environment is not None else get_environment()
    
    # Get path for environment
    if env in checkpoint_info:
        path_str = checkpoint_info[env]
    elif "local" in checkpoint_info:
        # Fallback to local if environment-specific path not found
        path_str = checkpoint_info["local"]
    else:
        # Fallback to defaults
        defaults = registry.get("defaults", {})
        default_dir = defaults.get(env, defaults.get("local", "checkpoints"))
        path_str = str(Path(default_dir) / f"{checkpoint_name}.pt")
    
    # Resolve path
    resolved_path = resolve_path(path_str, base_dir)
    
    # Warn if path doesn't exist (but don't fail - might be created later)
    if not resolved_path.exists():
        import warnings
        warnings.warn(
            f"Checkpoint path does not exist: {resolved_path}\n"
            f"  Registry reference: {checkpoint_ref}\n"
            f"  Environment: {env}",
            UserWarning
        )
    
    return resolved_path


def resolve_checkpoint_in_config(
    config: Dict[str, Any],
    base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Recursively resolve checkpoint registry references in a config dict.
    
    This function walks through the config and replaces any string values
    that start with "@" with their resolved paths from the registry.
    
    Args:
        config: Configuration dictionary (may be modified in-place)
        base_dir: Base directory for resolving paths (default: BASE_DIR)
                  Note: For registry references, BASE_DIR is always used.
                  This parameter is only for non-registry path resolution.
    
    Returns:
        Config with resolved checkpoint paths
    """
    # Use BASE_DIR for registry resolution (registry paths are relative to project root)
    registry_base_dir = BASE_DIR if base_dir is None else None
    
    if isinstance(config, dict):
        resolved = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("@"):
                # Resolve registry reference (always use BASE_DIR)
                resolved[key] = str(get_checkpoint_path(value, base_dir=registry_base_dir))
            elif isinstance(value, (dict, list)):
                # Recursively resolve nested structures
                resolved[key] = resolve_checkpoint_in_config(value, base_dir=base_dir)
            else:
                resolved[key] = value
        return resolved
    elif isinstance(config, list):
        return [resolve_checkpoint_in_config(item, base_dir=base_dir) for item in config]
    else:
        return config


def list_checkpoints() -> Dict[str, Dict[str, Any]]:
    """
    List all checkpoints in the registry.
    
    Returns:
        Dictionary mapping checkpoint names to their info
    """
    registry = load_registry()
    return registry.get("checkpoints", {})


def add_checkpoint(
    name: str,
    local_path: str,
    hpc_path: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Add a checkpoint to the registry (in-memory only, doesn't save to file).
    
    Args:
        name: Checkpoint name (without "@")
        local_path: Path for local environment
        hpc_path: Path for HPC environment (optional)
        description: Description of the checkpoint (optional)
    """
    registry = load_registry()
    checkpoints = registry.setdefault("checkpoints", {})
    
    checkpoint_info = {"local": local_path}
    if hpc_path:
        checkpoint_info["hpc"] = hpc_path
    if description:
        checkpoint_info["description"] = description
    
    checkpoints[name] = checkpoint_info


# Convenience function for common use case
def get_checkpoint(checkpoint_ref: str) -> Path:
    """
    Convenience function to get checkpoint path.
    
    Args:
        checkpoint_ref: Registry reference (e.g., "@vae_best") or direct path
    
    Returns:
        Resolved Path to checkpoint
    """
    return get_checkpoint_path(checkpoint_ref)

