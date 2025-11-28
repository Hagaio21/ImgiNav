#!/usr/bin/env python3
"""
Configuration loader for the data preparation pipeline.

Loads paths from paths.yaml and provides them to Python scripts.

Usage in Python scripts:
    from config_loader import load_config
    config = load_config("paths.yaml")
    output_root = config["output_dataset_root"]

Usage from command line (for shell scripts):
    python config_loader.py paths.yaml output_dataset_root
    python config_loader.py paths.yaml front3d_scenes_dir
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to paths.yaml
    
    Returns:
        Dictionary with resolved paths
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths
    base_dir = Path(config.get("base_dir", ".")).expanduser()
    
    # Make shards_dir and log_dir absolute
    shards_dir = config.get("shards_dir", "shards")
    if not os.path.isabs(shards_dir):
        shards_dir = str(base_dir / shards_dir)
    config["shards_dir"] = shards_dir
    
    log_dir = config.get("log_dir", "logs")
    if not os.path.isabs(log_dir):
        log_dir = str(base_dir / log_dir)
    config["log_dir"] = log_dir
    
    # Set default taxonomy path
    if not config.get("taxonomy_path"):
        config["taxonomy_path"] = str(Path(config["output_dataset_root"]) / "taxonomy" / "taxonomy.json")
    
    # Ensure base_dir is string
    config["base_dir"] = str(base_dir)
    
    return config


def get_config_value(config_path: str, key: str) -> Optional[str]:
    """
    Get a single configuration value.
    
    Args:
        config_path: Path to paths.yaml
        key: Configuration key to retrieve
    
    Returns:
        Configuration value as string, or None if not found
    """
    config = load_config(config_path)
    value = config.get(key)
    return str(value) if value is not None else None


def validate_config(config: Dict[str, Any], stage: int = 2) -> bool:
    """
    Validate configuration for a specific stage.
    
    Args:
        config: Configuration dictionary
        stage: Pipeline stage number (1-6)
    
    Returns:
        True if valid, raises exception otherwise
    """
    errors = []
    
    # Always required
    output_root = Path(config["output_dataset_root"])
    if not output_root.parent.exists():
        errors.append(f"Parent of output_dataset_root does not exist: {output_root.parent}")
    
    # Stage 1 and 2 need source paths
    if stage <= 2:
        scenes_dir = config.get("front3d_scenes_dir")
        if not scenes_dir or not Path(scenes_dir).exists():
            errors.append(f"front3d_scenes_dir not found: {scenes_dir}")
        
        model_info = config.get("front3d_model_info")
        if not model_info or not Path(model_info).exists():
            errors.append(f"front3d_model_info not found: {model_info}")
        
        if stage == 1:
            model_dir = config.get("front3d_model_dir")
            if not model_dir or not Path(model_dir).exists():
                errors.append(f"front3d_model_dir not found: {model_dir}")
    
    # Taxonomy needed for stages 1-3
    if stage <= 3:
        taxonomy = config.get("taxonomy_path")
        if not taxonomy or not Path(taxonomy).exists():
            errors.append(f"taxonomy_path not found: {taxonomy}")
    
    if errors:
        raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))
    
    return True


def print_config(config: Dict[str, Any]):
    """Print configuration in a readable format."""
    print("=" * 60)
    print("Pipeline Configuration")
    print("=" * 60)
    print(f"Output Dataset Root: {config['output_dataset_root']}")
    print()
    print("3D-FRONT/3D-FUTURE Sources:")
    print(f"  Scenes Dir:  {config.get('front3d_scenes_dir', 'N/A')}")
    print(f"  Model Info:  {config.get('front3d_model_info', 'N/A')}")
    print(f"  Model Dir:   {config.get('front3d_model_dir', 'N/A')}")
    print(f"  Texture Dir: {config.get('front3d_texture_dir', 'N/A')}")
    print()
    print("Pipeline Paths:")
    print(f"  Base Dir:    {config['base_dir']}")
    print(f"  Shards Dir:  {config['shards_dir']}")
    print(f"  Log Dir:     {config['log_dir']}")
    print(f"  Taxonomy:    {config.get('taxonomy_path', 'N/A')}")
    print("=" * 60)


def main():
    """Command-line interface for config loader."""
    parser = argparse.ArgumentParser(
        description="Load and query pipeline configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print all configuration
  python config_loader.py paths.yaml
  
  # Get a specific value (for use in shell scripts)
  python config_loader.py paths.yaml output_dataset_root
  python config_loader.py paths.yaml front3d_scenes_dir
  
  # Validate configuration for a stage
  python config_loader.py paths.yaml --validate --stage 2
"""
    )
    parser.add_argument("config", help="Path to paths.yaml")
    parser.add_argument("key", nargs="?", help="Configuration key to retrieve")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--stage", type=int, default=2, help="Stage number for validation (1-6)")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        
        if args.validate:
            validate_config(config, args.stage)
            print(f"Configuration valid for stage {args.stage}")
            return 0
        
        if args.key:
            # Output single value (for shell scripts)
            value = config.get(args.key)
            if value is not None:
                print(value)
                return 0
            else:
                print(f"Key not found: {args.key}", file=sys.stderr)
                return 1
        else:
            # Print full configuration
            print_config(config)
            return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
