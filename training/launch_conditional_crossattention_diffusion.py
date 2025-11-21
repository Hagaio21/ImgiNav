#!/usr/bin/env python3
"""
Launch script for conditional cross-attention diffusion training.

This script provides a simple interface to launch the conditional diffusion training
with cross-attention using embeddings.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Launch conditional cross-attention diffusion training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/diffusion/new_layouts/conditional_crossattention_diffusion.yaml",
        help="Path to experiment config YAML file (default: experiments/diffusion/new_layouts/conditional_crossattention_diffusion.yaml)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if exists"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force start from scratch (ignore existing checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Get absolute path to config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Resolve relative to project root
        project_root = Path(__file__).parent.parent
        config_path = (project_root / config_path).resolve()
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Build command
    base_dir = Path(__file__).parent.parent
    python_script = base_dir / "training" / "train_diffusion.py"
    
    cmd = [
        sys.executable,
        str(python_script),
        str(config_path)
    ]
    
    if args.resume:
        cmd.append("--resume")
    elif args.no_resume:
        cmd.append("--no-resume")
    
    print("=" * 60)
    print("Conditional Cross-Attention Diffusion Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Working directory: {base_dir}")
    print("=" * 60)
    print()
    
    # Run training
    try:
        subprocess.run(cmd, cwd=base_dir, check=True)
        print()
        print("=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"Training failed with exit code: {e.returncode}")
        print("=" * 60)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("Training interrupted by user")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

