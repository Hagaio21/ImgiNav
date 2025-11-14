#!/usr/bin/env python3
"""
Training pipeline script for Phase 2.1 experiments.
Runs both autoencoder and diffusion training sequentially.

This script:
1. Trains the autoencoder from the autoencoder config
2. Extracts the best checkpoint path
3. Updates the diffusion config with the autoencoder checkpoint path
4. Trains the diffusion model using the trained autoencoder

Usage:
    python training/train_pipeline_phase2.py --ae-config <autoencoder_config> --diffusion-config <diffusion_config>
"""

import torch
import sys
import os
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import (
    set_deterministic,
    load_config,
    get_device,
)
from training.train import main as train_ae_main
from training.train_diffusion import main as train_diffusion_main


def update_diffusion_config_with_ae_checkpoint(diffusion_config_path, ae_checkpoint_path):
    """
    Update diffusion config to point to the trained autoencoder checkpoint.
    
    This is for from-scratch training where the autoencoder checkpoint doesn't exist yet.
    The pipeline will train the autoencoder first, then update this config with the checkpoint path.
    
    Args:
        diffusion_config_path: Path to diffusion config YAML
        ae_checkpoint_path: Path to autoencoder best checkpoint (must exist)
    """
    print(f"\n{'='*60}")
    print("Updating diffusion config with autoencoder checkpoint")
    print(f"{'='*60}")
    
    # Verify checkpoint exists
    ae_checkpoint_abs = Path(ae_checkpoint_path).resolve()
    if not ae_checkpoint_abs.exists():
        raise FileNotFoundError(
            f"Autoencoder checkpoint not found: {ae_checkpoint_abs}\n"
            f"This should not happen - autoencoder training should have created it."
        )
    
    # Load diffusion config
    with open(diffusion_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update autoencoder checkpoint path
    if 'autoencoder' not in config:
        config['autoencoder'] = {}
    
    config['autoencoder']['checkpoint'] = str(ae_checkpoint_abs)
    
    # Save updated config (preserve comments and structure as much as possible)
    with open(diffusion_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Updated autoencoder checkpoint path to: {ae_checkpoint_abs}")
    print(f"Config saved to: {diffusion_config_path}")


def find_ae_checkpoint(ae_config):
    """
    Find the best checkpoint path from autoencoder config.
    
    Args:
        ae_config: Autoencoder config dictionary
        
    Returns:
        Path to best checkpoint, or None if not found
    """
    exp_name = ae_config.get("experiment", {}).get("name", "unnamed")
    save_path = ae_config.get("experiment", {}).get("save_path")
    
    if save_path is None:
        save_path = Path("outputs") / exp_name
    else:
        save_path = Path(save_path)
    
    # Check for best checkpoint
    best_checkpoint = save_path / f"{exp_name}_checkpoint_best.pt"
    
    if best_checkpoint.exists():
        return best_checkpoint
    
    # Also check in checkpoints subdirectory (some experiments use this)
    checkpoint_dir = save_path / "checkpoints"
    best_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_best.pt"
    
    if best_checkpoint.exists():
        return best_checkpoint
    
    return None


def train_autoencoder(ae_config_path):
    """
    Train autoencoder using the training script.
    
    Args:
        ae_config_path: Path to autoencoder config YAML
        
    Returns:
        Path to best checkpoint, or None if training failed
    """
    print(f"\n{'='*60}")
    print("PHASE 1: Training Autoencoder")
    print(f"{'='*60}")
    print(f"Config: {ae_config_path}")
    print(f"{'='*60}\n")
    
    # Import and run autoencoder training
    # We'll use subprocess to run the training script to ensure clean state
    import subprocess
    
    try:
        # Get absolute paths
        base_dir = Path(__file__).parent.parent
        ae_config_abs = Path(ae_config_path).resolve()
        
        # Run as module to ensure imports work correctly
        # train.py adds parent to sys.path, so direct script execution works
        train_script = base_dir / "training" / "train.py"
        result = subprocess.run(
            [sys.executable, str(train_script), str(ae_config_abs)],
            check=True,
            cwd=base_dir
        )
        print(f"\n{'='*60}")
        print("Autoencoder training completed successfully!")
        print(f"{'='*60}\n")
        
        # Find the checkpoint
        ae_config = load_config(ae_config_path)
        checkpoint_path = find_ae_checkpoint(ae_config)
        
        if checkpoint_path is None:
            print("WARNING: Could not find autoencoder checkpoint!")
            return None
        
        print(f"Found autoencoder checkpoint: {checkpoint_path}")
        return checkpoint_path
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Autoencoder training failed with exit code {e.returncode}")
        print(f"{'='*60}\n")
        return None
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Autoencoder training failed: {e}")
        print(f"{'='*60}\n")
        return None


def train_diffusion(diffusion_config_path):
    """
    Train diffusion model using the training script.
    
    Args:
        diffusion_config_path: Path to diffusion config YAML
        
    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Training Diffusion Model")
    print(f"{'='*60}")
    print(f"Config: {diffusion_config_path}")
    print(f"{'='*60}\n")
    
    # Import and run diffusion training
    import subprocess
    
    try:
        # Get absolute paths
        base_dir = Path(__file__).parent.parent
        diffusion_config_abs = Path(diffusion_config_path).resolve()
        
        # Run as module since train_diffusion.py doesn't add parent to sys.path
        # Using -m flag ensures Python can find the training module
        result = subprocess.run(
            [sys.executable, "-m", "training.train_diffusion", str(diffusion_config_abs)],
            check=True,
            cwd=base_dir
        )
        print(f"\n{'='*60}")
        print("Diffusion training completed successfully!")
        print(f"{'='*60}\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Diffusion training failed with exit code {e.returncode}")
        print(f"{'='*60}\n")
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Diffusion training failed: {e}")
        print(f"{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.1 training pipeline: Autoencoder + Diffusion"
    )
    parser.add_argument(
        "--ae-config",
        type=Path,
        required=True,
        help="Path to autoencoder experiment config YAML file"
    )
    parser.add_argument(
        "--diffusion-config",
        type=Path,
        required=True,
        help="Path to diffusion experiment config YAML file"
    )
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip autoencoder training (use existing checkpoint)"
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=Path,
        help="Path to existing autoencoder checkpoint (if --skip-ae)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 2.1 Training Pipeline")
    print("="*60)
    print(f"Autoencoder config: {args.ae_config}")
    print(f"Diffusion config: {args.diffusion_config}")
    print("="*60)
    
    # Validate config files exist
    if not args.ae_config.exists():
        print(f"ERROR: Autoencoder config not found: {args.ae_config}")
        sys.exit(1)
    
    if not args.diffusion_config.exists():
        print(f"ERROR: Diffusion config not found: {args.diffusion_config}")
        sys.exit(1)
    
    # Step 1: Train autoencoder (or use existing checkpoint)
    # First, check if autoencoder checkpoint already exists
    ae_config = load_config(args.ae_config)
    existing_checkpoint = find_ae_checkpoint(ae_config)
    
    if args.skip_ae:
        if args.ae_checkpoint is None:
            print("ERROR: --ae-checkpoint required when using --skip-ae")
            sys.exit(1)
        if not args.ae_checkpoint.exists():
            print(f"ERROR: Autoencoder checkpoint not found: {args.ae_checkpoint}")
            sys.exit(1)
        ae_checkpoint_path = args.ae_checkpoint
        print(f"\nSkipping autoencoder training, using checkpoint: {ae_checkpoint_path}")
    elif existing_checkpoint is not None:
        # Autoencoder checkpoint already exists - skip training
        ae_checkpoint_path = existing_checkpoint
        print(f"\n{'='*60}")
        print("Autoencoder checkpoint already exists - skipping training")
        print(f"{'='*60}")
        print(f"Found checkpoint: {ae_checkpoint_path}")
        print(f"{'='*60}\n")
    else:
        # No checkpoint found - train autoencoder
        ae_checkpoint_path = train_autoencoder(args.ae_config)
        if ae_checkpoint_path is None:
            print("ERROR: Autoencoder training failed or checkpoint not found")
            sys.exit(1)
    
    # Step 2: Update diffusion config with autoencoder checkpoint
    update_diffusion_config_with_ae_checkpoint(args.diffusion_config, ae_checkpoint_path)
    
    # Step 3: Train diffusion model
    success = train_diffusion(args.diffusion_config)
    
    if success:
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        print(f"Autoencoder checkpoint: {ae_checkpoint_path}")
        print(f"Diffusion config: {args.diffusion_config}")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Pipeline failed during diffusion training")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()

