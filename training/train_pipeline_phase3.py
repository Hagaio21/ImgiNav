#!/usr/bin/env python3
"""
Training pipeline script for Phase 3 experiments.
Fine-tunes unconditional diffusion models to add conditioning.

This script:
1. Loads an unconditional diffusion model checkpoint
2. Fine-tunes it with conditioning enabled (cfg_dropout_rate < 1.0)

Usage:
    python training/train_pipeline_phase3.py --unconditional-checkpoint <checkpoint_path> --diffusion-config <diffusion_config>
"""

import torch
import sys
import os
import yaml
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import (
    set_deterministic,
    load_config,
    get_device,
)
from training.train_diffusion import main as train_diffusion_main


def find_unconditional_checkpoint(unconditional_exp_name, unconditional_save_path=None):
    """
    Find the best checkpoint path from unconditional experiment.
    
    Args:
        unconditional_exp_name: Name of unconditional experiment
        unconditional_save_path: Optional explicit save path
        
    Returns:
        Path to best checkpoint, or None if not found
    """
    if unconditional_save_path:
        save_path = Path(unconditional_save_path)
    else:
        # Default path structure
        save_path = Path("/work3/s233249/ImgiNav/experiments/phase2") / unconditional_exp_name
    
    # Check for best checkpoint in checkpoints subdirectory
    checkpoint_dir = save_path / "checkpoints"
    best_checkpoint = checkpoint_dir / f"{unconditional_exp_name}_checkpoint_best.pt"
    
    if best_checkpoint.exists():
        return best_checkpoint
    
    # Also check in root directory
    best_checkpoint = save_path / f"{unconditional_exp_name}_checkpoint_best.pt"
    
    if best_checkpoint.exists():
        return best_checkpoint
    
    return None


def update_diffusion_config_with_unconditional_checkpoint(diffusion_config_path, unconditional_checkpoint_path):
    """
    Update diffusion config to load from unconditional checkpoint.
    
    Args:
        diffusion_config_path: Path to diffusion config YAML
        unconditional_checkpoint_path: Path to unconditional checkpoint (must exist)
    """
    print(f"\n{'='*60}")
    print("Updating diffusion config with unconditional checkpoint")
    print(f"{'='*60}")
    
    # Verify checkpoint exists
    unconditional_checkpoint_abs = Path(unconditional_checkpoint_path).resolve()
    if not unconditional_checkpoint_abs.exists():
        raise FileNotFoundError(
            f"Unconditional checkpoint not found: {unconditional_checkpoint_abs}\n"
            f"Please ensure the unconditional model has been trained first."
        )
    
    # Load diffusion config
    with open(diffusion_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add diffusion section if it doesn't exist
    if 'diffusion' not in config:
        config['diffusion'] = {}
    
    # Set stage1_checkpoint to unconditional checkpoint (for fine-tuning)
    config['diffusion']['stage1_checkpoint'] = str(unconditional_checkpoint_abs)
    
    # CRITICAL: Ensure UNet is NOT frozen (must be trainable for fine-tuning)
    if 'unet' not in config:
        config['unet'] = {}
    
    # Explicitly ensure UNet is not frozen
    if config['unet'].get('frozen', False):
        print("WARNING: UNet was marked as frozen in config! Setting to trainable...")
        config['unet']['frozen'] = False
    
    # Remove any freeze_blocks settings that would freeze parts of UNet
    if 'freeze_blocks' in config['unet']:
        print("WARNING: UNet had freeze_blocks setting! Removing to ensure full training...")
        del config['unet']['freeze_blocks']
    if 'freeze_downblocks' in config['unet']:
        print("WARNING: UNet had freeze_downblocks setting! Removing to ensure full training...")
        del config['unet']['freeze_downblocks']
    if 'freeze_upblocks' in config['unet']:
        print("WARNING: UNet had freeze_upblocks setting! Removing to ensure full training...")
        del config['unet']['freeze_upblocks']
    
    # Save updated config
    with open(diffusion_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Updated unconditional checkpoint path to: {unconditional_checkpoint_abs}")
    print(f"Verified UNet is trainable (frozen=false)")
    print(f"Config saved to: {diffusion_config_path}")
    print(f"{'='*60}\n")


def train_diffusion(diffusion_config_path):
    """
    Train (fine-tune) diffusion model using the training script.
    
    Args:
        diffusion_config_path: Path to diffusion config YAML
        
    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 3: Fine-tuning Diffusion Model (Adding Conditioning)")
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
        print("Diffusion fine-tuning completed successfully!")
        print(f"{'='*60}\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Diffusion fine-tuning failed with exit code {e.returncode}")
        print(f"{'='*60}\n")
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Diffusion fine-tuning failed: {e}")
        print(f"{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 training pipeline: Fine-tune unconditional models with conditioning"
    )
    parser.add_argument(
        "--unconditional-checkpoint",
        type=Path,
        help="Path to unconditional diffusion checkpoint (optional, can be auto-detected from config)"
    )
    parser.add_argument(
        "--unconditional-exp-name",
        type=str,
        help="Name of unconditional experiment (for auto-detection if checkpoint not provided)"
    )
    parser.add_argument(
        "--unconditional-save-path",
        type=Path,
        help="Save path of unconditional experiment (for auto-detection)"
    )
    parser.add_argument(
        "--diffusion-config",
        type=Path,
        required=True,
        help="Path to diffusion experiment config YAML file (for conditional fine-tuning)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 3 Training Pipeline: Fine-tuning Unconditional → Conditional")
    print("="*60)
    print(f"Diffusion config: {args.diffusion_config}")
    print("="*60)
    
    # Validate config file exists
    if not args.diffusion_config.exists():
        print(f"ERROR: Diffusion config not found: {args.diffusion_config}")
        sys.exit(1)
    
    # Step 1: Find unconditional checkpoint
    unconditional_checkpoint_path = None
    
    if args.unconditional_checkpoint:
        # Explicit checkpoint path provided
        if not args.unconditional_checkpoint.exists():
            print(f"ERROR: Unconditional checkpoint not found: {args.unconditional_checkpoint}")
            sys.exit(1)
        unconditional_checkpoint_path = args.unconditional_checkpoint
        print(f"\nUsing provided unconditional checkpoint: {unconditional_checkpoint_path}")
    else:
        # Try to auto-detect from config or exp name
        diffusion_config = load_config(args.diffusion_config)
        
        # Check if config already has unconditional checkpoint path
        unconditional_checkpoint = diffusion_config.get("diffusion", {}).get("stage1_checkpoint")
        if unconditional_checkpoint and Path(unconditional_checkpoint).exists():
            unconditional_checkpoint_path = Path(unconditional_checkpoint)
            print(f"\nFound unconditional checkpoint in config: {unconditional_checkpoint_path}")
        elif args.unconditional_exp_name:
            # Try to find checkpoint from experiment name
            unconditional_checkpoint_path = find_unconditional_checkpoint(
                args.unconditional_exp_name,
                args.unconditional_save_path
            )
            if unconditional_checkpoint_path:
                print(f"\nAuto-detected unconditional checkpoint: {unconditional_checkpoint_path}")
            else:
                print(f"\nERROR: Could not find unconditional checkpoint for experiment: {args.unconditional_exp_name}")
                print("Please provide --unconditional-checkpoint explicitly.")
                sys.exit(1)
        else:
            print("\nERROR: Must provide either:")
            print("  --unconditional-checkpoint <path>")
            print("  OR")
            print("  --unconditional-exp-name <name> (with optional --unconditional-save-path)")
            sys.exit(1)
    
    # Step 2: Update diffusion config with unconditional checkpoint
    update_diffusion_config_with_unconditional_checkpoint(
        args.diffusion_config,
        str(unconditional_checkpoint_path)
    )
    
    # Step 3: Verify config has conditional settings (cfg_dropout_rate < 1.0)
    diffusion_config = load_config(args.diffusion_config)
    cfg_dropout_rate = diffusion_config.get("training", {}).get("cfg_dropout_rate", 1.0)
    
    print(f"\n{'='*60}")
    print("Config Verification")
    print(f"{'='*60}")
    print(f"cfg_dropout_rate: {cfg_dropout_rate}")
    if cfg_dropout_rate >= 1.0:
        print("WARNING: cfg_dropout_rate >= 1.0 means unconditional training!")
        print("For Phase 3 (conditional fine-tuning), cfg_dropout_rate should be < 1.0 (e.g., 0.1)")
    else:
        print("OK: Conditional training enabled (cfg_dropout_rate < 1.0)")
    print(f"{'='*60}\n")
    
    # Step 4: Fine-tune diffusion model
    success = train_diffusion(args.diffusion_config)
    
    if success:
        print("\n" + "="*60)
        print("Phase 3 pipeline completed successfully!")
        print("="*60)
        print(f"Unconditional checkpoint: {unconditional_checkpoint_path}")
        print(f"Conditional config: {args.diffusion_config}")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Phase 3 pipeline failed during fine-tuning")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()

