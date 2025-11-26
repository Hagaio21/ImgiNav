#!/usr/bin/env python3
"""
List unfinished experiments by checking checkpoint epochs against target epochs.

Scans experiment directories and reports experiments that haven't reached their target epoch count.
"""

import torch
import yaml
from pathlib import Path
import argparse
from typing import Optional, Tuple, List


def get_experiment_epoch_from_checkpoint(checkpoint_path: Path) -> Optional[int]:
    """Extract current epoch from checkpoint file."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
        epoch = payload.get("epoch", None)
        return epoch
    except Exception as e:
        print(f"  Warning: Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def get_target_epochs_from_config(config_path: Path) -> Optional[int]:
    """Extract target epochs from config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Try epochs_target first (from experiment section)
        epochs_target = config.get("experiment", {}).get("epochs_target", None)
        if epochs_target is not None:
            return epochs_target
        
        # Fall back to epochs in training section
        epochs = config.get("training", {}).get("epochs", None)
        if epochs is not None:
            return epochs
        
        return None
    except Exception as e:
        print(f"  Warning: Failed to load config {config_path}: {e}")
        return None


def find_experiment_config(exp_dir: Path, exp_name: str) -> Optional[Path]:
    """Try to find the config file for this experiment."""
    # Common config locations
    config_dirs = [
        Path("experiments/diffusion/clip"),
        Path("experiments/diffusion/clip/regular"),
        Path("experiments/diffusion/clip/regular_rooms"),
        Path("experiments/diffusion/clip/regular_scenes"),
        Path("experiments/diffusion/clip/spatial"),
        Path("experiments/diffusion/clip/spatial_rooms"),
        Path("experiments/diffusion/clip/spatial_scenes"),
    ]
    
    # Try to find config file matching experiment name
    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        
        # Look for YAML files that might match
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                    config_exp_name = config.get("experiment", {}).get("name", "")
                    if config_exp_name == exp_name:
                        return yaml_file
            except Exception:
                continue
    
    return None


def scan_experiments(base_dir: Path, default_target_epochs: int = 1000) -> List[Tuple[str, int, int]]:
    """
    Scan experiment directory and find unfinished experiments.
    
    Returns:
        List of (exp_name, current_epoch, target_epochs) tuples for unfinished experiments
    """
    unfinished = []
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return unfinished
    
    print(f"Scanning experiments in: {base_dir}")
    print()
    
    # Find all experiment directories
    exp_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if len(exp_dirs) == 0:
        print("No experiment directories found.")
        return unfinished
    
    print(f"Found {len(exp_dirs)} experiment directories")
    print()
    
    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        
        # Skip non-experiment directories (like 'comparison_summary', 'vae_clip', etc.)
        if not exp_name.startswith("diff_clip_"):
            continue
        
        checkpoint_dir = exp_dir / "checkpoints"
        if not checkpoint_dir.exists():
            print(f"  {exp_name}: No checkpoints directory (not started)")
            continue
        
        # Look for latest checkpoint
        latest_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
        if not latest_checkpoint.exists():
            print(f"  {exp_name}: No latest checkpoint (not started)")
            continue
        
        # Get current epoch from checkpoint
        current_epoch = get_experiment_epoch_from_checkpoint(latest_checkpoint)
        if current_epoch is None:
            print(f"  {exp_name}: Failed to read epoch from checkpoint")
            continue
        
        # Try to find config to get target epochs
        config_path = find_experiment_config(exp_dir, exp_name)
        if config_path:
            target_epochs = get_target_epochs_from_config(config_path)
        else:
            target_epochs = None
        
        # Use default if config not found
        if target_epochs is None:
            target_epochs = default_target_epochs
        
        # Check if unfinished
        if current_epoch < target_epochs:
            unfinished.append((exp_name, current_epoch, target_epochs))
            print(f"  {exp_name}: {current_epoch}/{target_epochs} (unfinished)")
        else:
            print(f"  {exp_name}: {current_epoch}/{target_epochs} (complete)")
    
    return unfinished


def main():
    parser = argparse.ArgumentParser(
        description="List unfinished experiments by checking checkpoint epochs"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default="/work3/s233249/ImgiNav/experiments/clip",
        help="Base directory containing experiment folders (default: /work3/s233249/ImgiNav/experiments/clip)"
    )
    parser.add_argument(
        "--default-target",
        type=int,
        default=1000,
        help="Default target epochs if not found in config (default: 1000)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary of unfinished experiments (no per-experiment details)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Unfinished Experiments Scanner")
    print("=" * 80)
    print()
    
    unfinished = scan_experiments(args.base_dir, args.default_target)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unfinished experiments: {len(unfinished)}")
    print()
    
    if len(unfinished) > 0:
        print("Unfinished experiments:")
        print("-" * 80)
        for exp_name, current_epoch, target_epochs in unfinished:
            print(f"{exp_name} - {current_epoch}/{target_epochs}")
        print("-" * 80)
    else:
        print("All experiments are complete!")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

