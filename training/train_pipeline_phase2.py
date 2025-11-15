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
from training.train import main as train_ae_main
from training.train_diffusion import main as train_diffusion_main


def update_diffusion_config_manifest(diffusion_config_path, manifest_path):
    """
    Update diffusion config to use the embedded manifest.
    
    Args:
        diffusion_config_path: Path to diffusion config YAML
        manifest_path: Path to manifest with embedded latents
    """
    print(f"\n{'='*60}")
    print("Updating diffusion config with embedded manifest")
    print(f"{'='*60}")
    
    # Verify manifest exists
    manifest_abs = Path(manifest_path).resolve()
    if not manifest_abs.exists():
        raise FileNotFoundError(
            f"Embedded manifest not found: {manifest_abs}\n"
            f"Embedding step may have failed."
        )
    
    # Load diffusion config
    with open(diffusion_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update dataset manifest and outputs
    if 'dataset' not in config:
        config['dataset'] = {}
    
    old_manifest = config['dataset'].get('manifest', 'not set')
    config['dataset']['manifest'] = str(manifest_abs)
    config['dataset']['outputs'] = {
        'latent': 'latent_path'  # Use pre-embedded latents
    }
    
    # CRITICAL: Ensure UNet is NOT frozen (must be trainable for diffusion training)
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
    
    print(f"Updated manifest path:")
    print(f"  Old: {old_manifest}")
    print(f"  New: {manifest_abs}")
    print(f"Config saved to: {diffusion_config_path}")
    print(f"{'='*60}\n")


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
    
    # CRITICAL: Ensure UNet is NOT frozen (must be trainable for diffusion training)
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
    
    # Save updated config (preserve comments and structure as much as possible)
    with open(diffusion_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Updated autoencoder checkpoint path to: {ae_checkpoint_abs}")
    print(f"Verified UNet is trainable (frozen=false)")
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


def embed_dataset(ae_checkpoint_path, ae_config_path, input_manifest_path, output_manifest_path, batch_size=32, num_workers=8):
    """
    Embed dataset using the trained autoencoder.
    
    Args:
        ae_checkpoint_path: Path to autoencoder checkpoint
        ae_config_path: Path to autoencoder config
        input_manifest_path: Path to input manifest (with images)
        output_manifest_path: Path to output manifest (with latent_path column)
        batch_size: Batch size for encoding
        num_workers: Number of workers for data loading
        
    Returns:
        True if embedding succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 1.5: Embedding Dataset")
    print(f"{'='*60}")
    print(f"Autoencoder checkpoint: {ae_checkpoint_path}")
    print(f"Input manifest: {input_manifest_path}")
    print(f"Output manifest: {output_manifest_path}")
    print(f"{'='*60}\n")
    
    import subprocess
    
    try:
        # Get absolute paths
        base_dir = Path(__file__).parent.parent
        embed_script = base_dir / "data_preparation" / "create_embeddings.py"
        ae_checkpoint_abs = Path(ae_checkpoint_path).resolve()
        ae_config_abs = Path(ae_config_path).resolve()
        input_manifest_abs = Path(input_manifest_path).resolve()
        output_manifest_abs = Path(output_manifest_path).resolve()
        
        # Ensure output directory exists
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True)
        
        # Run embedding script
        # Note: --output is required but for layout manifest-based workflow, --output-manifest is what we actually use
        result = subprocess.run(
            [
                sys.executable,
                str(embed_script),
                "--type", "layout",
                "--manifest", str(input_manifest_abs),
                "--output", str(output_manifest_abs.parent),  # Required: output directory
                "--output-manifest", str(output_manifest_abs),  # Actual manifest path
                "--autoencoder-config", str(ae_config_abs),
                "--autoencoder-checkpoint", str(ae_checkpoint_abs),
                "--batch-size", str(batch_size),
                "--num-workers", str(num_workers),
                "--device", "cuda"
            ],
            check=True,
            cwd=base_dir
        )
        print(f"\n{'='*60}")
        print("Dataset embedding completed successfully!")
        print(f"{'='*60}\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset embedding failed with exit code {e.returncode}")
        print(f"{'='*60}\n")
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset embedding failed: {e}")
        print(f"{'='*60}\n")
        return False


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
    
    # Step 2: Embed dataset using the trained autoencoder
    # Determine paths for embedding
    diffusion_config = load_config(args.diffusion_config)
    exp_name = diffusion_config.get("experiment", {}).get("name", "unnamed")
    exp_save_path = diffusion_config.get("experiment", {}).get("save_path")
    
    if exp_save_path is None:
        exp_save_path = Path("outputs") / exp_name
    else:
        exp_save_path = Path(exp_save_path)
    
    # Create embeddings directory in experiment path
    embeddings_dir = exp_save_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Input manifest (original dataset)
    input_manifest = Path("/work3/s233249/ImgiNav/datasets/layouts.csv")
    if not input_manifest.exists():
        # Try alternative path
        input_manifest = Path("/work3/s233249/ImgiNav/datasets/augmented/manifest_images.csv")
    
    # Output manifest (with embedded latents, saved in experiment embeddings folder)
    output_manifest = embeddings_dir / "manifest_with_latents.csv"
    
    print(f"\n{'='*60}")
    print("Embedding dataset with trained autoencoder")
    print(f"{'='*60}")
    print(f"Input manifest: {input_manifest}")
    print(f"Output manifest: {output_manifest}")
    print(f"{'='*60}\n")
    
    if not input_manifest.exists():
        print(f"WARNING: Input manifest not found: {input_manifest}")
        print("Skipping embedding step. Diffusion will encode on-the-fly.")
        embedded_manifest = None
    else:
        # Check if embeddings already exist in experiment directory
        if output_manifest.exists():
            print(f"\n{'='*60}")
            print("Embedded manifest already exists in experiment directory")
            print(f"{'='*60}")
            print(f"Found: {output_manifest}")
            print("Skipping embedding step.")
            print(f"{'='*60}\n")
            embedded_manifest = output_manifest
        else:
            # Embed the dataset in the experiment folder
            print(f"Embedding dataset with new autoencoder...")
            success = embed_dataset(
                ae_checkpoint_path,
                args.ae_config,
                str(input_manifest),
                str(output_manifest),
                batch_size=32,
                num_workers=8
            )
            if success:
                embedded_manifest = output_manifest
            else:
                print("WARNING: Embedding failed. Diffusion will encode on-the-fly.")
                embedded_manifest = None
    
    # Step 3: Update diffusion config with autoencoder checkpoint
    update_diffusion_config_with_ae_checkpoint(args.diffusion_config, ae_checkpoint_path)
    
    # Step 4: Update manifest path if embedding succeeded
    if embedded_manifest is not None:
        update_diffusion_config_manifest(args.diffusion_config, str(embedded_manifest))
        
        # Verify the config was updated correctly
        final_config = load_config(args.diffusion_config)
        final_manifest = final_config.get("dataset", {}).get("manifest", "")
        final_outputs = final_config.get("dataset", {}).get("outputs", {})
        
        print(f"\n{'='*60}")
        print("Final Config Verification")
        print(f"{'='*60}")
        print(f"Manifest path: {final_manifest}")
        print(f"Outputs: {final_outputs}")
        
        # Verify UNet config is trainable
        unet_cfg = final_config.get('unet', {})
        unet_frozen = unet_cfg.get('frozen', False)
        has_freeze_blocks = 'freeze_blocks' in unet_cfg or 'freeze_downblocks' in unet_cfg or 'freeze_upblocks' in unet_cfg
        
        print(f"UNet config:")
        print(f"  frozen: {unet_frozen} (must be False)")
        if has_freeze_blocks:
            print(f"  WARNING: freeze_blocks settings found: {[k for k in ['freeze_blocks', 'freeze_downblocks', 'freeze_upblocks'] if k in unet_cfg]}")
        else:
            print(f"  freeze_blocks: None (OK)")
        
        if unet_frozen or has_freeze_blocks:
            print(f"  ERROR: UNet will be frozen! Training will fail!")
        else:
            print(f"  OK: UNet is trainable")
        
        print(f"{'='*60}\n")
    else:
        # If embedding failed or was skipped, we need to ensure the model has encoder
        # But actually, if embedding failed, we should fail the pipeline
        print(f"\n{'='*60}")
        print("ERROR: Dataset embedding is required but failed or was skipped")
        print(f"{'='*60}")
        print("The diffusion model expects 4-channel latents but the dataset")
        print("does not have pre-embedded latents. Please ensure:")
        print("1. Input manifest exists and is accessible")
        print("2. Embedding step completes successfully")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Step 5: Train diffusion model
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

