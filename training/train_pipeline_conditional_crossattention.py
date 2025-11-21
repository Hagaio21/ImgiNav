#!/usr/bin/env python3
"""
Complete training pipeline for conditional cross-attention diffusion.

This script:
1. Trains the 32x32 VAE autoencoder
2. Embeds the ControlNet dataset with the new VAE (preserves existing embeddings)
3. Calculates scale_factor from embedded latents
4. Updates diffusion config with VAE checkpoint, embedded manifest, and scale_factor
5. Trains the conditional cross-attention diffusion model

Usage:
    python training/train_pipeline_conditional_crossattention.py \
        --ae-config <autoencoder_config> \
        --diffusion-config <diffusion_config> \
        [--skip-ae] [--skip-embedding] [--skip-training]
"""

import torch
import sys
import os
import yaml
import argparse
import pandas as pd
import subprocess
import numpy as np
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
from training.train_diffusion import main as train_diffusion_main, calculate_scale_factor_from_dataset
from models.datasets.datasets import ManifestDataset


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
    
    # Also check in checkpoints subdirectory
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
    print("PHASE 1: Training 32x32 VAE Autoencoder")
    print(f"{'='*60}")
    print(f"Config: {ae_config_path}")
    print(f"{'='*60}\n")
    
    try:
        base_dir = Path(__file__).parent.parent
        ae_config_abs = Path(ae_config_path).resolve()
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
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return None


def embed_controlnet_dataset_with_vae(
    ae_checkpoint_path,
    ae_config_path,
    input_manifest_path,
    output_manifest_path,
    batch_size=32,
    num_workers=8
):
    """
    Embed ControlNet dataset with new VAE, preserving existing embeddings.
    
    This function:
    1. Loads the ControlNet manifest (which has graph_embedding_path and pov_embedding_path)
    2. Embeds layouts using the new VAE to create latent_path
    3. Creates a new manifest with all three: latent_path, graph_embedding_path, pov_embedding_path
    
    Args:
        ae_checkpoint_path: Path to VAE checkpoint
        ae_config_path: Path to VAE config
        input_manifest_path: Path to ControlNet manifest (with embeddings)
        output_manifest_path: Path to output manifest (with latents + embeddings)
        batch_size: Batch size for encoding
        num_workers: Number of workers
        
    Returns:
        True if embedding succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Embedding ControlNet Dataset with 32x32 VAE")
    print(f"{'='*60}")
    print(f"VAE checkpoint: {ae_checkpoint_path}")
    print(f"Input manifest: {input_manifest_path}")
    print(f"Output manifest: {output_manifest_path}")
    print(f"{'='*60}\n")
    
    try:
        base_dir = Path(__file__).parent.parent
        embed_script = base_dir / "data_preparation" / "create_embeddings.py"
        ae_checkpoint_abs = Path(ae_checkpoint_path).resolve()
        ae_config_abs = Path(ae_config_path).resolve()
        input_manifest_abs = Path(input_manifest_path).resolve()
        output_manifest_abs = Path(output_manifest_path).resolve()
        
        # Ensure output directory exists
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True)
        latents_dir = output_manifest_abs.parent / "latents"
        latents_dir.mkdir(parents=True, exist_ok=True)
        
        # Read input manifest to get layout paths
        df = pd.read_csv(input_manifest_abs, low_memory=False)
        
        # Create temporary manifest with just layout paths for embedding
        # The ControlNet manifest should have layout_path column
        temp_manifest = output_manifest_abs.parent / "temp_layouts_for_embedding.csv"
        layout_rows = []
        for _, row in df.iterrows():
            # Try to find layout_path in various possible column names
            layout_path = row.get("layout_path") or row.get("rgb_path") or row.get("image_path")
            if layout_path and pd.notna(layout_path) and layout_path != "":
                # Ensure absolute path
                if not Path(layout_path).is_absolute():
                    manifest_dir = input_manifest_abs.parent
                    layout_path = str((manifest_dir / layout_path).resolve())
                layout_rows.append({"layout_path": layout_path})
        
        if not layout_rows:
            raise ValueError("No layout paths found in input manifest! Check for 'layout_path', 'rgb_path', or 'image_path' columns.")
        
        temp_df = pd.DataFrame(layout_rows)
        temp_df.to_csv(temp_manifest, index=False)
        print(f"Created temporary manifest with {len(layout_rows)} layout paths")
        
        # Embed layouts using VAE
        temp_output_manifest = output_manifest_abs.parent / "temp_embedded_latents.csv"
        
        cmd = [
            sys.executable,
            str(embed_script),
            "--type", "layout",
            "--manifest", str(temp_manifest),
            "--output", str(output_manifest_abs.parent),  # Required: output directory
            "--output-manifest", str(temp_output_manifest),  # Actual manifest path
            "--output-latent-dir", str(latents_dir),  # Save latents in experiment folder
            "--autoencoder-config", str(ae_config_abs),
            "--autoencoder-checkpoint", str(ae_checkpoint_abs),
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
            "--device", "cuda"
        ]
        
        print(f"Running embedding command...")
        result = subprocess.run(cmd, check=True, cwd=base_dir)
        
        # Read embedded latents manifest
        if not temp_output_manifest.exists():
            raise FileNotFoundError(f"Embedded manifest not created: {temp_output_manifest}")
        
        embedded_df = pd.read_csv(temp_output_manifest, low_memory=False)
        
        # Create mapping: use absolute paths for matching
        latent_mapping = {}
        for _, emb_row in embedded_df.iterrows():
            emb_layout_path = emb_row.get("layout_path", "")
            emb_latent_path = emb_row.get("latent_path", "")
            if emb_layout_path and emb_latent_path:
                # Normalize paths for matching
                emb_layout_abs = str(Path(emb_layout_path).resolve())
                latent_mapping[emb_layout_abs] = emb_latent_path
        
        # Merge with original ControlNet manifest
        output_rows = []
        for _, row in df.iterrows():
            output_row = row.to_dict()
            
            # Add latent_path from VAE embedding
            layout_path = row.get("layout_path") or row.get("rgb_path") or row.get("image_path")
            if layout_path and pd.notna(layout_path) and layout_path != "":
                # Normalize path for matching
                if not Path(layout_path).is_absolute():
                    manifest_dir = input_manifest_abs.parent
                    layout_path_abs = str((manifest_dir / layout_path).resolve())
                else:
                    layout_path_abs = str(Path(layout_path).resolve())
                
                if layout_path_abs in latent_mapping:
                    output_row["latent_path"] = latent_mapping[layout_path_abs]
                else:
                    # Try direct match as fallback
                    if layout_path in latent_mapping:
                        output_row["latent_path"] = latent_mapping[layout_path]
                    else:
                        output_row["latent_path"] = ""
                        print(f"Warning: No latent found for layout_path: {layout_path}")
            else:
                output_row["latent_path"] = ""
            
            # Preserve existing embeddings (graph_embedding_path, pov_embedding_path)
            # These should already be in the row
            
            output_rows.append(output_row)
        
        # Write final manifest
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_manifest_abs, index=False)
        
        # Clean up temp files
        for temp_file in [temp_manifest, temp_output_manifest]:
            if temp_file.exists():
                temp_file.unlink()
        
        print(f"\n{'='*60}")
        print("Dataset embedding completed successfully!")
        print(f"Output manifest: {output_manifest_abs}")
        print(f"  - latent_path: Added from 32x32 VAE")
        print(f"  - graph_embedding_path: Preserved from ControlNet dataset")
        print(f"  - pov_embedding_path: Preserved from ControlNet dataset")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset embedding failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return False


def calculate_and_update_scale_factor(diffusion_config_path, embedded_manifest_path):
    """
    Calculate scale_factor from embedded latents and update config.
    
    Args:
        diffusion_config_path: Path to diffusion config
        embedded_manifest_path: Path to manifest with embedded latents
        
    Returns:
        Calculated scale_factor, or None if failed
    """
    print(f"\n{'='*60}")
    print("PHASE 3: Calculating Scale Factor")
    print(f"{'='*60}")
    
    try:
        # Build dataset to calculate scale factor
        from models.datasets.datasets import ManifestDataset
        
        dataset_config = {
            "manifest": str(embedded_manifest_path),
            "outputs": {"latent": "latent_path"},
            "filters": {"is_empty": [False]},
            "return_path": False
        }
        
        dataset = ManifestDataset.from_config(dataset_config)
        
        # Calculate scale factor
        scale_factor = calculate_scale_factor_from_dataset(
            dataset,
            num_samples=100,
            seed=42
        )
        
        # Update config
        with open(diffusion_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config["scale_factor"] = scale_factor
        
        with open(diffusion_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"Calculated scale_factor: {scale_factor:.6f}")
        print(f"Updated config: {diffusion_config_path}")
        print(f"{'='*60}\n")
        
        return scale_factor
        
    except Exception as e:
        print(f"WARNING: Failed to calculate scale_factor: {e}")
        print("  Will use default scale_factor=1.0 or value from config")
        print(f"{'='*60}\n")
        return None


class NumpySafeLoader(yaml.SafeLoader):
    """Custom YAML loader that handles numpy scalar types."""
    pass


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


# Register the constructor for numpy scalar types
NumpySafeLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor
)


def update_diffusion_config(
    diffusion_config_path,
    ae_checkpoint_path,
    embedded_manifest_path,
    scale_factor=None
):
    """
    Update diffusion config with VAE checkpoint, embedded manifest, and scale_factor.
    
    Args:
        diffusion_config_path: Path to diffusion config
        ae_checkpoint_path: Path to VAE checkpoint
        embedded_manifest_path: Path to embedded manifest
        scale_factor: Optional scale_factor to add
    """
    print(f"\n{'='*60}")
    print("PHASE 4: Updating Diffusion Config")
    print(f"{'='*60}")
    
    # Try loading with custom loader first, fallback to FullLoader if it fails
    try:
        with open(diffusion_config_path, 'r') as f:
            config = yaml.load(f, Loader=NumpySafeLoader)
    except (yaml.constructor.ConstructorError, yaml.YAMLError) as e:
        # If custom loader fails (e.g., other numpy types), try FullLoader
        print(f"Warning: Custom loader failed, trying FullLoader: {e}")
        with open(diffusion_config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Update autoencoder checkpoint
    if 'autoencoder' not in config:
        config['autoencoder'] = {}
    config['autoencoder']['checkpoint'] = str(Path(ae_checkpoint_path).resolve())
    
    # Update dataset manifest
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['manifest'] = str(Path(embedded_manifest_path).resolve())
    config['dataset']['outputs'] = {
        'latent': 'latent_path',
        'text_emb': 'graph_embedding_path',
        'pov_emb': 'pov_embedding_path'
    }
    
    # Update scale_factor if provided
    if scale_factor is not None:
        config['scale_factor'] = scale_factor
    
    # Save updated config
    with open(diffusion_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Updated config:")
    print(f"  - autoencoder.checkpoint: {ae_checkpoint_path}")
    print(f"  - dataset.manifest: {embedded_manifest_path}")
    print(f"  - dataset.outputs: latent, text_emb, pov_emb")
    if scale_factor:
        print(f"  - scale_factor: {scale_factor:.6f}")
    print(f"{'='*60}\n")


def train_diffusion(diffusion_config_path):
    """
    Train conditional cross-attention diffusion model.
    
    Args:
        diffusion_config_path: Path to diffusion config
        
    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 5: Training Conditional Cross-Attention Diffusion")
    print(f"{'='*60}")
    print(f"Config: {diffusion_config_path}")
    print(f"{'='*60}\n")
    
    try:
        base_dir = Path(__file__).parent.parent
        diffusion_config_abs = Path(diffusion_config_path).resolve()
        train_script = base_dir / "training" / "train_diffusion.py"
        
        result = subprocess.run(
            [sys.executable, str(train_script), str(diffusion_config_abs), "--resume"],
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
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for conditional cross-attention diffusion"
    )
    parser.add_argument(
        "--ae-config",
        type=Path,
        required=True,
        help="Path to 32x32 VAE autoencoder config YAML"
    )
    parser.add_argument(
        "--diffusion-config",
        type=Path,
        required=True,
        help="Path to conditional cross-attention diffusion config YAML"
    )
    parser.add_argument(
        "--controlnet-manifest",
        type=Path,
        help="Path to ControlNet manifest with embeddings (default: from diffusion config)"
    )
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip VAE training (use existing checkpoint)"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip dataset embedding (use existing embedded manifest)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip diffusion training (only do VAE and embedding)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Conditional Cross-Attention Diffusion Pipeline")
    print("="*60)
    print(f"VAE config: {args.ae_config}")
    print(f"Diffusion config: {args.diffusion_config}")
    print("="*60)
    print()
    
    # Load configs
    ae_config = load_config(args.ae_config)
    diffusion_config = load_config(args.diffusion_config)
    
    # Get paths
    ae_exp_name = ae_config.get("experiment", {}).get("name", "unnamed")
    ae_save_path = Path(ae_config.get("experiment", {}).get("save_path", "outputs"))
    
    diffusion_exp_name = diffusion_config.get("experiment", {}).get("name", "unnamed")
    diffusion_save_path = Path(diffusion_config.get("experiment", {}).get("save_path", "outputs"))
    
    # Get ControlNet manifest path
    if args.controlnet_manifest:
        controlnet_manifest = Path(args.controlnet_manifest).resolve()
    else:
        # Try to get from diffusion config
        controlnet_manifest = Path(diffusion_config.get("dataset", {}).get("manifest", ""))
        if not controlnet_manifest.exists():
            # Default ControlNet manifest
            controlnet_manifest = Path("/work3/s233249/ImgiNav/experiments/controlnet/new_layouts/controlnet_unet48_d4_new_layouts_seg/manifest_with_embeddings.csv")
    
    if not controlnet_manifest.exists():
        raise FileNotFoundError(f"ControlNet manifest not found: {controlnet_manifest}")
    
    # Output manifest (with embedded latents + preserved embeddings)
    embeddings_dir = diffusion_save_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedded_manifest = embeddings_dir / "manifest_with_latents_and_embeddings.csv"
    
    # Step 1: Train VAE (or find existing)
    ae_checkpoint = None
    if not args.skip_ae:
        ae_checkpoint = train_autoencoder(args.ae_config)
        if ae_checkpoint is None:
            print("ERROR: VAE training failed or checkpoint not found")
            sys.exit(1)
    else:
        ae_checkpoint = find_ae_checkpoint(ae_config)
        if ae_checkpoint is None:
            print("ERROR: --skip-ae specified but no checkpoint found")
            sys.exit(1)
        print(f"Using existing VAE checkpoint: {ae_checkpoint}")
    
    # Step 2: Embed dataset
    if not args.skip_embedding:
        success = embed_controlnet_dataset_with_vae(
            ae_checkpoint,
            args.ae_config,
            controlnet_manifest,
            embedded_manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        if not success:
            print("ERROR: Dataset embedding failed")
            sys.exit(1)
    else:
        if not embedded_manifest.exists():
            print(f"ERROR: --skip-embedding specified but embedded manifest not found: {embedded_manifest}")
            sys.exit(1)
        print(f"Using existing embedded manifest: {embedded_manifest}")
    
    # Step 3: Calculate scale factor
    scale_factor = calculate_and_update_scale_factor(
        args.diffusion_config,
        embedded_manifest
    )
    
    # Step 4: Update diffusion config
    update_diffusion_config(
        args.diffusion_config,
        ae_checkpoint,
        embedded_manifest,
        scale_factor
    )
    
    # Step 5: Train diffusion
    if not args.skip_training:
        success = train_diffusion(args.diffusion_config)
        if not success:
            print("ERROR: Diffusion training failed")
            sys.exit(1)
    else:
        print("Skipping diffusion training (--skip-training)")
    
    print("\n" + "="*60)
    print("Pipeline COMPLETE - SUCCESS")
    print("="*60)
    print(f"VAE checkpoint: {ae_checkpoint}")
    print(f"Embedded manifest: {embedded_manifest}")
    print(f"Scale factor: {scale_factor:.6f if scale_factor else 'N/A'}")
    print(f"Diffusion config: {args.diffusion_config}")
    print("="*60)


if __name__ == "__main__":
    main()

