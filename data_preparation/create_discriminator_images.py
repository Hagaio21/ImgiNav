#!/usr/bin/env python3
"""
Create discriminator dataset with real and generated images.

This script:
1. Loads a diffusion model checkpoint and generates images
2. Copies positive (non-empty, valid) images from a manifest to the "real" directory
3. Saves generated images to the "generated" directory
4. Creates a manifest with valid/invalid labels

Usage:
    python data_preparation/create_discriminator_images.py \
        --manifest datasets/layouts.csv \
        --diffusion_checkpoint checkpoints/diffusion_best.pt \
        --diffusion_config experiments/diffusion/config.yaml \
        --output_dir data_preparation/discriminator \
        --num_real 5000 \
        --num_generated 5000
"""

import argparse
import sys
from pathlib import Path
import shutil
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from training.utils import load_config, set_deterministic
from torchvision.utils import save_image


def select_real_images(manifest_path, num_samples=5000, seed=42):
    """
    Select real (non-empty, non-augmented) images from manifest.
    
    Args:
        manifest_path: Path to manifest CSV
        num_samples: Number of real images to select
        seed: Random seed for selection
    
    Returns:
        DataFrame with selected real images
    """
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    
    print(f"Loaded manifest with {len(df)} total samples")
    
    # Filter non-empty layouts
    if "is_empty" in df.columns:
        df = df[df["is_empty"] == False].copy()
        print(f"After filtering empty: {len(df)} samples")
    
    # Filter out augmented images if possible
    if "is_augmented" in df.columns:
        df = df[df["is_augmented"] == False].copy()
        print(f"After filtering augmented: {len(df)} samples")
    else:
        # Try to identify augmented images by path patterns
        layout_col = "layout_path" if "layout_path" in df.columns else "path"
        if layout_col in df.columns:
            def is_augmented_path(path_str):
                if pd.isna(path_str):
                    return True
                path_str = str(path_str).lower()
                aug_patterns = ["_rot", "_mirror", "_aug", "rot90", "rot180", "rot270", 
                               "mirror_rot", "augmented"]
                return any(pattern in path_str for pattern in aug_patterns)
            
            df["_is_augmented"] = df[layout_col].apply(is_augmented_path)
            df = df[df["_is_augmented"] == False].copy()
            df = df.drop(columns=["_is_augmented"])
            print(f"After filtering augmented (by path): {len(df)} samples")
    
    # Ensure we have enough samples
    if len(df) < num_samples:
        print(f"Warning: Only {len(df)} real images available, using all of them")
        num_samples = len(df)
    
    # Randomly select num_samples
    if len(df) > num_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(df), size=num_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
    
    print(f"Selected {len(df)} real images")
    return df


def copy_real_images(df_real, output_dir, layout_column="layout_path"):
    """
    Copy real images to the "real" directory.
    
    Args:
        df_real: DataFrame with real images
        output_dir: Output directory (will create "real" subdirectory)
        layout_column: Column name for layout paths
    
    Returns:
        List of paths to copied images
    """
    output_dir = Path(output_dir)
    real_dir = output_dir / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    
    copied_paths = []
    failed = 0
    
    print(f"Copying {len(df_real)} real images to {real_dir}...")
    
    for idx, row in tqdm(df_real.iterrows(), total=len(df_real), desc="Copying real images"):
        layout_path = Path(row[layout_column])
        
        if not layout_path.exists():
            print(f"Warning: Image not found: {layout_path}")
            failed += 1
            continue
        
        try:
            # Verify it's a valid image
            img = Image.open(layout_path)
            img.verify()
            
            # Copy to output directory
            output_path = real_dir / f"real_{idx:05d}.png"
            shutil.copy2(layout_path, output_path)
            copied_paths.append(output_path)
        except Exception as e:
            print(f"Warning: Failed to copy {layout_path}: {e}")
            failed += 1
    
    print(f"Copied {len(copied_paths)} real images, {failed} failed")
    return copied_paths


def generate_images(
    diffusion_checkpoint,
    diffusion_config,
    num_samples,
    output_dir,
    batch_size=8,
    num_steps=50,
    device="cuda",
    seed=42
):
    """
    Generate images using diffusion model.
    
    Args:
        diffusion_checkpoint: Path to diffusion checkpoint
        diffusion_config: Path to diffusion config YAML
        num_samples: Number of images to generate
        output_dir: Output directory (will create "generated" subdirectory)
        batch_size: Batch size for generation
        num_steps: Number of DDIM steps
        device: Device to use
        seed: Random seed
    
    Returns:
        List of paths to generated images
    """
    set_deterministic(seed)
    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading diffusion model from {diffusion_checkpoint}")
    config = load_config(diffusion_config)
    
    # Load model
    model, _ = DiffusionModel.load_checkpoint(
        diffusion_checkpoint,
        map_location=device,
        return_extra=True,
        config=config
    )
    model = model.to(device_obj)
    model.eval()
    
    print(f"Generating {num_samples} images...")
    
    generated_paths = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating images"):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Generate samples
            sample_output = model.sample(
                batch_size=current_batch_size,
                num_steps=num_steps,
                method="ddim",
                eta=0.0,
                device=device_obj,
                verbose=False
            )
            
            # Get RGB images
            if "rgb" in sample_output:
                images = sample_output["rgb"]
            else:
                raise ValueError("Model should return RGB images")
            
            # Save individual images
            for i in range(images.shape[0]):
                idx = batch_idx * batch_size + i
                image_path = generated_dir / f"generated_{idx:05d}.png"
                
                # Save image (images are already in [0, 1] range)
                save_image(images[i], image_path, normalize=False)
                generated_paths.append(image_path)
    
    print(f"Generated {len(generated_paths)} images")
    return generated_paths


def create_manifest(real_paths, generated_paths, output_dir):
    """
    Create manifest CSV with valid/invalid labels.
    
    Args:
        real_paths: List of paths to real images
        generated_paths: List of paths to generated images
        output_dir: Output directory
    
    Returns:
        Path to created manifest
    """
    output_dir = Path(output_dir)
    
    manifest_entries = []
    
    # Real images (valid=1)
    for path in real_paths:
        manifest_entries.append({
            "image_path": str(path),
            "label": 1,  # 1 = valid/real
            "is_valid": True,
            "type": "real"
        })
    
    # Generated images (valid=0)
    for path in generated_paths:
        manifest_entries.append({
            "image_path": str(path),
            "label": 0,  # 0 = invalid/generated
            "is_valid": False,
            "type": "generated"
        })
    
    # Create DataFrame
    df = pd.DataFrame(manifest_entries)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save manifest
    manifest_path = output_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    
    print(f"\nCreated discriminator manifest with {len(df)} samples:")
    print(f"  Real (valid): {len(df[df['label'] == 1])} samples")
    print(f"  Generated (invalid): {len(df[df['label'] == 0])} samples")
    print(f"  Saved to: {manifest_path}")
    
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Create discriminator dataset with real and generated images")
    parser.add_argument("--manifest", type=Path, required=True,
                       help="Path to manifest CSV with real images")
    parser.add_argument("--diffusion_checkpoint", type=Path, required=True,
                       help="Path to diffusion checkpoint")
    parser.add_argument("--diffusion_config", type=Path, required=True,
                       help="Path to diffusion config YAML")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for discriminator dataset")
    parser.add_argument("--num_real", type=int, default=5000,
                       help="Number of real images to copy (default: 5000)")
    parser.add_argument("--num_generated", type=int, default=5000,
                       help="Number of generated images (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation (default: 8)")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of DDIM steps for generation (default: 50)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--layout_column", type=str, default="layout_path",
                       help="Column name for layout paths in manifest (default: layout_path)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Creating Discriminator Dataset")
    print("="*60)
    
    # Step 1: Select real images
    print("\n[Step 1/4] Selecting real images from manifest...")
    df_real = select_real_images(
        args.manifest, 
        num_samples=args.num_real, 
        seed=args.seed
    )
    
    # Step 2: Copy real images
    print("\n[Step 2/4] Copying real images to 'real' directory...")
    real_paths = copy_real_images(
        df_real,
        output_dir,
        layout_column=args.layout_column
    )
    
    # Step 3: Generate images
    print("\n[Step 3/4] Generating images using diffusion model...")
    generated_paths = generate_images(
        args.diffusion_checkpoint,
        args.diffusion_config,
        args.num_generated,
        output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed
    )
    
    # Step 4: Create manifest
    print("\n[Step 4/4] Creating discriminator manifest...")
    manifest_path = create_manifest(
        real_paths,
        generated_paths,
        output_dir
    )
    
    print("\n" + "="*60)
    print("Discriminator Dataset Created Successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - Real images: {output_dir / 'real/'} ({len(real_paths)} files)")
    print(f"  - Generated images: {output_dir / 'generated/'} ({len(generated_paths)} files)")
    print(f"  - Manifest: {manifest_path}")
    print(f"\nManifest columns:")
    print(f"  - image_path: Path to image")
    print(f"  - label: 1 for real/valid, 0 for generated/invalid")
    print(f"  - is_valid: Boolean flag")
    print(f"  - type: 'real' or 'generated'")


if __name__ == "__main__":
    main()

