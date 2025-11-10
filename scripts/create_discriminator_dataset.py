#!/usr/bin/env python3
"""
Create discriminator dataset by:
1. Selecting 5000 real (non-augmented) images from manifest
2. Encoding them to get real latents
3. Generating 5000 "bad" latents from diffusion model
4. Saving both sets and creating a manifest with good/bad labels
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

from training.utils import load_config, set_deterministic, get_device
from models.diffusion import DiffusionModel
from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from torch.utils.data import DataLoader
from common.utils import is_augmented_path


def select_real_images(manifest_path, num_samples=5000, seed=42):
    """
    Select real (non-augmented) images from manifest.
    
    Args:
        manifest_path: Path to augmented manifest CSV
        num_samples: Number of real images to select
        seed: Random seed for selection
    
    Returns:
        DataFrame with selected real images
    """
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    
    print(f"Loaded manifest with {len(df)} total samples")
    
    # Filter for non-augmented images
    # Strategy: Look for images that are NOT augmented
    # Common patterns: augmented images might have "aug" in path, or there's an is_augmented column
    
    # Filter non-empty layouts first
    if "is_empty" in df.columns:
        df = df[df["is_empty"] == False].copy()
    
    # Check if there's an is_augmented column
    if "is_augmented" in df.columns:
        df_real = df[df["is_augmented"] == False].copy()
        print(f"Found {len(df_real)} non-augmented images (is_augmented=False)")
    else:
        # Check if layout_path contains "aug" or similar patterns
        # Original images typically don't have augmentation markers in path
        layout_col = "layout_path" if "layout_path" in df.columns else "path"
        
        # Filter out rows with NaN
        df = df.dropna(subset=[layout_col])
        
        # Try to identify augmented images by path patterns
        # Augmented images might have patterns like "_rot", "_mirror", "_aug", etc.
        if layout_col in df.columns:
            # Check for common augmentation patterns in filename
            df["_is_augmented"] = df[layout_col].apply(is_augmented_path)
            df_real = df[df["_is_augmented"] == False].copy()
            df_real = df_real.drop(columns=["_is_augmented"])
            print(f"Found {len(df_real)} non-augmented images (by path pattern)")
        else:
            # If we can't determine, use all images
            df_real = df.copy()
            print(f"Warning: Cannot identify augmented images, using all {len(df_real)} images")
    
    # Filter non-empty layouts (already done above, but double-check)
    if "is_empty" in df_real.columns:
        df_real = df_real[df_real["is_empty"] == False]
        print(f"After filtering empty: {len(df_real)} samples")
    
    # Ensure we have enough samples
    if len(df_real) < num_samples:
        print(f"Warning: Only {len(df_real)} real images available, using all of them")
        num_samples = len(df_real)
    
    # Randomly select num_samples
    if len(df_real) > num_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(df_real), size=num_samples, replace=False)
        df_real = df_real.iloc[indices].reset_index(drop=True)
    
    print(f"Selected {len(df_real)} real images")
    return df_real


def encode_real_images(df_real, autoencoder_checkpoint, output_dir, batch_size=32, device="cuda"):
    """
    Encode real images to latents.
    
    Args:
        df_real: DataFrame with real images
        autoencoder_checkpoint: Path to autoencoder checkpoint
        output_dir: Directory to save latents
        batch_size: Batch size for encoding
        device: Device to use
    
    Returns:
        List of paths to saved latent files
    """
    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    latents_dir = output_dir / "real_latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading autoencoder from {autoencoder_checkpoint}")
    autoencoder = Autoencoder.load_checkpoint(autoencoder_checkpoint, map_location=device)
    autoencoder = autoencoder.to(device_obj)
    autoencoder.eval()
    
    # Create temporary manifest for loading images
    temp_manifest = output_dir / "temp_real_manifest.csv"
    df_real.to_csv(temp_manifest, index=False)
    
    # Create dataset
    dataset = ManifestDataset(
        manifest=str(temp_manifest),
        outputs={"rgb": "layout_path"},
        return_path=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device_obj.type == "cuda"
    )
    
    latent_paths = []
    all_latents = []
    
    print(f"Encoding {len(df_real)} real images...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding")):
            rgb = batch["rgb"].to(device_obj)
            
            # Encode
            encoder_out = autoencoder.encoder({"rgb": rgb})
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
            
            # Save individual latents
            for i in range(latents.shape[0]):
                idx = batch_idx * batch_size + i
                if idx >= len(df_real):
                    break
                
                latent_path = latents_dir / f"real_latent_{idx:05d}.pt"
                torch.save(latents[i].cpu(), latent_path)
                latent_paths.append(latent_path)
            
            all_latents.append(latents.cpu())
    
    # Save all latents as single file
    all_latents_tensor = torch.cat(all_latents, dim=0)
    all_latents_path = output_dir / "real_latents_all.pt"
    torch.save(all_latents_tensor, all_latents_path)
    print(f"Saved all real latents to: {all_latents_path}")
    
    # Clean up temp manifest
    temp_manifest.unlink()
    
    return latent_paths, all_latents_path


def generate_bad_latents(
    diffusion_checkpoint,
    diffusion_config,
    num_samples,
    output_dir,
    batch_size=32,
    num_steps=100,
    device="cuda",
    seed=42
):
    """
    Generate bad latents from diffusion model.
    
    Args:
        diffusion_checkpoint: Path to diffusion checkpoint
        diffusion_config: Path to diffusion config
        num_samples: Number of bad latents to generate
        output_dir: Directory to save latents
        batch_size: Batch size for generation
        num_steps: Number of DDIM steps
        device: Device to use
        seed: Random seed
    
    Returns:
        Path to saved bad latents file
    """
    set_deterministic(seed)
    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    latents_dir = output_dir / "bad_latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading diffusion model from {diffusion_checkpoint}")
    config = load_config(diffusion_config)
    
    model, _ = DiffusionModel.load_checkpoint(
        diffusion_checkpoint,
        map_location=device,
        return_extra=True,
        config=config
    )
    model = model.to(device_obj)
    model.eval()
    
    print(f"Generating {num_samples} bad latents...")
    
    all_latents = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating bad latents"):
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
            
            # Get latents
            if "latent" in sample_output:
                latents = sample_output["latent"]
            else:
                raise ValueError("Model should return latents, not images")
            
            # Save individual latents
            for i in range(latents.shape[0]):
                idx = batch_idx * batch_size + i
                latent_path = latents_dir / f"bad_latent_{idx:05d}.pt"
                torch.save(latents[i].cpu(), latent_path)
            
            all_latents.append(latents.cpu())
    
    # Save all latents as single file
    all_latents_tensor = torch.cat(all_latents, dim=0)
    all_latents_path = output_dir / "bad_latents_all.pt"
    torch.save(all_latents_tensor, all_latents_path)
    print(f"Saved all bad latents to: {all_latents_path}")
    
    return all_latents_path


def create_discriminator_manifest(
    real_latent_paths,
    bad_latents_path,
    output_dir,
    num_real,
    num_bad
):
    """
    Create manifest for discriminator training with good/bad labels.
    
    Args:
        real_latent_paths: List of paths to real latent files
        bad_latents_path: Path to bad latents tensor file
        output_dir: Output directory
        num_real: Number of real latents
        num_bad: Number of bad latents
    
    Returns:
        Path to created manifest
    """
    output_dir = Path(output_dir)
    
    # Load bad latents to get individual paths
    bad_latents = torch.load(bad_latents_path)
    bad_latents_dir = bad_latents_path.parent / "bad_latents"
    
    # Create manifest entries
    manifest_entries = []
    
    # Real latents (label=1 for viable/good)
    for i, latent_path in enumerate(real_latent_paths):
        manifest_entries.append({
            "latent_path": str(latent_path),
            "label": 1,  # 1 = real/viable
            "is_viable": True
        })
    
    # Bad latents (label=0 for non-viable/bad)
    for i in range(num_bad):
        latent_path = bad_latents_dir / f"bad_latent_{i:05d}.pt"
        manifest_entries.append({
            "latent_path": str(latent_path),
            "label": 0,  # 0 = fake/non-viable
            "is_viable": False
        })
    
    # Create DataFrame
    df = pd.DataFrame(manifest_entries)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save manifest
    manifest_path = output_dir / "discriminator_manifest.csv"
    df.to_csv(manifest_path, index=False)
    
    print(f"\nCreated discriminator manifest with {len(df)} samples:")
    print(f"  Real (viable): {len(df[df['label'] == 1])} samples")
    print(f"  Bad (non-viable): {len(df[df['label'] == 0])} samples")
    print(f"  Saved to: {manifest_path}")
    
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Create discriminator dataset")
    parser.add_argument("--manifest", type=Path, required=True,
                       help="Path to augmented manifest CSV")
    parser.add_argument("--autoencoder_checkpoint", type=Path, required=True,
                       help="Path to autoencoder checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=Path, required=True,
                       help="Path to diffusion checkpoint")
    parser.add_argument("--diffusion_config", type=Path, required=True,
                       help="Path to diffusion config YAML")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for discriminator dataset")
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Number of real and bad samples (default: 5000 each)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding and generation")
    parser.add_argument("--num_steps", type=int, default=100,
                       help="Number of DDIM steps for generation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Creating Discriminator Dataset")
    print("="*60)
    
    # Step 1: Select real images
    print("\n[Step 1/4] Selecting real images from manifest...")
    df_real = select_real_images(args.manifest, num_samples=args.num_samples, seed=args.seed)
    
    # Step 2: Encode real images to latents
    print("\n[Step 2/4] Encoding real images to latents...")
    real_latent_paths, real_latents_all_path = encode_real_images(
        df_real,
        args.autoencoder_checkpoint,
        output_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Step 3: Generate bad latents
    print("\n[Step 3/4] Generating bad latents from diffusion model...")
    bad_latents_path = generate_bad_latents(
        args.diffusion_checkpoint,
        args.diffusion_config,
        args.num_samples,
        output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed
    )
    
    # Step 4: Create manifest
    print("\n[Step 4/4] Creating discriminator manifest...")
    manifest_path = create_discriminator_manifest(
        real_latent_paths,
        bad_latents_path,
        output_dir,
        num_real=len(real_latent_paths),
        num_bad=args.num_samples
    )
    
    print("\n" + "="*60)
    print("Discriminator Dataset Created Successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - Real latents: {output_dir / 'real_latents/'} ({len(real_latent_paths)} files)")
    print(f"  - Real latents (all): {real_latents_all_path}")
    print(f"  - Bad latents: {output_dir / 'bad_latents/'} ({args.num_samples} files)")
    print(f"  - Bad latents (all): {bad_latents_path}")
    print(f"  - Manifest: {manifest_path}")
    print(f"\nNext step: Train discriminator using:")
    print(f"  python training/train_discriminator.py \\")
    print(f"    --real_latents {real_latents_all_path} \\")
    print(f"    --fake_latents {bad_latents_path} \\")
    print(f"    --output_dir /path/to/discriminator_output")


if __name__ == "__main__":
    main()

