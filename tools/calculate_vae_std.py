#!/usr/bin/env python3
"""
Calculate VAE latent standard deviation to determine correct scale_factor for diffusion model.

This script:
1. Loads a manifest CSV file
2. Samples ~100 random latent files (.pt)
3. Calculates the global standard deviation
4. Prints the recommended scale_factor (1.0 / std)
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
import argparse
import random


def calculate_vae_std(manifest_path, latent_column="latent_path", num_samples=100, seed=42):
    """
    Calculate standard deviation of VAE latents from manifest.
    
    Args:
        manifest_path: Path to CSV manifest file
        latent_column: Name of column containing latent file paths
        num_samples: Number of random samples to use (default: 100)
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (std, scale_factor, num_loaded)
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load manifest
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, low_memory=False)
    
    # Check for latent column
    if latent_column not in df.columns:
        raise ValueError(f"Column '{latent_column}' not found in manifest. Available columns: {list(df.columns)}")
    
    # Filter out rows with missing latent paths
    df = df.dropna(subset=[latent_column])
    df = df[df[latent_column].str.strip() != ""]
    
    if len(df) == 0:
        raise ValueError(f"No valid latent paths found in manifest")
    
    print(f"Found {len(df)} rows with latent paths")
    
    # Sample random indices
    num_samples = min(num_samples, len(df))
    sampled_indices = random.sample(range(len(df)), num_samples)
    
    print(f"Sampling {num_samples} random latents...")
    
    # Load and collect latents
    manifest_dir = manifest_path.parent
    all_latent_values = []
    loaded_count = 0
    failed_count = 0
    
    for idx in sampled_indices:
        row = df.iloc[idx]
        latent_path_str = row[latent_column]
        
        # Resolve path (handle relative paths)
        latent_path = Path(latent_path_str)
        if not latent_path.is_absolute():
            latent_path = manifest_dir / latent_path
        
        if not latent_path.exists():
            print(f"Warning: Latent file not found: {latent_path}")
            failed_count += 1
            continue
        
        try:
            # Load latent tensor
            latent = torch.load(latent_path, map_location="cpu")
            
            # Convert to numpy and flatten
            if isinstance(latent, torch.Tensor):
                latent_np = latent.numpy()
            else:
                latent_np = np.array(latent)
            
            # Flatten to 1D array
            latent_flat = latent_np.flatten()
            all_latent_values.append(latent_flat)
            loaded_count += 1
            
            if loaded_count % 10 == 0:
                print(f"  Loaded {loaded_count}/{num_samples} latents...")
        
        except Exception as e:
            print(f"Warning: Failed to load {latent_path}: {e}")
            failed_count += 1
            continue
    
    if loaded_count == 0:
        raise RuntimeError("Failed to load any latent files")
    
    print(f"\nSuccessfully loaded {loaded_count} latents")
    if failed_count > 0:
        print(f"Failed to load {failed_count} latents")
    
    # Concatenate all latents and calculate global statistics
    print("Calculating statistics...")
    all_latents = np.concatenate(all_latent_values)
    
    mean = np.mean(all_latents)
    std = np.std(all_latents)
    min_val = np.min(all_latents)
    max_val = np.max(all_latents)
    
    # Calculate scale factor (1.0 / std to normalize to unit variance)
    scale_factor = 1.0 / std if std > 0 else 1.0
    
    # Print results
    print("\n" + "="*60)
    print("VAE Latent Statistics")
    print("="*60)
    print(f"Number of samples: {loaded_count}")
    print(f"Total values: {len(all_latents):,}")
    print(f"Mean: {mean:.6f}")
    print(f"Standard Deviation: {std:.6f}")
    print(f"Min: {min_val:.6f}")
    print(f"Max: {max_val:.6f}")
    print("="*60)
    print(f"\nRecommended scale_factor: {scale_factor:.6f}")
    print(f"  (This will normalize latents to unit variance)")
    print("\nAdd this to your diffusion config:")
    print(f"  scale_factor: {scale_factor:.6f}")
    print("="*60)
    
    return std, scale_factor, loaded_count


def main():
    parser = argparse.ArgumentParser(
        description="Calculate VAE latent standard deviation for diffusion model scaling"
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to CSV manifest file with latent_path column"
    )
    parser.add_argument(
        "--latent-column",
        type=str,
        default="latent_path",
        help="Name of column containing latent file paths (default: latent_path)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples to use (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        std, scale_factor, num_loaded = calculate_vae_std(
            args.manifest_path,
            latent_column=args.latent_column,
            num_samples=args.num_samples,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

