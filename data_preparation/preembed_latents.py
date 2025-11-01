#!/usr/bin/env python3
"""
Pre-embed dataset images into latents using a trained autoencoder encoder.
Creates a new manifest with latent_path column pointing to saved .pt files.
"""

import argparse
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.utils import load_config


def load_autoencoder_encoder(config_path, checkpoint_path, device="cuda"):
    """Load autoencoder and return encoder only."""
    print(f"Loading autoencoder from {checkpoint_path}")
    
    # Load config
    config = load_config(config_path)
    ae_cfg = config.get("autoencoder", config)
    
    # Load model
    model = Autoencoder.load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    return model.encoder


def encode_dataset(encoder, manifest_path, output_manifest_path, batch_size=32, 
                   num_workers=8, overwrite=False):

    manifest_path = Path(manifest_path)
    output_manifest_path = Path(output_manifest_path)
    manifest_dir = manifest_path.parent
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} samples")
    
    # Create dataset for loading images
    dataset = ManifestDataset(
        manifest=str(manifest_path),
        outputs={"rgb": "layout_path"},
        filters={"is_empty": [False]},
        return_path=False
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    
    device = next(encoder.parameters()).device
    
    # Process all samples
    latent_paths = []
    processed = 0
    skipped = 0
    failed = 0
    
    output_manifest_dir = output_manifest_path.parent
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Encoding images to latents...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding")):
            rgb_images = batch["rgb"].to(device, non_blocking=True)
            
            # Encode to latents
            encoder_out = encoder(rgb_images)  # Returns dict
            
            # Extract latent from dict
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]  # Use mu for VAE
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
            
            # Save latents for each sample in batch
            batch_start_idx = batch_idx * batch_size
            for i in range(len(rgb_images)):
                sample_idx = batch_start_idx + i
                if sample_idx >= len(df):
                    break
                
                row = df.iloc[sample_idx]
                
                # Determine output path for latent
                # Use same directory structure as input images
                layout_path = Path(row.get("layout_path", ""))
                if not layout_path:
                    latent_paths.append("")
                    failed += 1
                    continue
                
                # Create latent filename: same name, different extension
                latent_path = layout_path.with_suffix('.pt')
                
                # Skip if exists and not overwriting
                if latent_path.exists() and not overwrite:
                    latent_paths.append(str(latent_path))
                    skipped += 1
                    continue
                
                # Save latent tensor
                try:
                    latent_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(latents[i].cpu(), latent_path)
                    latent_paths.append(str(latent_path))
                    processed += 1
                except Exception as e:
                    print(f"Error saving latent for {layout_path}: {e}")
                    latent_paths.append("")
                    failed += 1
    
    # Create output manifest with latent_path column
    df_output = df.copy()
    df_output["latent_path"] = latent_paths
    
    # Save new manifest
    df_output.to_csv(output_manifest_path, index=False)
    
    print(f"\n✓ Encoding complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output manifest: {output_manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-embed dataset images into latents using trained autoencoder"
    )
    parser.add_argument(
        "--autoencoder-config",
        type=str,
        required=True,
        help="Path to autoencoder config YAML"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=str,
        required=True,
        help="Path to trained autoencoder checkpoint"
    )
    parser.add_argument(
        "--dataset-manifest",
        type=str,
        required=True,
        help="Path to input dataset manifest CSV"
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Path to save output manifest with latent_path column"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers (default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing latent files"
    )
    
    args = parser.parse_args()
    
    # Load encoder
    encoder = load_autoencoder_encoder(
        args.autoencoder_config,
        args.autoencoder_checkpoint,
        device=args.device
    )
    
    # Encode dataset
    encode_dataset(
        encoder,
        args.dataset_manifest,
        args.output_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

