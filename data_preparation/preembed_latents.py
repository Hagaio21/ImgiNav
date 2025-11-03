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
    
    # Apply the same filters that ManifestDataset will use
    # Filter out rows with NaN values in required columns
    required_cols = ["layout_path"]
    df = df.dropna(subset=required_cols)
    
    # Apply filters (same as ManifestDataset)
    filters = {"is_empty": [False]}
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            df = df[df[key].isin(value)]
        else:
            df = df[df[key] == value]
    df = df.reset_index(drop=True)
    
    print(f"After filtering (non-empty layouts): {len(df)} samples")
    
    # Create dataset for loading images (using filtered manifest)
    dataset = ManifestDataset(
        manifest=str(manifest_path),
        outputs={"rgb": "layout_path"},
        filters={"is_empty": [False]},
        return_path=False
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    
    # Verify that filtered dataframe matches dataset length
    if len(df) != len(dataset):
        raise ValueError(
            f"Mismatch between filtered dataframe length ({len(df)}) "
            f"and dataset length ({len(dataset)}). "
            "They should match after applying the same filters."
        )
    
    device = next(encoder.parameters()).device
    
    # Process all samples
    latent_paths = []
    processed = 0
    skipped = 0
    failed = 0
    
    # Collect latents for statistics computation
    all_latents = []
    
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
            
            # Collect latents for statistics (detach and move to CPU to save memory)
            all_latents.append(latents.detach().cpu())
            
            # Save latents for each sample in batch
            batch_start_idx = batch_idx * batch_size
            for i in range(len(rgb_images)):
                sample_idx = batch_start_idx + i
                if sample_idx >= len(df):
                    break
                
                row = df.iloc[sample_idx]
                
                # Determine output path for latent
                layout_path_str = row.get("layout_path", "")
                if not layout_path_str:
                    latent_paths.append("")
                    failed += 1
                    continue
                
                layout_path = Path(layout_path_str)
                
                # Convert to absolute path if relative
                if not layout_path.is_absolute():
                    # If relative, resolve relative to manifest directory
                    layout_path = (manifest_dir / layout_path).resolve()
                
                # For augmented dataset: create latents in /work3/s233249/ImgiNav/datasets/augmented/latents/
                # Structure: images/name.png -> latents/name.pt
                # Find the augmented directory in the path
                latent_base_dir = None
                if "augmented" in layout_path.parts:
                    # Find the augmented directory and use it as base
                    parts = list(layout_path.parts)
                    for i, part in enumerate(parts):
                        if part == "augmented":
                            # Use everything up to and including "augmented" as base
                            latent_base_dir = Path(*parts[:i+1])
                            break
                
                if latent_base_dir is None:
                    # Fallback: use manifest directory's parent (assuming it's in datasets/augmented/)
                    latent_base_dir = manifest_dir.parent if manifest_dir.name != "augmented" else manifest_dir
                
                # Create latent path: augmented/latents/filename.pt
                latent_dir = latent_base_dir / "latents"
                
                # Get filename from layout_path and change extension
                if layout_path.name:
                    latent_filename = Path(layout_path.name).with_suffix('.pt')
                    latent_path_full = latent_dir / latent_filename
                else:
                    # Fallback: use hash of path or index
                    latent_filename = f"latent_{sample_idx}.pt"
                    latent_path_full = latent_dir / latent_filename
                
                # Ensure path is absolute (for cross-environment compatibility)
                if not latent_path_full.is_absolute():
                    latent_path_full = latent_path_full.resolve()
                
                # Skip if exists and not overwriting
                if latent_path_full.exists() and not overwrite:
                    latent_paths.append(str(latent_path_full.resolve()))
                    skipped += 1
                    continue
                
                # Save latent tensor
                try:
                    latent_path_full.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(latents[i].cpu(), latent_path_full)
                    latent_paths.append(str(latent_path_full.resolve()))
                    processed += 1
                except Exception as e:
                    print(f"Error saving latent for {layout_path}: {e}")
                    latent_paths.append("")
                    failed += 1
    
    # Create output manifest with latent_path column
    # Ensure all paths are absolute for cross-environment compatibility
    df_output = df.copy()
    df_output["latent_path"] = ""
    
    for idx in range(len(df_output)):
        if latent_paths[idx]:
            path_obj = Path(latent_paths[idx])
            # Convert to absolute if not already
            if not path_obj.is_absolute():
                path_obj = (manifest_dir / path_obj).resolve()
            else:
                path_obj = path_obj.resolve()
            df_output.loc[idx, "latent_path"] = str(path_obj)
    
    # Ensure layout_path column is also absolute for consistency
    if "layout_path" in df_output.columns:
        df_output["layout_path"] = df_output["layout_path"].apply(
            lambda p: str((manifest_dir / Path(p)).resolve()) if p and not Path(p).is_absolute() else str(p) if p else ""
        )
    
    # Save new manifest
    df_output.to_csv(output_manifest_path, index=False)
    
    # Compute latent statistics
    print("\nComputing latent statistics...")
    if all_latents:
        # Concatenate all latents
        all_latents_tensor = torch.cat(all_latents, dim=0)
        
        # Flatten for global statistics
        latent_flat = all_latents_tensor.reshape(all_latents_tensor.shape[0], -1)
        
        # Compute global statistics
        latent_mean = latent_flat.mean().item()
        latent_std = latent_flat.std().item()
        
        # Compute per-channel statistics (if spatial dimensions exist)
        if all_latents_tensor.ndim == 4:  # [B, C, H, W]
            per_channel_mean = all_latents_tensor.mean(dim=(0, 2, 3)).cpu().numpy()  # [C]
            per_channel_std = all_latents_tensor.std(dim=(0, 2, 3)).cpu().numpy()  # [C]
            
            print(f"\n{'='*60}")
            print(f"Latent Statistics Summary")
            print(f"{'='*60}")
            print(f"Global Statistics:")
            print(f"  Mean: {latent_mean:.6f} (target: 0.0)")
            print(f"  Std:  {latent_std:.6f} (target: 1.0)")
            print(f"  Mean deviation: {abs(latent_mean):.6f}")
            print(f"  Std deviation:  {abs(latent_std - 1.0):.6f}")
            print(f"\nPer-Channel Statistics:")
            print(f"  Channel | Mean      | Std       | Mean Dev | Std Dev")
            print(f"  {'-'*55}")
            for ch_idx in range(len(per_channel_mean)):
                ch_mean = per_channel_mean[ch_idx]
                ch_std = per_channel_std[ch_idx]
                mean_dev = abs(ch_mean)
                std_dev = abs(ch_std - 1.0)
                print(f"  {ch_idx:7d} | {ch_mean:9.6f} | {ch_std:9.6f} | {mean_dev:9.6f} | {std_dev:9.6f}")
            print(f"{'='*60}")
            
            # Check if standardized
            if abs(latent_mean) < 0.1 and 0.9 < latent_std < 1.1:
                print("✓ Latents appear to be well-standardized (~N(0,1))")
            elif abs(latent_mean) < 0.5 and 0.5 < latent_std < 1.5:
                print("⚠ Latents are somewhat standardized but could be better")
            else:
                print("✗ Latents are NOT well-standardized - may need retraining")
        else:
            print(f"\nGlobal Statistics:")
            print(f"  Mean: {latent_mean:.6f} (target: 0.0)")
            print(f"  Std:  {latent_std:.6f} (target: 1.0)")
    else:
        print("⚠ No latents collected for statistics (all skipped?)")
    
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

