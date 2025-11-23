#!/usr/bin/env python3
"""
Simple script to embed layouts using CLIP VAEs.
Loads VAE from checkpoint, embeds layouts with 256x256 transform, updates manifest.
"""

import argparse
import pandas as pd
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder


def get_transform():
    """Get 256x256 transform as used in training."""
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def embed_layouts_with_vae(
    checkpoint_path,
    manifest_path,
    output_manifest_path,
    latent_dir,
    column_name,
    batch_size=32,
    num_workers=8,
    device="cuda"
):
    """
    Embed layouts using VAE and update manifest.
    
    Args:
        checkpoint_path: Path to VAE checkpoint
        manifest_path: Input manifest path
        output_manifest_path: Output manifest path (can be same as input)
        latent_dir: Directory to save latents
        column_name: Column name in manifest (e.g., "latent_path_vae_clip")
        batch_size: Batch size for encoding
        num_workers: Number of workers
        device: Device to use
    """
    print(f"\n{'='*60}")
    print(f"Embedding layouts with VAE")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_manifest_path}")
    print(f"Latent dir: {latent_dir}")
    print(f"Column: {column_name}")
    print(f"{'='*60}\n")
    
    # Load VAE from checkpoint
    print(f"Loading VAE from checkpoint...")
    autoencoder = Autoencoder.load_checkpoint(checkpoint_path, map_location="cpu")
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    print(f"✓ VAE loaded")
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} samples")
    
    # Filter rows with layout_path
    df = df.dropna(subset=["layout_path"])
    print(f"Found {len(df)} samples with layout_path")
    
    # Create latent directory
    latent_dir = Path(latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup transform
    transform = get_transform()
    
    # Process in batches
    latent_paths = []
    processed = 0
    failed = 0
    
    print(f"Embedding layouts...")
    for batch_idx in tqdm(range(0, len(df), batch_size), desc="Batches"):
        batch_df = df.iloc[batch_idx:batch_idx + batch_size]
        batch_images = []
        batch_indices = []
        
        # Load images
        for idx, row in batch_df.iterrows():
            layout_path = Path(row["layout_path"])
            if not layout_path.exists():
                print(f"Warning: Layout not found: {layout_path}")
                failed += 1
                continue
            
            try:
                img = Image.open(layout_path).convert("RGB")
                img_tensor = transform(img)
                batch_images.append(img_tensor)
                batch_indices.append(idx)
            except Exception as e:
                print(f"Error loading {layout_path}: {e}")
                failed += 1
                continue
        
        if not batch_images:
            continue
        
        # Stack batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Encode
        with torch.no_grad():
            encoder_out = autoencoder.encode(batch_tensor)
            
            # Extract latent
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
        
        # Save latents
        for i, orig_idx in enumerate(batch_indices):
            row = df.loc[orig_idx]
            layout_path = Path(row["layout_path"])
            
            # Generate latent filename from layout path
            layout_name = layout_path.stem
            latent_filename = f"{layout_name}.pt"
            latent_path = latent_dir / latent_filename
            
            # Save latent
            torch.save(latents[i].cpu(), latent_path)
            latent_paths.append((orig_idx, str(latent_path.resolve())))
            processed += 1
    
    print(f"✓ Processed {processed} layouts, {failed} failed")
    
    # Update manifest
    print(f"Updating manifest...")
    
    # Initialize column if it doesn't exist
    if column_name not in df.columns:
        df[column_name] = ""
    
    # Update latent paths
    for idx, latent_path in latent_paths:
        df.at[idx, column_name] = latent_path
    
    # Save manifest
    df.to_csv(output_manifest_path, index=False)
    print(f"✓ Manifest updated: {output_manifest_path}")
    
    # Cleanup
    del autoencoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return processed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Embed layouts using CLIP VAEs"
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=Path,
        required=True,
        help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="Input manifest path"
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Output manifest path (can be same as input)"
    )
    parser.add_argument(
        "--latent-dir",
        type=Path,
        required=True,
        help="Directory to save latents"
    )
    parser.add_argument(
        "--column-name",
        type=str,
        required=True,
        help="Column name in manifest (e.g., 'latent_path_vae_clip')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers (default: 8)"
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Layout-only mode (flag for compatibility)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.ae_checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.ae_checkpoint}")
        sys.exit(1)
    
    if not args.input_manifest.exists():
        print(f"ERROR: Manifest not found: {args.input_manifest}")
        sys.exit(1)
    
    # Ensure output directory exists
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    
    # Embed layouts
    processed, failed = embed_layouts_with_vae(
        checkpoint_path=args.ae_checkpoint,
        manifest_path=args.input_manifest,
        output_manifest_path=args.output_manifest,
        latent_dir=args.latent_dir,
        column_name=args.column_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if failed > 0:
        print(f"\nWARNING: {failed} layouts failed to process")
    
    print(f"\n{'='*60}")
    print(f"Embedding COMPLETE")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Column: {args.column_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

