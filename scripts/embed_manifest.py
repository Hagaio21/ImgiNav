#!/usr/bin/env python3
"""
Create layout embeddings from manifest CSV (manifest-based workflow).

This script is the preferred workflow for diffusion training.
It reads a manifest CSV, encodes RGB images to latents using an autoencoder,
and outputs a new manifest with latent_path column.

Usage:
    python scripts/embed_manifest.py \
        --manifest datasets/manifest.csv \
        --output-manifest datasets/manifest_with_latents.csv \
        --autoencoder-config config.yaml \
        --autoencoder-checkpoint checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from training.utils import load_config
from data_preparation.create_embeddings import create_layout_embeddings_from_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create layout embeddings from manifest CSV"
    )
    
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to input manifest CSV"
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Path for output manifest CSV with latent_path column"
    )
    parser.add_argument(
        "--autoencoder-config",
        required=True,
        help="Path to Autoencoder config YAML"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        required=True,
        help="Path to Autoencoder checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output-latent-dir",
        default=None,
        help="Directory to save latent files (default: uses dataset structure)"
    )
    parser.add_argument(
        "--diffusion-config",
        default=None,
        help="Path to diffusion config YAML (optional, used to get filters)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files"
    )
    
    args = parser.parse_args()
    
    # Load autoencoder model
    print(f"[INFO] Loading autoencoder from {args.autoencoder_checkpoint}")
    model = Autoencoder.load_checkpoint(args.autoencoder_checkpoint, map_location=args.device)
    model = model.to(args.device)
    model.eval()
    
    # Create embeddings from manifest
    create_layout_embeddings_from_manifest(
        encoder=model.encoder,
        manifest_path=args.manifest,
        output_manifest_path=args.output_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        device=args.device,
        autoencoder_config_path=args.autoencoder_config,
        output_latent_dir=args.output_latent_dir,
        diffusion_config_path=args.diffusion_config
    )


if __name__ == "__main__":
    main()

