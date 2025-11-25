#!/usr/bin/env python3
"""
Create layout embeddings by scanning directory (legacy workflow for inference/testing).

This script scans a directory for layout images and creates embeddings.
Used for inference/testing on raw folders.

Usage:
    python scripts/embed_folder.py \
        --data-root datasets/scenes \
        --autoencoder-config config.yaml \
        --autoencoder-checkpoint checkpoint.pt \
        --output-manifest datasets/embeddings_manifest.csv
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from training.utils import load_config
from data_preparation.create_embeddings import create_layout_embeddings_from_directory


def main():
    parser = argparse.ArgumentParser(
        description="Create layout embeddings by scanning directory"
    )
    
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root folder containing scenes/rooms with layout images"
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
        "--output-manifest",
        help="Output manifest CSV path (optional)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
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
    
    # Create embeddings from directory
    manifest_data = create_layout_embeddings_from_directory(
        model=model,
        data_root=Path(args.data_root),
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite
    )
    
    # Save manifest if requested
    if args.output_manifest and manifest_data:
        import pandas as pd
        df = pd.DataFrame(manifest_data)
        cols = ["scene", "type", "room_id", "layout_path", "layout_emb_path"]
        df = df[cols]
        output_path = Path(args.output_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, sep="|")
        print(f"\n[INFO] Manifest saved to {output_path}")


if __name__ == "__main__":
    main()

