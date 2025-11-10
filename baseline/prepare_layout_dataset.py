#!/usr/bin/env python3
"""
Prepare layout image dataset from manifest for SD fine-tuning.

Extracts layout images from manifest and saves them to a directory
for use with baseline/finetune_sd.py.

Usage:
    python baseline/prepare_layout_dataset.py \
        --manifest /path/to/manifest.csv \
        --output_dir datasets/sd_finetuning_images \
        --num_samples 5000 \
        --filter_empty
"""

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import is_augmented_path


def prepare_layout_dataset(
    manifest_path,
    output_dir,
    num_samples=None,
    filter_empty=True,
    filter_augmented=True,
    layout_column="layout_path"
):
    """
    Extract layout images from manifest for SD fine-tuning.
    
    Args:
        manifest_path: Path to manifest CSV
        output_dir: Directory to save images
        num_samples: Number of samples to extract (None = all)
        filter_empty: Filter out empty layouts
        filter_augmented: Filter out augmented images
        layout_column: Column name for layout paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} samples from manifest")
    
    # Filter empty layouts
    if filter_empty and "is_empty" in df.columns:
        df = df[df["is_empty"] == False].copy()
        print(f"After filtering empty: {len(df)} samples")
    
    # Filter augmented images
    if filter_augmented:
        if "is_augmented" in df.columns:
            df = df[df["is_augmented"] == False].copy()
            print(f"After filtering augmented: {len(df)} samples")
        else:
            # Filter by path pattern
            df["_is_augmented"] = df[layout_column].apply(is_augmented_path)
            df = df[df["_is_augmented"] == False].copy()
            df = df.drop(columns=["_is_augmented"])
            print(f"After filtering augmented (by path): {len(df)} samples")
    
    # Sample if requested
    if num_samples and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {num_samples} images")
    
    # Copy images
    print(f"\nCopying {len(df)} images to {output_dir}...")
    copied = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
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
            output_path = output_dir / f"layout_{idx:05d}.png"
            shutil.copy2(layout_path, output_path)
            copied += 1
        except Exception as e:
            print(f"Warning: Failed to copy {layout_path}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"  Copied: {copied} images")
    print(f"  Failed: {failed} images")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")
    print(f"\nNext step: Fine-tune SD using:")
    print(f"  python baseline/finetune_sd.py \\")
    print(f"    --dataset_dir {output_dir} \\")
    print(f"    --output_dir outputs/baseline_sd_finetuned")


def main():
    parser = argparse.ArgumentParser(description="Prepare layout dataset for SD fine-tuning")
    parser.add_argument("--manifest", type=Path, required=True,
                       help="Path to manifest CSV")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for images")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to extract (None = all)")
    parser.add_argument("--filter_empty", action="store_true", default=True,
                       help="Filter out empty layouts")
    parser.add_argument("--no_filter_empty", dest="filter_empty", action="store_false",
                       help="Don't filter empty layouts")
    parser.add_argument("--filter_augmented", action="store_true", default=True,
                       help="Filter out augmented images")
    parser.add_argument("--no_filter_augmented", dest="filter_augmented", action="store_false",
                       help="Don't filter augmented images")
    parser.add_argument("--layout_column", type=str, default="layout_path",
                       help="Column name for layout paths")
    
    args = parser.parse_args()
    
    prepare_layout_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        filter_empty=args.filter_empty,
        filter_augmented=args.filter_augmented,
        layout_column=args.layout_column
    )


if __name__ == "__main__":
    main()

