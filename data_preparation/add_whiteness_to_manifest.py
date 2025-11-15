#!/usr/bin/env python3
"""
Script to add whiteness_ratio column to manifest CSV.

The whiteness_ratio column indicates what fraction of pixels in an image are "white"
(where all RGB channels are above a threshold, default 0.9 in [0,1] range = ~230/255).

This allows filtering out overly empty/white images during training.

Usage:
    # Update manifest in-place (recommended)
    python data_preparation/add_whiteness_to_manifest.py \
        --manifest datasets/layouts.csv \
        --layout-column layout_path
    
    # Or save to a new file
    python data_preparation/add_whiteness_to_manifest.py \
        --manifest datasets/layouts.csv \
        --output datasets/layouts_with_whiteness.csv \
        --layout-column layout_path
    
    # Customize white threshold (default: 0.9 = ~230/255)
    python data_preparation/add_whiteness_to_manifest.py \
        --manifest datasets/layouts.csv \
        --layout-column layout_path \
        --white-threshold 0.85
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def compute_whiteness_ratio(image_path, white_threshold=0.9, manifest_dir=None):
    """
    Compute the ratio of white pixels in an image.
    
    A pixel is considered "white" if all RGB channels are above white_threshold.
    
    Args:
        image_path: Path to image file
        white_threshold: Pixel value threshold for "white" in [0, 1] range (default: 0.9 = ~230/255)
        manifest_dir: Directory of manifest (for resolving relative paths)
    
    Returns:
        float: Ratio of white pixels (0.0 to 1.0)
    """
    image_path = Path(image_path)
    
    # Handle relative paths
    if not image_path.is_absolute() and manifest_dir:
        image_path = manifest_dir / image_path
    
    if not image_path.exists():
        # Try parent directory
        if manifest_dir:
            image_path = manifest_dir.parent / image_path
            if not image_path.exists():
                return None
    
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.ascontiguousarray(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_array)
        
        # Convert to [C, H, W] format if needed
        if img_tensor.ndim == 3 and img_tensor.shape[2] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        # Normalize to [0, 1] range
        img_tensor = img_tensor / 255.0
        
        # Check if all channels (R, G, B) are above threshold for each pixel
        # Shape: [C, H, W] where C=3 (RGB)
        white_mask = (img_tensor > white_threshold).all(dim=0)  # All channels > threshold
        
        # Count white pixels
        total_pixels = white_mask.numel()
        white_pixels = white_mask.sum().item()
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return white_ratio
    
    except Exception as e:
        print(f"Warning: Error processing {image_path}: {e}")
        return None


def analyze_single_image(args_tuple):
    """
    Analyze a single image. Wrapper for multiprocessing.
    
    Args:
        args_tuple: (index, image_path_str, white_threshold, manifest_dir)
    
    Returns:
        (index, whiteness_ratio)
    """
    idx, image_path_str, white_threshold, manifest_dir = args_tuple
    whiteness_ratio = compute_whiteness_ratio(image_path_str, white_threshold, manifest_dir)
    return idx, whiteness_ratio


def add_whiteness_to_manifest(
    manifest_path: Path,
    output_path: Path = None,
    layout_column: str = "layout_path",
    white_threshold: float = 0.9,
    workers: int = None,
    overwrite: bool = False
):
    """
    Add whiteness_ratio column to manifest CSV.
    
    Args:
        manifest_path: Path to input manifest CSV
        output_path: Path to output manifest CSV (if None, overwrites input)
        layout_column: Name of column containing image paths
        white_threshold: Pixel value threshold for "white" in [0, 1] range
        workers: Number of parallel workers (None = auto)
        overwrite: Whether to overwrite existing whiteness_ratio column
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    
    if output_path is None:
        output_path = manifest_path
    else:
        output_path = Path(output_path)
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path, low_memory=False)
    print(f"Loaded {len(df)} samples")
    
    # Check if column already exists
    if "whiteness_ratio" in df.columns:
        if overwrite:
            print("Warning: whiteness_ratio column already exists, will be overwritten")
        else:
            print("Error: whiteness_ratio column already exists. Use --overwrite to replace it.")
            return
    
    # Check if layout column exists
    if layout_column not in df.columns:
        raise ValueError(f"Column '{layout_column}' not found in manifest. Available columns: {list(df.columns)}")
    
    # Filter out NaN paths
    valid_df = df[df[layout_column].notna()].copy()
    invalid_count = len(df) - len(valid_df)
    
    if invalid_count > 0:
        print(f"Warning: {invalid_count} rows have missing {layout_column}, will be skipped")
    
    # Prepare arguments for multiprocessing
    args_list = [
        (i, row[layout_column], white_threshold, manifest_dir)
        for i, row in valid_df.iterrows()
    ]
    
    # Initialize results
    results_dict = {}
    completed = 0
    total = len(valid_df)
    
    # Determine number of workers
    if workers is None:
        import os
        workers = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid memory issues
    
    print(f"\nComputing whiteness ratios using {workers} workers...")
    print(f"White threshold: {white_threshold} (~{int(white_threshold * 255)}/255)")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(analyze_single_image, args): i
            for i, args in enumerate(args_list)
        }
        
        # Collect results
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                idx, whiteness_ratio = future.result()
                results_dict[idx] = whiteness_ratio
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                results_dict[i] = None
            
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
    
    # Add column to dataframe
    df["whiteness_ratio"] = None
    
    for idx, whiteness_ratio in results_dict.items():
        original_idx = valid_df.index[idx]
        df.at[original_idx, "whiteness_ratio"] = whiteness_ratio
    
    # Fill missing values with NaN (will be filtered out if needed)
    df["whiteness_ratio"] = pd.to_numeric(df["whiteness_ratio"], errors='coerce')
    
    # Print statistics
    valid_ratios = df["whiteness_ratio"].dropna()
    if len(valid_ratios) > 0:
        print(f"\nWhiteness ratio statistics:")
        print(f"  Mean: {valid_ratios.mean():.4f}")
        print(f"  Median: {valid_ratios.median():.4f}")
        print(f"  Min: {valid_ratios.min():.4f}")
        print(f"  Max: {valid_ratios.max():.4f}")
        print(f"  Std: {valid_ratios.std():.4f}")
        
        # Show distribution
        print(f"\nWhiteness ratio distribution:")
        bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        for i in range(len(bins) - 1):
            count = ((valid_ratios >= bins[i]) & (valid_ratios < bins[i+1])).sum()
            pct = 100 * count / len(valid_ratios)
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:6d} ({pct:5.1f}%)")
        
        # Count very white images
        very_white = (valid_ratios >= 0.95).sum()
        print(f"\n  Very white images (≥95%): {very_white} ({100*very_white/len(valid_ratios):.1f}%)")
        
        # Suggest threshold
        print(f"\nSuggested filters:")
        print(f"  Remove very white (≥95%): whiteness_ratio__lt: 0.95")
        print(f"  Remove mostly white (≥90%): whiteness_ratio__lt: 0.90")
        print(f"  Remove quite white (≥80%): whiteness_ratio__lt: 0.80")
    else:
        print("\nWarning: No valid whiteness ratios computed!")
    
    # Save manifest
    print(f"\nSaving manifest to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved manifest with whiteness_ratio column")


def main():
    parser = argparse.ArgumentParser(
        description="Add whiteness_ratio column to manifest CSV"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to input manifest CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output manifest CSV (default: overwrites input)"
    )
    parser.add_argument(
        "--layout-column",
        type=str,
        default="layout_path",
        help="Name of column containing image paths (default: layout_path)"
    )
    parser.add_argument(
        "--white-threshold",
        type=float,
        default=0.9,
        help="Pixel value threshold for 'white' in [0, 1] range (default: 0.9 = ~230/255)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing whiteness_ratio column"
    )
    
    args = parser.parse_args()
    
    add_whiteness_to_manifest(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output) if args.output else None,
        layout_column=args.layout_column,
        white_threshold=args.white_threshold,
        workers=args.workers,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()

