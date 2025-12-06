#!/usr/bin/env python3
"""
Clean Dataset - Check layout quality and mark rejections in manifest.

Iterates sample by sample, checks if each layout image passes validation,
and adds a 'rejected' column to filter rejected samples.

Rejection Rules:
- Layout is rejected if missing:
  - Floor (≥ min_pixels)
  - Wall (≥ min_pixels)
  - Door OR Window (≥ min_pixels)
- Or has quality issues:
  - Mostly black (render failure)
  - Too little content (mostly background)

Usage:
    # Single process (all samples)
    python clean_dataset.py \\
        --manifest manifest_seg.csv \\
        --dataset-root dataset_v2 \\
        --output manifest_seg_cleaned.csv
    
    # Parallel processing (multiple workers)
    python clean_dataset.py \\
        --manifest manifest_seg.csv \\
        --dataset-root dataset_v2 \\
        --output manifest_seg_cleaned.csv \\
        --num-workers 8
    
    # Sharded processing (for array jobs)
    python clean_dataset.py \\
        --manifest manifest_seg.csv \\
        --dataset-root dataset_v2 \\
        --output manifest_seg_cleaned.csv \\
        --shard-id 0 \\
        --num-shards 100
    
    # Merge shards after processing
    python clean_dataset.py \\
        --merge-shards /path/to/shard/directory \\
        --output manifest_seg_cleaned.csv
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Segmentation Colors
# =============================================================================

SEG_COLORS = {
    "Floor": (180, 180, 180),
    "Wall": (60, 60, 60),
    "Door": (255, 100, 100),
    "Window": (100, 200, 255),
    "Background": (255, 255, 255),
}

COLOR_TOLERANCE = 10
WHITE_THRESHOLD = 245  # RGB values above this are considered white


# =============================================================================
# Layout Quality Check
# =============================================================================

def check_sample_layout(
    row_data: Tuple[int, dict, Path, int, float, float],
) -> Tuple[int, bool, str]:
    """
    Check a single sample's layout (for multiprocessing).
    
    Args:
        row_data: (index, row_dict, dataset_root, min_pixels, max_black_fraction, min_content_fraction)
    
    Returns:
        (index, is_valid, rejection_reason)
    """
    idx, row, dataset_root, min_pixels, max_black_fraction, min_content_fraction = row_data
    layout_path_val = row.get("layout_path", "")
    layout_path = layout_path_val if layout_path_val and str(layout_path_val) != "nan" else ""
    
    if not layout_path or layout_path == "":
        return idx, False, "NO_LAYOUT_PATH"
    
    layout_full = dataset_root / layout_path if not Path(layout_path).is_absolute() else Path(layout_path)
    if not layout_full.exists():
        return idx, False, "LAYOUT_NOT_FOUND"
    
    is_valid, rejection_reason = check_layout(
        layout_full,
        min_pixels=min_pixels,
        max_black_fraction=max_black_fraction,
        min_content_fraction=min_content_fraction,
    )
    
    return idx, is_valid, rejection_reason


def check_layout(
    image_path: Path,
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
) -> Tuple[bool, str]:
    """
    Check if a layout image is valid.
    
    Returns:
        (is_valid, rejection_reason)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)
    except Exception as e:
        return False, "LOAD_ERROR"
    
    total_pixels = pixels.shape[0] * pixels.shape[1]
    
    # Check for black (render failure)
    black_mask = np.all(pixels < 5, axis=2)
    black_fraction = black_mask.mean()
    
    if black_fraction > max_black_fraction:
        return False, "MOSTLY_BLACK"
    
    # Count semantic classes
    floor_pixels = 0
    wall_pixels = 0
    door_pixels = 0
    window_pixels = 0
    background_pixels = 0
    
    for class_name, color in SEG_COLORS.items():
        color_arr = np.array(color)
        diff = np.abs(pixels.astype(np.int16) - color_arr)
        mask = np.all(diff <= COLOR_TOLERANCE, axis=2)
        count = int(mask.sum())
        
        if class_name == "Floor":
            floor_pixels = count
        elif class_name == "Wall":
            wall_pixels = count
        elif class_name == "Door":
            door_pixels = count
        elif class_name == "Window":
            window_pixels = count
        elif class_name == "Background":
            background_pixels = count
    
    # Check content fraction
    content_fraction = (total_pixels - background_pixels) / total_pixels
    
    if content_fraction < min_content_fraction:
        return False, "TOO_LITTLE_CONTENT"
    
    # Check required classes
    has_floor = floor_pixels >= min_pixels
    has_wall = wall_pixels >= min_pixels
    has_door = door_pixels >= min_pixels
    has_window = window_pixels >= min_pixels
    
    if not has_floor:
        return False, "NO_FLOOR"
    if not has_wall:
        return False, "NO_WALL"
    if not has_door and not has_window:
        return False, "NO_DOOR_OR_WINDOW"
    
    return True, ""


# =============================================================================
# Merge Shards
# =============================================================================

def merge_shards(shards_dir: Path, output_path: Path) -> int:
    """
    Merge shard CSV files into a single manifest.
    
    Args:
        shards_dir: Directory containing shard CSV files (pattern: *_shard*.csv)
        output_path: Output path for merged manifest
    
    Returns:
        Exit code (0 for success)
    """
    logger.info(f"Merging shards from: {shards_dir}")
    
    if not shards_dir.exists():
        logger.error(f"Shards directory not found: {shards_dir}")
        return 1
    
    # Find all cleaned shard files (prefer *_cleaned.csv, fallback to any shard file)
    shard_files = sorted(shards_dir.glob("*_shard*_cleaned.csv"))
    if not shard_files:
        # Fallback: try any shard file
        shard_files = sorted(shards_dir.glob("*_shard*.csv"))
        # Exclude original shard files if cleaned ones exist
        cleaned_files = [f for f in shard_files if "_cleaned" in f.name]
        if cleaned_files:
            shard_files = cleaned_files
    
    if not shard_files:
        logger.error(f"No shard files found in {shards_dir} (looking for pattern: *_shard*_cleaned.csv)")
        return 1
    
    logger.info(f"Found {len(shard_files)} shard files")
    
    # Load and concatenate shards
    dfs = []
    for shard_file in tqdm(shard_files, desc="Loading shards"):
        try:
            df = pd.read_csv(shard_file, low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {shard_file}: {e}")
            continue
    
    if not dfs:
        logger.error("No valid shard files could be loaded")
        return 1
    
    # Merge all shards
    logger.info("Merging shards...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure rejected column exists
    if "rejected" not in merged_df.columns:
        merged_df["rejected"] = False
    
    # Sort by original index if available, or keep order
    if "index" in merged_df.columns:
        merged_df = merged_df.sort_values("index").reset_index(drop=True)
    
    rejected_count = merged_df["rejected"].sum()
    
    # Save merged manifest
    logger.info(f"Saving merged manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Merge Summary:")
    logger.info(f"  Shards merged: {len(dfs)}")
    logger.info(f"  Total rows: {len(merged_df)}")
    logger.info(f"  Rejected: {rejected_count} ({100*rejected_count/len(merged_df):.1f}%)")
    logger.info(f"  Accepted: {len(merged_df) - rejected_count}")
    logger.info(f"{'='*50}")
    
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check layout quality and mark rejections in manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--manifest", required=False, type=Path,
                        help="Input manifest CSV file (not needed for --merge-shards)")
    parser.add_argument("--dataset-root", required=False, type=Path,
                        help="Root directory of dataset (not needed for --merge-shards)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output manifest CSV with rejected column")
    
    # Layout checking parameters
    parser.add_argument("--min-pixels", type=int, default=100,
                        help="Minimum pixels required for floor/wall/door/window (default: 100)")
    parser.add_argument("--max-black-fraction", type=float, default=0.95,
                        help="Maximum fraction of black pixels (default: 0.95)")
    parser.add_argument("--min-content-fraction", type=float, default=0.05,
                        help="Minimum fraction of non-background content (default: 0.05)")
    
    # Sharding/parallel processing options
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, use 0 for auto-detect)")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Process only a specific shard (0-indexed, for array jobs)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards (required if --shard-id is set)")
    parser.add_argument("--merge-shards", type=Path, default=None,
                        help="Merge shard files from directory into single output (instead of processing)")
    
    args = parser.parse_args()
    
    # Handle merge mode
    if args.merge_shards is not None:
        return merge_shards(args.merge_shards, args.output)
    
    # Validate required arguments for processing mode
    if args.manifest is None:
        logger.error("--manifest is required (unless using --merge-shards)")
        return 1
    if args.dataset_root is None:
        logger.error("--dataset-root is required (unless using --merge-shards)")
        return 1
    
    # Validate sharding arguments
    if args.shard_id is not None:
        if args.num_shards is None:
            logger.error("--num-shards is required when --shard-id is specified")
            return 1
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            logger.error(f"--shard-id must be between 0 and {args.num_shards - 1}")
            return 1
    
    if not args.dataset_root.exists():
        logger.error(f"Dataset root not found: {args.dataset_root}")
        return 1
    
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        return 1
    
    # Load manifest
    logger.info(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest, low_memory=False)
    df.columns = df.columns.str.strip()
    logger.info(f"  Loaded {len(df)} rows")
    
    # Check for layout_path column
    if "layout_path" not in df.columns:
        logger.error("Manifest must have 'layout_path' column")
        return 1
    
    # Initialize rejected column if it doesn't exist
    if "rejected" not in df.columns:
        df["rejected"] = False
    else:
        # Reset existing rejected column
        df["rejected"] = False
    
    # Handle sharding: if shard-id is specified, process only that shard
    if args.shard_id is not None:
        total_samples = len(df)
        samples_per_shard = (total_samples + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * samples_per_shard
        end_idx = min((args.shard_id + 1) * samples_per_shard, total_samples)
        df = df.iloc[start_idx:end_idx].copy()
        logger.info(f"Processing shard {args.shard_id}/{args.num_shards}: samples {start_idx} to {end_idx-1} ({len(df)} samples)")
    
    # Determine number of workers
    if args.num_workers == 0:
        num_workers = mp.cpu_count()
    else:
        num_workers = args.num_workers
    
    # Check each sample individually
    if num_workers > 1 and len(df) > 1:
        logger.info(f"Checking {len(df)} samples using {num_workers} parallel workers...")
        
        # Prepare data for multiprocessing
        rows_data = [
            (idx, row.to_dict(), args.dataset_root, args.min_pixels, 
             args.max_black_fraction, args.min_content_fraction)
            for idx, row in df.iterrows()
        ]
        
        # Process in parallel
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(check_sample_layout, rows_data),
                total=len(rows_data),
                desc="Checking samples"
            ))
        
        # Update dataframe with results
        rejected_count = 0
        for idx, is_valid, rejection_reason in results:
            if not is_valid:
                df.at[idx, "rejected"] = True
                rejected_count += 1
    else:
        # Sequential processing
        logger.info(f"Checking {len(df)} samples sequentially...")
        rejected_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking samples"):
            layout_path = row["layout_path"] if pd.notna(row["layout_path"]) else ""
            
            # Normalize: treat NaN and empty string as empty
            if not layout_path or layout_path == "":
                layout_valid = False
            else:
                layout_full = args.dataset_root / layout_path if not Path(layout_path).is_absolute() else Path(layout_path)
                if not layout_full.exists():
                    layout_valid = False
                else:
                    layout_valid, _ = check_layout(
                        layout_full,
                        min_pixels=args.min_pixels,
                        max_black_fraction=args.max_black_fraction,
                        min_content_fraction=args.min_content_fraction,
                    )
            
            # Mark sample as rejected if layout is invalid
            if not layout_valid:
                df.at[idx, "rejected"] = True
                rejected_count += 1
    
    logger.info(f"  Marked {rejected_count} samples as rejected ({100*rejected_count/len(df):.1f}%)")
    
    # Save output
    logger.info(f"\nSaving cleaned manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # If processing a shard, append shard info to output filename
    if args.shard_id is not None:
        output_stem = args.output.stem
        output_suffix = args.output.suffix
        output_dir = args.output.parent
        shard_output = output_dir / f"{output_stem}_shard{args.shard_id:04d}{output_suffix}"
        df.to_csv(shard_output, index=False)
        logger.info(f"  Saved shard to: {shard_output}")
        logger.info(f"  To merge shards, run with --merge-shards pointing to output directory")
    else:
        df.to_csv(args.output, index=False)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Summary:")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Rejected: {rejected_count} ({100*rejected_count/len(df):.1f}%)")
    logger.info(f"  Accepted: {len(df) - rejected_count}")
    logger.info(f"{'='*50}")
    
    return 0


if __name__ == "__main__":
    exit(main())
