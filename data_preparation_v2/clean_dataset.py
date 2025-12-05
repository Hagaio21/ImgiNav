#!/usr/bin/env python3
"""
Clean Dataset - Check layout quality and mark rejections in manifest.

Iterates over manifest rows, checks if layouts are valid, and adds a 'rejected' column.
If a layout is bad, all rows with that layout are marked as rejected.

Rejection Rules:
- Layout is rejected if missing:
  - Floor (≥ min_pixels)
  - Wall (≥ min_pixels)
  - Door OR Window (≥ min_pixels)
- Or has quality issues:
  - Mostly black (render failure)
  - Too little content (mostly background)

Usage:
    python clean_dataset.py \\
        --manifest manifest_seg.csv \\
        --dataset-root dataset_v2 \\
        --output manifest_seg_cleaned.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set, Tuple

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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check layout quality and mark rejections in manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Input manifest CSV file")
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Root directory of dataset")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output manifest CSV with rejected column")
    
    # Layout checking parameters
    parser.add_argument("--min-pixels", type=int, default=100,
                        help="Minimum pixels required for floor/wall/door/window (default: 100)")
    parser.add_argument("--max-black-fraction", type=float, default=0.95,
                        help="Maximum fraction of black pixels (default: 0.95)")
    parser.add_argument("--min-content-fraction", type=float, default=0.05,
                        help="Minimum fraction of non-background content (default: 0.05)")
    
    args = parser.parse_args()
    
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
    
    # Get unique layouts and check them
    logger.info("Checking unique layouts...")
    unique_layouts = df["layout_path"].fillna("").unique()
    logger.info(f"  Found {len(unique_layouts)} unique layouts")
    
    bad_layouts: Set[str] = set()
    layout_cache: Dict[str, Tuple[bool, str]] = {}
    
    for layout_path in tqdm(unique_layouts, desc="Checking layouts"):
        # Normalize: treat NaN and empty string as empty
        if not layout_path or layout_path == "":
            bad_layouts.add("")
            layout_cache[""] = (False, "NO_LAYOUT_PATH")
            continue
        
        layout_full = args.dataset_root / layout_path if not Path(layout_path).is_absolute() else Path(layout_path)
        if not layout_full.exists():
            bad_layouts.add(layout_path)
            layout_cache[layout_path] = (False, "LAYOUT_NOT_FOUND")
            continue
        
        layout_valid, rejection_reason = check_layout(
            layout_full,
            min_pixels=args.min_pixels,
            max_black_fraction=args.max_black_fraction,
            min_content_fraction=args.min_content_fraction,
        )
        
        layout_cache[layout_path] = (layout_valid, rejection_reason)
        if not layout_valid:
            bad_layouts.add(layout_path)
    
    logger.info(f"  Found {len(bad_layouts)} bad layouts")
    
    # Mark all rows with bad layouts as rejected
    logger.info("Marking rejected rows...")
    rejected_count = 0
    
    for idx, row in df.iterrows():
        layout_path = row["layout_path"] if pd.notna(row["layout_path"]) else ""
        
        if layout_path in bad_layouts:
            df.at[idx, "rejected"] = True
            rejected_count += 1
    
    logger.info(f"  Marked {rejected_count} rows as rejected ({100*rejected_count/len(df):.1f}%)")
    
    # Breakdown by reason
    reason_counts = {}
    for layout_path, (is_valid, reason) in layout_cache.items():
        if not is_valid:
            reason_counts[reason] = reason_counts.get(reason, 0) + df[df["layout_path"].fillna("") == layout_path].shape[0]
    
    if reason_counts:
        logger.info("\nRejection breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count} rows")
    
    # Save output
    logger.info(f"\nSaving cleaned manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
