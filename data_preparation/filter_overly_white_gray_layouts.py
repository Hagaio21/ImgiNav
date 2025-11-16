#!/usr/bin/env python3
"""
Filter overly white or gray layout images and move them to a 'to_review' folder.

This script:
1. Scans layout images in the specified directory
2. Checks whiteness ratio (background pixels)
3. Checks grayness ratio (wall color pixels)
4. Moves suspicious images to a 'to_review' subfolder
5. Creates a manifest CSV of moved images for easy review

Usage:
    python data_preparation/filter_overly_white_gray_layouts.py \
        --layout_dir /work3/s233249/ImgiNav/datasets/layout_new \
        --max_whiteness 0.95 \
        --max_grayness 0.90
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_whiteness_ratio(canvas: np.ndarray, white_threshold: int = 230) -> float:
    """
    Compute the ratio of white pixels in the canvas.
    
    Args:
        canvas: Image array (H, W, 3) with values in [0, 255]
        white_threshold: Pixel value threshold for "white" (default: 230)
    
    Returns:
        float: Ratio of white pixels (0.0 to 1.0)
    """
    # Check if all channels are above threshold
    white_mask = (canvas > white_threshold).all(axis=2)
    total_pixels = white_mask.size
    white_pixels = white_mask.sum()
    return white_pixels / total_pixels if total_pixels > 0 else 0.0


def compute_grayness_ratio(canvas: np.ndarray, gray_colors: List[Tuple[int, int, int]], 
                          tolerance: int = 5) -> float:
    """
    Compute the ratio of gray/wall-colored pixels in the canvas.
    
    Args:
        canvas: Image array (H, W, 3) with values in [0, 255]
        gray_colors: List of RGB tuples representing gray/wall colors
        tolerance: Color matching tolerance (default: 5)
    
    Returns:
        float: Ratio of gray pixels (0.0 to 1.0)
    """
    pixels = canvas.reshape(-1, 3)
    gray_pixels = 0
    total_pixels = len(pixels)
    
    for gray_color in gray_colors:
        gray_arr = np.array(gray_color)
        # Check if pixel is within tolerance of gray color
        diff = np.abs(pixels.astype(np.int16) - gray_arr.astype(np.int16))
        matches = np.all(diff <= tolerance, axis=1)
        gray_pixels += matches.sum()
    
    # Avoid double counting (pixels that match multiple gray colors)
    # For simplicity, we'll use the maximum, but in practice we should track unique matches
    return min(gray_pixels / total_pixels, 1.0) if total_pixels > 0 else 0.0


def analyze_layout_image(image_path: Path, gray_colors: List[Tuple[int, int, int]], 
                        white_threshold: int = 230, gray_tolerance: int = 5) -> Dict:
    """
    Analyze a layout image for whiteness and grayness.
    
    Returns:
        Dict with whiteness_ratio, grayness_ratio, and should_filter flag
    """
    try:
        img = Image.open(image_path).convert("RGB")
        canvas = np.array(img)
        
        whiteness_ratio = compute_whiteness_ratio(canvas, white_threshold)
        grayness_ratio = compute_grayness_ratio(canvas, gray_colors, gray_tolerance)
        
        return {
            "whiteness_ratio": whiteness_ratio,
            "grayness_ratio": grayness_ratio,
            "total_ratio": whiteness_ratio + grayness_ratio  # Combined metric
        }
    except Exception as e:
        print(f"[warn] Error analyzing {image_path}: {e}")
        return {
            "whiteness_ratio": 0.0,
            "grayness_ratio": 0.0,
            "total_ratio": 0.0
        }


def filter_layouts(
    layout_dir: Path,
    to_review_dir: Path,
    max_whiteness: float = 0.90,
    max_grayness: float = 0.30,
    max_combined: float = 0.95,
    white_threshold: int = 230,
    gray_tolerance: int = 5,
    dry_run: bool = False
) -> None:
    """
    Filter overly white or gray layout images and move them to to_review folder.
    
    Args:
        layout_dir: Directory containing layout images
        to_review_dir: Directory to move filtered images to
        max_whiteness: Maximum whiteness ratio (default: 0.95)
        max_grayness: Maximum grayness ratio (default: 0.90)
        max_combined: Maximum combined whiteness+grayness ratio (default: 0.98)
        white_threshold: Pixel value threshold for white (default: 230)
        gray_tolerance: Color matching tolerance for gray (default: 5)
        dry_run: If True, don't move files, just report what would be moved
    """
    layout_dir = Path(layout_dir)
    to_review_dir = Path(to_review_dir)
    
    if not layout_dir.exists():
        raise ValueError(f"Layout directory does not exist: {layout_dir}")
    
    # Common gray/wall colors (light gray background and wall colors)
    gray_colors = [
        (240, 240, 240),  # Background color
        (200, 200, 200),  # Light gray
        (211, 211, 211),  # Light gray
        (220, 220, 220),  # Light gray
        (230, 230, 230),  # Light gray
    ]
    
    # Find all layout images
    layout_patterns = ["*_room_seg_layout.png", "*_scene_layout.png"]
    layout_files = []
    for pattern in layout_patterns:
        layout_files.extend(layout_dir.glob(pattern))
    
    layout_files = sorted(layout_files)
    
    print(f"[INFO] Found {len(layout_files)} layout images in {layout_dir}")
    print(f"[INFO] Filtering criteria:")
    print(f"  Max whiteness ratio: {max_whiteness}")
    print(f"  Max grayness ratio: {max_grayness}")
    print(f"  Max combined ratio: {max_combined}")
    if dry_run:
        print(f"[INFO] DRY RUN MODE - files will not be moved")
    
    # Analyze and filter
    moved_images = []
    kept_images = []
    
    for layout_path in tqdm(layout_files, desc="Analyzing layouts"):
        analysis = analyze_layout_image(
            layout_path, gray_colors, white_threshold, gray_tolerance
        )
        
        whiteness = analysis["whiteness_ratio"]
        grayness = analysis["grayness_ratio"]
        combined = analysis["total_ratio"]
        
        # Check if should be filtered
        should_filter = (
            whiteness > max_whiteness or
            grayness > max_grayness or
            combined > max_combined
        )
        
        if should_filter:
            reason = []
            if whiteness > max_whiteness:
                reason.append(f"whiteness {whiteness:.3f} > {max_whiteness}")
            if grayness > max_grayness:
                reason.append(f"grayness {grayness:.3f} > {max_grayness}")
            if combined > max_combined:
                reason.append(f"combined {combined:.3f} > {max_combined}")
            
            moved_images.append({
                "image_path": str(layout_path.relative_to(layout_dir.parent)),
                "filename": layout_path.name,
                "whiteness_ratio": f"{whiteness:.4f}",
                "grayness_ratio": f"{grayness:.4f}",
                "combined_ratio": f"{combined:.4f}",
                "filter_reason": "; ".join(reason)
            })
            
            # Move to to_review folder
            if not dry_run:
                to_review_dir.mkdir(parents=True, exist_ok=True)
                dest_path = to_review_dir / layout_path.name
                shutil.move(str(layout_path), str(dest_path))
        else:
            kept_images.append(layout_path.name)
    
    # Save manifest of moved images
    if moved_images:
        manifest_path = to_review_dir / "filtered_images_manifest.csv"
        if not dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["image_path", "filename", "whiteness_ratio", "grayness_ratio", 
                            "combined_ratio", "filter_reason"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(moved_images)
        
        print(f"\n[INFO] Filtered {len(moved_images)} images")
        print(f"[INFO] Manifest saved to: {manifest_path}")
        
        # Print statistics
        whiteness_vals = [float(m["whiteness_ratio"]) for m in moved_images]
        grayness_vals = [float(m["grayness_ratio"]) for m in moved_images]
        combined_vals = [float(m["combined_ratio"]) for m in moved_images]
        
        print(f"\n[INFO] Statistics of filtered images:")
        print(f"  Whiteness ratio: min={min(whiteness_vals):.3f}, max={max(whiteness_vals):.3f}, mean={np.mean(whiteness_vals):.3f}")
        print(f"  Grayness ratio: min={min(grayness_vals):.3f}, max={max(grayness_vals):.3f}, mean={np.mean(grayness_vals):.3f}")
        print(f"  Combined ratio: min={min(combined_vals):.3f}, max={max(combined_vals):.3f}, mean={np.mean(combined_vals):.3f}")
    else:
        print(f"\n[INFO] No images were filtered")
    
    print(f"\n[INFO] Kept {len(kept_images)} images in original location")
    
    if not dry_run:
        print(f"\n[INFO] Filtered images moved to: {to_review_dir}")
        print(f"[INFO] You can manually review and move back any images that should be kept")


def main():
    parser = argparse.ArgumentParser(
        description="Filter overly white or gray layout images and move them to review folder"
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        required=True,
        help="Directory containing layout images"
    )
    parser.add_argument(
        "--to_review_dir",
        type=Path,
        default=None,
        help="Directory to move filtered images to (default: layout_dir/to_review)"
    )
    parser.add_argument(
        "--max_whiteness",
        type=float,
        default=0.90,
        help="Maximum whiteness ratio to keep image (default: 0.90 = 90%%)"
    )
    parser.add_argument(
        "--max_grayness",
        type=float,
        default=0.30,
        help="Maximum grayness (wall color) ratio to keep image (default: 0.30 = 30%%)"
    )
    parser.add_argument(
        "--max_combined",
        type=float,
        default=0.95,
        help="Maximum combined (whiteness + grayness) ratio (default: 0.95 = 95%%)"
    )
    parser.add_argument(
        "--white_threshold",
        type=int,
        default=230,
        help="Pixel value threshold for white (default: 230)"
    )
    parser.add_argument(
        "--gray_tolerance",
        type=int,
        default=5,
        help="Color matching tolerance for gray (default: 5)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't move files, just report what would be moved"
    )
    
    args = parser.parse_args()
    
    layout_dir = Path(args.layout_dir)
    if args.to_review_dir:
        to_review_dir = Path(args.to_review_dir)
    else:
        to_review_dir = layout_dir / "to_review"
    
    filter_layouts(
        layout_dir=layout_dir,
        to_review_dir=to_review_dir,
        max_whiteness=args.max_whiteness,
        max_grayness=args.max_grayness,
        max_combined=args.max_combined,
        white_threshold=args.white_threshold,
        gray_tolerance=args.gray_tolerance,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

