#!/usr/bin/env python3
"""
Clean Dataset - Check layout quality and output rejections.

Scans layout segmentation images and checks for required semantic content.
Outputs a CSV of layout paths with rejection status.

A layout is rejected if missing:
- Floor (≥ min_pixels)
- Wall (≥ min_pixels)
- Door OR Window (≥ min_pixels)

Or has quality issues:
- Mostly black (render failure)
- Too little content (mostly background)

Usage:
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --output rejections.csv

    # With shard for parallel processing
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --shard-file shards/shard_000.txt \\
        --output rejections_shard_000.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
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


# =============================================================================
# Layout Quality Check
# =============================================================================

def check_layout(
    image_path: Path,
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
) -> Tuple[bool, str, Dict]:
    """
    Check if a layout image is valid.
    
    Returns:
        (is_valid, rejection_reason, details)
    """
    details = {
        "floor_pixels": 0,
        "wall_pixels": 0,
        "door_pixels": 0,
        "window_pixels": 0,
        "background_pixels": 0,
        "black_fraction": 0.0,
        "content_fraction": 0.0,
    }
    
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)
    except Exception as e:
        return False, f"LOAD_ERROR", details
    
    total_pixels = pixels.shape[0] * pixels.shape[1]
    
    # Check for black (render failure)
    black_mask = np.all(pixels < 5, axis=2)
    black_fraction = black_mask.mean()
    details["black_fraction"] = round(black_fraction, 4)
    
    if black_fraction > max_black_fraction:
        return False, "MOSTLY_BLACK", details
    
    # Count semantic classes
    for class_name, color in SEG_COLORS.items():
        color_arr = np.array(color)
        diff = np.abs(pixels.astype(np.int16) - color_arr)
        mask = np.all(diff <= COLOR_TOLERANCE, axis=2)
        count = int(mask.sum())
        
        if class_name == "Floor":
            details["floor_pixels"] = count
        elif class_name == "Wall":
            details["wall_pixels"] = count
        elif class_name == "Door":
            details["door_pixels"] = count
        elif class_name == "Window":
            details["window_pixels"] = count
        elif class_name == "Background":
            details["background_pixels"] = count
    
    # Check content fraction
    content_fraction = (total_pixels - details["background_pixels"]) / total_pixels
    details["content_fraction"] = round(content_fraction, 4)
    
    if content_fraction < min_content_fraction:
        return False, "TOO_LITTLE_CONTENT", details
    
    # Check required classes
    has_floor = details["floor_pixels"] >= min_pixels
    has_wall = details["wall_pixels"] >= min_pixels
    has_door = details["door_pixels"] >= min_pixels
    has_window = details["window_pixels"] >= min_pixels
    
    if not has_floor:
        return False, "NO_FLOOR", details
    if not has_wall:
        return False, "NO_WALL", details
    if not has_door and not has_window:
        return False, "NO_DOOR_OR_WINDOW", details
    
    return True, "", details


# =============================================================================
# File Discovery
# =============================================================================

def extract_scene_id(filename: str) -> Optional[str]:
    """Extract scene ID from filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if parts and len(parts[0]) >= 8:
        return parts[0]
    return None


def find_layouts(
    dataset_root: Path,
    scene_ids: Optional[Set[str]] = None,
) -> List[Path]:
    """Find all layout segmentation files."""
    layouts_seg_dir = dataset_root / "layouts" / "seg"
    if not layouts_seg_dir.exists():
        return []
    
    layouts = []
    for path in layouts_seg_dir.glob("*_seg_layout.png"):
        if scene_ids is not None:
            scene_id = extract_scene_id(path.name)
            if scene_id not in scene_ids:
                continue
        layouts.append(path)
    
    return layouts


# =============================================================================
# Main
# =============================================================================

def process_dataset(
    dataset_root: Path,
    output_path: Path,
    scene_ids: Optional[Set[str]] = None,
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
):
    """Check all layouts and write rejections CSV."""
    
    # Find layouts
    logger.info("Finding layouts...")
    layouts = find_layouts(dataset_root, scene_ids)
    logger.info(f"  Found {len(layouts)} layouts")
    
    if not layouts:
        logger.warning("No layouts found")
        return
    
    # Check each layout
    logger.info("Checking layouts...")
    results = []
    
    rejected_count = 0
    for layout_path in tqdm(layouts, desc="Checking"):
        is_valid, reason, details = check_layout(
            layout_path,
            min_pixels=min_pixels,
            max_black_fraction=max_black_fraction,
            min_content_fraction=min_content_fraction,
        )
        
        rejected = not is_valid
        if rejected:
            rejected_count += 1
        
        # Store both seg and tex paths
        seg_path = str(layout_path.relative_to(dataset_root))
        tex_path = seg_path.replace("/seg/", "/tex/").replace("_seg_", "_tex_")
        
        results.append({
            "layout_path_seg": seg_path,
            "layout_path_tex": tex_path,
            "rejected": rejected,
            "rejection_reason": reason,
            **details,
        })
    
    # Write output
    logger.info(f"Writing {len(results)} results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    columns = [
        "layout_path_seg", "layout_path_tex", "rejected", "rejection_reason",
        "floor_pixels", "wall_pixels", "door_pixels", "window_pixels",
        "background_pixels", "black_fraction", "content_fraction",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total layouts: {len(results)}")
    logger.info(f"  Rejected: {rejected_count} ({100*rejected_count/len(results):.1f}%)")
    logger.info(f"  Accepted: {len(results) - rejected_count}")
    
    # Breakdown by reason
    reason_counts: Dict[str, int] = {}
    for r in results:
        if r["rejected"]:
            reason = r["rejection_reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    if reason_counts:
        logger.info(f"\nRejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")


def load_scene_list(shard_file: Path) -> Set[str]:
    """Load scene IDs from shard file."""
    scene_ids = set()
    with open(shard_file, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scene_ids.add(scene_id)
    return scene_ids


def main():
    parser = argparse.ArgumentParser(
        description="Check layout quality and output rejections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-file", type=Path)
    
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--max-black-fraction", type=float, default=0.95)
    parser.add_argument("--min-content-fraction", type=float, default=0.05)
    
    args = parser.parse_args()
    
    if not args.dataset_root.exists():
        logger.error(f"Dataset root not found: {args.dataset_root}")
        return 1
    
    scene_ids = None
    if args.shard_file:
        scene_ids = load_scene_list(args.shard_file)
        logger.info(f"Loaded {len(scene_ids)} scene IDs")
    
    process_dataset(
        args.dataset_root,
        args.output,
        scene_ids=scene_ids,
        min_pixels=args.min_pixels,
        max_black_fraction=args.max_black_fraction,
        min_content_fraction=args.min_content_fraction,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())