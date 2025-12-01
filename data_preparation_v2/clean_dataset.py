#!/usr/bin/env python3
"""
Clean Dataset - Check layout quality and output rejections.

Scans layout segmentation images and checks for required semantic content.
Also checks POV images for palette matching with layouts.
Outputs a CSV of layout paths with rejection status.

A layout is rejected if missing:
- Floor (≥ min_pixels)
- Wall (≥ min_pixels)
- Door OR Window (≥ min_pixels)

Or has quality issues:
- Mostly black (render failure)
- Too little content (mostly background)

A POV is rejected if:
- POV palette doesn't match layout palette (excluding white)
- Used to identify POVs looking at walls or not seeing room interior

Usage:
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --output rejections.csv

    # With manifest file for POV checking
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --manifest manifest.csv \\
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
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

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

# White color threshold (for excluding white/background)
WHITE_THRESHOLD = 245  # RGB values above this are considered white


# =============================================================================
# Palette Extraction and Comparison
# =============================================================================

def is_white_color(rgb: Tuple[int, int, int]) -> bool:
    """Check if a color is white or near-white."""
    r, g, b = rgb
    return r >= WHITE_THRESHOLD and g >= WHITE_THRESHOLD and b >= WHITE_THRESHOLD


def extract_palette(
    image_path: Path,
    exclude_white: bool = True,
    color_tolerance: int = 10,
    min_pixel_fraction: float = 0.001,
) -> Set[Tuple[int, int, int]]:
    """
    Extract color palette from an image, excluding white.
    
    Args:
        image_path: Path to image
        exclude_white: Whether to exclude white/background colors
        color_tolerance: Tolerance for color quantization (colors within this distance are grouped)
        min_pixel_fraction: Minimum fraction of pixels a color must have to be included
    
    Returns:
        Set of RGB tuples representing the palette
    """
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return set()
    
    h, w = pixels.shape[:2]
    total_pixels = h * w
    
    # Flatten pixels
    pixels_flat = pixels.reshape(-1, 3)
    
    # Filter out white pixels if requested
    if exclude_white:
        white_mask = np.all(pixels_flat >= WHITE_THRESHOLD, axis=1)
        pixels_flat = pixels_flat[~white_mask]
    
    if len(pixels_flat) == 0:
        return set()
    
    # Quantize colors with tolerance (group similar colors)
    palette = set()
    color_counts = Counter()
    
    for pixel in pixels_flat:
        r, g, b = pixel
        
        # Quantize to reduce similar colors
        r_q = (r // color_tolerance) * color_tolerance
        g_q = (g // color_tolerance) * color_tolerance
        b_q = (b // color_tolerance) * color_tolerance
        
        color_counts[(r_q, g_q, b_q)] += 1
    
    # Filter by minimum pixel fraction
    min_pixels = int(total_pixels * min_pixel_fraction)
    for color, count in color_counts.items():
        if count >= min_pixels:
            palette.add(color)
    
    return palette


def compare_palettes(
    layout_palette: Set[Tuple[int, int, int]],
    pov_palette: Set[Tuple[int, int, int]],
    color_tolerance: int = 20,
    min_match_ratio: float = 0.3,
) -> Tuple[bool, float, Dict]:
    """
    Compare two palettes to see if they match.
    
    Args:
        layout_palette: Palette from layout image
        pov_palette: Palette from POV image
        color_tolerance: Maximum color distance for matching
        min_match_ratio: Minimum ratio of layout colors that must match in POV
    
    Returns:
        (is_match, match_ratio, details)
    """
    if len(layout_palette) == 0:
        return False, 0.0, {"layout_colors": 0, "pov_colors": len(pov_palette), "matched_colors": 0}
    
    if len(pov_palette) == 0:
        return False, 0.0, {"layout_colors": len(layout_palette), "pov_colors": 0, "matched_colors": 0}
    
    # Convert to numpy arrays for distance calculation
    layout_arr = np.array(list(layout_palette))
    pov_arr = np.array(list(pov_palette))
    
    # For each layout color, find closest POV color
    matched_count = 0
    for layout_color in layout_palette:
        layout_rgb = np.array(layout_color)
        
        # Calculate distances to all POV colors
        distances = np.linalg.norm(pov_arr - layout_rgb, axis=1)
        min_distance = np.min(distances)
        
        if min_distance <= color_tolerance:
            matched_count += 1
    
    match_ratio = matched_count / len(layout_palette)
    is_match = match_ratio >= min_match_ratio
    
    details = {
        "layout_colors": len(layout_palette),
        "pov_colors": len(pov_palette),
        "matched_colors": matched_count,
        "match_ratio": round(match_ratio, 4),
    }
    
    return is_match, match_ratio, details


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
# POV-Layout Palette Check
# =============================================================================

def check_pov_palette(
    layout_path: Path,
    pov_path: Path,
    dataset_root: Path,
    color_tolerance: int = 20,
    min_match_ratio: float = 0.3,
    palette_color_tolerance: int = 10,
) -> Tuple[bool, str, Dict]:
    """
    Check if POV palette matches layout palette.
    
    Returns:
        (is_valid, rejection_reason, details)
    """
    details = {
        "layout_colors": 0,
        "pov_colors": 0,
        "matched_colors": 0,
        "match_ratio": 0.0,
    }
    
    # Resolve paths
    layout_full = dataset_root / layout_path if not layout_path.is_absolute() else layout_path
    pov_full = dataset_root / pov_path if not pov_path.is_absolute() else pov_path
    
    if not layout_full.exists():
        return False, "LAYOUT_NOT_FOUND", details
    
    if not pov_full.exists():
        return False, "POV_NOT_FOUND", details
    
    # Extract palettes
    layout_palette = extract_palette(
        layout_full,
        exclude_white=True,
        color_tolerance=palette_color_tolerance,
    )
    
    pov_palette = extract_palette(
        pov_full,
        exclude_white=True,
        color_tolerance=palette_color_tolerance,
    )
    
    if len(layout_palette) == 0:
        return False, "LAYOUT_NO_PALETTE", details
    
    if len(pov_palette) == 0:
        return False, "POV_NO_PALETTE", details
    
    # Compare palettes
    is_match, match_ratio, match_details = compare_palettes(
        layout_palette,
        pov_palette,
        color_tolerance=color_tolerance,
        min_match_ratio=min_match_ratio,
    )
    
    details.update(match_details)
    
    if not is_match:
        return False, "POV_PALETTE_MISMATCH", details
    
    return True, "", details


# =============================================================================
# Main
# =============================================================================

def process_dataset(
    dataset_root: Path,
    output_path: Path,
    scene_ids: Optional[Set[str]] = None,
    manifest_path: Optional[Path] = None,
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
    enable_pov_check: bool = False,
    pov_color_tolerance: int = 20,
    pov_min_match_ratio: float = 0.3,
    palette_color_tolerance: int = 10,
):
    """
    Check all layouts and write rejections CSV.
    
    Always checks layout quality. If manifest_path is provided and enable_pov_check is True,
    also checks POV palette matching with layouts.
    """
    
    # =============================================================================
    # STEP 1: Check Layout Quality (always done)
    # =============================================================================
    logger.info("Finding layouts...")
    layouts = find_layouts(dataset_root, scene_ids)
    logger.info(f"  Found {len(layouts)} layouts")
    
    layout_results = []
    layout_rejected_count = 0
    
    if layouts:
        logger.info("Checking layout quality...")
        for layout_path in tqdm(layouts, desc="Checking layouts"):
            is_valid, reason, details = check_layout(
                layout_path,
                min_pixels=min_pixels,
                max_black_fraction=max_black_fraction,
                min_content_fraction=min_content_fraction,
            )
            
            rejected = not is_valid
            if rejected:
                layout_rejected_count += 1
            
            # Store both seg and tex paths
            seg_path = str(layout_path.relative_to(dataset_root))
            tex_path = seg_path.replace("/seg/", "/tex/").replace("_seg_", "_tex_")
            
            layout_results.append({
                "layout_path_seg": seg_path,
                "layout_path_tex": tex_path,
                "rejected": rejected,
                "rejection_reason": reason,
                **details,
            })
    
    # =============================================================================
    # STEP 2: Check POV Palette Matching (if enabled)
    # =============================================================================
    pov_results = []
    pov_rejected_count = 0
    
    if manifest_path and enable_pov_check:
        # Process manifest rows (check POV palette matching)
        logger.info("\nLoading manifest for POV checking...")
        try:
            df = pd.read_csv(manifest_path, low_memory=False)
            df.columns = df.columns.str.strip()
            logger.info(f"  Loaded {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            logger.warning("  Continuing with layout checks only")
        else:
            # Filter to rows with both layout and POV
            has_layout = df["layout_path"].notna() & (df["layout_path"] != "")
            has_pov = df["pov_path"].notna() & (df["pov_path"] != "")
            valid_rows = df[has_layout & has_pov].copy()
            logger.info(f"  Found {len(valid_rows)} rows with both layout and POV")
            
            if len(valid_rows) > 0:
                # Check each row
                logger.info("Checking POV palette matching...")
                
                for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Checking POVs"):
                    layout_path = row["layout_path"]
                    pov_path = row["pov_path"]
                    
                    # Check POV palette
                    is_valid, reason, details = check_pov_palette(
                        Path(layout_path),
                        Path(pov_path),
                        dataset_root,
                        color_tolerance=pov_color_tolerance,
                        min_match_ratio=pov_min_match_ratio,
                        palette_color_tolerance=palette_color_tolerance,
                    )
                    
                    rejected = not is_valid
                    if rejected:
                        pov_rejected_count += 1
                    
                    pov_results.append({
                        "path": pov_path,  # Use POV path as primary identifier
                        "layout_path": layout_path,
                        "pov_path": pov_path,
                        "rejected": rejected,
                        "rejection_reason": reason,
                        **details,
                    })
    
    # =============================================================================
    # STEP 3: Write Combined Output
    # =============================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write layout results
    if layout_results:
        logger.info(f"\nWriting {len(layout_results)} layout results to {output_path}")
        
        layout_columns = [
            "layout_path_seg", "layout_path_tex", "rejected", "rejection_reason",
            "floor_pixels", "wall_pixels", "door_pixels", "window_pixels",
            "background_pixels", "black_fraction", "content_fraction",
        ]
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=layout_columns)
            writer.writeheader()
            writer.writerows(layout_results)
        
        # Layout summary
        logger.info(f"\nLayout Quality Summary:")
        logger.info(f"  Total layouts: {len(layout_results)}")
        logger.info(f"  Rejected: {layout_rejected_count} ({100*layout_rejected_count/len(layout_results):.1f}%)")
        logger.info(f"  Accepted: {len(layout_results) - layout_rejected_count}")
        
        # Breakdown by reason
        layout_reason_counts: Dict[str, int] = {}
        for r in layout_results:
            if r["rejected"]:
                reason = r["rejection_reason"]
                layout_reason_counts[reason] = layout_reason_counts.get(reason, 0) + 1
        
        if layout_reason_counts:
            logger.info(f"\nLayout rejection reasons:")
            for reason, count in sorted(layout_reason_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {reason}: {count}")
    
    # Write POV results (append to same file or separate)
    if pov_results:
        pov_output_path = output_path.parent / f"{output_path.stem}_povs.csv"
        logger.info(f"\nWriting {len(pov_results)} POV results to {pov_output_path}")
        
        pov_columns = [
            "path", "layout_path", "pov_path", "rejected", "rejection_reason",
            "layout_colors", "pov_colors", "matched_colors", "match_ratio",
        ]
        
        with open(pov_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pov_columns)
            writer.writeheader()
            writer.writerows(pov_results)
        
        # POV summary
        logger.info(f"\nPOV Palette Matching Summary:")
        logger.info(f"  Total POVs: {len(pov_results)}")
        logger.info(f"  Rejected: {pov_rejected_count} ({100*pov_rejected_count/len(pov_results):.1f}%)")
        logger.info(f"  Accepted: {len(pov_results) - pov_rejected_count}")
        
        # Breakdown by reason
        pov_reason_counts: Dict[str, int] = {}
        for r in pov_results:
            if r["rejected"]:
                reason = r["rejection_reason"]
                pov_reason_counts[reason] = pov_reason_counts.get(reason, 0) + 1
        
        if pov_reason_counts:
            logger.info(f"\nPOV rejection reasons:")
            for reason, count in sorted(pov_reason_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {reason}: {count}")
    
    # Overall summary
    if layout_results or pov_results:
        total_rejected = layout_rejected_count + pov_rejected_count
        total_checked = len(layout_results) + len(pov_results)
        logger.info(f"\n{'='*50}")
        logger.info(f"Overall Summary:")
        logger.info(f"  Total checked: {total_checked} (layouts: {len(layout_results)}, POVs: {len(pov_results)})")
        logger.info(f"  Total rejected: {total_rejected} (layouts: {layout_rejected_count}, POVs: {pov_rejected_count})")
        logger.info(f"{'='*50}")


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
    parser.add_argument("--manifest", type=Path,
                        help="Manifest CSV file (required for POV palette checking)")
    
    # Layout checking parameters
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--max-black-fraction", type=float, default=0.95)
    parser.add_argument("--min-content-fraction", type=float, default=0.05)
    
    # POV palette checking parameters
    parser.add_argument("--check-pov-palette", action="store_true",
                        help="Check POV palette matching with layout (requires --manifest)")
    parser.add_argument("--pov-color-tolerance", type=int, default=20,
                        help="Color distance tolerance for palette matching (default: 20)")
    parser.add_argument("--pov-min-match-ratio", type=float, default=0.3,
                        help="Minimum ratio of layout colors that must match in POV (default: 0.3)")
    parser.add_argument("--palette-color-tolerance", type=int, default=10,
                        help="Color quantization tolerance for palette extraction (default: 10)")
    
    args = parser.parse_args()
    
    if not args.dataset_root.exists():
        logger.error(f"Dataset root not found: {args.dataset_root}")
        return 1
    
    if args.check_pov_palette and not args.manifest:
        logger.error("--check-pov-palette requires --manifest")
        return 1
    
    if args.manifest and not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        return 1
    
    scene_ids = None
    if args.shard_file:
        scene_ids = load_scene_list(args.shard_file)
        logger.info(f"Loaded {len(scene_ids)} scene IDs")
    
    process_dataset(
        args.dataset_root,
        args.output,
        scene_ids=scene_ids,
        manifest_path=args.manifest,
        min_pixels=args.min_pixels,
        max_black_fraction=args.max_black_fraction,
        min_content_fraction=args.min_content_fraction,
        enable_pov_check=args.check_pov_palette,
        pov_color_tolerance=args.pov_color_tolerance,
        pov_min_match_ratio=args.pov_min_match_ratio,
        palette_color_tolerance=args.palette_color_tolerance,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())