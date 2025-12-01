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
- Room has content (not empty) AND POV is only shades of gray (above threshold)
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
import shutil
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
    min_pixel_fraction: float = 0.005,
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
    
    Uses bidirectional matching: checks if layout colors match POV colors
    AND if POV colors match layout colors, then uses the better ratio.
    
    Args:
        layout_palette: Palette from layout image
        pov_palette: Palette from POV image
        color_tolerance: Maximum color distance for matching
        min_match_ratio: Minimum ratio of colors that must match
    
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
    
    # Check layout -> POV matching
    layout_matched = 0
    for layout_color in layout_palette:
        layout_rgb = np.array(layout_color)
        distances = np.linalg.norm(pov_arr - layout_rgb, axis=1)
        min_distance = np.min(distances)
        if min_distance <= color_tolerance:
            layout_matched += 1
    
    # Check POV -> layout matching
    pov_matched = 0
    for pov_color in pov_palette:
        pov_rgb = np.array(pov_color)
        distances = np.linalg.norm(layout_arr - pov_rgb, axis=1)
        min_distance = np.min(distances)
        if min_distance <= color_tolerance:
            pov_matched += 1
    
    # Use the better match ratio (more lenient)
    layout_ratio = layout_matched / len(layout_palette) if len(layout_palette) > 0 else 0.0
    pov_ratio = pov_matched / len(pov_palette) if len(pov_palette) > 0 else 0.0
    match_ratio = max(layout_ratio, pov_ratio)
    
    # Also check intersection-based ratio (even more lenient)
    # Count how many colors from either palette have a match in the other
    total_colors = len(layout_palette) + len(pov_palette)
    total_matched = layout_matched + pov_matched
    intersection_ratio = total_matched / total_colors if total_colors > 0 else 0.0
    
    # Use the best ratio (most lenient)
    final_ratio = max(match_ratio, intersection_ratio)
    is_match = final_ratio >= min_match_ratio
    
    details = {
        "layout_colors": len(layout_palette),
        "pov_colors": len(pov_palette),
        "matched_colors": layout_matched + pov_matched,
        "match_ratio": round(final_ratio, 4),
    }
    
    return is_match, final_ratio, details


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
# POV Grayscale Check
# =============================================================================

def check_pov_grayscale(
    layout_path: Path,
    pov_path: Path,
    dataset_root: Path,
    grayscale_threshold: float = 0.9,
    grayscale_color_tolerance: int = 10,
) -> Tuple[bool, str, Dict]:
    """
    Check if POV is only shades of gray when room has content.
    
    Rejects if:
    - Layout has content (not empty)
    - POV is mostly grayscale (above threshold)
    
    Returns:
        (is_valid, rejection_reason, details)
    """
    details = {
        "layout_has_content": False,
        "pov_grayscale_fraction": 0.0,
    }
    
    # Resolve paths
    layout_full = dataset_root / layout_path if not layout_path.is_absolute() else layout_path
    pov_full = dataset_root / pov_path if not pov_path.is_absolute() else pov_path
    
    if not layout_full.exists():
        return False, "LAYOUT_NOT_FOUND", details
    
    if not pov_full.exists():
        return False, "POV_NOT_FOUND", details
    
    # Check if layout has content (not empty)
    try:
        layout_img = Image.open(layout_full).convert("RGB")
        layout_pixels = np.array(layout_img)
        
        # Count non-background pixels
        total_pixels = layout_pixels.shape[0] * layout_pixels.shape[1]
        background_color = np.array(SEG_COLORS["Background"])
        bg_diff = np.abs(layout_pixels.astype(np.int16) - background_color)
        bg_mask = np.all(bg_diff <= COLOR_TOLERANCE, axis=2)
        content_pixels = total_pixels - bg_mask.sum()
        content_fraction = content_pixels / total_pixels
        
        # Room is not empty if it has at least 5% content
        layout_has_content = content_fraction >= 0.05
        details["layout_has_content"] = layout_has_content
    except Exception as e:
        logger.warning(f"Failed to check layout content: {e}")
        layout_has_content = False
    
    # Check if POV is grayscale
    try:
        pov_img = Image.open(pov_full).convert("RGB")
        pov_pixels = np.array(pov_img)
        
        # Flatten pixels
        h, w = pov_pixels.shape[:2]
        pixels_flat = pov_pixels.reshape(-1, 3)
        
        # Check if each pixel is grayscale (R, G, B are similar)
        # A pixel is grayscale if max(R,G,B) - min(R,G,B) <= tolerance
        pixel_ranges = np.max(pixels_flat, axis=1) - np.min(pixels_flat, axis=1)
        grayscale_mask = pixel_ranges <= grayscale_color_tolerance
        
        grayscale_fraction = grayscale_mask.mean()
        details["pov_grayscale_fraction"] = round(grayscale_fraction, 4)
        
        # Reject if room has content AND POV is mostly grayscale
        if layout_has_content and grayscale_fraction >= grayscale_threshold:
            return False, "POV_REJECTED", details
        
    except Exception as e:
        logger.warning(f"Failed to check POV grayscale: {e}")
        return False, "POV_CHECK_ERROR", details
    
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
    Check samples from manifest and write rejections CSV.
    
    If manifest_path is provided, checks each sample (layout + POV) and rejects
    the sample if either the layout quality check or POV palette check fails.
    
    If manifest_path is not provided, falls back to checking all layouts only.
    """
    
    # =============================================================================
    # MODE 1: Sample-based checking (with manifest and sample_id)
    # =============================================================================
    if manifest_path and manifest_path.exists():
        logger.info("Loading manifest with sample_ids...")
        try:
            df = pd.read_csv(manifest_path, low_memory=False)
            df.columns = df.columns.str.strip()
            logger.info(f"  Loaded {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return
        
        # Check if sample_id column exists
        if "sample_id" not in df.columns:
            logger.error("Manifest must have 'sample_id' column. Run add_sample_id.py first.")
            return
        
        # Filter to rows with layout (and optionally POV if POV checking enabled)
        has_layout = df["layout_path"].notna() & (df["layout_path"] != "")
        if enable_pov_check:
            has_pov = df["pov_path"].notna() & (df["pov_path"] != "")
            valid_rows = df[has_layout & has_pov].copy()
            logger.info(f"  Found {len(valid_rows)} samples with both layout and POV")
        else:
            valid_rows = df[has_layout].copy()
            logger.info(f"  Found {len(valid_rows)} samples with layout")
        
        if len(valid_rows) == 0:
            logger.warning("No valid samples found in manifest")
            return
        
        # Filter by scene_ids if provided (for shard processing)
        if scene_ids:
            valid_rows = valid_rows[valid_rows["scene_id"].isin(scene_ids)].copy()
            logger.info(f"  Filtered to {len(valid_rows)} samples in shard")
        
        if len(valid_rows) == 0:
            logger.warning("No samples in shard")
            return
        
        # Check each sample
        check_desc = "Checking samples (layout" + (" + POV" if enable_pov_check else "") + ")"
        logger.info(check_desc)
        results = []
        rejected_count = 0
        
        for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Checking samples"):
            sample_id = row["sample_id"]
            layout_path = row["layout_path"]
            pov_path = row.get("pov_path", "") if enable_pov_check else ""
            
            rejection_reasons = []
            rejection_details = {}
            
            # =====================================================================
            # STEP 1: Check layout quality (ALWAYS done for every sample)
            # =====================================================================
            layout_valid = True
            layout_reason = ""
            layout_details = {}
            layout_full = None
            
            if layout_path:
                layout_full = dataset_root / layout_path if not Path(layout_path).is_absolute() else Path(layout_path)
                if layout_full.exists():
                    layout_valid, layout_reason, layout_details = check_layout(
                        layout_full,
                        min_pixels=min_pixels,
                        max_black_fraction=max_black_fraction,
                        min_content_fraction=min_content_fraction,
                    )
                else:
                    layout_valid = False
                    layout_reason = "LAYOUT_NOT_FOUND"
            else:
                layout_valid = False
                layout_reason = "NO_LAYOUT_PATH"
            
            if not layout_valid:
                rejection_reasons.append(f"LAYOUT:{layout_reason}")
                rejection_details.update({f"layout_{k}": v for k, v in layout_details.items()})
            
            # =====================================================================
            # STEP 2: Check POV grayscale (ONLY if POV checking enabled)
            # Reject if room has content AND POV is only shades of gray
            # =====================================================================
            pov_valid = True
            pov_reason = ""
            pov_details = {}
            pov_full = None
            
            if enable_pov_check:
                if pov_path:
                    pov_full = dataset_root / pov_path if not Path(pov_path).is_absolute() else Path(pov_path)
                    if layout_full and layout_full.exists() and pov_full.exists():
                        pov_valid, pov_reason, pov_details = check_pov_grayscale(
                            layout_full,
                            pov_full,
                            dataset_root,
                            grayscale_threshold=pov_min_match_ratio,  # Reuse this param for grayscale threshold
                            grayscale_color_tolerance=pov_color_tolerance,  # Reuse this param for color tolerance
                        )
                    else:
                        pov_valid = False
                        if not layout_full.exists():
                            pov_reason = "LAYOUT_NOT_FOUND"
                        elif not pov_full.exists():
                            pov_reason = "POV_NOT_FOUND"
                else:
                    pov_valid = False
                    pov_reason = "NO_POV_PATH"
                
                if not pov_valid:
                    rejection_reasons.append(f"POV:{pov_reason}")
                    rejection_details.update({f"pov_{k}": v for k, v in pov_details.items()})
            
            # =====================================================================
            # STEP 3: Determine if sample is rejected
            # Sample is rejected if layout fails OR (if POV checking enabled) POV fails
            # =====================================================================
            is_rejected = not layout_valid or (enable_pov_check and not pov_valid)
            if is_rejected:
                rejected_count += 1
                
                # Copy rejected images to separate directories
                # Only copy the images that caused the rejection
                if not layout_valid and layout_full and layout_full.exists():
                    rejected_layouts_dir = dataset_root / "rejected_layouts"
                    rejected_layouts_dir.mkdir(parents=True, exist_ok=True)
                    # Copy with sample_id in filename to avoid conflicts
                    dest_name = f"{sample_id}_{layout_full.name}"
                    dest_path = rejected_layouts_dir / dest_name
                    try:
                        shutil.copy2(layout_full, dest_path)
                    except Exception as e:
                        logger.warning(f"Failed to copy rejected layout {layout_full}: {e}")
                
                if enable_pov_check and not pov_valid and pov_full and pov_full.exists():
                    rejected_pov_dir = dataset_root / "rejected_pov"
                    rejected_pov_dir.mkdir(parents=True, exist_ok=True)
                    # Copy with sample_id in filename to avoid conflicts
                    dest_name = f"{sample_id}_{pov_full.name}"
                    dest_path = rejected_pov_dir / dest_name
                    try:
                        shutil.copy2(pov_full, dest_path)
                    except Exception as e:
                        logger.warning(f"Failed to copy rejected POV {pov_full}: {e}")
            
            # Build result row - sample_id is the primary key for merging back into manifest
            # Only include essential fields, not all the detail metrics
            result = {
                "sample_id": sample_id,
                "rejected": is_rejected,
                "rejection_reason": "|".join(rejection_reasons) if rejection_reasons else "",
                "layout_valid": layout_valid,
                "pov_valid": pov_valid if enable_pov_check else True,
            }
            results.append(result)
        
        # Write output
        logger.info(f"\nWriting {len(results)} sample results to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build column list - only essential fields
        columns = [
            "sample_id",
            "rejected",
            "rejection_reason",
            "layout_valid",
            "pov_valid",
        ]
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Sample Quality Summary:")
        logger.info(f"  Total samples: {len(results)}")
        logger.info(f"  Rejected: {rejected_count} ({100*rejected_count/len(results):.1f}%)")
        logger.info(f"  Accepted: {len(results) - rejected_count}")
        
        # Breakdown by reason
        reason_counts: Dict[str, int] = {}
        layout_rejections = 0
        pov_rejections = 0
        
        for r in results:
            if r["rejected"]:
                reasons = r["rejection_reason"].split("|")
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    if reason.startswith("LAYOUT:"):
                        layout_rejections += 1
                    elif reason.startswith("POV:"):
                        pov_rejections += 1
        
        logger.info(f"\nRejection breakdown:")
        logger.info(f"  Layout rejections: {layout_rejections} (always checked)")
        if enable_pov_check:
            logger.info(f"  POV rejections: {pov_rejections} (checked when POV checking enabled)")
        else:
            logger.info(f"  POV checking: DISABLED")
        
        if reason_counts:
            logger.info(f"\nRejection reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {reason}: {count}")
        logger.info(f"{'='*50}")
        
        return
    
    # =============================================================================
    # MODE 2: Fallback - Layout-only checking (no manifest)
    # =============================================================================
    logger.info("No manifest provided, checking all layouts...")
    layouts = find_layouts(dataset_root, scene_ids)
    logger.info(f"  Found {len(layouts)} layouts")
    
    if not layouts:
        logger.warning("No layouts found")
        return
    
    results = []
    rejected_count = 0
    
    logger.info("Checking layouts...")
    for layout_path in tqdm(layouts, desc="Checking layouts"):
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
    logger.info(f"\nWriting {len(results)} layout results to {output_path}")
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
    logger.info(f"\nLayout Quality Summary:")
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
    parser.add_argument("--manifest", type=Path,
                        help="Manifest CSV file (required for POV palette checking)")
    
    # Layout checking parameters
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--max-black-fraction", type=float, default=0.95)
    parser.add_argument("--min-content-fraction", type=float, default=0.05)
    
    # POV grayscale checking parameters
    parser.add_argument("--check-pov-palette", action="store_true",
                        help="Check if POV is only grayscale when room has content (requires --manifest)")
    parser.add_argument("--pov-color-tolerance", type=int, default=10,
                        help="Color tolerance for grayscale detection (max R,G,B difference, default: 10)")
    parser.add_argument("--pov-min-match-ratio", type=float, default=0.9,
                        help="Grayscale threshold - reject if POV grayscale fraction >= this (default: 0.9)")
    parser.add_argument("--palette-color-tolerance", type=int, default=10,
                        help="Unused (kept for backward compatibility)")
    
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