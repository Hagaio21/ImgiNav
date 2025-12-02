#!/usr/bin/env python3
"""
Clean Dataset - Check layout quality and output rejections.

Scans layout segmentation images and checks for required semantic content.
Also checks POV images for uniformity when room has content.
Outputs a CSV of sample rejections.

Rejection Rules:
1. If layout is bad -> reject ALL samples with that layout
   A layout is rejected if missing:
   - Floor (≥ min_pixels)
   - Wall (≥ min_pixels)
   - Door OR Window (≥ min_pixels)
   Or has quality issues:
   - Mostly black (render failure)
   - Too little content (mostly background)

2. If POV is bad and room is empty -> do NOT reject sample
   (Empty rooms may have uniform POVs, which is acceptable)

3. If POV is bad and room is not empty -> reject that sample
   A POV is rejected if:
   - Room has content (not empty) AND POV is too uniform (too one-colored, above threshold)
   - Used to identify POVs looking at blank walls or empty spaces where something should be visible

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
# POV Uniformity Check
# =============================================================================

def check_pov_uniformity(
    layout_path: Path,
    pov_path: Path,
    dataset_root: Path,
    max_dominant_color_fraction: float = 0.7,
    color_tolerance: int = 20,
) -> Tuple[bool, str, Dict]:
    """
    Check if POV is too uniform (too one-colored).
    
    Rejection logic:
    - If room is empty (no content) -> do NOT reject (returns True), even if POV is uniform
    - If room has content AND POV is too uniform -> reject (returns False)
    
    This ensures that empty rooms with uniform POVs are acceptable, but rooms with
    content should have diverse POVs showing the content.
    
    Returns:
        (is_valid, rejection_reason, details)
    """
    details = {
        "layout_has_content": False,
        "pov_dominant_color_fraction": 0.0,
        "pov_num_colors": 0,
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
    
    # Check if POV is too uniform (too one-colored)
    try:
        pov_img = Image.open(pov_full).convert("RGB")
        pov_pixels = np.array(pov_img)
        
        # Flatten pixels
        h, w = pov_pixels.shape[:2]
        total_pixels = h * w
        pixels_flat = pov_pixels.reshape(-1, 3)
        
        # Quantize colors to group similar colors together
        # This helps identify dominant colors even with slight variations
        quantized_colors = []
        for pixel in pixels_flat:
            r, g, b = pixel
            r_q = (r // color_tolerance) * color_tolerance
            g_q = (g // color_tolerance) * color_tolerance
            b_q = (b // color_tolerance) * color_tolerance
            quantized_colors.append((r_q, g_q, b_q))
        
        # Count color frequencies
        color_counts = Counter(quantized_colors)
        details["pov_num_colors"] = len(color_counts)
        
            # Find the most dominant color fraction
            if len(color_counts) > 0:
                max_count = max(color_counts.values())
                dominant_fraction = max_count / total_pixels
                details["pov_dominant_color_fraction"] = round(dominant_fraction, 4)
                
                # Reject ONLY if room has content AND POV is too uniform
                # If room is empty, do NOT reject (even if POV is uniform)
                if layout_has_content and dominant_fraction >= max_dominant_color_fraction:
                    return False, "POV_REJECTED", details
        
    except Exception as e:
        logger.warning(f"Failed to check POV uniformity: {e}")
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
    
    Rejection logic:
    1. First pass: Check all unique layouts. If a layout is bad, mark it for rejection.
    2. Second pass: For each sample:
       - If layout is bad -> reject ALL samples with that layout
       - If POV is bad and room is empty -> do NOT reject sample
       - If POV is bad and room is not empty -> reject that sample
    
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
        
        # Filter by scene_ids if provided (for shard processing)
        if scene_ids:
            df = df[df["scene_id"].isin(scene_ids)].copy()
            logger.info(f"  Filtered to {len(df)} samples in shard")
        
        if len(df) == 0:
            logger.warning("No samples in shard")
            return
        
        # Count samples with/without required paths for logging
        has_layout = df["layout_path"].notna() & (df["layout_path"] != "")
        if enable_pov_check:
            has_pov = df["pov_path"].notna() & (df["pov_path"] != "")
            valid_rows = df[has_layout & has_pov].copy()
            logger.info(f"  Found {len(valid_rows)} samples with both layout and POV")
            logger.info(f"  Found {len(df) - len(valid_rows)} samples missing layout or POV (will be marked as rejected)")
        else:
            valid_rows = df[has_layout].copy()
            logger.info(f"  Found {len(valid_rows)} samples with layout")
            logger.info(f"  Found {len(df) - len(valid_rows)} samples missing layout (will be marked as rejected)")
        
        # =====================================================================
        # STEP 1: First pass - Check all unique layouts and build bad layouts set
        # If a layout is bad, ALL samples with that layout will be rejected
        # =====================================================================
        logger.info("First pass: Checking all unique layouts...")
        bad_layouts: Set[str] = set()
        layout_cache: Dict[str, Tuple[bool, str, Dict]] = {}  # Cache layout check results
        
        # Get unique layouts, handling NaN and empty strings consistently
        unique_layouts = df["layout_path"].fillna("").unique()
        logger.info(f"  Found {len(unique_layouts)} unique layouts")
        
        for layout_path in tqdm(unique_layouts, desc="Checking layouts"):
            # Normalize: treat NaN and empty string as empty
            if not layout_path or layout_path == "":
                bad_layouts.add("")  # Empty path is bad
                layout_cache[""] = (False, "NO_LAYOUT_PATH", {})
                continue
            
            layout_full = dataset_root / layout_path if not Path(layout_path).is_absolute() else Path(layout_path)
            if not layout_full.exists():
                bad_layouts.add(layout_path)
                layout_cache[layout_path] = (False, "LAYOUT_NOT_FOUND", {})
                continue
            
            layout_valid, layout_reason, layout_details = check_layout(
                layout_full,
                min_pixels=min_pixels,
                max_black_fraction=max_black_fraction,
                min_content_fraction=min_content_fraction,
            )
            
            layout_cache[layout_path] = (layout_valid, layout_reason, layout_details)
            if not layout_valid:
                bad_layouts.add(layout_path)
        
        logger.info(f"  Found {len(bad_layouts)} bad layouts (will reject all samples with these layouts)")
        
        # =====================================================================
        # STEP 2: Second pass - Check each sample
        # =====================================================================
        check_desc = "Checking samples (layout" + (" + POV" if enable_pov_check else "") + ")"
        logger.info(check_desc)
        results = []
        rejected_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking samples"):
            sample_id = row["sample_id"]
            layout_path = row["layout_path"] if pd.notna(row["layout_path"]) else ""
            pov_path = row.get("pov_path", "") if enable_pov_check else ""
            
            rejection_reasons = []
            rejection_details = {}
            
            # =====================================================================
            # STEP 2a: Check if layout is bad (from first pass)
            # If layout is bad, reject ALL samples with that layout
            # =====================================================================
            layout_valid = True
            layout_reason = ""
            layout_details = {}
            layout_full = None
            
            if layout_path in bad_layouts:
                # Layout is bad - reject this sample
                layout_valid = False
                cached_result = layout_cache.get(layout_path, (False, "UNKNOWN", {}))
                layout_reason = cached_result[1]
                layout_details = cached_result[2]
                rejection_reasons.append(f"LAYOUT:{layout_reason}")
                rejection_details.update({f"layout_{k}": v for k, v in layout_details.items()})
            elif layout_path:
                # Layout is good (from cache)
                cached_result = layout_cache.get(layout_path, (True, "", {}))
                layout_valid = cached_result[0]
                layout_reason = cached_result[1]
                layout_details = cached_result[2]
                layout_full = dataset_root / layout_path if not Path(layout_path).is_absolute() else Path(layout_path)
            else:
                # No layout path
                layout_valid = False
                layout_reason = "NO_LAYOUT_PATH"
                rejection_reasons.append(f"LAYOUT:{layout_reason}")
            
            # =====================================================================
            # STEP 2b: Check POV uniformity (ONLY if POV checking enabled)
            # Reject ONLY if: room has content (not empty) AND POV is too uniform
            # Do NOT reject if: room is empty (even if POV is bad)
            # =====================================================================
            pov_valid = True
            pov_reason = ""
            pov_details = {}
            pov_full = None
            
            if enable_pov_check:
                if pov_path and pd.notna(pov_path):
                    pov_full = dataset_root / pov_path if not Path(pov_path).is_absolute() else Path(pov_path)
                    if layout_full and layout_full.exists() and pov_full.exists():
                        pov_valid, pov_reason, pov_details = check_pov_uniformity(
                            layout_full,
                            pov_full,
                            dataset_root,
                            max_dominant_color_fraction=pov_min_match_ratio,  # Reuse this param for uniformity threshold
                            color_tolerance=pov_color_tolerance,  # Reuse this param for color quantization
                        )
                        # Note: check_pov_uniformity only rejects if room has content AND POV is too uniform
                        # If room is empty, it returns True (valid), which is what we want
                    else:
                        pov_valid = False
                        if not layout_full or not layout_full.exists():
                            pov_reason = "LAYOUT_NOT_FOUND"
                        elif not pov_full.exists():
                            pov_reason = "POV_NOT_FOUND"
                else:
                    pov_valid = False
                    pov_reason = "NO_POV_PATH"
                
                # Only add POV rejection reason if POV is actually invalid
                # (check_pov_uniformity already handles the "room empty" case correctly)
                if not pov_valid:
                    rejection_reasons.append(f"POV:{pov_reason}")
                    rejection_details.update({f"pov_{k}": v for k, v in pov_details.items()})
            
            # =====================================================================
            # STEP 3: Determine if sample is rejected
            # Sample is rejected if:
            # - Layout is bad (rejects ALL samples with that layout), OR
            # - POV is bad AND room is not empty (POV check already handles this)
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
    
    # POV uniformity checking parameters
    parser.add_argument("--check-pov-palette", action="store_true",
                        help="Check if POV is too uniform (too one-colored) when room has content (requires --manifest)")
    parser.add_argument("--pov-color-tolerance", type=int, default=20,
                        help="Color quantization tolerance for uniformity detection (default: 20)")
    parser.add_argument("--pov-min-match-ratio", type=float, default=0.7,
                        help="Uniformity threshold - reject if dominant color fraction >= this (default: 0.7)")
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