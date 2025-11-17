#!/usr/bin/env python3
"""
Clean layout images by detecting floor color, calculating bounding box and density.
Images with no floor color or low floor density are moved to a failed folder.
"""

import argparse
import shutil
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from common.taxonomy import Taxonomy


def get_floor_color(taxonomy: Taxonomy) -> Tuple[int, int, int]:
    """Get the floor color from taxonomy."""
    floor_id = 2052  # Category ID for floor
    color = taxonomy.get_color(floor_id, mode="category")
    return tuple(color)


def find_floor_pixels(image: np.ndarray, floor_color: Tuple[int, int, int], 
                     tolerance: int = 0) -> np.ndarray:
    """
    Find pixels matching the floor color.
    
    Args:
        image: Image array of shape (H, W, 3)
        floor_color: RGB tuple (R, G, B)
        tolerance: Color matching tolerance (0 = exact match)
    
    Returns:
        Boolean mask of floor pixels
    """
    floor_rgb = np.array(floor_color, dtype=np.uint8)
    
    if tolerance == 0:
        # Exact match
        mask = np.all(image == floor_rgb, axis=2)
    else:
        # Match within tolerance
        diff = np.abs(image.astype(np.int16) - floor_rgb)
        mask = np.all(diff <= tolerance, axis=2)
    
    return mask


def calculate_floor_bbox(floor_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculate bounding box around floor pixels.
    
    Args:
        floor_mask: Boolean mask of floor pixels
    
    Returns:
        (x_min, y_min, x_max, y_max) or None if no floor pixels found
    """
    if not np.any(floor_mask):
        return None
    
    # Find all floor pixel coordinates
    y_coords, x_coords = np.where(floor_mask)
    
    if len(x_coords) == 0:
        return None
    
    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(np.min(y_coords))
    y_max = int(np.max(y_coords))
    
    return (x_min, y_min, x_max, y_max)


def calculate_floor_density(floor_mask: np.ndarray, 
                           bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    Calculate floor density (percentage of floor pixels).
    
    Args:
        floor_mask: Boolean mask of floor pixels
        bbox: Optional bounding box (x_min, y_min, x_max, y_max)
              If provided, density is calculated within bbox only
    
    Returns:
        Density as a float between 0.0 and 1.0
    """
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        region_mask = floor_mask[y_min:y_max+1, x_min:x_max+1]
        total_pixels = region_mask.size
        floor_pixels = np.sum(region_mask)
    else:
        total_pixels = floor_mask.size
        floor_pixels = np.sum(floor_mask)
    
    if total_pixels == 0:
        return 0.0
    
    return float(floor_pixels) / float(total_pixels)


def clean_image(image: np.ndarray, floor_mask: np.ndarray, 
               bbox: Optional[Tuple[int, int, int, int]] = None,
               background_color: Tuple[int, int, int] = (240, 240, 240)) -> np.ndarray:
    """
    Clean image by masking non-floor areas outside bbox.
    
    Args:
        image: Original image array
        floor_mask: Boolean mask of floor pixels
        bbox: Optional bounding box (x_min, y_min, x_max, y_max)
        background_color: Color to use for masked areas
    
    Returns:
        Cleaned image array
    """
    cleaned = image.copy()
    bg_color = np.array(background_color, dtype=np.uint8)
    
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        # Create mask for area outside bbox
        mask = np.ones_like(floor_mask, dtype=bool)
        mask[y_min:y_max+1, x_min:x_max+1] = False
        # Set pixels outside bbox to background
        cleaned[mask] = bg_color
    else:
        # Mask all non-floor pixels
        cleaned[~floor_mask] = bg_color
    
    return cleaned


def process_image(image_path: Path, taxonomy: Taxonomy, 
                 min_density: float = 0.1,
                 tolerance: int = 0,
                 clean_output: Optional[Path] = None) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
    """
    Process a single image.
    
    Args:
        image_path: Path to input image
        taxonomy: Taxonomy instance
        min_density: Minimum floor density threshold (0.0 to 1.0)
        tolerance: Color matching tolerance
        clean_output: Optional path to save cleaned image
    
    Returns:
        (success, density, bbox) where success is True if image passes checks
    """
    # Load image
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image_array = np.array(img)
    except Exception as e:
        print(f"[ERROR] Failed to load {image_path}: {e}", flush=True)
        return (False, 0.0, None)
    
    # Get floor color
    floor_color = get_floor_color(taxonomy)
    
    # Find floor pixels
    floor_mask = find_floor_pixels(image_array, floor_color, tolerance=tolerance)
    
    # Check if floor color exists
    if not np.any(floor_mask):
        return (False, 0.0, None)
    
    # Calculate bounding box
    bbox = calculate_floor_bbox(floor_mask)
    if bbox is None:
        return (False, 0.0, None)
    
    # Calculate density within bbox
    density = calculate_floor_density(floor_mask, bbox)
    
    # Check if density meets threshold
    if density < min_density:
        return (False, density, bbox)
    
    # Save cleaned image if requested
    if clean_output is not None:
        cleaned_image = clean_image(image_array, floor_mask, bbox)
        cleaned_pil = Image.fromarray(cleaned_image)
        clean_output.parent.mkdir(parents=True, exist_ok=True)
        cleaned_pil.save(clean_output)
    
    return (True, density, bbox)


def main():
    parser = argparse.ArgumentParser(
        description="Clean layout images by detecting floor color and calculating density"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing layout images"
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="config/taxonomy.json",
        help="Path to taxonomy.json file"
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.1,
        help="Minimum floor density threshold (0.0 to 1.0, default: 0.1)"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Color matching tolerance (0 = exact match, default: 0)"
    )
    parser.add_argument(
        "--failed-dir",
        type=str,
        default=None,
        help="Directory to move failed images (default: input_dir/failed)"
    )
    parser.add_argument(
        "--clean-dir",
        type=str,
        default=None,
        help="Directory to save cleaned images (optional, default: don't save)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=["png", "jpg", "jpeg"],
        help="Image file extensions to process (default: png jpg jpeg)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy file does not exist: {taxonomy_path}")
    
    # Load taxonomy
    taxonomy = Taxonomy(taxonomy_path)
    
    # Setup output directories
    if args.failed_dir is None:
        failed_dir = input_dir / "failed"
    else:
        failed_dir = Path(args.failed_dir)
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    clean_dir = None
    if args.clean_dir is not None:
        clean_dir = Path(args.clean_dir)
        clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = [ext.lower() for ext in args.extensions]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*.{ext}"))
        image_files.extend(input_dir.glob(f"*.{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"[WARN] No images found in {input_dir} with extensions {image_extensions}", flush=True)
        return
    
    print(f"[INFO] Found {len(image_files)} images to process", flush=True)
    print(f"[INFO] Minimum density threshold: {args.min_density}", flush=True)
    print(f"[INFO] Color tolerance: {args.tolerance}", flush=True)
    
    # Process images
    success_count = 0
    failed_count = 0
    no_floor_count = 0
    low_density_count = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Process image
        success, density, bbox = process_image(
            image_path,
            taxonomy,
            min_density=args.min_density,
            tolerance=args.tolerance,
            clean_output=clean_dir / image_path.name if clean_dir else None
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            
            # Determine failure reason
            if density == 0.0 and bbox is None:
                no_floor_count += 1
                reason = "no floor color"
            else:
                low_density_count += 1
                reason = f"low density ({density:.3f} < {args.min_density})"
            
            # Move to failed directory
            failed_path = failed_dir / image_path.name
            try:
                shutil.move(str(image_path), str(failed_path))
                tqdm.write(f"[FAILED] {image_path.name}: {reason} -> moved to {failed_dir}")
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to move {image_path.name}: {e}")
    
    # Print summary
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"Total images processed: {len(image_files)}", flush=True)
    print(f"Successful: {success_count}", flush=True)
    print(f"Failed: {failed_count}", flush=True)
    print(f"  - No floor color: {no_floor_count}", flush=True)
    print(f"  - Low density: {low_density_count}", flush=True)
    print(f"Failed images moved to: {failed_dir}", flush=True)
    if clean_dir:
        print(f"Cleaned images saved to: {clean_dir}", flush=True)


if __name__ == "__main__":
    main()

