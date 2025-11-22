#!/usr/bin/env python3
"""
Recolor new layouts by swapping floor and wall colors.

This script:
1. Updates taxonomy.json to swap floor and wall colors:
   - Floor (2052): from dark gray [50, 50, 50] to light gray [200, 200, 200]
   - Wall (2053): from light gray [200, 200, 200] to dark gray [50, 50, 50]
2. Optionally recolor existing layout images by swapping pixel colors

Usage:
    python recolor_new_layouts.py --taxonomy config/taxonomy.json
    python recolor_new_layouts.py --taxonomy config/taxonomy.json --layouts-dir path/to/layouts --recolor-images
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from common.utils import write_json, safe_mkdir


# Current colors (before swap)
FLOOR_ID = "2052"
WALL_ID = "2053"
FLOOR_COLOR_OLD = [50, 50, 50]  # dark gray
WALL_COLOR_OLD = [200, 200, 200]  # light gray

# New colors (after swap)
FLOOR_COLOR_NEW = [200, 200, 200]  # light gray (was wall color)
WALL_COLOR_NEW = [50, 50, 50]  # dark gray (was floor color)


def update_taxonomy_colors(taxonomy_path: Path, backup: bool = True) -> bool:
    """
    Update taxonomy.json to swap floor and wall colors.
    
    Args:
        taxonomy_path: Path to taxonomy.json
        backup: Whether to create a backup of the original file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"[INFO] Reading taxonomy from {taxonomy_path}")
    
    if not taxonomy_path.exists():
        print(f"[ERROR] Taxonomy file not found: {taxonomy_path}")
        return False
    
    # Load taxonomy
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    
    # Create backup if requested
    if backup:
        backup_path = taxonomy_path.with_suffix(".json.backup")
        print(f"[INFO] Creating backup: {backup_path}")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    
    # Check current colors
    id2color = taxonomy.get("id2color", {})
    
    if FLOOR_ID not in id2color:
        print(f"[ERROR] Floor ID {FLOOR_ID} not found in taxonomy")
        return False
    
    if WALL_ID not in id2color:
        print(f"[ERROR] Wall ID {WALL_ID} not found in taxonomy")
        return False
    
    floor_color_current = id2color[FLOOR_ID]
    wall_color_current = id2color[WALL_ID]
    
    print(f"[INFO] Current colors:")
    print(f"  Floor ({FLOOR_ID}): {floor_color_current}")
    print(f"  Wall ({WALL_ID}): {wall_color_current}")
    
    # Swap colors
    id2color[FLOOR_ID] = FLOOR_COLOR_NEW
    id2color[WALL_ID] = WALL_COLOR_NEW
    
    print(f"[INFO] New colors:")
    print(f"  Floor ({FLOOR_ID}): {FLOOR_COLOR_NEW}")
    print(f"  Wall ({WALL_ID}): {WALL_COLOR_NEW}")
    
    # Save updated taxonomy
    print(f"[INFO] Saving updated taxonomy to {taxonomy_path}")
    write_json(taxonomy, taxonomy_path)
    
    print(f"[INFO] ✓ Taxonomy updated successfully")
    return True


def recolor_image(image_path: Path, output_path: Optional[Path] = None, 
                  floor_color_old: Tuple[int, int, int] = tuple(FLOOR_COLOR_OLD),
                  wall_color_old: Tuple[int, int, int] = tuple(WALL_COLOR_OLD),
                  floor_color_new: Tuple[int, int, int] = tuple(FLOOR_COLOR_NEW),
                  wall_color_new: Tuple[int, int, int] = tuple(WALL_COLOR_NEW),
                  overwrite: bool = True) -> bool:
    """
    Recolor a layout image by swapping floor and wall pixel colors.
    
    Args:
        image_path: Path to input image
        output_path: Path to save recolored image (if None, overwrites input)
        floor_color_old: Old floor color (RGB tuple)
        wall_color_old: Old wall color (RGB tuple)
        floor_color_new: New floor color (RGB tuple)
        wall_color_new: New wall color (RGB tuple)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Create masks for floor and wall pixels
        # Floor pixels: match old floor color (with small tolerance for compression artifacts)
        floor_mask = np.all(np.abs(img_array - np.array(floor_color_old)) < 5, axis=2)
        # Wall pixels: match old wall color
        wall_mask = np.all(np.abs(img_array - np.array(wall_color_old)) < 5, axis=2)
        
        # Create output array (copy of input)
        output_array = img_array.copy()
        
        # Replace floor pixels with new floor color
        output_array[floor_mask] = floor_color_new
        
        # Replace wall pixels with new wall color
        output_array[wall_mask] = wall_color_new
        
        # Save recolored image
        save_path = output_path if output_path else image_path
        
        # Check if file exists and overwrite flag
        if save_path.exists() and not overwrite:
            return False
        
        output_img = Image.fromarray(output_array)
        safe_mkdir(save_path.parent)
        output_img.save(save_path)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to recolor {image_path}: {e}")
        return False


def recolor_layouts_directory(layouts_dir: Path, output_dir: Optional[Path] = None,
                              recursive: bool = True, 
                              pattern: str = "*.png", overwrite: bool = True) -> int:
    """
    Recolor all layout images in a directory.
    
    Args:
        layouts_dir: Directory containing layout images
        output_dir: Directory to save recolored images (if None, creates "recolored" folder next to layouts_dir)
        recursive: Whether to search recursively
        pattern: File pattern to match (default: "*.png")
        overwrite: Whether to overwrite existing files
        
    Returns:
        Number of images successfully recolored
    """
    if not layouts_dir.exists():
        print(f"[ERROR] Layouts directory not found: {layouts_dir}")
        return 0
    
    # Determine output directory
    if output_dir is None:
        # Create "recolored" folder next to layouts_dir
        output_dir = layouts_dir.parent / f"{layouts_dir.name}_recolored"
    
    output_dir = Path(output_dir)
    safe_mkdir(output_dir)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Find all layout images
    if recursive:
        image_paths = list(layouts_dir.rglob(pattern))
    else:
        image_paths = list(layouts_dir.glob(pattern))
    
    if not image_paths:
        print(f"[WARN] No images found matching pattern '{pattern}' in {layouts_dir}")
        return 0
    
    print(f"[INFO] Found {len(image_paths)} images to recolor")
    
    # Recolor each image, preserving directory structure
    success_count = 0
    for image_path in tqdm(image_paths, desc="Recoloring images"):
        # Preserve relative path structure
        relative_path = image_path.relative_to(layouts_dir)
        output_path = output_dir / relative_path
        
        if recolor_image(image_path, output_path=output_path, overwrite=overwrite):
            success_count += 1
    
    print(f"[INFO] ✓ Successfully recolored {success_count}/{len(image_paths)} images")
    print(f"[INFO] Recolored images saved to: {output_dir}")
    return success_count


def recolor_layouts_from_manifest(manifest_path: Path, overwrite: bool = True) -> int:
    """
    Recolor layout images listed in a manifest CSV.
    
    Args:
        manifest_path: Path to manifest CSV with 'layout_path' column
        overwrite: Whether to overwrite existing files
        
    Returns:
        Number of images successfully recolored
    """
    import pandas as pd
    from common.file_io import read_manifest
    
    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        return 0
    
    # Read manifest
    rows = read_manifest(manifest_path)
    
    if not rows:
        print(f"[WARN] Manifest is empty: {manifest_path}")
        return 0
    
    # Collect layout paths
    layout_paths = []
    for row in rows:
        layout_path_str = row.get("layout_path", "")
        if layout_path_str:
            layout_path = Path(layout_path_str)
            if layout_path.exists():
                layout_paths.append(layout_path)
            else:
                print(f"[WARN] Layout file not found: {layout_path}")
    
    if not layout_paths:
        print(f"[WARN] No valid layout paths found in manifest")
        return 0
    
    print(f"[INFO] Found {len(layout_paths)} layout images in manifest")
    
    # Recolor each image
    success_count = 0
    for layout_path in tqdm(layout_paths, desc="Recoloring images"):
        if recolor_image(layout_path, overwrite=overwrite):
            success_count += 1
    
    print(f"[INFO] ✓ Successfully recolored {success_count}/{len(layout_paths)} images")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Swap floor and wall colors in taxonomy and optionally recolor layout images"
    )
    
    parser.add_argument(
        "--taxonomy",
        required=True,
        type=Path,
        help="Path to taxonomy.json file"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup of taxonomy.json before updating (default: True)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Don't create backup of taxonomy.json"
    )
    
    parser.add_argument(
        "--recolor-images",
        action="store_true",
        help="Also recolor existing layout images"
    )
    
    parser.add_argument(
        "--layouts-dir",
        type=Path,
        help="Directory containing layout images to recolor (used with --recolor-images)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save recolored images (default: creates 'recolored' folder next to layouts-dir)"
    )
    
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest CSV with layout_path column (used with --recolor-images, alternative to --layouts-dir)"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search for images recursively in layouts-dir (default: True)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="File pattern to match when searching layouts-dir (default: '*.png')"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing files when recoloring (default: True)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Update taxonomy
    print("=" * 60)
    print("Step 1: Updating taxonomy colors")
    print("=" * 60)
    success = update_taxonomy_colors(args.taxonomy, backup=args.backup)
    
    if not success:
        print("[ERROR] Failed to update taxonomy. Aborting.")
        return 1
    
    # Step 2: Recolor images if requested
    if args.recolor_images:
        print("\n" + "=" * 60)
        print("Step 2: Recoloring layout images")
        print("=" * 60)
        
        if args.manifest:
            # Recolor from manifest
            recolor_layouts_from_manifest(args.manifest, overwrite=args.overwrite)
        elif args.layouts_dir:
            # Recolor from directory
            recolor_layouts_directory(
                args.layouts_dir,
                output_dir=args.output_dir,
                recursive=args.recursive,
                pattern=args.pattern,
                overwrite=args.overwrite
            )
        else:
            print("[WARN] --recolor-images specified but neither --layouts-dir nor --manifest provided")
            print("[INFO] Skipping image recoloring")
    else:
        print("\n[INFO] Image recoloring skipped (use --recolor-images to enable)")
    
    print("\n" + "=" * 60)
    print("✓ Complete!")
    print("=" * 60)
    print(f"Taxonomy updated: {args.taxonomy}")
    if args.backup:
        print(f"Backup created: {args.taxonomy.with_suffix('.json.backup')}")
    
    return 0


if __name__ == "__main__":
    exit(main())

