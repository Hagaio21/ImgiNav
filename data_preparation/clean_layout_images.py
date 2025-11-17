#!/usr/bin/env python3
"""
Clean layout images by detecting floor color, calculating bounding box and density.
Images with no floor color or low floor density are moved to a failed folder.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

from common.taxonomy import Taxonomy
from utils.layout_analysis import count_distinct_colors


def parse_layout_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse layout filename to extract scene_id, type, and room_id.
    
    Formats:
    - Room: {scene_id}_{room_id}_room_seg_layout.png
    - Scene: {scene_id}_scene_layout.png
    
    Returns:
        (scene_id, type, room_id)
    """
    stem = Path(filename).stem
    
    # Check for scene layout: {scene_id}_scene_layout
    scene_match = re.match(r"^(.+?)_scene_layout$", stem)
    if scene_match:
        scene_id = scene_match.group(1)
        return (scene_id, "scene", "scene")
    
    # Check for room layout: {scene_id}_{room_id}_room_seg_layout
    room_match = re.match(r"^(.+?)_(\d{4})_room_seg_layout$", stem)
    if room_match:
        scene_id = room_match.group(1)
        room_id = room_match.group(2)
        return (scene_id, "room", room_id)
    
    # Fallback: try more flexible pattern for room layouts
    room_match_flex = re.match(r"^(.+?)_(.+?)_room_seg_layout$", stem)
    if room_match_flex:
        scene_id = room_match_flex.group(1)
        room_id = room_match_flex.group(2)
        return (scene_id, "room", room_id)
    
    raise ValueError(f"Could not parse layout filename: {filename}")


def check_if_empty(layout_path: Path, min_colors: int = 4) -> bool:
    """
    Check if a layout image is empty (too few colors).
    
    Args:
        layout_path: Path to layout image
        min_colors: Minimum number of distinct colors (excluding background)
    
    Returns:
        True if empty, False otherwise
    """
    try:
        color_count = count_distinct_colors(
            layout_path, 
            exclude_background=True, 
            min_pixel_threshold=10
        )
        return color_count < min_colors
    except Exception as e:
        print(f"[warn] Error checking emptiness for {layout_path}: {e}", flush=True)
        return True  # Assume empty if we can't check


def get_floor_color(taxonomy: Taxonomy) -> Tuple[int, int, int]:
    """Get the floor color from taxonomy."""
    floor_id = 2052  # Category ID for floor
    color = taxonomy.get_color(floor_id, mode="category")
    return tuple(color)


def get_wall_color(taxonomy: Taxonomy) -> Tuple[int, int, int]:
    """Get the wall color from taxonomy."""
    wall_id = 2053  # Category ID for wall
    color = taxonomy.get_color(wall_id, mode="category")
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


def find_wall_pixels(image: np.ndarray, wall_color: Tuple[int, int, int], 
                    tolerance: int = 0) -> np.ndarray:
    """
    Find pixels matching the wall color.
    
    Args:
        image: Image array of shape (H, W, 3)
        wall_color: RGB tuple (R, G, B)
        tolerance: Color matching tolerance (0 = exact match)
    
    Returns:
        Boolean mask of wall pixels
    """
    wall_rgb = np.array(wall_color, dtype=np.uint8)
    
    if tolerance == 0:
        # Exact match
        mask = np.all(image == wall_rgb, axis=2)
    else:
        # Match within tolerance
        diff = np.abs(image.astype(np.int16) - wall_rgb)
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


def create_manifest_from_cleaned(cleaned_dir: Path, output_csv: Path, min_colors: int = 4):
    """
    Create manifest CSV from cleaned layout images.
    
    Args:
        cleaned_dir: Directory containing cleaned layout images
        output_csv: Output manifest CSV path
        min_colors: Minimum colors for empty check
    """
    if not cleaned_dir.exists():
        print(f"[WARN] Cleaned directory does not exist: {cleaned_dir}", flush=True)
        return
    
    # Find all image files
    image_files = []
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        image_files.extend(cleaned_dir.glob(f"*.{ext}"))
    
    # Remove duplicates and filter to files only
    image_files = list(set(image_files))
    image_files = [f for f in image_files if f.is_file()]
    image_files = sorted(image_files)
    
    print(f"[INFO] Found {len(image_files)} cleaned images for manifest", flush=True)
    
    # Prepare manifest rows
    manifest_rows: List[Dict[str, str]] = []
    
    for layout_path in tqdm(image_files, desc="Creating manifest"):
        try:
            # Parse filename - try to parse, use fallback if it doesn't match expected patterns
            try:
                scene_id, layout_type, room_id = parse_layout_filename(layout_path.name)
            except ValueError:
                # If filename doesn't match expected patterns, use filename as scene_id
                print(f"[warn] File with unexpected name format, using fallback: {layout_path.name}", flush=True)
                scene_id = layout_path.stem  # Use filename without extension as scene_id
                layout_type = "unknown"  # Mark as unknown type
                room_id = "unknown"  # Mark as unknown room
            
            # Always use absolute paths
            layout_path_str = str(layout_path.resolve())
            
            # Check if empty
            is_empty = 0
            if check_if_empty(layout_path, min_colors=min_colors):
                is_empty = 1
            
            manifest_rows.append({
                "scene_id": scene_id,
                "type": layout_type,
                "room_id": room_id,
                "layout_path": layout_path_str,
                "is_empty": str(is_empty)
            })
            
        except Exception as e:
            print(f"[warn] Error processing {layout_path} for manifest: {e}", flush=True)
            continue
    
    # Write manifest CSV
    fieldnames = ["scene_id", "type", "room_id", "layout_path", "is_empty"]
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    
    # Print statistics
    scene_count = sum(1 for r in manifest_rows if r["type"] == "scene")
    room_count = sum(1 for r in manifest_rows if r["type"] == "room")
    empty_count = sum(1 for r in manifest_rows if r["is_empty"] == "1")
    
    print(f"\nManifest statistics:", flush=True)
    print(f"  Total entries: {len(manifest_rows)}", flush=True)
    print(f"  Scene layouts: {scene_count}", flush=True)
    print(f"  Room layouts: {room_count}", flush=True)
    print(f"  Empty layouts: {empty_count}", flush=True)
    print(f"  Valid layouts: {len(manifest_rows) - empty_count}", flush=True)


def process_image(image_path: Path, taxonomy: Taxonomy, 
                 min_density: float = 0.1,
                 tolerance: int = 0,
                 clean_output: Optional[Path] = None) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]], str]:
    """
    Process a single image.
    
    Args:
        image_path: Path to input image
        taxonomy: Taxonomy instance
        min_density: Minimum floor density threshold (0.0 to 1.0)
        tolerance: Color matching tolerance
        clean_output: Optional path to save cleaned image
    
    Returns:
        (success, density, bbox, reason) where success is True if image passes checks,
        reason is failure reason if success is False
    """
    # Load image
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image_array = np.array(img)
    except Exception as e:
        print(f"[ERROR] Failed to load {image_path}: {e}", flush=True)
        return (False, 0.0, None, "load_error")
    
    # Get floor and wall colors
    floor_color = get_floor_color(taxonomy)
    wall_color = get_wall_color(taxonomy)
    
    # Find floor and wall pixels
    floor_mask = find_floor_pixels(image_array, floor_color, tolerance=tolerance)
    wall_mask = find_wall_pixels(image_array, wall_color, tolerance=tolerance)
    
    # Count pixels
    floor_pixel_count = np.sum(floor_mask)
    wall_pixel_count = np.sum(wall_mask)
    
    # Check if floor color exists
    if not np.any(floor_mask):
        return (False, 0.0, None, "no floor color")
    
    # Check if wall color is more than floor color
    if wall_pixel_count > floor_pixel_count:
        return (False, 0.0, None, f"wall > floor (wall: {wall_pixel_count}, floor: {floor_pixel_count})")
    
    # Calculate bounding box
    bbox = calculate_floor_bbox(floor_mask)
    if bbox is None:
        return (False, 0.0, None, "no floor bbox")
    
    # Calculate density within bbox
    density = calculate_floor_density(floor_mask, bbox)
    
    # Check if density meets threshold
    if density < min_density:
        return (False, density, bbox, f"low density ({density:.3f} < {min_density})")
    
    # Save cleaned image if requested
    if clean_output is not None:
        try:
            cleaned_image = clean_image(image_array, floor_mask, bbox)
            cleaned_pil = Image.fromarray(cleaned_image)
            clean_output.parent.mkdir(parents=True, exist_ok=True)
            cleaned_pil.save(clean_output)
        except Exception as e:
            print(f"[ERROR] Failed to save cleaned image {clean_output}: {e}", flush=True)
            return (False, density, bbox, f"save_error: {e}")
    
    return (True, density, bbox, "success")


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
        "--output-manifest",
        type=str,
        default=None,
        help="Output path for manifest CSV (optional, creates manifest from cleaned images)"
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
    
    # Find all images - process ALL files with image extensions, regardless of filename
    image_extensions = [ext.lower() for ext in args.extensions]
    image_files = []
    for ext in image_extensions:
        # Find all files with this extension (case-insensitive)
        image_files.extend(input_dir.glob(f"*.{ext}"))
        image_files.extend(input_dir.glob(f"*.{ext.upper()}"))
        image_files.extend(input_dir.glob(f"*.{ext.capitalize()}"))
    
    # Remove duplicates (in case of case variations)
    image_files = list(set(image_files))
    # Filter to only actual files (not directories)
    image_files = [f for f in image_files if f.is_file()]
    image_files = sorted(image_files)
    
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
    wall_exceeds_floor_count = 0
    save_error_count = 0
    skipped_count = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Process image
            success, density, bbox, reason = process_image(
                image_path,
                taxonomy,
                min_density=args.min_density,
                tolerance=args.tolerance,
                clean_output=clean_dir / image_path.name if clean_dir else None
            )
            
            if success:
                success_count += 1
                # Verify cleaned image was actually saved (if clean_dir specified)
                if clean_dir is not None:
                    cleaned_path = clean_dir / image_path.name
                    if not cleaned_path.exists():
                        # If save failed silently, mark as failed
                        failed_count += 1
                        success_count -= 1
                        save_error_count += 1
                        reason = "save_error: file not found after save"
                        # Copy original to failed directory
                        failed_path = failed_dir / image_path.name
                        try:
                            shutil.copy2(str(image_path), str(failed_path))
                            tqdm.write(f"[FAILED] {image_path.name}: {reason} -> copied to {failed_dir}")
                        except Exception as e:
                            tqdm.write(f"[ERROR] Failed to copy {image_path.name}: {e}")
            else:
                failed_count += 1
                
                # Track failure reasons
                if "no floor color" in reason:
                    no_floor_count += 1
                elif "wall > floor" in reason:
                    wall_exceeds_floor_count += 1
                elif "low density" in reason:
                    low_density_count += 1
                elif "save_error" in reason:
                    save_error_count += 1
                elif "load_error" in reason:
                    skipped_count += 1
                else:
                    skipped_count += 1  # Track any other errors
                
                # Copy to failed directory (preserve original)
                failed_path = failed_dir / image_path.name
                try:
                    shutil.copy2(str(image_path), str(failed_path))
                    tqdm.write(f"[FAILED] {image_path.name}: {reason} -> copied to {failed_dir}")
                except Exception as e:
                    tqdm.write(f"[ERROR] Failed to copy {image_path.name}: {e}")
        except Exception as e:
            # Catch any unexpected errors during processing
            failed_count += 1
            skipped_count += 1
            reason = f"unexpected_error: {e}"
            tqdm.write(f"[ERROR] Unexpected error processing {image_path.name}: {e}")
            # Try to copy to failed directory anyway
            failed_path = failed_dir / image_path.name
            try:
                shutil.copy2(str(image_path), str(failed_path))
            except Exception as copy_e:
                tqdm.write(f"[ERROR] Also failed to copy {image_path.name}: {copy_e}")
    
    # Print summary
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    # Count actual files in output directories
    cleaned_count = 0
    failed_count_actual = 0
    if clean_dir and clean_dir.exists():
        cleaned_count = len(list(clean_dir.glob("*")))
    if failed_dir.exists():
        failed_count_actual = len(list(failed_dir.glob("*")))
    
    print(f"Total images found: {len(image_files)}", flush=True)
    print(f"Total images processed: {success_count + failed_count}", flush=True)
    print(f"Successful: {success_count}", flush=True)
    print(f"Failed: {failed_count}", flush=True)
    print(f"  - No floor color: {no_floor_count}", flush=True)
    print(f"  - Wall > floor: {wall_exceeds_floor_count}", flush=True)
    print(f"  - Low density: {low_density_count}", flush=True)
    print(f"  - Save errors: {save_error_count}", flush=True)
    print(f"  - Load errors: {skipped_count}", flush=True)
    if len(image_files) != (success_count + failed_count):
        print(f"  - Unprocessed: {len(image_files) - (success_count + failed_count)}", flush=True)
    print(f"\nActual file counts in directories:", flush=True)
    if clean_dir:
        print(f"  Cleaned directory: {cleaned_count} files", flush=True)
    print(f"  Failed directory: {failed_count_actual} files", flush=True)
    total_in_dirs = cleaned_count + failed_count_actual
    if len(image_files) != total_in_dirs:
        print(f"\n⚠ WARNING: File count mismatch!", flush=True)
        print(f"  Expected: {len(image_files)} files", flush=True)
        print(f"  Found in dirs: {total_in_dirs} files", flush=True)
        print(f"  Missing: {len(image_files) - total_in_dirs} files", flush=True)
    print(f"\nFailed images copied to: {failed_dir}", flush=True)
    if clean_dir:
        print(f"Cleaned images saved to: {clean_dir}", flush=True)
    
    # Create manifest from cleaned images if requested
    if clean_dir and args.output_manifest:
        print(f"\n{'='*60}", flush=True)
        print("Creating manifest from cleaned images", flush=True)
        print(f"{'='*60}", flush=True)
        
        output_manifest = Path(args.output_manifest)
        create_manifest_from_cleaned(clean_dir, output_manifest, min_colors=4)
        
        print(f"Manifest created: {output_manifest}", flush=True)


if __name__ == "__main__":
    main()

