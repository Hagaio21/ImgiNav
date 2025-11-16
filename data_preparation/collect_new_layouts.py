#!/usr/bin/env python3
"""
Collect new layout images from layout_new folder into a manifest CSV.

The manifest will include:
- scene_id: Scene identifier
- type: "scene" or "room"
- room_id: Room identifier (empty for scene layouts)
- layout_path: Path to the layout image (relative to data root or absolute)
- is_empty: Whether the layout is empty (0 or 1)

Usage:
    python data_preparation/collect_new_layouts.py \
        --layout_dir /work3/s233249/ImgiNav/datasets/layout_new \
        --output /work3/s233249/ImgiNav/datasets/layouts_new.csv \
        --data_root /work3/s233249/ImgiNav/datasets
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from tqdm import tqdm
from PIL import Image

from utils.layout_analysis import count_distinct_colors


def parse_layout_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse layout filename to extract scene_id, type, and room_id.
    
    Formats:
    - Room: {scene_id}_{room_id}_room_seg_layout.png
      Example: eff72f0f-049a-47b3-8685-b1b6af535435_3021_room_seg_layout.png
      scene_id = eff72f0f-049a-47b3-8685-b1b6af535435
      room_id = 3021 (4 digits, coded type)
    
    - Scene: {scene_id}_scene_layout.png
      Example: eff72f0f-049a-47b3-8685-b1b6af535435_scene_layout.png
      scene_id = eff72f0f-049a-47b3-8685-b1b6af535435
      room_id = "scene"
    
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
    # The room_id is typically 4 digits (coded type)
    room_match = re.match(r"^(.+?)_(\d{4})_room_seg_layout$", stem)
    if room_match:
        scene_id = room_match.group(1)
        room_id = room_match.group(2)  # 4-digit room code
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
        print(f"[warn] Error checking emptiness for {layout_path}: {e}")
        return True  # Assume empty if we can't check


def collect_new_layouts(
    layout_dir: Path,
    output_csv: Path,
    data_root: Optional[Path] = None,
    check_empty: bool = True,
    min_colors: int = 4,
    use_relative_paths: bool = True
) -> None:
    """
    Collect new layout images into a manifest CSV.
    
    Args:
        layout_dir: Directory containing layout_new images
        output_csv: Output manifest CSV path
        data_root: Root directory for relative paths (if None, uses absolute paths)
        check_empty: Whether to check if layouts are empty
        min_colors: Minimum colors for non-empty check
        use_relative_paths: If True and data_root provided, use relative paths
    """
    layout_dir = Path(layout_dir)
    output_csv = Path(output_csv)
    
    if not layout_dir.exists():
        raise ValueError(f"Layout directory does not exist: {layout_dir}")
    
    # Find all layout images
    layout_patterns = ["*_room_seg_layout.png", "*_scene_layout.png"]
    layout_files = []
    for pattern in layout_patterns:
        layout_files.extend(layout_dir.glob(pattern))
    
    layout_files = sorted(layout_files)
    
    print(f"[INFO] Found {len(layout_files)} layout images in {layout_dir}")
    
    # Prepare manifest rows
    manifest_rows: List[Dict[str, str]] = []
    
    for layout_path in tqdm(layout_files, desc="Processing layouts"):
        try:
            # Parse filename
            scene_id, layout_type, room_id = parse_layout_filename(layout_path.name)
            
            # Determine layout path (relative or absolute)
            if use_relative_paths and data_root:
                try:
                    layout_path_str = str(layout_path.relative_to(data_root))
                except ValueError:
                    # If not under data_root, use absolute path
                    layout_path_str = str(layout_path.resolve())
            else:
                layout_path_str = str(layout_path.resolve())
            
            # Check if empty
            is_empty = 0
            if check_empty:
                if check_if_empty(layout_path, min_colors=min_colors):
                    is_empty = 1
            
            manifest_rows.append({
                "scene_id": scene_id,
                "type": layout_type,
                "room_id": room_id,  # Keep room_id as-is (4 digits for rooms, "scene" for scenes)
                "layout_path": layout_path_str,
                "is_empty": str(is_empty)
            })
            
        except Exception as e:
            print(f"[warn] Error processing {layout_path}: {e}")
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
    
    print(f"\n[INFO] Manifest created: {output_csv}")
    print(f"[INFO] Statistics:")
    print(f"  Total layouts: {len(manifest_rows)}")
    print(f"  Scene layouts: {scene_count}")
    print(f"  Room layouts: {room_count}")
    print(f"  Empty layouts: {empty_count}")
    print(f"  Valid layouts: {len(manifest_rows) - empty_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect new layout images into a manifest CSV"
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        required=True,
        help="Directory containing layout_new images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest CSV path"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Root directory for relative paths (optional)"
    )
    parser.add_argument(
        "--no-check-empty",
        action="store_true",
        help="Skip empty layout checking (faster)"
    )
    parser.add_argument(
        "--min-colors",
        type=int,
        default=4,
        help="Minimum colors for non-empty check (default: 4)"
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Use absolute paths instead of relative paths"
    )
    
    args = parser.parse_args()
    
    collect_new_layouts(
        layout_dir=args.layout_dir,
        output_csv=args.output,
        data_root=args.data_root,
        check_empty=not args.no_check_empty,
        min_colors=args.min_colors,
        use_relative_paths=not args.absolute_paths
    )


if __name__ == "__main__":
    main()

