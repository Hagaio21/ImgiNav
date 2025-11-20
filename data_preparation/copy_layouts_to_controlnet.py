#!/usr/bin/env python3
"""
Copy layouts from manifest to ControlNet dataset structure.

This script:
1. Reads layouts manifest CSV
2. Copies layout image files to dataset/controlnet/layouts/
3. Creates embeddings directory
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """Copy a file, creating parent directories if needed."""
    if dst.exists() and not overwrite:
        return True
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """Create and return dictionary of dataset directories."""
    dirs = {
        "layouts": base_dir / "controlnet" / "layouts",
        "layouts_embeddings": base_dir / "controlnet" / "layouts" / "embeddings",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def copy_layouts_to_dataset(layouts_manifest: Path, output_dirs: Dict[str, Path],
                           overwrite: bool = False) -> Dict[Tuple[str, str], str]:
    """
    Copy layout files to dataset directory.
    Returns mapping: (scene_id, room_id, type) -> layout_path
    """
    print("Processing layouts...")
    rows = read_manifest(layouts_manifest)
    
    layout_mapping = {}
    skipped = 0
    
    for row in tqdm(rows, desc="Copying layouts"):
        layout_path_str = row.get("layout_path", "")
        if not layout_path_str:
            skipped += 1
            continue
        
        layout_path = Path(layout_path_str)
        if not layout_path.exists():
            print(f"Warning: Layout file not found: {layout_path}")
            skipped += 1
            continue
        
        # Check if empty
        is_empty = row.get("is_empty", "0")
        if str(is_empty) in ("1", "true", "True"):
            skipped += 1
            continue
        
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        layout_type = row.get("type", "")
        
        # Create unique filename
        if layout_type == "scene":
            filename = f"{scene_id}_scene_layout.png"
        else:
            filename = f"{scene_id}_{room_id}_room_layout.png"
        
        dst = output_dirs["layouts"] / filename
        
        if copy_file(layout_path, dst, overwrite):
            key = (scene_id, room_id, layout_type)
            layout_mapping[key] = str(dst)
    
    print(f"✓ Processed {len(layout_mapping)} layouts, skipped {skipped}")
    return layout_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Copy layouts from manifest to ControlNet dataset structure"
    )
    
    parser.add_argument(
        "--layouts-manifest",
        required=True,
        type=Path,
        help="Path to layouts manifest CSV"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Base directory for dataset (will create controlnet subdirectories)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.layouts_manifest.exists():
        parser.error(f"Layouts manifest not found: {args.layouts_manifest}")
    
    # Setup directories
    print("Setting up directories...")
    output_dirs = setup_directories(args.dataset_dir)
    
    # Copy layouts
    layout_mapping = copy_layouts_to_dataset(
        args.layouts_manifest,
        output_dirs,
        overwrite=args.overwrite
    )
    
    print(f"\n✓ Layouts processing complete!")
    print(f"  Dataset directory: {args.dataset_dir / 'controlnet'}")
    print(f"  Processed {len(layout_mapping)} layouts")


if __name__ == "__main__":
    main()

