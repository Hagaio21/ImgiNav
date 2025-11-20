#!/usr/bin/env python3
"""
Copy POVs from manifest to ControlNet dataset structure.

This script:
1. Reads POVs manifest CSV
2. Copies POV image files to dataset/controlnet/povs/tex/ or dataset/controlnet/povs/seg/
3. Creates embeddings directories
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
        "pov_tex": base_dir / "controlnet" / "povs" / "tex",
        "pov_seg": base_dir / "controlnet" / "povs" / "seg",
        "pov_tex_embeddings": base_dir / "controlnet" / "povs" / "tex" / "embeddings",
        "pov_seg_embeddings": base_dir / "controlnet" / "povs" / "seg" / "embeddings",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def copy_povs_to_dataset(povs_manifest: Path, output_dirs: Dict[str, Path], 
                         overwrite: bool = False) -> Dict[Tuple[str, str, str], str]:
    """
    Copy POV files to appropriate directories (tex/seg).
    Returns mapping: (scene_id, room_id, type) -> pov_path
    """
    print("Processing POVs...")
    rows = read_manifest(povs_manifest)
    
    pov_mapping = {}
    skipped = 0
    
    for row in tqdm(rows, desc="Copying POVs"):
        pov_path_str = row.get("pov_path", "")
        if not pov_path_str:
            skipped += 1
            continue
        
        pov_path = Path(pov_path_str)
        if not pov_path.exists():
            print(f"Warning: POV file not found: {pov_path}")
            skipped += 1
            continue
        
        # Check if empty
        is_empty = row.get("is_empty", "0")
        if str(is_empty) in ("1", "true", "True"):
            skipped += 1
            continue
        
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        pov_type = row.get("type", "")
        
        # Determine output directory
        if pov_type == "tex":
            output_dir = output_dirs["pov_tex"]
        elif pov_type == "seg":
            output_dir = output_dirs["pov_seg"]
        else:
            print(f"Warning: Unknown POV type '{pov_type}', skipping")
            skipped += 1
            continue
        
        # Create unique filename (use original filename)
        filename = pov_path.name
        dst = output_dir / filename
        
        if copy_file(pov_path, dst, overwrite):
            key = (scene_id, room_id, pov_type)
            pov_mapping[key] = str(dst)
    
    print(f"✓ Processed {len(pov_mapping)} POVs, skipped {skipped}")
    return pov_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Copy POVs from manifest to ControlNet dataset structure"
    )
    
    parser.add_argument(
        "--povs-manifest",
        required=True,
        type=Path,
        help="Path to POVs manifest CSV"
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
    if not args.povs_manifest.exists():
        parser.error(f"POVs manifest not found: {args.povs_manifest}")
    
    # Setup directories
    print("Setting up directories...")
    output_dirs = setup_directories(args.dataset_dir)
    
    # Copy POVs
    pov_mapping = copy_povs_to_dataset(
        args.povs_manifest,
        output_dirs,
        overwrite=args.overwrite
    )
    
    print(f"\n✓ POVs processing complete!")
    print(f"  Dataset directory: {args.dataset_dir / 'controlnet'}")
    print(f"  Processed {len(pov_mapping)} POVs")


if __name__ == "__main__":
    main()

