#!/usr/bin/env python3
"""
Update manifest CSV files to use recolored layouts.

Replaces layout paths from 'layouts' to 'layouts_recolored' in manifest files.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.file_io import read_manifest, create_manifest


def update_manifest_paths(
    manifest_path: Path,
    old_path: str,
    new_path: str
) -> int:
    """
    Update layout_path in manifest to use new path.
    
    Args:
        manifest_path: Path to manifest CSV
        old_path: Old path to replace (e.g., 'layouts')
        new_path: New path (e.g., 'layouts_recolored')
        
    Returns:
        Number of rows updated
    """
    print(f"[INFO] Reading manifest: {manifest_path}")
    rows = read_manifest(manifest_path)
    
    if not rows:
        print("[WARN] Manifest is empty")
        return 0
    
    updated_count = 0
    
    for row in rows:
        layout_path_str = row.get("layout_path", "")
        if not layout_path_str:
            continue
        
        if old_path in layout_path_str:
            # Replace old path with new path
            new_layout_path = layout_path_str.replace(old_path, new_path)
            row["layout_path"] = new_layout_path
            updated_count += 1
    
    if updated_count > 0:
        # Get fieldnames from first row
        fieldnames = list(rows[0].keys())
        
        # Save updated manifest
        print(f"[INFO] Saving updated manifest: {manifest_path}")
        create_manifest(rows, manifest_path, fieldnames)
        print(f"[INFO] ✓ Updated {updated_count} layout paths")
    else:
        print(f"[INFO] No paths to update (no paths containing '{old_path}')")
    
    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Update manifest CSV to use recolored layouts"
    )
    
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to manifest CSV file"
    )
    
    parser.add_argument(
        "--old-path",
        default="/work3/s233249/ImgiNav/datasets/controlnet/layouts",
        help="Old path to replace (default: /work3/s233249/ImgiNav/datasets/controlnet/layouts)"
    )
    
    parser.add_argument(
        "--new-path",
        default="/work3/s233249/ImgiNav/datasets/controlnet/layouts_recolored",
        help="New path (default: /work3/s233249/ImgiNav/datasets/controlnet/layouts_recolored)"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup of original manifest (default: True)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Don't create backup"
    )
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        return 1
    
    # Create backup if requested
    if args.backup:
        backup_path = args.manifest.with_suffix(f".csv.backup")
        print(f"[INFO] Creating backup: {backup_path}")
        import shutil
        shutil.copy2(args.manifest, backup_path)
    
    updated = update_manifest_paths(
        args.manifest,
        args.old_path,
        args.new_path
    )
    
    print(f"\n[INFO] ✓ Manifest update complete")
    print(f"[INFO] Updated {updated} layout paths in {args.manifest}")
    
    return 0


if __name__ == "__main__":
    exit(main())

