#!/usr/bin/env python3
"""
Move all images back from to_review folder to the main layout_new folder.

This script undoes the filter operation by moving all images from
layout_new/to_review/ back to layout_new/

Usage:
    python data_preparation/move_back_from_review.py \
        --layout_dir /work3/s233249/ImgiNav/datasets/layout_new
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def move_back_from_review(
    layout_dir: Path,
    to_review_dir: Path = None,
    dry_run: bool = False
) -> None:
    """
    Move all images from to_review folder back to layout_new folder.
    
    Args:
        layout_dir: Main directory containing layout images (layout_new)
        to_review_dir: Directory containing filtered images (default: layout_dir/to_review)
        dry_run: If True, don't move files, just report what would be moved
    """
    layout_dir = Path(layout_dir)
    if to_review_dir is None:
        to_review_dir = layout_dir / "to_review"
    else:
        to_review_dir = Path(to_review_dir)
    
    if not to_review_dir.exists():
        print(f"[WARNING] to_review directory does not exist: {to_review_dir}")
        print("[INFO] Nothing to move back.")
        return
    
    # Find all image files in to_review
    image_patterns = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(to_review_dir.glob(pattern))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"[INFO] No images found in {to_review_dir}")
        return
    
    print(f"[INFO] Found {len(image_files)} images in {to_review_dir}")
    print(f"[INFO] Moving back to {layout_dir}")
    
    if dry_run:
        print(f"[INFO] DRY RUN MODE - files will not be moved")
    
    moved_count = 0
    skipped_count = 0
    
    for image_path in tqdm(image_files, desc="Moving images back"):
        dest_path = layout_dir / image_path.name
        
        # Check if destination already exists
        if dest_path.exists():
            print(f"[WARNING] Destination already exists: {dest_path.name}, skipping")
            skipped_count += 1
            continue
        
        if not dry_run:
            try:
                shutil.move(str(image_path), str(dest_path))
                moved_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to move {image_path.name}: {e}")
                skipped_count += 1
        else:
            moved_count += 1
    
    print(f"\n[INFO] Moved {moved_count} images back to {layout_dir}")
    if skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} images (already exist or errors)")
    
    if not dry_run and moved_count > 0:
        print(f"[INFO] All images moved back successfully")
        print(f"[INFO] You can now delete the to_review folder if desired: rm -r {to_review_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Move all images back from to_review folder to layout_new folder"
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        required=True,
        help="Main directory containing layout images (layout_new)"
    )
    parser.add_argument(
        "--to_review_dir",
        type=Path,
        default=None,
        help="Directory containing filtered images (default: layout_dir/to_review)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't move files, just report what would be moved"
    )
    
    args = parser.parse_args()
    
    move_back_from_review(
        layout_dir=args.layout_dir,
        to_review_dir=args.to_review_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

