#!/usr/bin/env python3
"""
Stage 4b: Cleanup POVs

Removes POV images that are monocolor (single color = bad render).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_monocolor(image_path: Path, threshold: float = 0.02) -> bool:
    """
    Check if an image is essentially monocolor.
    
    Args:
        image_path: Path to image
        threshold: Max std deviation to consider monocolor (0-1 scale)
    
    Returns:
        True if image is monocolor
    """
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img, dtype=np.float32) / 255.0
        
        # Check standard deviation across all pixels
        std = pixels.std()
        
        return std < threshold
    except Exception as e:
        logger.warning(f"Could not check {image_path}: {e}")
        return False


def cleanup_povs(pov_dir: Path, threshold: float = 0.02, dry_run: bool = False):
    """
    Remove monocolor POV images.
    
    Args:
        pov_dir: Directory containing POV images
        threshold: Std deviation threshold for monocolor detection
        dry_run: If True, only report what would be deleted
    """
    # Find all POV images
    tex_dir = pov_dir / "tex"
    seg_dir = pov_dir / "seg"
    
    removed_count = 0
    checked_count = 0
    
    for subdir in [tex_dir, seg_dir]:
        if not subdir.exists():
            continue
        
        for img_path in subdir.glob("*.png"):
            checked_count += 1
            
            if is_monocolor(img_path, threshold):
                if dry_run:
                    logger.info(f"Would remove: {img_path.name}")
                else:
                    # Remove both tex and seg versions
                    img_path.unlink()
                    logger.info(f"Removed: {img_path.name}")
                
                removed_count += 1
    
    logger.info(f"\nChecked {checked_count} images, {'would remove' if dry_run else 'removed'} {removed_count} monocolor images")


def main():
    parser = argparse.ArgumentParser(description="Cleanup monocolor POV images")
    parser.add_argument("--pov-dir", required=True, help="Directory containing POV images")
    parser.add_argument("--threshold", type=float, default=0.02, 
                        help="Std deviation threshold (0-1). Lower = stricter. Default: 0.02")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Only report what would be deleted, don't actually delete")
    args = parser.parse_args()
    
    pov_dir = Path(args.pov_dir)
    
    if not pov_dir.exists():
        logger.error(f"POV directory not found: {pov_dir}")
        return
    
    cleanup_povs(pov_dir, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()