#!/usr/bin/env python3
"""
Clean Dataset - Remove bad POV and layout images.

Checks for quality issues in both POV (first-person) and layout (top-down) images.
Syncs deletions between tex/seg variants and cleans up embeddings.

POV Quality Checks:
- Monocolor: Single color (bad render)
- Mostly black: >threshold pixels very dark (failed lighting)
- Mostly white: >threshold pixels very bright (overexposed)
- Low entropy: Lacks visual complexity (staring at wall)

Layout Quality Checks:
- Dominant background: Too much background color (empty room)
- Single dominant color: One color covers >threshold of image
- Too few colors: Segmentation lacks object variety
- Small room area: Actual room content is tiny
- High fragmentation: Many disconnected small regions (artifacts)

Usage:
    # Dry run (report only)
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --dry-run

    # Clean POVs only
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --skip-layouts

    # Clean layouts only
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --skip-povs

    # Move to rejected folder instead of deleting
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --move-to-rejected

    # Custom thresholds
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --background-threshold 0.85 \\
        --dominant-color-threshold 0.90
"""

import argparse
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from collections import Counter
from enum import Enum, auto

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Quality Issue Types
# =============================================================================

class QualityIssue(Enum):
    """Types of quality issues detected."""
    # POV issues
    POV_MONOCOLOR = auto()
    POV_MOSTLY_BLACK = auto()
    POV_MOSTLY_WHITE = auto()
    POV_LOW_ENTROPY = auto()
    
    # Layout issues
    LAYOUT_DOMINANT_BACKGROUND = auto()
    LAYOUT_SINGLE_DOMINANT_COLOR = auto()
    LAYOUT_TOO_FEW_COLORS = auto()
    LAYOUT_SMALL_ROOM_AREA = auto()
    LAYOUT_HIGH_FRAGMENTATION = auto()
    
    # Sync issues
    MISSING_PAIR = auto()  # tex exists but seg doesn't (or vice versa)


@dataclass
class QualityReport:
    """Report for a single image."""
    path: Path
    is_valid: bool
    issues: List[QualityIssue] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self):
        if self.is_valid:
            return f"✓ {self.path.name}"
        issues_str = ", ".join(i.name for i in self.issues)
        details_str = ", ".join(f"{k}={v:.3f}" for k, v in self.details.items())
        return f"✗ {self.path.name}: {issues_str} ({details_str})"


@dataclass 
class CleanupStats:
    """Statistics from cleanup operation."""
    checked_povs: int = 0
    checked_layouts: int = 0
    removed_povs: int = 0
    removed_layouts: int = 0
    removed_embeddings: int = 0
    issues_by_type: Dict[QualityIssue, int] = field(default_factory=lambda: {i: 0 for i in QualityIssue})
    
    def print_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("CLEANUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"POVs:    checked={self.checked_povs}, removed={self.removed_povs}")
        logger.info(f"Layouts: checked={self.checked_layouts}, removed={self.removed_layouts}")
        logger.info(f"Embeddings removed: {self.removed_embeddings}")
        logger.info("\nIssues by type:")
        for issue, count in sorted(self.issues_by_type.items(), key=lambda x: -x[1]):
            if count > 0:
                logger.info(f"  {issue.name}: {count}")


# =============================================================================
# Image Quality Checks
# =============================================================================

def compute_entropy(pixels: np.ndarray) -> float:
    """Compute Shannon entropy of pixel values (0-8 for 8-bit images)."""
    # Flatten and convert to grayscale-ish by averaging channels
    if pixels.ndim == 3:
        gray = pixels.mean(axis=2)
    else:
        gray = pixels
    
    # Quantize to reduce noise (256 bins -> 64 bins)
    quantized = (gray * 63).astype(np.int32)
    
    # Compute histogram
    hist, _ = np.histogram(quantized.flatten(), bins=64, range=(0, 64))
    hist = hist[hist > 0]  # Remove zeros
    
    # Normalize
    p = hist / hist.sum()
    
    # Shannon entropy
    entropy = -np.sum(p * np.log2(p))
    
    return entropy


def check_pov_quality(
    image_path: Path,
    monocolor_threshold: float = 0.02,
    black_threshold: float = 0.95,
    white_threshold: float = 0.95,
    black_value: float = 0.05,
    white_value: float = 0.95,
    entropy_threshold: float = 2.0,
) -> QualityReport:
    """
    Check POV image quality.
    
    Args:
        image_path: Path to POV image
        monocolor_threshold: Max std to consider monocolor
        black_threshold: Fraction of pixels that must be dark to flag as "mostly black"
        white_threshold: Fraction of pixels that must be bright to flag as "mostly white"
        black_value: Pixel value (0-1) below which is considered "black"
        white_value: Pixel value (0-1) above which is considered "white"
        entropy_threshold: Min entropy to be considered valid
    
    Returns:
        QualityReport with issues found
    """
    report = QualityReport(path=image_path, is_valid=True)
    
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        logger.warning(f"Could not load {image_path}: {e}")
        report.is_valid = False
        report.issues.append(QualityIssue.POV_MONOCOLOR)
        return report
    
    # Compute statistics
    std = pixels.std()
    mean = pixels.mean()
    
    # Per-pixel brightness (average across RGB)
    brightness = pixels.mean(axis=2)
    black_fraction = (brightness < black_value).mean()
    white_fraction = (brightness > white_value).mean()
    
    # Entropy
    entropy = compute_entropy(pixels)
    
    # Store details
    report.details = {
        "std": std,
        "mean": mean,
        "black_frac": black_fraction,
        "white_frac": white_fraction,
        "entropy": entropy,
    }
    
    # Check issues
    if std < monocolor_threshold:
        report.issues.append(QualityIssue.POV_MONOCOLOR)
    
    if black_fraction > black_threshold:
        report.issues.append(QualityIssue.POV_MOSTLY_BLACK)
    
    if white_fraction > white_threshold:
        report.issues.append(QualityIssue.POV_MOSTLY_WHITE)
    
    if entropy < entropy_threshold:
        report.issues.append(QualityIssue.POV_LOW_ENTROPY)
    
    report.is_valid = len(report.issues) == 0
    return report


def check_layout_quality(
    image_path: Path,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    background_threshold: float = 0.85,
    dominant_color_threshold: float = 0.90,
    min_unique_colors: int = 3,
    min_room_area_fraction: float = 0.05,
    max_fragmentation: float = 0.3,
    color_tolerance: int = 5,
) -> QualityReport:
    """
    Check layout image quality.
    
    Args:
        image_path: Path to layout image
        background_color: RGB tuple for background (default black)
        background_threshold: Max fraction of background allowed
        dominant_color_threshold: Max fraction for any single non-bg color
        min_unique_colors: Min number of distinct colors (for segmentation)
        min_room_area_fraction: Min fraction of image that should be room content
        max_fragmentation: Max ratio of small disconnected regions
        color_tolerance: Tolerance for color matching (0-255)
    
    Returns:
        QualityReport with issues found
    """
    report = QualityReport(path=image_path, is_valid=True)
    
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)
    except Exception as e:
        logger.warning(f"Could not load {image_path}: {e}")
        report.is_valid = False
        report.issues.append(QualityIssue.LAYOUT_DOMINANT_BACKGROUND)
        return report
    
    height, width = pixels.shape[:2]
    total_pixels = height * width
    
    # Identify background pixels
    bg = np.array(background_color)
    bg_mask = np.all(np.abs(pixels.astype(np.int16) - bg) <= color_tolerance, axis=2)
    background_fraction = bg_mask.mean()
    
    # Non-background pixels
    non_bg_mask = ~bg_mask
    room_area_fraction = non_bg_mask.mean()
    
    # Count unique colors (quantize to reduce noise)
    # Quantize to nearest 8 to handle anti-aliasing
    quantized = (pixels // 8) * 8
    flat_colors = quantized.reshape(-1, 3)
    non_bg_colors = flat_colors[non_bg_mask.flatten()]
    
    if len(non_bg_colors) > 0:
        # Count unique colors
        color_tuples = [tuple(c) for c in non_bg_colors]
        color_counts = Counter(color_tuples)
        unique_colors = len(color_counts)
        
        # Find dominant non-background color
        most_common_color, most_common_count = color_counts.most_common(1)[0]
        dominant_color_fraction = most_common_count / len(non_bg_colors)
    else:
        unique_colors = 0
        dominant_color_fraction = 1.0
    
    # Fragmentation check (ratio of small connected components)
    # Simple version: count isolated pixels vs total room pixels
    fragmentation = 0.0
    if room_area_fraction > 0.01:
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(non_bg_mask)
            if num_features > 0:
                component_sizes = ndimage.sum(non_bg_mask, labeled, range(1, num_features + 1))
                small_components = sum(1 for s in component_sizes if s < 100)  # <100 pixels
                fragmentation = small_components / num_features if num_features > 0 else 0
        except ImportError:
            # Skip fragmentation check if scipy not available
            pass
    
    # Store details
    report.details = {
        "bg_frac": background_fraction,
        "room_frac": room_area_fraction,
        "unique_colors": unique_colors,
        "dominant_frac": dominant_color_fraction,
        "fragmentation": fragmentation,
    }
    
    # Check issues
    if background_fraction > background_threshold:
        report.issues.append(QualityIssue.LAYOUT_DOMINANT_BACKGROUND)
    
    if dominant_color_fraction > dominant_color_threshold and unique_colors > 1:
        report.issues.append(QualityIssue.LAYOUT_SINGLE_DOMINANT_COLOR)
    
    if unique_colors < min_unique_colors:
        report.issues.append(QualityIssue.LAYOUT_TOO_FEW_COLORS)
    
    if room_area_fraction < min_room_area_fraction:
        report.issues.append(QualityIssue.LAYOUT_SMALL_ROOM_AREA)
    
    if fragmentation > max_fragmentation:
        report.issues.append(QualityIssue.LAYOUT_HIGH_FRAGMENTATION)
    
    report.is_valid = len(report.issues) == 0
    return report


# =============================================================================
# File Management
# =============================================================================

def find_paired_files(
    dataset_root: Path,
    subdir: str,
    variants: List[str] = ["tex", "seg"],
) -> Dict[str, Dict[str, Path]]:
    """
    Find all files and their paired variants.
    
    Returns:
        Dict of base_name -> {variant: path}
    """
    paired = {}
    
    for variant in variants:
        variant_dir = dataset_root / subdir / variant
        if not variant_dir.exists():
            continue
        
        for img_path in variant_dir.glob("*.png"):
            base_name = img_path.stem
            if base_name not in paired:
                paired[base_name] = {}
            paired[base_name][variant] = img_path
    
    return paired


def find_embedding_for_image(
    image_path: Path,
    dataset_root: Path,
    image_type: str,  # "pov" or "layout"
) -> Optional[Path]:
    """Find the embedding file corresponding to an image."""
    # Extract scene_id and room_id from filename
    # Expected format: {scene_id}_{room_id}_*.png or similar
    stem = image_path.stem
    parts = stem.rsplit("_", 1)
    
    if len(parts) < 2:
        return None
    
    # Try to find embedding
    if image_type == "pov":
        # POV embeddings: povs/embeddings_{variant}/{scene_id}_{room_id}_pov.pt
        variant = image_path.parent.name
        emb_dir = dataset_root / "povs" / f"embeddings_{variant}"
        # The stem might be scene_room or scene_room_angle
        # We need scene_room for the embedding
        scene_room = "_".join(stem.split("_")[:2])
        emb_path = emb_dir / f"{scene_room}_pov.pt"
        if emb_path.exists():
            return emb_path
    
    return None


def remove_or_move(
    path: Path,
    move_to_rejected: bool,
    dry_run: bool,
) -> bool:
    """
    Remove file or move to parallel rejected directory.
    
    e.g., layouts/tex/image.png -> layouts/tex_rejected/image.png
          povs/seg/image.png -> povs/seg_rejected/image.png
          povs/embeddings_tex/emb.pt -> povs/embeddings_tex_rejected/emb.pt
    """
    if dry_run:
        return True
    
    try:
        if move_to_rejected:
            # Create parallel rejected folder: parent/variant -> parent/variant_rejected
            parent = path.parent.parent  # e.g., layouts/ or povs/
            variant = path.parent.name   # e.g., tex, seg, embeddings_tex
            rejected_variant = f"{variant}_rejected"
            dest_dir = parent / rejected_variant
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / path.name
            shutil.move(str(path), str(dest))
        else:
            path.unlink()
        return True
    except Exception as e:
        logger.warning(f"Failed to remove {path}: {e}")
        return False


# =============================================================================
# Main Cleanup Functions
# =============================================================================

def clean_povs(
    dataset_root: Path,
    move_to_rejected: bool,
    dry_run: bool,
    stats: CleanupStats,
    **quality_params,
) -> Set[str]:
    """
    Clean POV images.
    
    Returns:
        Set of removed base names (for syncing with other data)
    """
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING POV IMAGES")
    logger.info("=" * 60)
    
    pov_dir = dataset_root / "povs"
    if not pov_dir.exists():
        logger.warning(f"POV directory not found: {pov_dir}")
        return set()
    
    # Find paired files
    paired = find_paired_files(dataset_root, "povs", ["tex", "seg"])
    logger.info(f"Found {len(paired)} POV image sets")
    
    removed_bases = set()
    
    for base_name, variants in tqdm(paired.items(), desc="Checking POVs"):
        stats.checked_povs += 1
        
        should_remove = False
        issues_found = []
        
        # Check each variant
        for variant, path in variants.items():
            report = check_pov_quality(path, **quality_params)
            
            if not report.is_valid:
                should_remove = True
                issues_found.extend(report.issues)
                if not dry_run:
                    logger.debug(str(report))
        
        # Check for missing pairs
        if len(variants) == 1:
            should_remove = True
            issues_found.append(QualityIssue.MISSING_PAIR)
        
        # Remove all variants if any has issues
        if should_remove:
            removed_bases.add(base_name)
            
            for issue in issues_found:
                stats.issues_by_type[issue] += 1
            
            for variant, path in variants.items():
                if dry_run:
                    logger.info(f"Would remove: {path.relative_to(dataset_root)}")
                else:
                    if remove_or_move(path, move_to_rejected, dry_run):
                        stats.removed_povs += 1
                
                # Also remove embedding
                emb_path = find_embedding_for_image(path, dataset_root, "pov")
                if emb_path and emb_path.exists():
                    if dry_run:
                        logger.info(f"Would remove embedding: {emb_path.relative_to(dataset_root)}")
                    else:
                        if remove_or_move(emb_path, move_to_rejected, dry_run):
                            stats.removed_embeddings += 1
    
    logger.info(f"POVs: {stats.removed_povs} removed out of {stats.checked_povs} checked")
    return removed_bases


def clean_layouts(
    dataset_root: Path,
    move_to_rejected: bool,
    dry_run: bool,
    stats: CleanupStats,
    **quality_params,
) -> Set[str]:
    """
    Clean layout images.
    
    Returns:
        Set of removed base names (for syncing with other data)
    """
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING LAYOUT IMAGES")
    logger.info("=" * 60)
    
    layouts_dir = dataset_root / "layouts"
    if not layouts_dir.exists():
        logger.warning(f"Layouts directory not found: {layouts_dir}")
        return set()
    
    # Find paired files
    paired = find_paired_files(dataset_root, "layouts", ["tex", "seg"])
    logger.info(f"Found {len(paired)} layout image sets")
    
    removed_bases = set()
    
    for base_name, variants in tqdm(paired.items(), desc="Checking layouts"):
        stats.checked_layouts += 1
        
        should_remove = False
        issues_found = []
        
        # Check each variant
        for variant, path in variants.items():
            report = check_layout_quality(path, **quality_params)
            
            if not report.is_valid:
                should_remove = True
                issues_found.extend(report.issues)
                if not dry_run:
                    logger.debug(str(report))
        
        # Check for missing pairs
        if len(variants) == 1:
            should_remove = True
            issues_found.append(QualityIssue.MISSING_PAIR)
        
        # Remove all variants if any has issues
        if should_remove:
            removed_bases.add(base_name)
            
            for issue in issues_found:
                stats.issues_by_type[issue] += 1
            
            for variant, path in variants.items():
                if dry_run:
                    logger.info(f"Would remove: {path.relative_to(dataset_root)}")
                else:
                    if remove_or_move(path, move_to_rejected, dry_run):
                        stats.removed_layouts += 1
    
    logger.info(f"Layouts: {stats.removed_layouts} removed out of {stats.checked_layouts} checked")
    return removed_bases


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean dataset by removing bad POV and layout images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Dataset root directory")
    
    # Mode
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report, don't delete")
    parser.add_argument("--move-to-rejected", action="store_true",
                        help="Move to rejected/ folder instead of deleting")
    parser.add_argument("--skip-povs", action="store_true",
                        help="Skip POV cleaning")
    parser.add_argument("--skip-layouts", action="store_true",
                        help="Skip layout cleaning")
    
    # POV thresholds
    pov_group = parser.add_argument_group("POV quality thresholds")
    pov_group.add_argument("--pov-monocolor-threshold", type=float, default=0.02,
                           help="Max std for monocolor (default: 0.02)")
    pov_group.add_argument("--pov-black-threshold", type=float, default=0.95,
                           help="Fraction dark pixels for 'mostly black' (default: 0.95)")
    pov_group.add_argument("--pov-white-threshold", type=float, default=0.95,
                           help="Fraction bright pixels for 'mostly white' (default: 0.95)")
    pov_group.add_argument("--pov-entropy-threshold", type=float, default=2.0,
                           help="Min entropy for valid image (default: 2.0)")
    
    # Layout thresholds
    layout_group = parser.add_argument_group("Layout quality thresholds")
    layout_group.add_argument("--background-threshold", type=float, default=0.85,
                              help="Max background fraction (default: 0.85)")
    layout_group.add_argument("--dominant-color-threshold", type=float, default=0.90,
                              help="Max single color fraction (default: 0.90)")
    layout_group.add_argument("--min-unique-colors", type=int, default=3,
                              help="Min unique colors for segmentation (default: 3)")
    layout_group.add_argument("--min-room-area", type=float, default=0.05,
                              help="Min room area fraction (default: 0.05)")
    
    # Verbosity
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show details for each removed file")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return 1
    
    # Setup rejected mode
    move_to_rejected = args.move_to_rejected
    if move_to_rejected:
        logger.info("Moving rejected files to parallel *_rejected folders")
    
    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")
    
    stats = CleanupStats()
    
    # Clean POVs
    if not args.skip_povs:
        pov_params = {
            "monocolor_threshold": args.pov_monocolor_threshold,
            "black_threshold": args.pov_black_threshold,
            "white_threshold": args.pov_white_threshold,
            "entropy_threshold": args.pov_entropy_threshold,
        }
        clean_povs(dataset_root, move_to_rejected, args.dry_run, stats, **pov_params)
    
    # Clean layouts
    if not args.skip_layouts:
        layout_params = {
            "background_threshold": args.background_threshold,
            "dominant_color_threshold": args.dominant_color_threshold,
            "min_unique_colors": args.min_unique_colors,
            "min_room_area_fraction": args.min_room_area,
        }
        clean_layouts(dataset_root, move_to_rejected, args.dry_run, stats, **layout_params)
    
    # Print summary
    stats.print_summary()
    
    if args.dry_run:
        logger.info("\nThis was a dry run. Use without --dry-run to actually remove files.")
    
    return 0


if __name__ == "__main__":
    exit(main())