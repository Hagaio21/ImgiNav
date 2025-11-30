#!/usr/bin/env python3
"""
Clean Dataset - Quality Assessment Mode

Instead of deleting files, this script:
1. Checks quality of POV and layout images
2. Outputs a rejections CSV with quality decisions
3. Supports sharding for parallel processing (each shard writes its own output)

The rejections CSV can then be joined with the manifest to add a 'rejected' column,
allowing filtering during training without actually deleting files.

Sharding Strategy:
- Each shard processes a subset of scenes
- Each shard writes to its own output file (e.g., rejections_shard_001.csv)
- After all shards complete, merge with: merge_rejections.py
- Finally, update manifest with: update_manifest_rejections.py

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
    # Full dataset (single process)
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --output rejections.csv

    # Process specific shard (for parallel processing)
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --shard-file shards/shard_001.txt \\
        --output rejections_shard_001.csv

    # Skip certain checks
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --skip-layouts \\
        --output rejections.csv

    # Custom thresholds
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --background-threshold 0.85 \\
        --dominant-color-threshold 0.90 \\
        --output rejections.csv
"""

import argparse
import csv
import logging
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
class RejectionRecord:
    """A single rejection decision."""
    image_type: str  # 'pov' or 'layout'
    variant: str  # 'tex' or 'seg'
    base_name: str  # e.g., "scene123_Bedroom_door0"
    path: str  # relative path from dataset root
    rejected: bool
    rejection_reasons: List[str]
    details: Dict[str, float]


@dataclass 
class CleanupStats:
    """Statistics from cleanup operation."""
    checked_povs: int = 0
    checked_layouts: int = 0
    rejected_povs: int = 0
    rejected_layouts: int = 0
    issues_by_type: Dict[QualityIssue, int] = field(default_factory=lambda: {i: 0 for i in QualityIssue})
    
    def print_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("QUALITY CHECK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"POVs:    checked={self.checked_povs}, rejected={self.rejected_povs}")
        logger.info(f"Layouts: checked={self.checked_layouts}, rejected={self.rejected_layouts}")
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

def extract_scene_id(filename: str) -> Optional[str]:
    """
    Extract scene ID from filename.
    
    Expected format: {scene_id}_{room_id}_*.png
    Scene IDs are UUIDs (36 chars with dashes) or similar long identifiers.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        scene_id = parts[0]
        if len(scene_id) >= 8:
            return scene_id
    return None


def find_paired_files(
    dataset_root: Path,
    subdir: str,
    variants: List[str] = ["tex", "seg"],
    scene_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Find all files and their paired variants.
    
    Args:
        dataset_root: Root directory of dataset
        subdir: Subdirectory (e.g., "povs", "layouts")
        variants: List of variants to find (e.g., ["tex", "seg"])
        scene_ids: Optional set of scene IDs to filter by
    
    Returns:
        Dict of base_name -> {variant: path}
    """
    paired = {}
    
    for variant in variants:
        variant_dir = dataset_root / subdir / variant
        if not variant_dir.exists():
            continue
        
        for img_path in variant_dir.glob("*.png"):
            # Filter by scene ID if provided
            if scene_ids is not None:
                scene_id = extract_scene_id(img_path.name)
                if scene_id is None or scene_id not in scene_ids:
                    continue
            
            base_name = img_path.stem
            if base_name not in paired:
                paired[base_name] = {}
            paired[base_name][variant] = img_path
    
    return paired


# =============================================================================
# Main Quality Check Functions
# =============================================================================

def check_povs(
    dataset_root: Path,
    stats: CleanupStats,
    scene_ids: Optional[Set[str]] = None,
    **quality_params,
) -> List[RejectionRecord]:
    """
    Check POV image quality.
    
    Returns:
        List of RejectionRecord for each image set
    """
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING POV IMAGES")
    if scene_ids is not None:
        logger.info(f"Filtering by {len(scene_ids)} scene IDs")
    logger.info("=" * 60)
    
    pov_dir = dataset_root / "povs"
    if not pov_dir.exists():
        logger.warning(f"POV directory not found: {pov_dir}")
        return []
    
    # Find paired files
    paired = find_paired_files(dataset_root, "povs", ["tex", "seg"], scene_ids=scene_ids)
    logger.info(f"Found {len(paired)} POV image sets")
    
    records = []
    
    for base_name, variants in tqdm(paired.items(), desc="Checking POVs"):
        stats.checked_povs += 1
        
        should_reject = False
        issues_found = []
        all_details = {}
        
        # Check each variant
        for variant, path in variants.items():
            report = check_pov_quality(path, **quality_params)
            
            if not report.is_valid:
                should_reject = True
                issues_found.extend(report.issues)
            
            # Prefix details with variant
            for k, v in report.details.items():
                all_details[f"{variant}_{k}"] = v
        
        # Check for missing pairs
        if len(variants) == 1:
            should_reject = True
            issues_found.append(QualityIssue.MISSING_PAIR)
        
        # Record decision
        if should_reject:
            stats.rejected_povs += 1
            for issue in issues_found:
                stats.issues_by_type[issue] += 1
        
        # Create record for each variant
        for variant, path in variants.items():
            rel_path = path.relative_to(dataset_root)
            records.append(RejectionRecord(
                image_type="pov",
                variant=variant,
                base_name=base_name,
                path=str(rel_path),
                rejected=should_reject,
                rejection_reasons=[i.name for i in issues_found],
                details=all_details,
            ))
    
    logger.info(f"POVs: {stats.rejected_povs} rejected out of {stats.checked_povs} checked")
    return records


def check_layouts(
    dataset_root: Path,
    stats: CleanupStats,
    scene_ids: Optional[Set[str]] = None,
    **quality_params,
) -> List[RejectionRecord]:
    """
    Check layout image quality.
    
    Returns:
        List of RejectionRecord for each image set
    """
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING LAYOUT IMAGES")
    if scene_ids is not None:
        logger.info(f"Filtering by {len(scene_ids)} scene IDs")
    logger.info("=" * 60)
    
    layouts_dir = dataset_root / "layouts"
    if not layouts_dir.exists():
        logger.warning(f"Layouts directory not found: {layouts_dir}")
        return []
    
    # Find paired files
    paired = find_paired_files(dataset_root, "layouts", ["tex", "seg"], scene_ids=scene_ids)
    logger.info(f"Found {len(paired)} layout image sets")
    
    records = []
    
    for base_name, variants in tqdm(paired.items(), desc="Checking layouts"):
        stats.checked_layouts += 1
        
        should_reject = False
        issues_found = []
        all_details = {}
        
        # Check each variant
        for variant, path in variants.items():
            report = check_layout_quality(path, **quality_params)
            
            if not report.is_valid:
                should_reject = True
                issues_found.extend(report.issues)
            
            # Prefix details with variant
            for k, v in report.details.items():
                all_details[f"{variant}_{k}"] = v
        
        # Check for missing pairs
        if len(variants) == 1:
            should_reject = True
            issues_found.append(QualityIssue.MISSING_PAIR)
        
        # Record decision
        if should_reject:
            stats.rejected_layouts += 1
            for issue in issues_found:
                stats.issues_by_type[issue] += 1
        
        # Create record for each variant
        for variant, path in variants.items():
            rel_path = path.relative_to(dataset_root)
            records.append(RejectionRecord(
                image_type="layout",
                variant=variant,
                base_name=base_name,
                path=str(rel_path),
                rejected=should_reject,
                rejection_reasons=[i.name for i in issues_found],
                details=all_details,
            ))
    
    logger.info(f"Layouts: {stats.rejected_layouts} rejected out of {stats.checked_layouts} checked")
    return records


# =============================================================================
# Output
# =============================================================================

def write_rejections(records: List[RejectionRecord], output_path: Path):
    """Write rejection records to CSV."""
    if not records:
        logger.warning("No records to write")
        return
    
    logger.info(f"Writing {len(records)} rejection records to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all detail keys across all records
    all_detail_keys = set()
    for record in records:
        all_detail_keys.update(record.details.keys())
    detail_keys = sorted(all_detail_keys)
    
    columns = [
        "image_type", "variant", "base_name", "path", "rejected", "rejection_reasons"
    ] + detail_keys
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for record in records:
            row = {
                "image_type": record.image_type,
                "variant": record.variant,
                "base_name": record.base_name,
                "path": record.path,
                "rejected": record.rejected,
                "rejection_reasons": "|".join(record.rejection_reasons),
            }
            # Add details
            for key in detail_keys:
                row[key] = record.details.get(key, "")
            
            writer.writerow(row)
    
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  Written: {size_kb:.2f} KB")


# =============================================================================
# Main
# =============================================================================

def load_scene_list(shard_file: Path) -> Set[str]:
    """Load scene IDs from a shard file (one per line)."""
    scene_ids = set()
    if not shard_file.exists():
        logger.warning(f"Shard file not found: {shard_file}")
        return scene_ids
    
    with open(shard_file, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scene_ids.add(scene_id)
    
    return scene_ids


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset quality and output rejection decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Dataset root directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV file for rejection decisions")
    
    # Sharding
    parser.add_argument("--shard-file", type=Path,
                        help="Path to shard file with scene IDs (one per line)")
    
    # Skip options
    parser.add_argument("--skip-povs", action="store_true",
                        help="Skip POV checking")
    parser.add_argument("--skip-layouts", action="store_true",
                        help="Skip layout checking")
    
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
                        help="Show details for each checked file")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return 1
    
    # Load scene IDs from shard file if provided
    scene_ids = None
    if args.shard_file:
        logger.info(f"Loading scene IDs from shard file: {args.shard_file}")
        scene_ids = load_scene_list(args.shard_file)
        logger.info(f"Loaded {len(scene_ids)} scene IDs from shard file")
        if len(scene_ids) == 0:
            logger.warning("No scene IDs found in shard file, nothing to process")
            return 0
    
    stats = CleanupStats()
    all_records = []
    
    # Check POVs
    if not args.skip_povs:
        pov_params = {
            "monocolor_threshold": args.pov_monocolor_threshold,
            "black_threshold": args.pov_black_threshold,
            "white_threshold": args.pov_white_threshold,
            "entropy_threshold": args.pov_entropy_threshold,
        }
        pov_records = check_povs(dataset_root, stats, scene_ids=scene_ids, **pov_params)
        all_records.extend(pov_records)
    
    # Check layouts
    if not args.skip_layouts:
        layout_params = {
            "background_threshold": args.background_threshold,
            "dominant_color_threshold": args.dominant_color_threshold,
            "min_unique_colors": args.min_unique_colors,
            "min_room_area_fraction": args.min_room_area,
        }
        layout_records = check_layouts(dataset_root, stats, scene_ids=scene_ids, **layout_params)
        all_records.extend(layout_records)
    
    # Write output
    write_rejections(all_records, args.output)
    
    # Print summary
    stats.print_summary()
    
    return 0


if __name__ == "__main__":
    exit(main())