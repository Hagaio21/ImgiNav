#!/usr/bin/env python3
"""
Add Rejections to Manifest

Joins the rejections CSV with manifest CSVs to add rejected column.

Usage:
    python add_rejections_to_manifest.py \\
        --rejections rejections_merged.csv \\
        --manifest-seg manifest_seg.csv \\
        --manifest-tex manifest_tex.csv \\
        --output-dir manifests/
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_rejections(rejections_path: Path) -> Dict[str, dict]:
    """
    Load rejections CSV into a dict keyed by both seg and tex paths.
    
    Returns:
        Dict mapping layout_path -> {rejected, rejection_reason}
    """
    rejections = {}
    
    with open(rejections_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg_path = row.get("layout_path_seg", "")
            tex_path = row.get("layout_path_tex", "")
            rejected = row.get("rejected", "False")
            reason = row.get("rejection_reason", "")
            
            info = {
                "rejected": rejected,
                "rejection_reason": reason,
            }
            
            if seg_path:
                rejections[seg_path] = info
            if tex_path:
                rejections[tex_path] = info
    
    return rejections


def add_rejections_to_manifest(
    manifest_path: Path,
    rejections: Dict[str, dict],
    output_path: Path,
):
    """Add rejected column to manifest."""
    
    logger.info(f"Processing: {manifest_path}")
    
    # Read manifest
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames)
        rows = list(reader)
    
    logger.info(f"  Read {len(rows)} rows")
    
    # Add new columns
    output_columns = columns + ["rejected", "rejection_reason"]
    
    # Match rejections
    matched = 0
    rejected_count = 0
    
    for row in rows:
        layout_path = row.get("layout_path", "")
        
        if layout_path in rejections:
            matched += 1
            info = rejections[layout_path]
            row["rejected"] = info["rejected"]
            row["rejection_reason"] = info["rejection_reason"]
            if info["rejected"] == "True":
                rejected_count += 1
        else:
            # Layout not in rejections - assume not rejected
            row["rejected"] = "False"
            row["rejection_reason"] = ""
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"  Matched: {matched}/{len(rows)} layouts")
    logger.info(f"  Rejected: {rejected_count}")
    logger.info(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add rejection column to manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--rejections", required=True, type=Path,
                        help="Rejections CSV from clean_dataset.py")
    parser.add_argument("--manifest-seg", type=Path,
                        help="Input manifest_seg.csv")
    parser.add_argument("--manifest-tex", type=Path,
                        help="Input manifest_tex.csv")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for new manifests")
    
    args = parser.parse_args()
    
    if not args.rejections.exists():
        logger.error(f"Rejections file not found: {args.rejections}")
        return 1
    
    # Load rejections
    logger.info(f"Loading rejections from: {args.rejections}")
    rejections = load_rejections(args.rejections)
    logger.info(f"  Loaded {len(rejections)} layout paths")
    
    # Count rejected
    rejected_layouts = sum(1 for v in rejections.values() if v["rejected"] == "True")
    logger.info(f"  Rejected layouts: {rejected_layouts}")
    
    # Process manifests
    if args.manifest_seg and args.manifest_seg.exists():
        output_seg = args.output_dir / "manifest_seg_with_rejections.csv"
        add_rejections_to_manifest(args.manifest_seg, rejections, output_seg)
    
    if args.manifest_tex and args.manifest_tex.exists():
        output_tex = args.output_dir / "manifest_tex_with_rejections.csv"
        add_rejections_to_manifest(args.manifest_tex, rejections, output_tex)
    
    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())