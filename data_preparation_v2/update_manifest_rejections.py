#!/usr/bin/env python3
"""
Update Manifest with Rejections - Add rejected column to manifest.

Takes a manifest CSV and a rejections CSV, and produces a new manifest
with a 'rejected' column added. This allows filtering during training
without actually deleting files.

The join is done on the path columns:
- Manifest has: pov_path, layout_path
- Rejections has: path

A row is marked as rejected if ANY of its referenced paths are rejected.

Usage:
    # Update manifest with rejections
    python update_manifest_rejections.py \\
        --manifest manifest_tex.csv \\
        --rejections rejections_merged.csv \\
        --output manifest_tex_filtered.csv

    # Keep only non-rejected rows (actually filter, not just add column)
    python update_manifest_rejections.py \\
        --manifest manifest_tex.csv \\
        --rejections rejections_merged.csv \\
        --output manifest_tex_clean.csv \\
        --filter-rejected
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


def load_rejections(rejections_path: Path) -> Set[str]:
    """Load set of rejected paths from rejections CSV."""
    rejected_paths = set()
    
    with open(rejections_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("rejected", "").lower() == "true":
                path = row.get("path", "")
                if path:
                    rejected_paths.add(path)
    
    return rejected_paths


def update_manifest(
    manifest_path: Path,
    rejections_path: Path,
    output_path: Path,
    filter_rejected: bool = False,
):
    """Update manifest with rejected column."""
    logger.info(f"Loading rejections from: {rejections_path}")
    rejected_paths = load_rejections(rejections_path)
    logger.info(f"  Found {len(rejected_paths)} rejected paths")
    
    logger.info(f"Processing manifest: {manifest_path}")
    
    # Read manifest
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_columns = reader.fieldnames or []
        rows = list(reader)
    
    logger.info(f"  Read {len(rows)} rows")
    
    # Add rejected column
    output_columns = original_columns + ["rejected", "rejection_paths"]
    
    output_rows = []
    rejected_count = 0
    
    for row in rows:
        # Check if any referenced path is rejected
        pov_path = row.get("pov_path", "")
        layout_path = row.get("layout_path", "")
        
        rejection_paths = []
        if pov_path and pov_path in rejected_paths:
            rejection_paths.append(pov_path)
        if layout_path and layout_path in rejected_paths:
            rejection_paths.append(layout_path)
        
        is_rejected = len(rejection_paths) > 0
        
        if is_rejected:
            rejected_count += 1
        
        if filter_rejected and is_rejected:
            continue  # Skip rejected rows
        
        row["rejected"] = is_rejected
        row["rejection_paths"] = "|".join(rejection_paths)
        output_rows.append(row)
    
    # Write output
    logger.info(f"Writing {len(output_rows)} rows to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(output_rows)
    
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  Written: {size_kb:.2f} KB")
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Original rows: {len(rows)}")
    logger.info(f"  Rejected: {rejected_count} ({100*rejected_count/len(rows):.1f}%)")
    if filter_rejected:
        logger.info(f"  Output rows: {len(output_rows)} (filtered)")
    else:
        logger.info(f"  Output rows: {len(output_rows)} (with rejected column)")


def main():
    parser = argparse.ArgumentParser(
        description="Update manifest with rejection information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Input manifest CSV file")
    parser.add_argument("--rejections", required=True, type=Path,
                        help="Rejections CSV file (from clean_dataset.py)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output manifest CSV file")
    parser.add_argument("--filter-rejected", action="store_true",
                        help="Remove rejected rows instead of just adding column")
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        return 1
    
    if not args.rejections.exists():
        logger.error(f"Rejections file not found: {args.rejections}")
        return 1
    
    update_manifest(
        args.manifest,
        args.rejections,
        args.output,
        args.filter_rejected,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())