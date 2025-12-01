#!/usr/bin/env python3
"""
Add Sample ID to Manifest

Adds a unique sample_id column to manifest CSV files. Each row gets a unique ID
that can be used to track samples through the cleaning/rejection process.

Usage:
    python add_sample_id.py \\
        --manifest manifest_tex.csv \\
        --output manifest_tex_with_ids.csv

    # Process multiple manifests
    python add_sample_id.py \\
        --manifest-seg manifest_seg.csv \\
        --manifest-tex manifest_tex.csv \\
        --output-dir manifests/
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_sample_id_to_manifest(
    manifest_path: Path,
    output_path: Path,
    id_prefix: Optional[str] = None,
):
    """Add sample_id column to manifest."""
    
    logger.info(f"Processing: {manifest_path}")
    
    # Read manifest
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames)
        rows = list(reader)
    
    logger.info(f"  Read {len(rows)} rows")
    
    # Check if sample_id already exists
    if "sample_id" in columns:
        logger.warning("  sample_id column already exists, skipping")
        return
    
    # Add sample_id as first column
    output_columns = ["sample_id"] + columns
    
    # Generate sample IDs
    if id_prefix is None:
        # Use manifest stem as prefix (e.g., "manifest_tex" -> "tex")
        prefix = manifest_path.stem.replace("manifest_", "")
        if prefix == manifest_path.stem:
            prefix = "sample"
    else:
        prefix = id_prefix
    
    # Add sample_id to each row
    for idx, row in enumerate(rows, start=1):
        # Format: {prefix}_{zero_padded_index}
        # e.g., tex_00000001, seg_00000001
        sample_id = f"{prefix}_{idx:08d}"
        row["sample_id"] = sample_id
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"  Added sample_id column")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Sample ID range: {prefix}_00000001 to {prefix}_{len(rows):08d}")


def main():
    parser = argparse.ArgumentParser(
        description="Add sample_id column to manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--manifest", type=Path,
                        help="Input manifest CSV")
    parser.add_argument("--manifest-seg", type=Path,
                        help="Input manifest_seg.csv")
    parser.add_argument("--manifest-tex", type=Path,
                        help="Input manifest_tex.csv")
    parser.add_argument("--output", type=Path,
                        help="Output manifest CSV (for single manifest)")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (for multiple manifests)")
    parser.add_argument("--id-prefix", type=str,
                        help="Prefix for sample IDs (default: auto-detect from filename)")
    
    args = parser.parse_args()
    
    manifests = []
    if args.manifest:
        if not args.output:
            logger.error("--output required when using --manifest")
            return 1
        manifests.append((args.manifest, args.output))
    
    if args.manifest_seg or args.manifest_tex:
        if not args.output_dir:
            logger.error("--output-dir required when using --manifest-seg or --manifest-tex")
            return 1
        
        if args.manifest_seg:
            output_seg = args.output_dir / "manifest_seg_with_ids.csv"
            manifests.append((args.manifest_seg, output_seg))
        
        if args.manifest_tex:
            output_tex = args.output_dir / "manifest_tex_with_ids.csv"
            manifests.append((args.manifest_tex, output_tex))
    
    if not manifests:
        logger.error("No manifests specified. Use --manifest or --manifest-seg/--manifest-tex")
        return 1
    
    # Process each manifest
    for manifest_path, output_path in manifests:
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}, skipping")
            continue
        
        add_sample_id_to_manifest(
            manifest_path,
            output_path,
            id_prefix=args.id_prefix,
        )
    
    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())

