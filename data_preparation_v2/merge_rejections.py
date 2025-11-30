#!/usr/bin/env python3
"""
Merge Rejections - Combine rejection CSVs from multiple shards.

After running clean_dataset.py in parallel across multiple shards,
use this script to merge all the rejection files into a single CSV.

Usage:
    # Merge all shard files
    python merge_rejections.py \\
        --input-dir rejections/ \\
        --output rejections_merged.csv

    # Merge specific files
    python merge_rejections.py \\
        --input-files rejections_shard_001.csv rejections_shard_002.csv \\
        --output rejections_merged.csv

    # Merge with glob pattern
    python merge_rejections.py \\
        --input-pattern "rejections/rejections_shard_*.csv" \\
        --output rejections_merged.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Set
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_csv_files(input_files: List[Path], output_path: Path):
    """Merge multiple CSV files with the same schema."""
    if not input_files:
        logger.error("No input files to merge")
        return
    
    logger.info(f"Merging {len(input_files)} files...")
    
    # Collect all columns across all files (in case some have different detail columns)
    all_columns: List[str] = []
    all_rows: List[dict] = []
    seen_paths: Set[str] = set()  # Detect duplicates
    
    for input_file in input_files:
        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            continue
        
        logger.info(f"  Reading: {input_file}")
        
        with open(input_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Update column list
            if reader.fieldnames:
                for col in reader.fieldnames:
                    if col not in all_columns:
                        all_columns.append(col)
            
            # Read rows
            file_rows = 0
            duplicates = 0
            for row in reader:
                path = row.get("path", "")
                if path in seen_paths:
                    duplicates += 1
                    continue
                seen_paths.add(path)
                all_rows.append(row)
                file_rows += 1
            
            logger.info(f"    Rows: {file_rows}, Duplicates skipped: {duplicates}")
    
    if not all_rows:
        logger.warning("No rows to write")
        return
    
    # Write merged output
    logger.info(f"Writing {len(all_rows)} rows to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(all_rows)
    
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  Written: {size_kb:.2f} KB")
    
    # Summary stats
    rejected_count = sum(1 for r in all_rows if r.get("rejected", "").lower() == "true")
    logger.info(f"\nMerge Summary:")
    logger.info(f"  Total files merged: {len(input_files)}")
    logger.info(f"  Total records: {len(all_rows)}")
    logger.info(f"  Rejected: {rejected_count} ({100*rejected_count/len(all_rows):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge rejection CSVs from multiple shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (mutually exclusive-ish, but can combine)
    parser.add_argument("--input-dir", type=Path,
                        help="Directory containing rejection CSV files")
    parser.add_argument("--input-files", nargs="+", type=Path,
                        help="Specific CSV files to merge")
    parser.add_argument("--input-pattern", type=str,
                        help="Glob pattern for input files (e.g., 'rejections_shard_*.csv')")
    
    # Output
    parser.add_argument("--output", required=True, type=Path,
                        help="Output merged CSV file")
    
    args = parser.parse_args()
    
    # Collect input files
    input_files = []
    
    if args.input_dir:
        input_files.extend(sorted(args.input_dir.glob("*.csv")))
    
    if args.input_files:
        input_files.extend(args.input_files)
    
    if args.input_pattern:
        input_files.extend([Path(p) for p in sorted(glob.glob(args.input_pattern))])
    
    if not input_files:
        logger.error("No input files specified. Use --input-dir, --input-files, or --input-pattern")
        return 1
    
    # Deduplicate input files
    input_files = list(dict.fromkeys(input_files))
    
    merge_csv_files(input_files, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())