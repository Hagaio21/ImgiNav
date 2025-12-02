#!/usr/bin/env python3
"""
Add Rejections to Manifest

Joins the rejections CSV with manifest CSVs to add rejected column.

Usage:
    # Update specific manifests
    python add_rejections_to_manifest.py \\
        --rejections rejections_merged.csv \\
        --manifest-seg manifest_seg.csv \\
        --manifest-tex manifest_tex.csv \\
        --output-dir manifests/
    
    # Update any manifest(s) with custom output paths
    python add_rejections_to_manifest.py \\
        --rejections rejections_merged.csv \\
        --manifest manifest_seg.csv manifest_vae_latent.csv \\
        --output manifest_seg_with_rejections.csv manifest_vae_latent_with_rejections.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_rejections(rejections_path: Path, original_manifest_path: Path = None) -> Dict[str, dict]:
    """
    Load rejections CSV into a dict keyed by sample_id and optionally layout_path.
    
    If original_manifest_path is provided, also creates a mapping from layout_path -> sample_id
    to help match manifests that don't have sample_id.
    
    Returns:
        Tuple of:
        - Dict mapping sample_id -> {rejected, rejection_reason, ...}
        - Dict mapping layout_path -> sample_id (if original_manifest provided)
    """
    rejections = {}
    layout_to_sample_id = {}
    
    # Load original manifest to create layout_path -> sample_id mapping
    if original_manifest_path and original_manifest_path.exists():
        logger.info(f"Loading original manifest for path mapping: {original_manifest_path}")
        with open(original_manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get("sample_id", "")
                layout_path = row.get("layout_path", "")
                if sample_id and layout_path:
                    layout_to_sample_id[layout_path] = sample_id
    
    with open(rejections_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id", "")
            rejected = row.get("rejected", "False")
            reason = row.get("rejection_reason", "")
            
            if sample_id:
                # New format: use sample_id
                rejections[sample_id] = {
                    "rejected": rejected,
                    "rejection_reason": reason,
                    **{k: v for k, v in row.items() if k not in ["sample_id", "rejected", "rejection_reason"]}
                }
            else:
                # Old format: fallback to layout_path
                seg_path = row.get("layout_path_seg", "")
                tex_path = row.get("layout_path_tex", "")
                
                info = {
                    "rejected": rejected,
                    "rejection_reason": reason,
                }
                
                if seg_path:
                    rejections[seg_path] = info
                if tex_path:
                    rejections[tex_path] = info
    
    return rejections, layout_to_sample_id


def add_rejections_to_manifest(
    manifest_path: Path,
    rejections: Dict[str, dict],
    output_path: Path,
    layout_to_sample_id: Dict[str, str] = None,
):
    """Add rejected column to manifest."""
    
    logger.info(f"Processing: {manifest_path}")
    
    # Read manifest
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames)
        rows = list(reader)
    
    logger.info(f"  Read {len(rows)} rows")
    
    # Check if manifest has sample_id column
    has_sample_id = "sample_id" in columns
    
    # If no sample_id and we have layout_to_sample_id mapping, add sample_id
    if not has_sample_id and layout_to_sample_id:
        logger.info("  Adding sample_id column from layout_path mapping")
        for row in rows:
            layout_path = row.get("layout_path", "")
            if layout_path and layout_path in layout_to_sample_id:
                row["sample_id"] = layout_to_sample_id[layout_path]
            else:
                row["sample_id"] = ""  # Empty if no match
        
        # Update columns list
        if "sample_id" not in columns:
            columns = columns + ["sample_id"]
        has_sample_id = True
    
    # Add new columns
    output_columns = columns + ["rejected", "rejection_reason"]
    
    # Match rejections
    matched = 0
    rejected_count = 0
    empty_sample_ids = 0
    missing_in_rejections = 0
    
    for row in rows:
        matched_this_row = False
        
        if has_sample_id:
            # Try to match by sample_id first
            sample_id = row.get("sample_id", "")
            if not sample_id or sample_id == "":
                empty_sample_ids += 1
            elif sample_id in rejections:
                matched += 1
                matched_this_row = True
                info = rejections[sample_id]
                row["rejected"] = info["rejected"]
                row["rejection_reason"] = info["rejection_reason"]
                if str(info["rejected"]).lower() == "true":
                    rejected_count += 1
            else:
                # sample_id exists but not in rejections
                missing_in_rejections += 1
        
        # If not matched by sample_id, try matching by layout_path -> sample_id
        if not matched_this_row and layout_to_sample_id:
            layout_path = row.get("layout_path", "")
            if layout_path and layout_path in layout_to_sample_id:
                sample_id = layout_to_sample_id[layout_path]
                if sample_id in rejections:
                    matched += 1
                    matched_this_row = True
                    info = rejections[sample_id]
                    row["rejected"] = info["rejected"]
                    row["rejection_reason"] = info["rejection_reason"]
                    if str(info["rejected"]).lower() == "true":
                        rejected_count += 1
        
        # If not matched, try direct layout_path match (old format fallback)
        if not matched_this_row:
            layout_path = row.get("layout_path", "")
            if layout_path and layout_path in rejections:
                matched += 1
                matched_this_row = True
                info = rejections[layout_path]
                row["rejected"] = info["rejected"]
                row["rejection_reason"] = info["rejection_reason"]
                if str(info["rejected"]).lower() == "true":
                    rejected_count += 1
        
        # If still not matched, assume not rejected
        if not matched_this_row:
            row["rejected"] = "False"
            row["rejection_reason"] = ""
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)
    
    match_key = "samples" if has_sample_id else "layouts"
    logger.info(f"  Matched: {matched}/{len(rows)} {match_key}")
    logger.info(f"  Rejected: {rejected_count}")
    if has_sample_id:
        unmatched = len(rows) - matched
        if unmatched > 0:
            logger.warning(f"  Unmatched: {unmatched} samples")
            if empty_sample_ids > 0:
                logger.warning(f"    - Empty/missing sample_id: {empty_sample_ids}")
            if missing_in_rejections > 0:
                logger.warning(f"    - sample_id not found in rejections: {missing_in_rejections}")
    logger.info(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add rejection column to manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--rejections", required=True, type=Path,
                        help="Rejections CSV from clean_dataset.py (merged)")
    parser.add_argument("--original-manifest", type=Path,
                        help="Original manifest with sample_id (e.g., manifest_seg.csv) - used to map layout_path -> sample_id for manifests without sample_id")
    parser.add_argument("--manifest-seg", type=Path,
                        help="Input manifest_seg.csv (legacy option)")
    parser.add_argument("--manifest-tex", type=Path,
                        help="Input manifest_tex.csv (legacy option)")
    parser.add_argument("--manifest", type=Path, nargs="+",
                        help="Input manifest file(s) to update (can specify multiple)")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory for new manifests (used with --manifest-seg/--manifest-tex)")
    parser.add_argument("--output", type=Path, nargs="+",
                        help="Output file path(s) for --manifest (must match number of manifests)")
    
    args = parser.parse_args()
    
    if not args.rejections.exists():
        logger.error(f"Rejections file not found: {args.rejections}")
        return 1
    
    # Load rejections and path mapping
    logger.info(f"Loading rejections from: {args.rejections}")
    rejections, layout_to_sample_id = load_rejections(args.rejections, args.original_manifest)
    
    # Check if rejections use sample_id or layout_path
    first_key = next(iter(rejections.keys())) if rejections else ""
    uses_sample_id = first_key and not ("/" in first_key or "\\" in first_key)
    key_type = "sample_ids" if uses_sample_id else "layout paths"
    logger.info(f"  Loaded {len(rejections)} {key_type}")
    
    # Count rejected
    rejected_count = sum(1 for v in rejections.values() 
                        if str(v.get("rejected", "")).lower() == "true")
    logger.info(f"  Rejected: {rejected_count}")
    
    # Process manifests
    processed = 0
    
    # New flexible mode: --manifest with --output
    if args.manifest:
        if not args.output:
            logger.error("--output is required when using --manifest")
            return 1
        
        if len(args.manifest) != len(args.output):
            logger.error(f"Number of manifests ({len(args.manifest)}) must match number of outputs ({len(args.output)})")
            return 1
        
        for manifest_path, output_path in zip(args.manifest, args.output):
            if not manifest_path.exists():
                logger.warning(f"Manifest not found: {manifest_path}, skipping")
                continue
            
            add_rejections_to_manifest(manifest_path, rejections, output_path, layout_to_sample_id)
            processed += 1
    
    # Legacy mode: --manifest-seg/--manifest-tex with --output-dir
    if args.output_dir:
        if args.manifest_seg and args.manifest_seg.exists():
            output_seg = args.output_dir / "manifest_seg_with_rejections.csv"
            add_rejections_to_manifest(args.manifest_seg, rejections, output_seg, layout_to_sample_id)
            processed += 1
        
        if args.manifest_tex and args.manifest_tex.exists():
            output_tex = args.output_dir / "manifest_tex_with_rejections.csv"
            add_rejections_to_manifest(args.manifest_tex, rejections, output_tex, layout_to_sample_id)
            processed += 1
    
    if processed == 0:
        logger.warning("No manifests were processed. Check your arguments.")
        return 1
    
    logger.info(f"\nDone! Processed {processed} manifest(s)")
    return 0


if __name__ == "__main__":
    exit(main())