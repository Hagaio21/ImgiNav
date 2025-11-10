#!/usr/bin/env python3
"""
Script to add content_category column to manifest CSV.

content_category is the number of distinct colors/classes in the layout image.

Usage:
    python data_preparation/add_layout_columns_to_manifest.py \
        --manifest datasets/augmented/manifest.csv \
        --output datasets/augmented/manifest_with_category.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preparation.utils.layout_analysis import count_distinct_colors


def analyze_single_layout(args_tuple):
    """
    Analyze a single layout image. Wrapper for multiprocessing.
    
    Args:
        args_tuple: (index, layout_path_str, manifest_dir, exclude_background, min_pixel_threshold)
    
    Returns:
        (index, num_classes int) - number of distinct colors/classes in the layout
    """
    idx, layout_path_str, manifest_dir, exclude_background, min_pixel_threshold = args_tuple
    
    layout_path = Path(layout_path_str)
    
    # Handle relative paths
    if not layout_path.is_absolute():
        layout_path = manifest_dir / layout_path
    
    # Try to resolve the path
    if not layout_path.exists():
        # Try to find it relative to manifest directory
        layout_path = manifest_dir.parent / layout_path
        if not layout_path.exists():
            return idx, 0
    
    # Count distinct colors/classes in layout
    num_classes = count_distinct_colors(
        layout_path,
        exclude_background=exclude_background,
        min_pixel_threshold=min_pixel_threshold
    )
    return idx, num_classes


def add_content_category_to_manifest(
    manifest_path: Path,
    output_path: Path,
    layout_column: str = "layout_path",
    workers: int = None,
    exclude_background: bool = True,
    min_pixel_threshold: int = 0,
    overwrite: bool = False
):
    """
    Add content_category column to manifest CSV.
    
    content_category is the number of distinct colors/classes in the layout image.
    
    Args:
        manifest_path: Path to input manifest CSV
        output_path: Path to output manifest CSV
        layout_column: Name of column containing layout paths
        workers: Number of parallel workers (default: CPU count)
        exclude_background: If True, exclude white/gray background colors from count
        min_pixel_threshold: Minimum pixels for a color to be counted
        overwrite: If True, overwrite existing column
    """
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} rows")
    
    # Check if column already exists
    if "content_category" in df.columns and not overwrite:
        print(f"Warning: content_category column already exists. Use --overwrite to replace it.")
        print("Aborting. Use --overwrite flag to replace existing column.")
        return
    
    # Check for layout column
    if layout_column not in df.columns:
        # Try common alternatives
        alternatives = ["path", "image_path", "layout_image"]
        for alt in alternatives:
            if alt in df.columns:
                layout_column = alt
                print(f"Using column '{layout_column}' for layout paths")
                break
        else:
            raise ValueError(f"Layout column '{layout_column}' not found. Available columns: {list(df.columns)}")
    
    # Get manifest directory for resolving relative paths
    manifest_dir = manifest_path.parent
    
    # Prepare arguments for multiprocessing
    if workers is None:
        import multiprocessing
        workers = multiprocessing.cpu_count()
    
    print(f"Analyzing layouts with {workers} workers...")
    print(f"  - exclude_background={exclude_background}")
    print(f"  - min_pixel_threshold={min_pixel_threshold}")
    
    # Filter out rows with missing layout paths
    valid_mask = df[layout_column].notna()
    valid_df = df[valid_mask].copy()
    invalid_count = (~valid_mask).sum()
    
    if invalid_count > 0:
        print(f"Warning: {invalid_count} rows have missing layout paths, will be set to defaults")
    
    # Prepare arguments for parallel processing
    idx_mapping = {i: orig_idx for i, orig_idx in enumerate(valid_df.index)}
    args_list = [
        (i, row[layout_column], manifest_dir, exclude_background, min_pixel_threshold)
        for i, (_, row) in enumerate(valid_df.iterrows())
    ]
    
    # Process in parallel
    results_dict = {}
    completed = 0
    total = len(valid_df)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(analyze_single_layout, args): i
            for i, args in enumerate(args_list)
        }
        
        # Collect results
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                idx, result = future.result()
                results_dict[idx] = result
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                results_dict[i] = 0
            
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
    
    # Add column to dataframe
    df["content_category"] = None
    
    for idx, value in results_dict.items():
        original_idx = idx_mapping[idx]
        df.at[original_idx, "content_category"] = value
    
    # Fill missing values and convert to int
    df["content_category"] = df["content_category"].fillna(0).astype(int)
    
    # Print statistics
    print(f"\ncontent_category statistics:")
    print(f"  Mean: {df['content_category'].mean():.2f}")
    print(f"  Median: {df['content_category'].median():.2f}")
    print(f"  Min: {df['content_category'].min()}")
    print(f"  Max: {df['content_category'].max()}")
    print(f"  Std: {df['content_category'].std():.2f}")
    print(f"\ncontent_category distribution:")
    value_counts = df["content_category"].value_counts().sort_index()
    for val, count in value_counts.head(20).items():
        print(f"  {val:3d} classes: {count:6d} ({100*count/len(df):.1f}%)")
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved updated manifest to {output_path}")
    print(f"Total rows: {len(df)}")
    non_null_count = df["content_category"].notna().sum()
    print(f"Rows with content_category: {non_null_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Add content_category column to manifest CSV (number of distinct colors/classes)"
    )
    parser.add_argument("--manifest", type=Path, required=True,
                       help="Path to input manifest CSV")
    parser.add_argument("--output", type=Path, required=True,
                       help="Path to output manifest CSV")
    parser.add_argument("--layout_column", type=str, default="layout_path",
                       help="Column name for layout paths (default: layout_path)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--include_background", action="store_true",
                       help="Include background/white colors in count (default: exclude)")
    parser.add_argument("--min_pixel_threshold", type=int, default=0,
                       help="Minimum pixels for a color to be counted (default: 0)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing column if present")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Adding content_category to Manifest")
    print("="*60)
    print(f"Input manifest: {args.manifest}")
    print(f"Output manifest: {args.output}")
    print(f"Layout column: {args.layout_column}")
    print(f"Exclude background: {not args.include_background}")
    print(f"Min pixel threshold: {args.min_pixel_threshold}")
    print("="*60)
    
    add_content_category_to_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        layout_column=args.layout_column,
        workers=args.workers,
        exclude_background=not args.include_background,
        min_pixel_threshold=args.min_pixel_threshold,
        overwrite=args.overwrite
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()

