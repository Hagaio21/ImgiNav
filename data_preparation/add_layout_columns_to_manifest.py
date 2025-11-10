#!/usr/bin/env python3
"""
Script to add content_category and class_combination columns to manifest CSV.

- content_category: Number of distinct object classes (categories) in the layout image.
- class_combination: Sorted comma-separated string of category names (e.g., "Bed,Chair,Table").
                    Useful for identifying common vs rare layout combinations.

Only counts colors that map to actual object categories in the taxonomy.

Usage:
    # Update manifest in-place (recommended)
    python data_preparation/add_layout_columns_to_manifest.py \
        --manifest datasets/augmented/manifest.csv \
        --taxonomy config/taxonomy.json
    
    # Or save to a new file
    python data_preparation/add_layout_columns_to_manifest.py \
        --manifest datasets/augmented/manifest.csv \
        --output datasets/augmented/manifest_with_category.csv \
        --taxonomy config/taxonomy.json
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy
from data_preparation.utils.layout_analysis import (
    build_color_to_category_mapping,
    count_distinct_object_classes,
    get_object_class_combination
)


def analyze_single_layout(args_tuple):
    """
    Analyze a single layout image. Wrapper for multiprocessing.
    
    Args:
        args_tuple: (index, layout_path_str, taxonomy_path, manifest_dir)
    
    Returns:
        (index, dict) with 'content_category' (num_classes int) and 'class_combination' (string)
    """
    idx, layout_path_str, taxonomy_path, manifest_dir = args_tuple
    
    # Load taxonomy and color mapping (rebuild in each worker to avoid pickling issues)
    # Only use super-categories
    taxonomy = Taxonomy(taxonomy_path)
    color_to_category = build_color_to_category_mapping(taxonomy, super_categories_only=True)
    
    layout_path = Path(layout_path_str)
    
    # Handle relative paths
    if not layout_path.is_absolute():
        layout_path = manifest_dir / layout_path
    
    # Try to resolve the path
    if not layout_path.exists():
        # Try to find it relative to manifest directory
        layout_path = manifest_dir.parent / layout_path
        if not layout_path.exists():
            return idx, {"content_category": 0, "class_combination": "unknown"}
    
    # Count distinct colors (for content_category) - just count all distinct colors
    num_classes = count_distinct_object_classes(layout_path)
    
    # Get the combination string (for class_combination) - use super-categories only
    class_combination = get_object_class_combination(
        layout_path,
        color_to_category=color_to_category
    )
    
    return idx, {"content_category": num_classes, "class_combination": class_combination}


def add_content_category_to_manifest(
    manifest_path: Path,
    taxonomy_path: Path,
    output_path: Path = None,
    layout_column: str = "layout_path",
    workers: int = None,
    overwrite: bool = False
):
    """
    Add content_category and class_combination columns to manifest CSV.
    
    - content_category: Number of distinct object classes (categories) in the layout image.
    - class_combination: Sorted comma-separated string of category names present (e.g., "Bed,Chair,Table").
                        Useful for identifying common vs rare layout combinations.
    
    Only counts colors that map to actual object categories in the taxonomy.
    
    Args:
        manifest_path: Path to input manifest CSV (will be updated in-place if output_path is None)
        taxonomy_path: Path to taxonomy.json
        output_path: Path to output manifest CSV (optional, if None updates manifest_path in-place)
        layout_column: Name of column containing layout paths
        workers: Number of parallel workers (default: CPU count)
        overwrite: If True, overwrite existing columns
    """
    manifest_path = Path(manifest_path)
    taxonomy_path = Path(taxonomy_path)
    
    # If no output path specified, update in-place
    if output_path is None:
        output_path = manifest_path
    else:
        output_path = Path(output_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy not found: {taxonomy_path}")
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} rows")
    
    # Check if columns already exist
    existing_columns = [col for col in ["content_category", "class_combination"] if col in df.columns]
    if existing_columns and not overwrite:
        print(f"Warning: Columns already exist: {existing_columns}")
        print("Aborting. Use --overwrite flag to replace existing columns.")
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
    print(f"  - Taxonomy: {taxonomy_path}")
    print(f"  - Counting all colors that map to categories (no pixel threshold)")
    
    # Filter out rows with missing layout paths
    valid_mask = df[layout_column].notna()
    valid_df = df[valid_mask].copy()
    invalid_count = (~valid_mask).sum()
    
    if invalid_count > 0:
        print(f"Warning: {invalid_count} rows have missing layout paths, will be set to defaults")
    
    # Prepare arguments for parallel processing
    idx_mapping = {i: orig_idx for i, orig_idx in enumerate(valid_df.index)}
    args_list = [
        (i, row[layout_column], taxonomy_path, manifest_dir)
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
                results_dict[i] = {"content_category": 0, "class_combination": "unknown"}
            
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
    
    # Add columns to dataframe
    df["content_category"] = None
    df["class_combination"] = None
    
    for idx, result_dict in results_dict.items():
        original_idx = idx_mapping[idx]
        df.at[original_idx, "content_category"] = result_dict["content_category"]
        df.at[original_idx, "class_combination"] = result_dict["class_combination"]
    
    # Fill missing values and convert to int
    df["content_category"] = df["content_category"].fillna(0).astype(int)
    df["class_combination"] = df["class_combination"].fillna("unknown")
    
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
    
    # Print class_combination statistics
    print(f"\nclass_combination statistics:")
    print(f"  Total unique combinations: {df['class_combination'].nunique()}")
    print(f"\nTop 20 most common combinations:")
    combo_counts = df["class_combination"].value_counts()
    for combo, count in combo_counts.head(20).items():
        print(f"  {combo:50s}: {count:6d} ({100*count/len(df):.1f}%)")
    
    print(f"\nRare combinations (appearing only once):")
    rare_combos = combo_counts[combo_counts == 1]
    print(f"  Count: {len(rare_combos)} unique combinations")
    if len(rare_combos) > 0:
        print(f"  Examples: {', '.join(rare_combos.head(10).index.tolist())}")
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved updated manifest to {output_path}")
    print(f"Total rows: {len(df)}")
    non_null_count = df["content_category"].notna().sum()
    print(f"Rows with content_category: {non_null_count}")
    non_null_combo = df["class_combination"].notna().sum()
    print(f"Rows with class_combination: {non_null_combo}")


def main():
    parser = argparse.ArgumentParser(
        description="Add content_category and class_combination columns to manifest CSV"
    )
    parser.add_argument("--manifest", type=Path, required=True,
                       help="Path to input manifest CSV (will be updated in-place if --output not specified)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Path to output manifest CSV (optional, if not specified updates manifest in-place)")
    parser.add_argument("--taxonomy", type=Path, required=True,
                       help="Path to taxonomy.json")
    parser.add_argument("--layout_column", type=str, default="layout_path",
                       help="Column name for layout paths (default: layout_path)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing column if present")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Adding content_category to Manifest")
    print("="*60)
    print(f"Input manifest: {args.manifest}")
    if args.output:
        print(f"Output manifest: {args.output}")
    else:
        print(f"Output: Updating manifest in-place")
    print(f"Taxonomy: {args.taxonomy}")
    print(f"Layout column: {args.layout_column}")
    print("="*60)
    
    add_content_category_to_manifest(
        manifest_path=args.manifest,
        taxonomy_path=args.taxonomy,
        output_path=args.output,
        layout_column=args.layout_column,
        workers=args.workers,
        overwrite=args.overwrite
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()

