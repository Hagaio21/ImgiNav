#!/usr/bin/env python3
"""
Fix misaligned CSV manifest files.

This script fixes CSV files where column headers have been inserted as data values,
causing column misalignment.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def fix_manifest_csv(input_path: Path, output_path: Path, expected_columns: list = None):
    """
    Fix a misaligned CSV manifest file.
    
    Args:
        input_path: Path to the broken CSV file
        output_path: Path to save the fixed CSV file
        expected_columns: Optional list of expected column names in order
    """
    print(f"Reading CSV from: {input_path}")
    
    # Read CSV with error handling
    try:
        # First, read as text to check for issues
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Found {len(lines)} lines in CSV")
        
        # Read header
        if len(lines) == 0:
            print("ERROR: CSV file is empty")
            sys.exit(1)
        
        header_line = lines[0].strip()
        headers = [h.strip() for h in header_line.split(',')]
        print(f"Header has {len(headers)} columns: {headers[-5:]}...")  # Show last 5
        
        # Check if expected_columns is provided
        if expected_columns:
            # Use expected columns if provided
            if len(headers) != len(expected_columns):
                print(f"WARNING: Header has {len(headers)} columns, expected {len(expected_columns)}")
                print("Using expected columns instead")
                headers = expected_columns
        else:
            # Try to detect the correct header
            # Look for the last expected column name
            if 'latent_path_vae_clip_spatial' in headers:
                # Header seems correct, but check if it's in the wrong position
                idx = headers.index('latent_path_vae_clip_spatial')
                if idx < len(headers) - 1:
                    print(f"WARNING: 'latent_path_vae_clip_spatial' found at index {idx}, expected at end")
        
        # Read CSV with pandas, handling potential issues
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
        
        print(f"Loaded {len(df)} rows")
        print(f"DataFrame has {len(df.columns)} columns")
        
        # Check for misalignment - look for rows where column name appears as value
        problematic_rows = []
        for idx, row in df.iterrows():
            # Check if any cell contains a column header name (indicating misalignment)
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip() in df.columns:
                    if str(row[col]).strip() == col:
                        continue  # Skip if it's the same column
                    problematic_rows.append((idx, col, str(row[col])))
        
        if problematic_rows:
            print(f"\nWARNING: Found {len(problematic_rows)} potentially misaligned cells")
            for idx, col, val in problematic_rows[:5]:  # Show first 5
                print(f"  Row {idx}, Column '{col}': '{val}'")
        
        # Ensure all expected columns exist
        if expected_columns:
            for col in expected_columns:
                if col not in df.columns:
                    print(f"Adding missing column: {col}")
                    df[col] = ""
        
        # Reorder columns to match expected order if provided
        if expected_columns:
            # Get columns that exist in both
            existing_cols = [c for c in expected_columns if c in df.columns]
            # Add any extra columns that weren't expected
            extra_cols = [c for c in df.columns if c not in expected_columns]
            df = df[existing_cols + extra_cols]
        
        # Check for rows with wrong number of columns
        # This is tricky with CSV, but we can check if any row has NaN in unexpected places
        print(f"\nChecking data integrity...")
        
        # Remove any rows where the last expected column has a column name as value
        if expected_columns and len(expected_columns) > 0:
            last_col = expected_columns[-1]
            if last_col in df.columns:
                # Find rows where last column contains a column name (indicating misalignment)
                mask = df[last_col].isin(df.columns)
                if mask.any():
                    print(f"WARNING: Found {mask.sum()} rows where '{last_col}' contains a column name")
                    print("These rows will be cleaned...")
                    # Replace column names with empty string in the last column
                    df.loc[mask, last_col] = ""
        
        # Save fixed CSV
        print(f"\nSaving fixed CSV to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Fixed CSV saved successfully")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names: {list(df.columns)[-3:]}")
        
    except Exception as e:
        print(f"ERROR: Failed to fix CSV: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_default_expected_columns():
    """Get the default expected columns for the shared embeddings manifest."""
    return [
        "scene_id",
        "room_id",
        "type",
        "layout_path",
        "graph_json_path",
        "graph_text_path",
        "pov_path",
        "latent_path",
        "pov_embedding_path",
        "graph_embedding_path",
        "latent_path_new_layouts_VAE_32x32_structural_256_clip",
        "latent_ae_new_layouts_VAE_32x32_structural_256_clip",
        "latent_path_vae_clip",
        "latent_ae_vae_clip",
        "latent_path_vae_clip_spatial",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Fix misaligned CSV manifest files"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--expected-columns",
        type=str,
        nargs="+",
        help="Expected column names in order (optional, defaults to shared embeddings manifest columns)"
    )
    parser.add_argument(
        "--use-default-columns",
        action="store_true",
        help="Use default expected columns for shared embeddings manifest"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    expected_columns = args.expected_columns
    if args.use_default_columns and not expected_columns:
        expected_columns = get_default_expected_columns()
        print(f"Using default expected columns ({len(expected_columns)} columns)")
    
    fix_manifest_csv(
        args.input,
        args.output,
        expected_columns=expected_columns
    )


if __name__ == "__main__":
    main()

