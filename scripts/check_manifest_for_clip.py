#!/usr/bin/env python3
"""
Check if manifest has the required columns for CLIP loss training.

Required columns:
- layout_path: Path to layout images
- graph_embedding_path: Path to graph/text embeddings (.pt files)
- pov_embedding_path: Path to POV embeddings (.pt files)
"""

import pandas as pd
import sys
from pathlib import Path

def check_manifest(manifest_path):
    """Check if manifest has required columns and valid data."""
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return False
    
    print(f"Checking manifest: {manifest_path}")
    print("=" * 60)
    
    # Read manifest
    df = pd.read_csv(manifest_path, low_memory=False)
    print(f"Total rows: {len(df)}")
    print()
    
    # Check required columns
    required_cols = {
        "layout_path": "Layout images (RGB)",
        "graph_embedding_path": "Graph/text embeddings (.pt files)",
        "pov_embedding_path": "POV embeddings (.pt files)"
    }
    
    missing_cols = []
    for col, desc in required_cols.items():
        if col not in df.columns:
            missing_cols.append(col)
            print(f"❌ Missing column: {col} ({desc})")
        else:
            # Check for non-null values
            non_null = df[col].notna().sum()
            print(f"✓ Column: {col} ({desc})")
            print(f"  Non-null values: {non_null}/{len(df)}")
            
            # Check if files exist (sample first 10)
            if non_null > 0:
                sample_paths = df[col].dropna().head(10)
                existing = 0
                for path_str in sample_paths:
                    path = Path(path_str)
                    if not path.is_absolute():
                        # Try relative to manifest directory
                        path = manifest_path.parent / path
                    if path.exists():
                        existing += 1
                print(f"  Sample check (first 10): {existing}/10 files exist")
    
    print()
    if missing_cols:
        print("=" * 60)
        print("❌ MANIFEST IS MISSING REQUIRED COLUMNS")
        print("=" * 60)
        print(f"Missing columns: {', '.join(missing_cols)}")
        print()
        print("You need to create embeddings first using:")
        print("  python training/embed_controlnet_dataset.py \\")
        print("    --ae-checkpoint <vae_checkpoint> \\")
        print("    --ae-config <vae_config> \\")
        print("    --input-manifest <input_manifest> \\")
        print("    --output-manifest <output_manifest>")
        return False
    else:
        print("=" * 60)
        print("✓ MANIFEST HAS ALL REQUIRED COLUMNS")
        print("=" * 60)
        print()
        print("You can now train the VAE with CLIP loss using:")
        print("  python training/train.py experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_manifest_for_clip.py <manifest_path>")
        print()
        print("Example:")
        print("  python scripts/check_manifest_for_clip.py /work3/s233249/ImgiNav/datasets/controlnet/manifest_seg.csv")
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    success = check_manifest(manifest_path)
    sys.exit(0 if success else 1)

