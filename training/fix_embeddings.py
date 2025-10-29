import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil


def fix_existing_embeddings(manifest_path, backup_dir=None):
    """
    Fix existing embeddings by removing the extra batch dimension.
    This is faster than recomputing from images.
    """
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    # Filter to entries with embeddings
    df_with_emb = df[df["layout_emb"].notna()]
    print(f"Found {len(df_with_emb)} entries with embeddings")
    
    # Create backup directory if specified
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backing up original embeddings to: {backup_dir}")
    
    fixed_count = 0
    already_correct = 0
    error_count = 0
    
    for _, row in tqdm(df_with_emb.iterrows(), total=len(df_with_emb), desc="Fixing embeddings"):
        emb_path = row["layout_emb"]
        
        try:
            # Load embedding
            emb = torch.load(emb_path, map_location='cpu')
            
            # Check if it needs fixing
            if emb.dim() == 4 and emb.shape[0] == 1:
                # Backup original if requested
                if backup_dir:
                    backup_path = os.path.join(backup_dir, Path(emb_path).name)
                    shutil.copy2(emb_path, backup_path)
                
                # Fix by removing batch dimension
                emb_fixed = emb.squeeze(0)
                
                # Save fixed embedding
                torch.save(emb_fixed, emb_path)
                fixed_count += 1
                
            elif emb.dim() == 3:
                # Already correct shape
                already_correct += 1
            else:
                raise ValueError(f"Unexpected shape: {emb.shape}")
                
        except Exception as e:
            print(f"\nError processing {emb_path}: {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Already correct: {already_correct}")
    print(f"  Errors: {error_count}")
    
    return fixed_count, already_correct, error_count


def verify_shapes(manifest_path, sample_size=10):
    """Verify shapes after fixing."""
    print(f"\nVerifying shapes from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df_with_emb = df[df["layout_emb"].notna()]
    
    samples = df_with_emb.sample(n=min(sample_size, len(df_with_emb)), random_state=42)
    
    shapes = []
    for _, row in samples.iterrows():
        try:
            emb = torch.load(row["layout_emb"], map_location='cpu')
            shapes.append(emb.shape)
            print(f"  {Path(row['layout_emb']).name}: {emb.shape}")
        except Exception as e:
            print(f"  Error: {row['layout_emb']}: {e}")
    
    unique_shapes = set(shapes) if shapes else set()
    if len(unique_shapes) == 1:
        print(f"\n✓ All sampled embeddings have shape: {shapes[0]}")
    else:
        print(f"\n⚠ Found {len(unique_shapes)} different shapes: {unique_shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix existing embeddings by removing batch dimension.")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV with embeddings")
    parser.add_argument("--backup_dir", help="Directory to backup original embeddings (optional)")
    parser.add_argument("--verify_only", action="store_true", help="Only verify shapes without fixing")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_shapes(args.manifest)
    else:
        fix_existing_embeddings(args.manifest, args.backup_dir)
        verify_shapes(args.manifest)