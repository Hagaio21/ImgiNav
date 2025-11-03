#!/usr/bin/env python3
"""
Create augmented dataset with original images + their augmentations.
This script only creates augmentations (no embedding) - CPU-bound task.

Generates augmented images and creates a manifest with all samples.
All files stored in dataset/augmented/images/

Usage:
    python data_preparation/create_augmentations.py \
        --dataset-manifest datasets/layouts_latents.csv \
        --output-dir datasets/augmented \
        --use-mirror-rotation
"""

import argparse
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preparation.create_augmented_dataset import (
    is_white_image,
    apply_mirror_flip,
    apply_rotation_90,
    generate_all_augmentations
)


def create_augmentations(
    manifest_path,
    output_dir,
    use_mirror_rotation=True,
    overwrite=False
):
    """
    Create augmented dataset with original + augmented images (no embedding).
    
    Args:
        manifest_path: Path to input manifest
        output_dir: Directory to save augmented dataset
        use_mirror_rotation: If True, generate 7 variants (3 rotations + 1 mirror + 3 rotated mirrors)
        overwrite: Whether to overwrite existing files
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} original samples")
    
    # Filter non-empty layouts
    if "is_empty" in df.columns:
        df = df[df["is_empty"] == False]
        print(f"After filtering empty: {len(df)} samples")
    
    # Filter out NaN layout paths
    layout_col = "layout_path" if "layout_path" in df.columns else "path"
    df = df.dropna(subset=[layout_col])
    df = df.reset_index(drop=True)
    
    print(f"After filtering NaN paths: {len(df)} samples")
    
    # Filter out white images (>95% whitish pixels)
    print(f"\nFiltering white images (removing images with >95% whitish pixels)...")
    white_images = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking white images"):
        image_path = Path(row[layout_col])
        if not image_path.exists():
            continue
        
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            img_array = np.ascontiguousarray(img, dtype=np.float32)
            img_tensor = torch.from_numpy(img_array)
            
            # Convert to [C, H, W] format
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            # Normalize to [0, 1] range for white detection
            img_tensor = img_tensor / 255.0
            
            # Check if white (threshold 0.9 in [0, 1] range = ~230 in [0, 255])
            if is_white_image(img_tensor, threshold=0.95, white_threshold=0.9):
                white_images.append(image_path)
            else:
                valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Error checking white image {image_path}: {e}")
            valid_indices.append(idx)
    
    # Filter dataframe
    df_filtered = df.loc[valid_indices].reset_index(drop=True)
    
    print(f"Removed {len(white_images)} white images")
    print(f"After filtering white images: {len(df_filtered)} samples")
    
    df = df_filtered
    
    # Determine augmentation strategy
    if use_mirror_rotation:
        num_augmentations = 7
        aug_description = "7 variants (3 rotations + mirror + 3 rotated mirrors)"
        aug_names = ["rot90", "rot180", "rot270", "mirror", "mirror_rot90", "mirror_rot180", "mirror_rot270"]
    else:
        num_augmentations = 3
        aug_description = "3 random variants (deprecated)"
        aug_names = [f"aug{i+1:02d}" for i in range(num_augmentations)]
    
    print(f"\n" + "="*60)
    print(f"Augmentation Strategy")
    print("="*60)
    print(f"  Original samples: {len(df)}")
    print(f"  Augmentation method: {aug_description}")
    print(f"  Augmentations per sample: {num_augmentations}")
    print(f"  Total expected samples: {len(df) * (1 + num_augmentations)}")
    
    # ============================================================
    # PHASE 1: Copy all original images to target location
    # ============================================================
    print(f"\n" + "="*60)
    print("PHASE 1: Copying original images to target location...")
    print("="*60)
    
    copied_images = {}
    failed = 0
    
    for idx in tqdm(range(len(df)), desc="Copying originals"):
        row = df.iloc[idx]
        original_image_path = Path(row[layout_col])
        
        if not original_image_path.exists():
            print(f"Warning: Original image not found: {original_image_path}")
            failed += 1
            continue
        
        base_name = original_image_path.stem
        copied_path = images_dir / f"{base_name}.png"
        
        try:
            if not copied_path.exists() or overwrite:
                shutil.copy2(original_image_path, copied_path)
            copied_images[idx] = copied_path
        except Exception as e:
            print(f"Error copying {original_image_path}: {e}")
            failed += 1
            continue
    
    print(f"Copied {len(copied_images)} original images")
    if failed > 0:
        print(f"Failed to copy {failed} images")
    
    # ============================================================
    # PHASE 2: Augment all copied images (in place)
    # ============================================================
    print(f"\n" + "="*60)
    print("PHASE 2: Creating augmentations from copied images...")
    print("="*60)
    
    augmented_images = {}
    failed = 0
    
    # Use CPU for augmentation (no GPU needed)
    device = torch.device("cpu")
    
    for idx, copied_path in tqdm(copied_images.items(), desc="Creating augmentations"):
        row = df.iloc[idx]
        base_name = copied_path.stem
        
        # Load copied image
        try:
            img = Image.open(copied_path).convert("RGB")
            img_array = np.ascontiguousarray(img, dtype=np.float32)
            img_tensor = torch.from_numpy(img_array)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1) / 127.5 - 1.0  # [0, 255] -> [-1, 1]
            img_tensor = img_tensor.to(device)
        except Exception as e:
            print(f"Error loading copied image {copied_path}: {e}")
            failed += 1
            continue
        
        # Create augmentations
        if use_mirror_rotation:
            # Generate all deterministic augmentations
            augmentation_variants = generate_all_augmentations(img_tensor)
            augmentation_variants = augmentation_variants[1:]  # Skip original (first one)
            
            for aug_tensor, aug_name in zip(augmentation_variants, aug_names):
                try:
                    aug_path = images_dir / f"{base_name}_{aug_name}.png"
                    
                    # Save augmented image
                    if not aug_path.exists() or overwrite:
                        aug_np = aug_tensor.cpu().permute(1, 2, 0).numpy()
                        aug_np = ((aug_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        aug_img = Image.fromarray(aug_np)
                        aug_img.save(aug_path)
                    
                    augmented_images[(idx, aug_name)] = aug_path
                except Exception as e:
                    print(f"Error creating augmentation {aug_name} for {copied_path}: {e}")
                    failed += 1
                    continue
        else:
            # Legacy: random augmentations (deprecated)
            for aug_idx, aug_name in enumerate(aug_names):
                try:
                    # Apply old-style random augmentation
                    angle = torch.rand(1, device=device).item() * 30 - 15
                    angle_rad = angle * np.pi / 180.0
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    img_tensor_batch = img_tensor.unsqueeze(0)
                    _, _, h, w = img_tensor_batch.shape
                    tx = (torch.rand(1, device=device).item() * 2 - 1) * 0.05 * w
                    ty = (torch.rand(1, device=device).item() * 2 - 1) * 0.05 * h
                    theta = torch.tensor([
                        [cos_a, -sin_a, tx],
                        [sin_a, cos_a, ty]
                    ], dtype=torch.float32, device=device).unsqueeze(0)
                    grid = F.affine_grid(theta, img_tensor_batch.size(), align_corners=False)
                    aug_tensor = F.grid_sample(
                        img_tensor_batch, grid, mode='bilinear', 
                        padding_mode='reflection', align_corners=False
                    ).squeeze(0)
                    
                    aug_path = images_dir / f"{base_name}_{aug_name}.png"
                    
                    if not aug_path.exists() or overwrite:
                        aug_np = aug_tensor.cpu().permute(1, 2, 0).numpy()
                        aug_np = ((aug_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        aug_img = Image.fromarray(aug_np)
                        aug_img.save(aug_path)
                    
                    augmented_images[(idx, aug_name)] = aug_path
                except Exception as e:
                    print(f"Error creating augmentation {aug_name} for {copied_path}: {e}")
                    failed += 1
                    continue
    
    print(f"Created {len(augmented_images)} augmented images")
    if failed > 0:
        print(f"Failed to create {failed} augmentations")
    
    # ============================================================
    # Create manifest with all images (no embedding yet)
    # ============================================================
    print(f"\n" + "="*60)
    print("Creating manifest with all images...")
    print("="*60)
    
    all_rows = []
    
    # Process originals
    for idx, copied_path in tqdm(copied_images.items(), desc="Adding originals to manifest"):
        row = df.iloc[idx]
        
        # Create manifest row - preserve all original metadata
        manifest_row = row.to_dict()
        manifest_row["layout_path"] = str(copied_path.relative_to(output_dir.parent))
        manifest_row["latent_path"] = None  # Will be filled by embedding script
        manifest_row["is_empty"] = False
        manifest_row["is_augmented"] = False
        manifest_row["augmentation_id"] = 0
        manifest_row["augmentation_type"] = None
        
        all_rows.append(manifest_row)
    
    # Process augmentations
    for (idx, aug_name), aug_path in tqdm(augmented_images.items(), desc="Adding augmented to manifest"):
        row = df.iloc[idx]
        
        # Create manifest row - preserve all original metadata
        manifest_row = row.to_dict()
        manifest_row["layout_path"] = str(aug_path.relative_to(output_dir.parent))
        manifest_row["latent_path"] = None  # Will be filled by embedding script
        manifest_row["is_empty"] = False
        manifest_row["is_augmented"] = True
        manifest_row["augmentation_id"] = aug_names.index(aug_name) + 1 if aug_name in aug_names else None
        manifest_row["augmentation_type"] = aug_name
        
        all_rows.append(manifest_row)
    
    # Create output manifest
    output_df = pd.DataFrame(all_rows)
    manifest_output_path = output_dir / "manifest_images.csv"  # Separate manifest for images only
    output_df.to_csv(manifest_output_path, index=False)
    
    print(f"\n" + "="*60)
    print("✓ Augmentation creation complete!")
    print("="*60)
    print(f"  Original samples: {len(copied_images)}")
    print(f"  Augmented samples: {len(augmented_images)}")
    print(f"  Total images: {len(output_df)}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"  Images manifest: {manifest_output_path}")
    print(f"\nNext step: Run embedding script to create latents")
    print(f"  python data_preparation/embed_images.py \\")
    print(f"    --images-manifest {manifest_output_path} \\")
    print(f"    --autoencoder-config <config> \\")
    print(f"    --autoencoder-checkpoint <checkpoint>")
    
    return manifest_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create augmented dataset with original + augmented images (no embedding)"
    )
    parser.add_argument(
        "--dataset-manifest",
        type=str,
        required=True,
        help="Path to input dataset manifest CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/augmented",
        help="Output directory for augmented dataset"
    )
    parser.add_argument(
        "--use-mirror-rotation",
        action="store_true",
        default=True,
        help="Use mirror + 90-degree rotations (7 variants per sample)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    create_augmentations(
        manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        use_mirror_rotation=args.use_mirror_rotation,
        overwrite=args.overwrite
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

