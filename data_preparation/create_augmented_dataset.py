#!/usr/bin/env python3
"""
Create augmented dataset with original images + their augmentations.
Generates augmented images, encodes them to latents, and creates a manifest.
All files stored in dataset/augmented/
"""

import argparse
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.utils import load_config


def apply_spatial_augmentation(image_tensor, rotation_range=15.0, translate_range=0.05):
    """
    Apply random rotation and translation to an image tensor.
    
    Args:
        image_tensor: Tensor of shape [C, H, W] or [1, C, H, W]
        rotation_range: Maximum rotation in degrees
        translate_range: Maximum translation as fraction of size
    
    Returns:
        Augmented image tensor
    """
    # Ensure batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    batch_size = image_tensor.shape[0]
    device = image_tensor.device
    
    # Random rotation angle in degrees
    angle = torch.rand(1, device=device).item() * (2 * rotation_range) - rotation_range
    
    # Random translation (x, y shifts)
    max_translate = translate_range
    tx = (torch.rand(1, device=device).item() * 2 - 1) * max_translate
    ty = (torch.rand(1, device=device).item() * 2 - 1) * max_translate
    
    # Create affine transformation matrix
    angle_rad = angle * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    _, _, h, w = image_tensor.shape
    translate_x = tx * w
    translate_y = ty * h
    
    theta = torch.tensor([
        [cos_a, -sin_a, translate_x],
        [sin_a, cos_a, translate_y]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Create grid for sampling
    grid = F.affine_grid(theta, image_tensor.size(), align_corners=False)
    
    # Apply transformation with reflection padding
    augmented = F.grid_sample(
        image_tensor, 
        grid, 
        mode='bilinear', 
        padding_mode='reflection',
        align_corners=False
    )
    
    # Remove batch dimension if it was added
    if augmented.shape[0] == 1:
        augmented = augmented.squeeze(0)
    
    return augmented


def create_augmented_dataset(
    encoder,
    manifest_path,
    output_dir,
    num_augmentations=3,
    rotation_range=15.0,
    translate_range=0.05,
    batch_size=32,
    num_workers=8,
    overwrite=False
):
    """
    Create augmented dataset with original + augmented images and latents.
    
    Args:
        encoder: Autoencoder encoder for creating latents
        manifest_path: Path to input manifest
        output_dir: Directory to save augmented dataset (dataset/augmented/)
        num_augmentations: Number of augmented versions per original image
        rotation_range: Maximum rotation in degrees
        translate_range: Maximum translation as fraction
        batch_size: Batch size for encoding
        num_workers: Number of DataLoader workers
        overwrite: Whether to overwrite existing files
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    latents_dir = output_dir / "latents"
    images_dir.mkdir(exist_ok=True)
    latents_dir.mkdir(exist_ok=True)
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} original samples")
    
    # Filter non-empty layouts
    if "is_empty" in df.columns:
        df = df[df["is_empty"] == False]
    
    # Filter out NaN layout paths
    layout_col = "layout_path" if "layout_path" in df.columns else "path"
    df = df.dropna(subset=[layout_col])
    df = df.reset_index(drop=True)
    
    print(f"After filtering: {len(df)} samples")
    
    # Create dataset for loading images
    dataset = ManifestDataset(
        manifest=str(manifest_path),
        outputs={"rgb": layout_col},
        return_path=True
    )
    
    # Filter dataset to match df
    if len(dataset) != len(df):
        # Create filtered dataset
        dataset = ManifestDataset(
            manifest=str(manifest_path),
            outputs={"rgb": layout_col},
            filters={"is_empty": [False]} if "is_empty" in df.columns else None,
            return_path=True
        )
    
    device = next(encoder.parameters()).device
    
    # Process all samples
    all_rows = []
    processed_originals = 0
    processed_augmented = 0
    skipped = 0
    failed = 0
    
    print(f"\nCreating augmented dataset...")
    print(f"  Original samples: {len(df)}")
    print(f"  Augmentations per sample: {num_augmentations}")
    print(f"  Total expected samples: {len(df) * (1 + num_augmentations)}")
    
    with torch.no_grad():
        for idx in tqdm(range(len(df)), desc="Processing"):
            row = df.iloc[idx]
            
            # Get original image path
            original_image_path = Path(row[layout_col])
            if not original_image_path.exists():
                print(f"Warning: Original image not found: {original_image_path}")
                failed += 1
                continue
            
            # Load original image
            try:
                original_img = Image.open(original_image_path).convert("RGB")
                # Convert to tensor [C, H, W] in [-1, 1] range
                img_array = np.ascontiguousarray(original_img, dtype=np.float32)
                img_tensor = torch.from_numpy(img_array)
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.permute(2, 0, 1) / 127.5 - 1.0  # [0, 255] -> [-1, 1]
                img_tensor = img_tensor.to(device)
            except Exception as e:
                print(f"Error loading image {original_image_path}: {e}")
                failed += 1
                continue
            
            # Create base filename from original (without extension)
            base_name = original_image_path.stem
            scene_id = row.get("scene_id", "unknown")
            room_id = row.get("room_id", "unknown")
            
            # Process original image
            # Save original image to augmented directory
            original_output_path = images_dir / f"{base_name}_orig.png"
            try:
                if not original_output_path.exists() or overwrite:
                    shutil.copy2(original_image_path, original_output_path)
                
                # Encode to latent
                latent_output_path = latents_dir / f"{base_name}_orig.pt"
                if not latent_output_path.exists() or overwrite:
                    encoder_out = encoder(img_tensor.unsqueeze(0))
                    if "latent" in encoder_out:
                        latent = encoder_out["latent"][0].cpu()
                    elif "mu" in encoder_out:
                        latent = encoder_out["mu"][0].cpu()
                    else:
                        raise ValueError(f"Encoder output must contain 'latent' or 'mu'")
                    torch.save(latent, latent_output_path)
                
                # Add to manifest
                all_rows.append({
                    "layout_path": str(original_output_path.relative_to(output_dir.parent)),
                    "latent_path": str(latent_output_path.relative_to(output_dir.parent)),
                    "scene_id": scene_id,
                    "room_id": room_id,
                    "is_empty": row.get("is_empty", False),
                    "is_augmented": False,
                    "augmentation_id": 0,
                    "original_path": str(original_image_path)
                })
                processed_originals += 1
            except Exception as e:
                print(f"Error processing original {original_image_path}: {e}")
                failed += 1
                continue
            
            # Create augmented versions
            for aug_idx in range(num_augmentations):
                try:
                    # Apply augmentation
                    augmented_tensor = apply_spatial_augmentation(
                        img_tensor,
                        rotation_range=rotation_range,
                        translate_range=translate_range
                    )
                    
                    # Save augmented image
                    aug_output_path = images_dir / f"{base_name}_aug{aug_idx+1:02d}.png"
                    if not aug_output_path.exists() or overwrite:
                        # Convert back to PIL and save
                        aug_np = augmented_tensor.cpu().permute(1, 2, 0).numpy()
                        aug_np = ((aug_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        aug_img = Image.fromarray(aug_np)
                        aug_img.save(aug_output_path)
                    
                    # Encode to latent
                    latent_output_path = latents_dir / f"{base_name}_aug{aug_idx+1:02d}.pt"
                    if not latent_output_path.exists() or overwrite:
                        encoder_out = encoder(augmented_tensor.unsqueeze(0))
                        if "latent" in encoder_out:
                            latent = encoder_out["latent"][0].cpu()
                        elif "mu" in encoder_out:
                            latent = encoder_out["mu"][0].cpu()
                        else:
                            raise ValueError(f"Encoder output must contain 'latent' or 'mu'")
                        torch.save(latent, latent_output_path)
                    
                    # Add to manifest
                    all_rows.append({
                        "layout_path": str(aug_output_path.relative_to(output_dir.parent)),
                        "latent_path": str(latent_output_path.relative_to(output_dir.parent)),
                        "scene_id": scene_id,
                        "room_id": room_id,
                        "is_empty": row.get("is_empty", False),
                        "is_augmented": True,
                        "augmentation_id": aug_idx + 1,
                        "original_path": str(original_image_path)
                    })
                    processed_augmented += 1
                    
                except Exception as e:
                    print(f"Error creating augmentation {aug_idx+1} for {original_image_path}: {e}")
                    failed += 1
                    continue
    
    # Create output manifest
    output_df = pd.DataFrame(all_rows)
    manifest_output_path = output_dir / "manifest.csv"
    output_df.to_csv(manifest_output_path, index=False)
    
    print(f"\n✓ Augmented dataset creation complete!")
    print(f"  Original samples processed: {processed_originals}")
    print(f"  Augmented samples created: {processed_augmented}")
    print(f"  Total samples: {len(output_df)}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"  Manifest: {manifest_output_path}")
    
    return manifest_output_path


def load_autoencoder_encoder(config_path, checkpoint_path, device="cuda"):
    """Load autoencoder and return encoder only."""
    print(f"Loading autoencoder from {checkpoint_path}")
    
    # Load config
    config = load_config(config_path)
    ae_cfg = config.get("autoencoder", config)
    
    # Load model
    model = Autoencoder.load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    return model.encoder


def main():
    parser = argparse.ArgumentParser(
        description="Create augmented dataset with original + augmented images and latents"
    )
    parser.add_argument(
        "--autoencoder-config",
        type=str,
        required=True,
        help="Path to autoencoder config YAML"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=str,
        required=True,
        help="Path to trained autoencoder checkpoint"
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
        default="dataset/augmented",
        help="Output directory for augmented dataset (default: dataset/augmented)"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=3,
        help="Number of augmented versions per original image (default: 3)"
    )
    parser.add_argument(
        "--rotation-range",
        type=float,
        default=15.0,
        help="Maximum rotation in degrees (default: 15.0)"
    )
    parser.add_argument(
        "--translate-range",
        type=float,
        default=0.05,
        help="Maximum translation as fraction of size (default: 0.05 = 5%)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (not used, but kept for compatibility)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers (not used, but kept for compatibility)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    # Load encoder
    encoder = load_autoencoder_encoder(
        args.autoencoder_config,
        args.autoencoder_checkpoint,
        device=args.device
    )
    
    # Create augmented dataset
    create_augmented_dataset(
        encoder,
        args.dataset_manifest,
        args.output_dir,
        num_augmentations=args.num_augmentations,
        rotation_range=args.rotation_range,
        translate_range=args.translate_range,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

