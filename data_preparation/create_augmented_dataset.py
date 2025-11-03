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


def is_white_image(image_tensor, threshold=0.95, white_threshold=0.9):
    """
    Check if an image is mostly white (>95% whitish pixels by default).
    
    A "whitish pixel" is one where all RGB channels are above the threshold.
    
    Args:
        image_tensor: Tensor of shape [C, H, W] in range [0, 1]
        threshold: Fraction of pixels that must be white (default: 0.95 = 95%)
        white_threshold: Pixel value threshold for "white" in [0, 1] range (default: 0.9 = 230/255)
    
    Returns:
        True if image is mostly white, False otherwise
    """
    # image_tensor is in [0, 1] range after normalization
    # Check if all channels (R, G, B) are above threshold for each pixel
    # Shape: [C, H, W] where C=3 (RGB)
    white_mask = (image_tensor > white_threshold).all(dim=0)  # All channels > threshold
    
    # Count white pixels
    total_pixels = white_mask.numel()
    white_pixels = white_mask.sum().item()
    white_ratio = white_pixels / total_pixels
    
    return white_ratio >= threshold


def apply_mirror_flip(image_tensor):
    """
    Apply horizontal mirror (flip) to an image tensor.
    
    Args:
        image_tensor: Tensor of shape [C, H, W] or [1, C, H, W]
    
    Returns:
        Mirrored image tensor
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Flip horizontally (along width dimension)
    flipped = torch.flip(image_tensor, dims=[-1])
    
    if flipped.shape[0] == 1:
        flipped = flipped.squeeze(0)
    
    return flipped


def apply_rotation_90(image_tensor, k):
    """
    Apply 90-degree rotation to an image tensor.
    
    Args:
        image_tensor: Tensor of shape [C, H, W] or [1, C, H, W]
        k: Number of 90-degree clockwise rotations (0, 1, 2, 3 for 0°, 90°, 180°, 270°)
    
    Returns:
        Rotated image tensor
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Rotate k*90 degrees clockwise
    rotated = torch.rot90(image_tensor, k=k, dims=[-2, -1])
    
    if rotated.shape[0] == 1:
        rotated = rotated.squeeze(0)
    
    return rotated


def generate_all_augmentations(image_tensor):
    """
    Generate all augmentation variants: 1 original + 7 augmented.
    
    Variants:
    - Original (0°)
    - Original rotated 90° (1)
    - Original rotated 180° (2)
    - Original rotated 270° (3)
    - Mirrored (0°)
    - Mirrored rotated 90° (1)
    - Mirrored rotated 180° (2)
    - Mirrored rotated 270° (3)
    
    Returns:
        List of 8 augmented tensors (original + 7 variants)
    """
    variants = []
    
    # Original + 3 rotations (0°, 90°, 180°, 270°)
    for k in range(4):
        rotated = apply_rotation_90(image_tensor, k)
        variants.append(rotated)
    
    # Mirrored + 3 rotations (0°, 90°, 180°, 270°)
    mirrored = apply_mirror_flip(image_tensor)
    for k in range(4):
        rotated = apply_rotation_90(mirrored, k)
        variants.append(rotated)
    
    return variants


def create_augmented_dataset(
    encoder,
    manifest_path,
    output_dir,
    use_mirror_rotation=True,
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
        use_mirror_rotation: If True, generate 7 variants (3 rotations + 1 mirror + 3 rotated mirrors). 
                            If False, generate 3 random variants using old method (deprecated).
        batch_size: Batch size for encoding (not used currently)
        num_workers: Number of DataLoader workers (not used currently)
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
            # Include in valid indices if we can't check (better safe than sorry)
            valid_indices.append(idx)
    
    # Filter dataframe
    df_filtered = df.loc[valid_indices].reset_index(drop=True)
    
    print(f"Removed {len(white_images)} white images")
    print(f"After filtering white images: {len(df_filtered)} samples")
    
    df = df_filtered
    
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
    
    # Determine augmentation strategy
    if use_mirror_rotation:
        num_augmentations = 7  # 3 rotations + 1 mirror + 3 rotated mirrors
        aug_description = "7 variants (3 rotations + mirror + 3 rotated mirrors)"
    else:
        num_augmentations = 3  # Legacy support
        aug_description = "3 random variants (deprecated)"
    
    print(f"\nCreating augmented dataset...")
    print(f"  Original samples: {len(df)}")
    print(f"  Augmentation method: {aug_description}")
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
                
                # Add to manifest (paths relative to output_dir for portability)
                all_rows.append({
                    "layout_path": str(original_output_path.relative_to(output_dir.parent)),
                    "latent_path": str(latent_output_path.relative_to(output_dir.parent)),
                    "scene_id": scene_id,
                    "room_id": room_id,
                    "is_empty": False,  # We've already filtered empty images
                    "is_augmented": False,
                    "augmentation_id": 0,
                    "original_path": str(original_image_path) if original_image_path.is_absolute() else str(original_image_path)
                })
                processed_originals += 1
            except Exception as e:
                print(f"Error processing original {original_image_path}: {e}")
                failed += 1
                continue
            
            # Create augmented versions
            if use_mirror_rotation:
                # Generate all deterministic augmentations
                augmentation_variants = generate_all_augmentations(img_tensor)
                # Skip the first one (original, already saved)
                augmentation_variants = augmentation_variants[1:]
                
                aug_names = [
                    "rot90", "rot180", "rot270",  # 3 rotations
                    "mirror", "mirror_rot90", "mirror_rot180", "mirror_rot270"  # mirror + 3 rotations
                ]
                
                for aug_idx, (augmented_tensor, aug_name) in enumerate(zip(augmentation_variants, aug_names)):
                    try:
                        # Save augmented image
                        aug_output_path = images_dir / f"{base_name}_{aug_name}.png"
                        if not aug_output_path.exists() or overwrite:
                            # Convert back to PIL and save
                            aug_np = augmented_tensor.cpu().permute(1, 2, 0).numpy()
                            aug_np = ((aug_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                            aug_img = Image.fromarray(aug_np)
                            aug_img.save(aug_output_path)
                        
                        # Encode to latent
                        latent_output_path = latents_dir / f"{base_name}_{aug_name}.pt"
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
                            "augmentation_type": aug_name,
                            "original_path": str(original_image_path)
                        })
                        processed_augmented += 1
                        
                    except Exception as e:
                        print(f"Error creating augmentation {aug_name} for {original_image_path}: {e}")
                        failed += 1
                        continue
            else:
                # Legacy: random augmentations (deprecated)
                for aug_idx in range(num_augmentations):
                    try:
                        # Apply old-style random augmentation (legacy - deprecated)
                        # Note: apply_spatial_augmentation removed, using simple rotation as fallback
                        angle = torch.rand(1).item() * 30 - 15  # -15 to +15 degrees
                        angle_rad = angle * np.pi / 180.0
                        cos_a = np.cos(angle_rad)
                        sin_a = np.sin(angle_rad)
                        device = img_tensor.device
                        if img_tensor.dim() == 3:
                            img_tensor_batch = img_tensor.unsqueeze(0)
                        else:
                            img_tensor_batch = img_tensor
                        _, _, h, w = img_tensor_batch.shape
                        tx = (torch.rand(1).item() * 2 - 1) * 0.05 * w
                        ty = (torch.rand(1).item() * 2 - 1) * 0.05 * h
                        theta = torch.tensor([
                            [cos_a, -sin_a, tx],
                            [sin_a, cos_a, ty]
                        ], dtype=torch.float32, device=device).unsqueeze(0)
                        grid = F.affine_grid(theta, img_tensor_batch.size(), align_corners=False)
                        augmented_tensor = F.grid_sample(
                            img_tensor_batch, grid, mode='bilinear', 
                            padding_mode='reflection', align_corners=False
                        )
                        if augmented_tensor.shape[0] == 1:
                            augmented_tensor = augmented_tensor.squeeze(0)
                        
                        # Save augmented image
                        aug_output_path = images_dir / f"{base_name}_aug{aug_idx+1:02d}.png"
                        if not aug_output_path.exists() or overwrite:
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
        "--use-mirror-rotation",
        action="store_true",
        default=True,
        help="Use mirror + 90-degree rotations (7 variants per sample). If False, uses old random augmentation (default: True)"
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
        use_mirror_rotation=args.use_mirror_rotation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

