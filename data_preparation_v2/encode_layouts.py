#!/usr/bin/env python3
"""
Standalone script to encode layout images to latents using a trained VAE.

This script:
1. Loads a trained VAE checkpoint
2. Encodes layout images to latents
3. Saves latents in dataset_v2/layouts/latents/{variant}_{vae_name}/
4. Creates a separate manifest with latent paths (keeps original manifest unchanged)

Usage:
    python encode_layouts.py \
        --vae-checkpoint outputs/autoencoders/v2/vae_seg_256_clip/checkpoints/vae_seg_256_clip_checkpoint_best.pt \
        --manifest dataset_v2/manifests/manifest_seg.csv \
        --dataset-root dataset_v2 \
        --variant seg \
        --batch-size 32 \
        --device cuda

Output Structure:
    dataset_v2/
    ├── layouts/
    │   └── latents/
    │       └── seg_{vae_name}/
    │           └── {scene_id}_{room_id}_layout.pt
    └── manifests/
        └── {vae_name}_latent_manifest_seg.csv  # Copy of manifest_seg.csv with latent column
"""

import argparse
import logging
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.utils import load_config, get_device, move_batch_to_device, to_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_vae_name(checkpoint_path: Path) -> str:
    """
    Extract VAE name from checkpoint path.
    
    Examples:
        vae_seg_256_clip_checkpoint_best.pt -> vae_seg_256_clip
        outputs/autoencoders/v2/vae_tex_512/checkpoints/vae_tex_512_checkpoint_best.pt -> vae_tex_512
    """
    name = checkpoint_path.stem
    # Remove common suffixes
    for suffix in ["_checkpoint_best", "_checkpoint_latest", "_checkpoint_epoch_", "_checkpoint"]:
        if suffix in name:
            name = name.split(suffix)[0]
            break
    return name


def encode_layouts(
    vae_checkpoint: Path,
    manifest: Path,
    dataset_root: Path,
    variant: str,
    output_dir: Path,
    batch_size: int = 32,
    device: str = "cuda",
    num_workers: int = 4,
    skip_existing: bool = True,
    output_manifest: Path = None,
    latent_column_name: str = None,
):
    """
    Encode layout images to latents using a trained VAE.
    
    Args:
        vae_checkpoint: Path to VAE checkpoint
        manifest: Path to manifest CSV
        dataset_root: Root directory of dataset
        variant: Layout variant (seg or tex)
        output_dir: Output directory for latents
        batch_size: Batch size for encoding
        device: Device to use
        num_workers: Number of data loading workers
        skip_existing: Skip files that already exist
        output_manifest: Path to output manifest (if None, auto-generates name)
        latent_column_name: Name for latent column (if None, uses default)
    """
    device_obj = to_device(device)
    
    # Load VAE
    logger.info(f"Loading VAE from {vae_checkpoint}")
    vae = Autoencoder.load_checkpoint(vae_checkpoint, map_location=device_obj)
    vae.eval()
    vae = vae.to(device_obj)
    
    # Extract VAE name for output directory
    vae_name = extract_vae_name(vae_checkpoint)
    output_subdir = output_dir / f"{variant}_{vae_name}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_subdir}")
    
    # Load manifest
    df = pd.read_csv(manifest)
    logger.info(f"Loaded manifest: {len(df)} rows")
    
    # Filter to layouts with the specified variant
    layout_col = f"layout_{variant}_path"
    if layout_col not in df.columns:
        # Try alternative column names
        alt_cols = [f"layout_path", "layout_seg_path", "layout_tex_path"]
        layout_col = None
        for col in alt_cols:
            if col in df.columns:
                layout_col = col
                break
        
        if layout_col is None:
            raise ValueError(f"Layout column not found. Available columns: {list(df.columns)}")
    
    # Filter to rows with layout paths
    has_layout = df[layout_col].notna() & (df[layout_col] != "")
    df_filtered = df[has_layout].copy()
    logger.info(f"Found {len(df_filtered)} rows with layout paths")
    
    # Filter already processed if skip_existing
    if skip_existing:
        to_process = []
        for _, row in df_filtered.iterrows():
            scene_id = row.get("scene_id", "")
            room_id = row.get("room_id", "")
            if not scene_id or not room_id:
                continue
            
            latent_path = output_subdir / f"{scene_id}_{room_id}_layout.pt"
            if not latent_path.exists():
                to_process.append(row)
        
        original_count = len(df_filtered)
        df_filtered = pd.DataFrame(to_process)
        logger.info(f"Processing {len(df_filtered)} layouts, {original_count - len(df_filtered)} already exist")
    
    if len(df_filtered) == 0:
        logger.info("All layouts already encoded, skipping")
        return
    
    # Create dataset
    # Use the same transform config as training (from VAE config if available)
    transform_cfg = None
    if hasattr(vae, '_init_kwargs') and 'encoder' in vae._init_kwargs:
        # Try to get transform from encoder config
        encoder_cfg = vae._init_kwargs.get('encoder', {})
        if 'transform' in encoder_cfg:
            transform_cfg = encoder_cfg['transform']
    
    # Default transform: resize to 256x256, normalize to [-1, 1]
    if transform_cfg is None:
        from torchvision import transforms
        transform_cfg = {
            "type": "Compose",
            "transforms": [
                {"type": "Resize", "size": (256, 256), "interpolation": "nearest"},
                {"type": "ToTensor"},
                {"type": "Normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
            ]
        }
    
    # Filter dataset to only rows we need to process
    # Create a filtered dataset with only the rows we need
    # We need to preserve the original column name for the dataset
    filtered_df = df_filtered[["scene_id", "room_id", layout_col]].copy()
    filtered_df.rename(columns={layout_col: "layout_path"}, inplace=True)
    
    # Create a temporary manifest for the filtered dataset
    temp_manifest = output_subdir / "temp_manifest.csv"
    filtered_df.to_csv(temp_manifest, index=False)
    
    dataset = ManifestDataset(
        manifest=temp_manifest,
        outputs={"rgb": "layout_path"},
        transform=transform_cfg,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device_obj.type == "cuda" else False,
    )
    
    # Encode layouts
    logger.info(f"Encoding {len(dataset)} layouts...")
    # Track latent paths by (scene_id, room_id)
    latent_path_map = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding layouts")):
            batch = move_batch_to_device(batch, device_obj)
            
            # Get RGB images
            rgb = batch.get("rgb")
            if rgb is None:
                logger.warning(f"Batch {batch_idx} has no RGB data, skipping")
                continue
            
            # Encode to latents
            encoder_out = vae.encode(rgb)
            
            # Get latent tensor (handle both VAE and regular AE)
            if "mu" in encoder_out:
                # VAE: use mu for deterministic encoding
                latent = encoder_out["mu"]
            elif "latent" in encoder_out:
                latent = encoder_out["latent"]
            elif "latent_features" in encoder_out:
                latent = encoder_out["latent_features"]
            else:
                # Try to get first tensor value
                latent = next(iter(encoder_out.values()))
            
            # Get scene_id and room_id for this batch
            # We need to track which samples correspond to which rows
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(df_filtered))
            batch_rows = df_filtered.iloc[batch_start:batch_end]
            
            # Save each latent
            for i, (_, row) in enumerate(batch_rows.iterrows()):
                scene_id = row.get("scene_id", "")
                room_id = row.get("room_id", "")
                
                if not scene_id or not room_id:
                    logger.warning(f"Row {batch_start + i} missing scene_id or room_id, skipping")
                    continue
                
                # Get latent for this sample
                sample_latent = latent[i].cpu()
                
                # Save latent
                latent_path = output_subdir / f"{scene_id}_{room_id}_layout.pt"
                torch.save(sample_latent, latent_path)
                rel_path = str(latent_path.relative_to(dataset_root))
                latent_path_map[(scene_id, room_id)] = rel_path
    
    # Clean up temp manifest
    if temp_manifest.exists():
        temp_manifest.unlink()
    
    logger.info(f"Encoded {len(latent_path_map)} layouts")
    logger.info(f"Latents saved to: {output_subdir}")
    
    # Also check for existing latents if skip_existing was True
    # Use the original df loaded at the start (before filtering)
    if skip_existing:
        for _, row in df.iterrows():
            scene_id = row.get("scene_id", "")
            room_id = row.get("room_id", "")
            if not scene_id or not room_id:
                continue
            
            key = (scene_id, room_id)
            if key not in latent_path_map:
                # Check if latent already exists
                latent_path = output_subdir / f"{scene_id}_{room_id}_layout.pt"
                if latent_path.exists():
                    rel_path = str(latent_path.relative_to(dataset_root))
                    latent_path_map[key] = rel_path
    
    # Create output manifest with latent column
    if output_manifest is None:
        # Auto-generate manifest name: <vae_name>_latent_manifest_<variant>.csv
        manifest_dir = manifest.parent
        manifest_stem = manifest.stem
        # Extract base name (e.g., "manifest_seg" -> "manifest_seg")
        output_manifest = manifest_dir / f"{vae_name}_latent_{manifest_stem}.csv"
    
    logger.info(f"Creating latent manifest: {output_manifest}")
    
    # Use the original manifest (df) - make a copy to avoid modifying the original
    df_output = df.copy()
    
    # Determine latent column name
    if latent_column_name is None:
        # Use default based on VAE type
        if "clip" in vae_name.lower():
            latent_column_name = "latent_path_vae_clip"
        else:
            latent_column_name = "layout_latent_path"
    
    # Add latent column
    def get_latent_path(row):
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        if not scene_id or not room_id:
            return ""
        return latent_path_map.get((scene_id, room_id), "")
    
    df_output[latent_column_name] = df_output.apply(get_latent_path, axis=1)
    
    # Save output manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_manifest, index=False)
    
    latent_count = (df_output[latent_column_name] != "").sum()
    logger.info(f"Saved latent manifest: {output_manifest}")
    logger.info(f"  Total rows: {len(df_output)}")
    logger.info(f"  Rows with latents: {latent_count}")
    logger.info(f"  Latent column: {latent_column_name}")
    
    return list(latent_path_map.values())


def main():
    parser = argparse.ArgumentParser(description="Encode layout images to latents using a trained VAE")
    parser.add_argument("--vae-checkpoint", required=True, type=Path, help="Path to VAE checkpoint")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest CSV")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Dataset root directory")
    parser.add_argument("--variant", required=True, choices=["seg", "tex"], help="Layout variant (seg or tex)")
    parser.add_argument("--output-dir", type=Path, default=None, 
                       help="Output directory for latents (default: dataset_root/layouts/latents)")
    parser.add_argument("--output-manifest", type=Path, default=None,
                       help="Output manifest path (default: <vae_name>_latent_<manifest_name>.csv)")
    parser.add_argument("--latent-column-name", type=str, default=None,
                       help="Name for latent column (default: latent_path_vae_clip for CLIP VAE, layout_latent_path otherwise)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing latents")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = args.dataset_root / "layouts" / "latents"
    else:
        output_dir = args.output_dir
    
    # Encode layouts
    encode_layouts(
        vae_checkpoint=args.vae_checkpoint,
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        variant=args.variant,
        output_dir=output_dir,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        skip_existing=not args.overwrite,
        output_manifest=args.output_manifest,
        latent_column_name=args.latent_column_name,
    )


if __name__ == "__main__":
    main()

