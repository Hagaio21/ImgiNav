#!/usr/bin/env python3
"""
Embed Layouts with VAE - Create layout embeddings for diffusion training

Takes a POV-normalized manifest and creates VAE latent embeddings for each layout.
Adds a latent_embedding_path column to the manifest pointing to saved embeddings.

Usage:
    python embed_layouts_with_vae.py \
        --manifest dataset_v2/manifests/manifest_seg_pov_normalized.csv \
        --dataset-root dataset_v2 \
        --vae-checkpoint experiments/v2/autoencoders/vae_clip_v2/checkpoints/vae_clip_v2_checkpoint_best.pt \
        --output-manifest dataset_v2/manifests/manifest_seg_pov_normalized_with_latents.csv \
        --batch-size 32
"""

import argparse
import logging
import sys
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_latent_statistics(all_latents):
    """
    Compute statistics over all collected latents.
    
    Args:
        all_latents: List of latent tensors [C, H, W]
    
    Returns:
        Dictionary with statistics
    """
    if not all_latents or len(all_latents) == 0:
        return {}
    
    # Stack all latents [N, C, H, W]
    all_latents_tensor = torch.stack(all_latents, dim=0)
    
    # Flatten for global statistics
    latent_flat = all_latents_tensor.reshape(all_latents_tensor.shape[0], -1)
    
    # Compute global statistics
    latent_mean = latent_flat.mean().item()
    latent_std = latent_flat.std().item()
    latent_min = latent_flat.min().item()
    latent_max = latent_flat.max().item()
    
    # Compute percentiles
    latent_flat_np = latent_flat.cpu().numpy()
    percentiles = {
        "p0.1": float(np.percentile(latent_flat_np, 0.1)),
        "p1": float(np.percentile(latent_flat_np, 1)),
        "p5": float(np.percentile(latent_flat_np, 5)),
        "p10": float(np.percentile(latent_flat_np, 10)),
        "p25": float(np.percentile(latent_flat_np, 25)),
        "p50": float(np.percentile(latent_flat_np, 50)),
        "p75": float(np.percentile(latent_flat_np, 75)),
        "p90": float(np.percentile(latent_flat_np, 90)),
        "p95": float(np.percentile(latent_flat_np, 95)),
        "p99": float(np.percentile(latent_flat_np, 99)),
        "p99.9": float(np.percentile(latent_flat_np, 99.9)),
    }
    
    stats = {
        "mean": latent_mean,
        "std": latent_std,
        "min": latent_min,
        "max": latent_max,
        "percentiles": percentiles,
    }
    
    # Compute per-channel statistics if spatial dimensions exist
    if all_latents_tensor.ndim == 4:  # [N, C, H, W]
        N, C, H, W = all_latents_tensor.shape
        # Per-channel mean and std
        per_channel_mean = all_latents_tensor.mean(dim=(0, 2, 3)).cpu().numpy()  # [C]
        per_channel_std = all_latents_tensor.std(dim=(0, 2, 3)).cpu().numpy()  # [C]
        # Per-channel min/max
        latents_reshaped = all_latents_tensor.permute(1, 0, 2, 3).reshape(C, -1)  # [C, N*H*W]
        per_channel_min = latents_reshaped.min(dim=1)[0].cpu().numpy()  # [C]
        per_channel_max = latents_reshaped.max(dim=1)[0].cpu().numpy()  # [C]
        
        stats["per_channel"] = {
            "mean": per_channel_mean.tolist(),
            "std": per_channel_std.tolist(),
            "min": per_channel_min.tolist(),
            "max": per_channel_max.tolist(),
        }
        stats["num_channels"] = C
        stats["spatial_shape"] = [H, W]
    
    return stats


def save_latent_statistics(stats: dict, output_file: Path, vae_name: str):
    """
    Save latent statistics to a text file with recommendations for diffusion training.
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Latent Statistics for {vae_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Global statistics
        f.write("GLOBAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean:     {stats['mean']:.6f}\n")
        f.write(f"Std:      {stats['std']:.6f}\n")
        f.write(f"Min:      {stats['min']:.6f}\n")
        f.write(f"Max:      {stats['max']:.6f}\n")
        f.write(f"Range:    {stats['max'] - stats['min']:.6f}\n\n")
        
        # Percentiles
        f.write("PERCENTILES\n")
        f.write("-" * 80 + "\n")
        for p_name, p_value in stats['percentiles'].items():
            f.write(f"{p_name:>6}: {p_value:>12.6f}\n")
        f.write("\n")
        
        # Per-channel statistics
        if "per_channel" in stats:
            f.write("PER-CHANNEL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of channels: {stats['num_channels']}\n")
            f.write(f"Spatial shape: {stats['spatial_shape']}\n\n")
            
            per_ch = stats['per_channel']
            f.write(f"{'Channel':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
            f.write("-" * 80 + "\n")
            for ch in range(stats['num_channels']):
                f.write(f"{ch:<10} {per_ch['mean'][ch]:<12.6f} {per_ch['std'][ch]:<12.6f} "
                       f"{per_ch['min'][ch]:<12.6f} {per_ch['max'][ch]:<12.6f}\n")
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS FOR DIFFUSION TRAINING\n")
        f.write("=" * 80 + "\n\n")
        
        # Scale factor (1.0 / std to normalize to unit variance)
        scale_factor = 1.0 / stats['std'] if stats['std'] > 0 else 1.0
        f.write(f"scale_factor: {scale_factor:.6f}\n")
        f.write("  Description: Multiplicative factor to normalize latents to unit variance\n")
        f.write("  Formula: 1.0 / std\n\n")
        
        # Clamp values - recommended based on percentiles with margin
        clamp_min_percentile = stats['percentiles']['p0.1']
        clamp_max_percentile = stats['percentiles']['p99.9']
        recommended_min = clamp_min_percentile - 0.5
        recommended_max = clamp_max_percentile + 0.5
        
        f.write("latent_clamp_min / latent_clamp_max:\n")
        f.write(f"  latent_clamp_min: {recommended_min:.6f}\n")
        f.write(f"  latent_clamp_max: {recommended_max:.6f}\n")
        f.write("  (Based on 0.1%-99.9% percentiles with 0.5 margin)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("YAML Config Values:\n")
        f.write("=" * 80 + "\n")
        f.write(f"latent_clamp_min: {recommended_min:.6f}\n")
        f.write(f"latent_clamp_max: {recommended_max:.6f}\n")
        f.write(f"scale_factor: {scale_factor:.6f}\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create layout embeddings with VAE for diffusion training"
    )
    parser.add_argument("--manifest", required=True, help="Path to input manifest CSV")
    parser.add_argument("--dataset-root", required=True, help="Root directory of dataset")
    parser.add_argument("--vae-checkpoint", required=True, help="Path to VAE checkpoint")
    parser.add_argument("--output-manifest", required=True, help="Path for output manifest")
    parser.add_argument("--vae-name", default="vae_clip_v2", help="Name of VAE (for output directory)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    output_manifest_path = Path(args.output_manifest)
    vae_checkpoint = Path(args.vae_checkpoint)
    output_dir = dataset_root / "layouts_pov" / "latents" / args.vae_name
    
    # Validate inputs
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        sys.exit(1)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    if not vae_checkpoint.exists():
        logger.error(f"VAE checkpoint not found: {vae_checkpoint}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("Layout Embedding with VAE")
    logger.info("=" * 70)
    logger.info(f"Dataset Root: {dataset_root}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"VAE Checkpoint: {vae_checkpoint}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    
    # Load VAE
    logger.info("Loading VAE...")
    from models.autoencoder import VAE
    vae = VAE.load_checkpoint(vae_checkpoint, map_location=args.device, strict=False)
    vae = vae.to(args.device)
    vae.eval()
    logger.info("VAE loaded successfully")
    
    # Load manifest
    logger.info("Loading manifest...")
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Get unique layouts
    has_layout = df["layout_path"].notna() & (df["layout_path"] != "")
    layouts_df = df[has_layout][["layout_path", "scene_id", "room_id", "pov_id"]].drop_duplicates()
    logger.info(f"Found {len(layouts_df)} unique layouts to embed")
    
    # Filter existing embeddings
    to_process = []
    existing_count = 0
    for _, row in layouts_df.iterrows():
        scene_id, room_id, pov_id = row["scene_id"], row["room_id"], row["pov_id"]
        layout_path = row["layout_path"]
        variant = "seg" if "seg" in layout_path else "tex"
        emb_path = output_dir / f"{scene_id}_{room_id}_{pov_id}_{variant}_latent.pt"
        if emb_path.exists():
            existing_count += 1
        else:
            to_process.append(row)
    
    layouts_df = pd.DataFrame(to_process)
    logger.info(f"{len(to_process)} to process, {existing_count} skipped (existing)")
    
    # Embed layouts
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    embedding_map = {}
    all_latents = []  # Collect all latents for statistics
    
    if len(layouts_df) > 0:
        logger.info(f"Embedding {len(layouts_df)} layouts...")
        with torch.no_grad():
            for start in tqdm(range(0, len(layouts_df), args.batch_size), desc="Layout embeddings"):
                batch = layouts_df.iloc[start:start + args.batch_size]
                
                images = []
                valid_rows = []
                
                for _, row in batch.iterrows():
                    layout_path = dataset_root / row["layout_path"]
                    if not layout_path.exists():
                        logger.warning(f"Layout not found: {layout_path}")
                        continue
                    
                    try:
                        img = Image.open(layout_path).convert("RGB")
                        images.append(transform(img))
                        valid_rows.append(row)
                    except Exception as e:
                        logger.warning(f"Failed to load {layout_path}: {e}")
                
                if not images:
                    continue
                
                # Encode to latent space
                batch_tensor = torch.stack(images).to(args.device)
                mu, logvar = vae.encoder(batch_tensor)
                latents = mu.cpu()  # Use mean for deterministic encoding
                
                # Save embeddings and collect for statistics
                for latent, row in zip(latents, valid_rows):
                    scene_id = row["scene_id"]
                    room_id = row["room_id"]
                    pov_id = row["pov_id"]
                    layout_path = row["layout_path"]
                    variant = "seg" if "seg" in layout_path else "tex"
                    
                    emb_name = f"{scene_id}_{room_id}_{pov_id}_{variant}_latent.pt"
                    emb_path = output_dir / emb_name
                    torch.save(latent, emb_path)
                    
                    # Collect latent for statistics (add batch dimension)
                    all_latents.append(latent.unsqueeze(0))
                    
                    rel_path = str(emb_path.relative_to(dataset_root))
                    embedding_map[layout_path] = rel_path
    
    # Build embedding map for all layouts (including existing) and load existing latents for stats
    has_layout = df["layout_path"].notna() & (df["layout_path"] != "")
    for _, row in df[has_layout][["layout_path", "scene_id", "room_id", "pov_id"]].drop_duplicates().iterrows():
        layout_path = row["layout_path"]
        if layout_path not in embedding_map:
            scene_id = row["scene_id"]
            room_id = row["room_id"]
            pov_id = row["pov_id"]
            variant = "seg" if "seg" in layout_path else "tex"
            emb_path = output_dir / f"{scene_id}_{room_id}_{pov_id}_{variant}_latent.pt"
            if emb_path.exists():
                rel_path = str(emb_path.relative_to(dataset_root))
                embedding_map[layout_path] = rel_path
                # Load existing latent for statistics
                if len(all_latents) < 10000:  # Limit to avoid memory issues
                    try:
                        existing_latent = torch.load(emb_path)
                        # Ensure it has batch dimension [1, C, H, W]
                        if existing_latent.dim() == 3:
                            existing_latent = existing_latent.unsqueeze(0)
                        all_latents.append(existing_latent)
                    except Exception as e:
                        logger.warning(f"Failed to load {emb_path} for statistics: {e}")
    
    # Compute and save latent statistics
    if len(all_latents) > 0:
        logger.info("")
        logger.info("Computing latent statistics...")
        stats = compute_latent_statistics(all_latents)
        
        # Save statistics text file
        stats_file = output_dir / f"{args.vae_name}_latent_statistics.txt"
        save_latent_statistics(stats, stats_file, args.vae_name)
        logger.info(f"  Saved statistics: {stats_file}")
        
        # Save statistics JSON for easy loading
        stats_json = output_dir / f"{args.vae_name}_latent_statistics.json"
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Saved statistics JSON: {stats_json}")
        
        # Print recommendations
        scale_factor = 1.0 / stats['std'] if stats['std'] > 0 else 1.0
        clamp_min = stats['percentiles']['p0.1'] - 0.5
        clamp_max = stats['percentiles']['p99.9'] + 0.5
        logger.info("")
        logger.info("Recommended values for diffusion configs:")
        logger.info(f"  latent_clamp_min: {clamp_min:.6f}")
        logger.info(f"  latent_clamp_max: {clamp_max:.6f}")
        logger.info(f"  scale_factor: {scale_factor:.6f}")
    
    # Create output manifest
    logger.info("Creating output manifest...")
    output_df = df.copy()
    output_df["latent_embedding_path"] = output_df["layout_path"].map(lambda x: embedding_map.get(x, ""))
    
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_manifest_path, index=False)
    
    logger.info(f"Saved: {output_manifest_path}")
    logger.info(f"Rows: {len(output_df)}")
    logger.info(f"Embeddings with paths: {(output_df['latent_embedding_path'] != '').sum()}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Layout embedding completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
