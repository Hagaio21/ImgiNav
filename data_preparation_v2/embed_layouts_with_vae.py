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
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
                
                # Save embeddings
                for latent, row in zip(latents, valid_rows):
                    scene_id = row["scene_id"]
                    room_id = row["room_id"]
                    pov_id = row["pov_id"]
                    layout_path = row["layout_path"]
                    variant = "seg" if "seg" in layout_path else "tex"
                    
                    emb_name = f"{scene_id}_{room_id}_{pov_id}_{variant}_latent.pt"
                    emb_path = output_dir / emb_name
                    torch.save(latent, emb_path)
                    
                    rel_path = str(emb_path.relative_to(dataset_root))
                    embedding_map[layout_path] = rel_path
    
    # Build embedding map for all layouts (including existing)
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
