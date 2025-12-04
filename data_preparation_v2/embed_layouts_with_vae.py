#!/usr/bin/env python3
"""
Embed Layouts with VAE - Create layout embeddings for diffusion training

Takes a POV-normalized manifest and creates VAE latent embeddings for each layout.
Adds a latent_embedding_path column to the manifest pointing to saved embeddings.

The script:
1. Loads a pre-trained VAE checkpoint
2. Encodes each layout image to latent space
3. Saves embeddings as .pt files
4. Creates output manifest with embedding paths (relative to dataset_root)

Usage:
    python embed_layouts_with_vae.py \
        --manifest dataset_v2/manifests/manifest_seg_pov_normalized.csv \
        --dataset-root dataset_v2 \
        --vae-checkpoint experiments/v2/autoencoders/vae_clip_v2/checkpoints/vae_clip_v2_checkpoint_best.pt \
        --output-manifest dataset_v2/manifests/manifest_seg_pov_normalized_with_latents.csv \
        --batch-size 32

Output Structure:
    dataset_v2/
    └── layouts_pov/latents/vae_clip_v2/
        ├── {scene_id}_{room_id}_{pov_id}_{variant}_latent.pt
        ├── {scene_id}_{room_id}_{pov_id}_{variant}_latent.pt
        └── ...

Output Manifest:
    - Copy of input manifest
    - Additional column: latent_embedding_path (relative path from dataset_root)
"""

import argparse
import logging
import sys
import os
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_vae_checkpoint(checkpoint_path: Path, device: str = "cuda", config_path: Optional[Path] = None) -> nn.Module:
    """
    Load VAE from checkpoint.
    
    The checkpoint should contain the full model state_dict.
    We'll look for a config file in the same directory as the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading VAE checkpoint from: {checkpoint_path}")
    
    # If user provided explicit config path, prefer it
    if config_path is not None:
        config_path = Path(config_path)

    # Try to find config in the same directory or parent directories if not provided
    if config_path is None:
        config_path = checkpoint_path.parent / "autoencoder_config.yaml"
        if not config_path.exists():
            config_path = checkpoint_path.parent.parent / "output" / "autoencoder_config.yaml"
        if not config_path.exists():
            config_path = checkpoint_path.parent.parent.parent / "output" / "autoencoder_config.yaml"
    if not config_path.exists():
        config_path = checkpoint_path.parent.parent / "output" / "autoencoder_config.yaml"
    if not config_path.exists():
        config_path = checkpoint_path.parent.parent.parent / "output" / "autoencoder_config.yaml"

    # Import project modules (prefer models.autoencoder.VAE)
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    VAEClass = None
    try:
        from models.autoencoder import VAE as _V
        VAEClass = _V
    except Exception:
        try:
            from modules.autoencoder import AutoEncoder as _AE
            VAEClass = _AE
        except Exception as e:
            raise RuntimeError(f"Failed to import VAE/AutoEncoder class from project: {e}")

    # If config found, build model from config
    if config_path and config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        # Load YAML config
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Try classmethod from_config, else try direct construction
        try:
            if hasattr(VAEClass, "from_config"):
                vae = VAEClass.from_config(cfg)
            else:
                # Some configs nest model under 'model' key
                model_cfg = cfg.get("model", cfg)
                # Attempt to construct with model_cfg if it's a dict
                if isinstance(model_cfg, dict):
                    vae = VAEClass(**model_cfg)
                else:
                    vae = VAEClass(model_cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate VAE from config: {e}")

        # Load checkpoint state
        state = torch.load(checkpoint_path, map_location="cpu")

        # Resolve state_dict
        if isinstance(state, dict):
            if "model" in state:
                state_dict = state["model"]
            elif "state_dict" in state:
                state_dict = state["state_dict"]
            elif "state_dict" in state.get("payload", {}):
                state_dict = state["payload"]["state_dict"]
            else:
                state_dict = state
        else:
            state_dict = state

        try:
            vae.load_state_dict(state_dict, strict=False)
        except Exception as e:
            # Best-effort: if state_dict contains nested keys, try to find a plausible nested dict
            raise RuntimeError(f"Failed to load checkpoint into VAE: {e}")

        vae = vae.to(device)
        vae.eval()

        logger.info(f"VAE loaded successfully from config")
        return vae

    # No config found - instruct caller to provide config path
    raise FileNotFoundError(
        f"Autoencoder config not found near checkpoint {checkpoint_path}. Provide --vae-config pointing to the autoencoder config YAML."
    )


def get_layout_transform():
    """Get transform for layout images (same as training)."""
    return transforms.Compose([
        transforms.ToTensor(),
        # No normalization - keep raw pixel values for VAE
    ])


def embed_layouts(
    df: pd.DataFrame,
    dataset_root: Path,
    vae: nn.Module,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    skip_existing: bool = True,
) -> Dict[str, str]:
    """
    Embed all layouts using VAE encoder.
    
    Returns: dict of layout_path -> relative_embedding_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique layouts
    has_layout = df["layout_path"].notna() & (df["layout_path"] != "")
    layouts_df = df[has_layout][["layout_path", "scene_id", "room_id", "pov_id"]].drop_duplicates()
    
    logger.info(f"Found {len(layouts_df)} unique layouts to embed")
    
    # Filter already processed if skip_existing
    if skip_existing:
        to_process = []
        existing_count = 0
        for _, row in layouts_df.iterrows():
            scene_id, room_id, pov_id = row["scene_id"], row["room_id"], row["pov_id"]
            # Try to infer variant from layout path
            layout_path = row["layout_path"]
            variant = "seg" if "seg" in layout_path else "tex"
            emb_path = output_dir / f"{scene_id}_{room_id}_{pov_id}_{variant}_latent.pt"
            
            if emb_path.exists():
                existing_count += 1
            else:
                to_process.append(row)
        
        layouts_df = pd.DataFrame(to_process)
        logger.info(f"  {len(to_process)} to process, {existing_count} skipped (existing)")
    
    if len(layouts_df) == 0:
        logger.info("All embeddings exist, skipping encoding")
        embedding_map = {}
        has_layout = df["layout_path"].notna() & (df["layout_path"] != "")
        for _, row in df[has_layout][["layout_path", "scene_id", "room_id", "pov_id"]].drop_duplicates().iterrows():
            scene_id, room_id, pov_id = row["scene_id"], row["room_id"], row["pov_id"]
            layout_path = row["layout_path"]
            variant = "seg" if "seg" in layout_path else "tex"
            emb_path = output_dir / f"{scene_id}_{room_id}_{pov_id}_{variant}_latent.pt"
            if emb_path.exists():
                rel_path = str(emb_path.relative_to(dataset_root))
                embedding_map[layout_path] = rel_path
        return embedding_map
    
    logger.info(f"Embedding {len(layouts_df)} layouts...")
    
    transform = get_layout_transform()
    embedding_map = {}
    
    with torch.no_grad():
        for start in tqdm(range(0, len(layouts_df), batch_size), desc="Layout embeddings"):
            batch = layouts_df.iloc[start:start + batch_size]
            
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
            batch_tensor = torch.stack(images).to(device)
            mu, logvar = vae.encoder(batch_tensor)
            # Use mean for deterministic encoding
            latents = mu.cpu()
            
            # Save embeddings
            for latent, row in zip(latents, valid_rows):
                scene_id = row["scene_id"]
                room_id = row["room_id"]
                pov_id = row["pov_id"]
                layout_path = row["layout_path"]
                
                # Infer variant from layout path
                variant = "seg" if "seg" in layout_path else "tex"
                
                emb_name = f"{scene_id}_{room_id}_{pov_id}_{variant}_latent.pt"
                emb_path = output_dir / emb_name
                
                torch.save(latent, emb_path)
                
                rel_path = str(emb_path.relative_to(dataset_root))
                embedding_map[layout_path] = rel_path
    
    # Add existing embeddings to map
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
    
    return embedding_map


def main():
    parser = argparse.ArgumentParser(
        description="Create layout embeddings with VAE for diffusion training"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to input POV-normalized manifest CSV"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of dataset (for relative paths)"
    )
    parser.add_argument(
        "--vae-checkpoint",
        required=True,
        help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "--vae-config",
        required=False,
        help="Path to VAE config YAML (optional, required if not discoverable near checkpoint)"
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Path for output manifest with embedding paths"
    )
    parser.add_argument(
        "--vae-name",
        default="vae_clip_v2",
        help="Name of VAE (for output directory naming)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip layouts with existing embeddings"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    output_manifest_path = Path(args.output_manifest)
    vae_checkpoint = Path(args.vae_checkpoint)
    
    # Output directory for embeddings
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
    logger.info(f"Output Manifest: {output_manifest_path}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    
    # Load manifest
    logger.info("Loading manifest...")
    df = pd.read_csv(manifest_path)
    logger.info(f"  Loaded {len(df)} rows")
    
    # Load VAE
    logger.info("")
    logger.info("Loading VAE...")
    try:
        vae = load_vae_checkpoint(vae_checkpoint, device=args.device, config_path=(Path(args.vae_config) if args.vae_config else None))
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Provide --vae-config pointing to the autoencoder YAML or place autoencoder_config.yaml near the checkpoint.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load VAE: {e}")
        raise
    
    # Embed layouts
    logger.info("")
    embedding_map = embed_layouts(
        df,
        dataset_root,
        vae,
        output_dir,
        device=args.device,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )
    
    # Add embedding paths to manifest
    logger.info("")
    logger.info("Creating output manifest...")
    
    output_df = df.copy()
    output_df["latent_embedding_path"] = output_df["layout_path"].map(
        lambda x: embedding_map.get(x, "")
    )
    
    # Save output manifest
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_manifest_path, index=False)
    
    logger.info(f"  Saved: {output_manifest_path}")
    logger.info(f"  Size: {output_manifest_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"  Rows: {len(output_df)}")
    logger.info(f"  Embeddings with paths: {(output_df['latent_embedding_path'] != '').sum()}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Layout embedding completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
