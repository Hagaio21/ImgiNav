#!/usr/bin/env python3
"""
stage6_create_layout_embeddings.py
----------------------------------
Encodes every layout image under a dataset root into latent embeddings
using a trained AutoEncoder (VAE).

Outputs *_layout_emb.pt next to each layout file and generates a
manifest CSV of all layout paths and their corresponding embedding paths.
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import yaml
import pandas as pd
from typing import Tuple, List, Dict, Any

# Assuming 'autoencoder.py' is in 'modules/' as per your original script
from modules.autoencoder import AutoEncoder


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model_from_experiment(config_path, checkpoint_path, device="cuda"):
    """Rebuild AutoEncoder and load weights."""
    print(f"[INFO] Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "model" not in config:
        raise KeyError(f"'model' key missing in config file {config_path}")

    model_cfg = config["model"]
    # Use from_shape as in the original script
    model = AutoEncoder.from_shape(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        latent_channels=model_cfg["latent_channels"],
        image_size=model_cfg["image_size"],
        latent_base=model_cfg["latent_base"],
        norm=model_cfg.get("norm"),
        act=model_cfg.get("act", "relu"),
        dropout=model_cfg.get("dropout", 0.0),
        num_classes=model_cfg.get("num_classes", None),
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    
    # Handle potential DataParallel prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    return model


# -----------------------------------------------------------------------------
# Path Parsing
# -----------------------------------------------------------------------------
def parse_layout_path(path: Path, data_root: Path) -> Tuple[str, str, str]:
    """
    Extracts scene, type, and room_id from a path.
    Assumes structure:
    - data_root/<scene_id>/<room_id>/..._layout.png  (for rooms)
    - data_root/<scene_id>/..._layout.png          (for scenes)
    """
    relative_parts = path.relative_to(data_root).parts
    
    if len(relative_parts) == 3:
        # e.g., ('scene_000', 'room_0', 'pano_layout.png')
        scene_id = relative_parts[0]
        type = "room"
        room_id = relative_parts[1]
    elif len(relative_parts) == 2:
        # e.g., ('scene_000', 'scene_layout.png')
        scene_id = relative_parts[0]
        type = "scene"
        room_id = "none"  # As requested
    else:
        raise ValueError(f"Unexpected path structure: {path.relative_to(data_root)}")
    return scene_id, type, room_id

# -----------------------------------------------------------------------------
# Image encoding
# -----------------------------------------------------------------------------
def encode_layouts(model, data_root, device="cuda", batch_size=32, overwrite=False) -> List[Dict[str, Any]]:
    """Find all layout PNGs, encode them, and return manifest data."""
    data_root = Path(data_root)
    layout_paths = sorted(list(data_root.rglob("*layout.png")))

    if not layout_paths:
        raise RuntimeError(f"No layout images found under {data_root}")

    # --- 1. Prepare transform ---
    # FIX: Access image_size from the encoder attribute
    image_size = model.encoder.image_size
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    # --- 2. Scan paths and build manifest/job list ---
    manifest_data = []
    paths_to_encode = []
    skipped_existing = 0
    skipped_parsing = 0

    print(f"[INFO] Scanning {len(layout_paths)} layout paths...")
    for path in tqdm(layout_paths, desc="Scanning paths", unit="file"):
        try:
            scene_id, type, room_id = parse_layout_path(path, data_root)
        except ValueError as e:
            print(f"[WARN] Skipping file with unexpected path: {e}")
            skipped_parsing += 1
            continue
        
        # Define output path
        out_path = path.with_name(path.stem + "_emb.pt")

        entry = {
            "scene": scene_id,
            "type": type,
            "room_id": room_id,
            "layout_path": str(path.relative_to(data_root)),
            "layout_emb_path": str(out_path.relative_to(data_root))
        }
        manifest_data.append(entry)

        if not out_path.exists() or overwrite:
            paths_to_encode.append((path, out_path))
        else:
            skipped_existing += 1

    # --- 3. Encode images in batches ---
    total_to_encode = len(paths_to_encode)
    success = 0
    failed_load = 0

    print(f"[INFO] Found {total_to_encode} layouts to encode.")
    with torch.no_grad():
        for i in tqdm(range(0, total_to_encode, batch_size), desc="Encoding layouts", unit="batch"):
            
            batch_job_paths = paths_to_encode[i:i + batch_size]
            imgs, valid_out_paths = [], []
            
            for in_path, out_path in batch_job_paths:
                try:
                    img = Image.open(in_path).convert("RGB")
                    imgs.append(transform(img))
                    valid_out_paths.append(out_path)
                except Exception as e:
                    print(f"[WARN] Failed to load {in_path}: {e}")
                    failed_load += 1

            if not imgs:
                continue

            imgs_tensor = torch.stack(imgs).to(device)
            # Use deterministic encoding (mu)
            z = model.encode_latent(imgs_tensor, deterministic=True) 

            for j, out_path in enumerate(valid_out_paths):
                emb = z[j].cpu()
                torch.save(emb, out_path)
                success += 1

    # --- 4. Print summary and return manifest ---
    print("\n[SUMMARY]")
    print(f"  Total layouts found: {len(layout_paths)}")
    print(f"  Skipped (bad path):  {skipped_parsing}")
    print(f"  Skipped (existing):  {skipped_existing}")
    print(f"  Total to encode:     {total_to_encode}")
    print(f"    Encoded new:     {success}")
    print(f"    Failed to load:  {failed_load}")
    
    return manifest_data


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Encode layout images into latent embeddings.")
    ap.add_argument("--config", required=True, help="Path to experiment YAML config")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--data_root", required=True, help="Root folder containing scenes/rooms")
    ap.add_argument("--manifest_out", default="layout_embeddings.csv", help="Output manifest file name (saved in data_root)")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    ap.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing embedding files")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[INFO] Using device: {device}")
    
    model = load_model_from_experiment(args.config, args.ckpt, device=device)
    
    manifest_data = encode_layouts(
        model, 
        args.data_root, 
        device=device, 
        batch_size=args.batch_size, 
        overwrite=args.overwrite
    )

    if manifest_data:
        df = pd.DataFrame(manifest_data)
        # Reorder columns to your spec
        cols = ["scene", "type", "room_id", "layout_path", "layout_emb_path"]
        df = df[cols]
        
        # Save it
        output_csv_path = Path(args.data_root) / args.manifest_out
        df.to_csv(output_csv_path, index=False, sep="|") # Use '|' separator
        print(f"\n[INFO] Manifest saved to {output_csv_path}")
    else:
        print("\n[INFO] No layout images found, manifest not created.")


if __name__ == "__main__":
    main()