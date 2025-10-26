#!/usr/bin/env python3
"""
stage6_create_layout_embeddings.py
----------------------------------
Encodes every layout image under a dataset root into latent embeddings
using a trained AutoEncoder (VEA).

Outputs *_layout_emb.pt next to each layout file.
Logs progress and summary statistics.
No manifest required as input.
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import yaml
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
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    return model


# -----------------------------------------------------------------------------
# Image encoding
# -----------------------------------------------------------------------------
def encode_layouts(model, data_root, device="cuda", batch_size=32):
    """Find all layout PNGs and encode them to latent space."""
    data_root = Path(data_root)
    layout_paths = sorted(list(data_root.rglob("*layout.png")))

    if not layout_paths:
        raise RuntimeError(f"No layout images found under {data_root}")

    transform = T.Compose([
        T.Resize((model.image_size, model.image_size)),
        T.ToTensor(),
    ])

    total = len(layout_paths)
    success = failed = 0

    print(f"[INFO] Found {total} layout images to encode.")
    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Encoding layouts", unit="batch"):
            batch_paths = layout_paths[i:i + batch_size]
            imgs, valid_paths = [], []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    imgs.append(transform(img))
                    valid_paths.append(path)
                except Exception:
                    failed += 1

            if not imgs:
                continue

            imgs = torch.stack(imgs).to(device)
            z = model.encode_latent(imgs)

            for j, layout_path in enumerate(valid_paths):
                out_path = layout_path.with_name(layout_path.stem.replace("_layout", "") + "_layout_emb.pt")
                emb = z[j].cpu()
                torch.save(emb, out_path)
                success += 1

    print("\n[SUMMARY]")
    print(f"  Encoded layouts: {success}")
    print(f"  Failed to load:  {failed}")
    print(f"  Total processed: {total}")
    print("[INFO] Embedding files written next to layouts.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Encode layout images into latent embeddings.")
    ap.add_argument("--config", required=True, help="Path to experiment YAML config")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--data_root", required=True, help="Root folder containing scenes/rooms")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    ap.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_from_experiment(args.config, args.ckpt, device=device)
    encode_layouts(model, args.data_root, device=device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
