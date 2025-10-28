import os, sys, yaml, random, torch, argparse, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import argparse
from modules.autoencoder import AutoEncoder


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)


def main(args):
    os.makedirs(args.output_latent_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1] Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)

    print(f"[2] Loading AutoEncoder from {args.autoencoder_config}")
    ae = AutoEncoder.from_config(args.autoencoder_config)
    ae.load_state_dict(torch.load(args.autoencoder_ckpt, map_location=device))
    ae.eval().to(device)

    latent_paths = []

    # Pre-create base subdirectories
    os.makedirs(os.path.join(args.output_latent_dir, "rooms"), exist_ok=True)
    os.makedirs(os.path.join(args.output_latent_dir, "scenes"), exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding layouts"):
        layout_path = row["layout_path"]
        scene_id = str(row["scene_id"])
        room_id = str(row.get("room_id", "scene"))
        entry_type = row.get("type", "scene")

        try:
            img = load_image(layout_path).unsqueeze(0).to(device)
            with torch.no_grad():
                z = ae.encode_latent(img)

            subdir = "rooms" if entry_type == "room" else "scenes"
            latent_file = os.path.join(
                args.output_latent_dir, subdir, f"{scene_id}_{room_id}_{entry_type}.pt"
            )

            torch.save(z.half().cpu(), latent_file)
            latent_paths.append(latent_file)

        except Exception as e:
            print(f"Failed on {layout_path}: {e}")
            latent_paths.append(None)

    print("[3] Adding 'layout_emb' column and saving new manifest...")
    df["layout_emb"] = latent_paths
    df.to_csv(args.new_manifest, index=False)
    print(f"[✓] Saved new manifest → {args.new_manifest}")
    print(f"[✓] Latent files in → {args.output_latent_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute latent embeddings for layouts.")
    parser.add_argument("--manifest", required=True, help="Path to dataset CSV (layouts.csv)")
    parser.add_argument("--autoencoder_config", required=True, help="Path to AE config YAML")
    parser.add_argument("--autoencoder_ckpt", required=True, help="Path to AE checkpoint")
    parser.add_argument("--output_latent_dir", required=True, help="Directory to save latent .pt files")
    parser.add_argument("--new_manifest", required=True, help="Path to save new manifest CSV")
    args = parser.parse_args()

    main(args)
