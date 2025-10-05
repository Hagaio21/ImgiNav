#!/usr/bin/env python3
import argparse
import os, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.append(str(Path(__file__).parent.parent / "modules"))
from datasets import LayoutDataset, collate_skip_none
from autoencoder import AutoEncoder


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to autoencoder YAML config")
    ap.add_argument("--ckpt", required=True, help="Path to autoencoder checkpoint .pt/.pth")
    ap.add_argument("--manifest", required=True, help="Input CSV manifest of layouts")
    ap.add_argument("--out_manifest", required=True, help="Output CSV manifest with embeddings column")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--format", choices=["pt", "npy"], default="pt")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load autoencoder ---
    ae = AutoEncoder.from_config(args.config)
    ckpt = torch.load(args.ckpt, map_location=device)
    ae.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    ae.to(device)
    ae.eval()

    # --- Transform (match training setup) ---
    transform = T.Compose([
        T.Resize((512, 512)),  # adjust if config differs
        T.ToTensor()
    ])

    # --- Dataset + Loader ---
    ds = LayoutDataset(args.manifest, transform=transform, mode="all", skip_empty=False, return_embeddings=False)
    dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate_skip_none
                    )

    # --- Prepare manifest update ---
    df = pd.read_csv(args.manifest)
    df = pd.read_csv(args.manifest)
    if "embedding_path" not in df.columns:
        df["embedding_path"] = None
    if "embedding_dim" not in df.columns:
        df["embedding_dim"] = None


    # --- Encode and save with progress bar ---
    total = len(ds)
    with torch.no_grad():
        pbar = tqdm(total=total, desc="Embedding layouts", unit="layout")
        for batch in dl:
            imgs = batch["layout"].to(device)
            z = ae.encoder(imgs)

            for i in range(len(imgs)):
                scene_id = batch["scene_id"][i]
                room_id = batch["room_id"][i]
                typ = batch["type"][i]
                is_empty = batch["is_empty"][i]
                layout_path = Path(batch["path"][i])

                if is_empty:
                    pbar.update(1)
                    continue

                if typ == "room":
                    fname = f"{scene_id}_{room_id}_layout_emb.{args.format}"
                else:  # scene
                    fname = f"{scene_id}_layout_emb.{args.format}"

                out_path = layout_path.parent / fname
                emb = z[i].cpu()

                if args.format == "pt":
                    torch.save(emb, out_path)
                    emb_dim = emb.numel()
                else:
                    import numpy as np
                    np.save(out_path.with_suffix(".npy"), emb.numpy())

                mask = (df["scene_id"] == scene_id) & (df["room_id"] == room_id) & (df["type"] == typ)
                df.loc[mask, "embedding_path"] = str(out_path.resolve())
                df.loc[mask, "embedding_dim"] = emb_dim
                pbar.update(1)

        pbar.close()

    df.to_csv(args.out_manifest, index=False)
    print(f"[INFO] Wrote updated manifest to {args.out_manifest}")


if __name__ == "__main__":
    main()