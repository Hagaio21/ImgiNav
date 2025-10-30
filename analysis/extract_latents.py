import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import glob
import yaml
import torch
import pandas as pd
import random
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use("Agg")  # headless backend for HPC
from torch.utils.data import DataLoader
from torchvision import transforms
from modules.autoencoder import AutoEncoder
from modules.datasets import LayoutDataset, collate_skip_none

#!/usr/bin/env python3

# ----------------------------------------------------------------------
def set_deterministic(seed: int = 0):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------
@torch.no_grad()
def extract_experiment_latents(exp_path, dataloader, device, output_root, sample_vis=8):
    """Run encoder on entire dataset and save latents + metadata."""
    exp_name = os.path.basename(os.path.normpath(exp_path))
    out_dir = os.path.join(output_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    cfg_path = os.path.join(exp_path, "output", "autoencoder_config.yaml")
    ckpt_path = os.path.join(exp_path, "checkpoints", "ae_latest.pt")
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        print(f"[WARN] Missing files for {exp_name}, skipping.")
        return

    print(f"Loading model for {exp_name}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = AutoEncoder.from_config(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    all_latents, scene_ids, types, room_ids = [], [], [], []
    example_inputs, example_recons = None, None

    for i, batch in enumerate(dataloader):
        if batch is None:
            continue

        imgs = batch["layout"]
        if not torch.is_tensor(imgs):
            imgs = torch.stack([transforms.ToTensor()(im) for im in imgs])

        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        lat = out["mu"].detach().cpu()
        all_latents.append(lat)

        scene_ids.extend(batch["scene_id"])
        types.extend(batch["type"])
        room_ids.extend(batch["room_id"])

        if example_inputs is None:
            recons = out["recon"].detach().cpu()
            example_inputs = imgs.cpu()[:sample_vis]
            example_recons = recons[:sample_vis]

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} batches")

    latents = torch.cat(all_latents)
    torch.save(
        {
            "latents": latents,
            "scene_id": scene_ids,
            "type": types,
            "room_id": room_ids,
        },
        os.path.join(out_dir, "latents.pt"),
    )
    print(f"Saved latents {latents.shape} → {out_dir}/latents.pt")

    if example_inputs is not None:
        inputs = (example_inputs - example_inputs.min()) / (
            example_inputs.max() - example_inputs.min() + 1e-8
        )
        recons = (example_recons - example_recons.min()) / (
            example_recons.max() - example_recons.min() + 1e-8
        )
        grid = make_grid(torch.cat([inputs, recons], dim=0), nrow=sample_vis)
        save_image(grid, os.path.join(out_dir, "examples.png"))
        print(f"Saved recon grid {out_dir}/examples.png")

    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------
def main(manifest, output_dir, exp_root, batch_size=32, num_workers=8):
    set_deterministic(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([transforms.ToTensor()])
    dataset = LayoutDataset(manifest, transform=tfm)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        pin_memory=True,
    )

    exp_paths = sorted(glob.glob(os.path.join(exp_root, "AEVAE_sweep", "*/")))
    print(f"Found {len(exp_paths)} experiments.")
    os.makedirs(output_dir, exist_ok=True)

    for i, exp_path in enumerate(exp_paths, 1):
        print(f"[{i}/{len(exp_paths)}] {exp_path}")
        extract_experiment_latents(exp_path, dataloader, device, output_dir)
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract AE/VAE latents (GPU, full dataset).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    main(args.manifest, args.output_dir, args.exp_root, args.batch_size, args.num_workers)
