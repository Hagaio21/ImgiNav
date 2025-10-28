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

# Deterministic setup
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


# ---------------------------------------------------------------------
@torch.no_grad()
def extract_experiment_latents(exp_path, dataloader, device, output_root, save_recons=True, sample_vis=8):
    """Extract and save latents (mu) for one experiment."""
    exp_name = os.path.basename(os.path.normpath(exp_path))
    out_dir = os.path.join(output_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    cfg_path = os.path.join(exp_path, "output", "autoencoder_config.yaml")
    ckpt_path = os.path.join(exp_path, "checkpoints", "ae_latest.pt")

    print(f"Loading {exp_name}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = AutoEncoder.from_config(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    all_mu = []
    example_inputs, example_recons = None, None

    for i, batch in enumerate(dataloader):
        if batch is None:
            continue

        imgs = batch["layout"]
        # handle raw PIL images if dataset doesn’t convert automatically
        if not torch.is_tensor(imgs):
            imgs = torch.stack([transforms.ToTensor()(im) for im in imgs])

        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        mu = out["mu"].detach().cpu()
        all_mu.append(mu)

        # optionally keep a few sample visuals
        if save_recons and example_inputs is None:
            recons = out["recon"].detach().cpu()
            example_inputs = imgs.cpu()[:sample_vis]
            example_recons = recons[:sample_vis]

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} batches")

    latents = torch.cat(all_mu, dim=0)
    torch.save(latents, os.path.join(out_dir, "latents.pt"))
    print(f"Saved latents: {latents.shape} -> {out_dir}/latents.pt")

    if save_recons and example_inputs is not None:
        # normalize to [0,1]
        inputs = (example_inputs - example_inputs.min()) / (example_inputs.max() - example_inputs.min() + 1e-8)
        recons = (example_recons - example_recons.min()) / (example_recons.max() - example_recons.min() + 1e-8)
        grid = make_grid(torch.cat([inputs, recons], dim=0), nrow=sample_vis)
        save_image(grid, os.path.join(out_dir, "examples.png"))
        print(f"Saved recon example grid to {out_dir}/examples.png")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------
def main(manifest, output_dir, exp_root, batch_size=32, num_workers=8):
    set_deterministic(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision import transforms

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
    print(f"Found {len(exp_paths)} experiments under {exp_root}/AEVAE_sweep/")

    os.makedirs(output_dir, exist_ok=True)
    for i, exp_path in enumerate(exp_paths, 1):
        print(f"[{i}/{len(exp_paths)}] Extracting latents for {exp_path}")
        extract_experiment_latents(exp_path, dataloader, device, output_dir)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract AE/VAE latent representations for full dataset.")
    parser.add_argument("--manifest", required=True, help="Path to dataset manifest CSV.")
    parser.add_argument("--output_dir", required=True, help="Output directory for latent .pt files.")
    parser.add_argument("--exp_root", required=True, help="Root dir containing AEVAE_sweep experiments.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    main(args.manifest, args.output_dir, args.exp_root, args.batch_size, args.num_workers)