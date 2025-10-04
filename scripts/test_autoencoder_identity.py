"""
test_autoencoder_identity.py
Check if a trained AutoEncoder behaves like an identity mapping.
Saves metrics and diagnostic plots to tests/identity_check_<model_name>/
"""

from __future__ import annotations
import os, sys, argparse, yaml, random, time, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from torchvision import utils, transforms
from pathlib import Path

# optional SSIM
try:
    from torchmetrics.image import structural_similarity_index_measure as ssim_tm
    TORCHMETRICS_AVAILABLE = True
except Exception:
    from skimage.metrics import structural_similarity as ssim_sk
    TORCHMETRICS_AVAILABLE = False

sys.path.append(str(Path(__file__).parent.parent / "modules"))
from datasets import LayoutDataset
from autoencoder import AutoEncoder


# ----------------- Utility -----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_model_name(config_path: str) -> str:
    base = os.path.basename(config_path)
    name = os.path.splitext(base)[0]
    if "_" in name:
        return "_".join(name.split("_")[1:])  # drop leading 'config'
    return name


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ----------------- Main Class -----------------
class IdentityTester:
    def __init__(self, cfg, checkpoint_path, manifest_path, device="auto", batch_size=64):
        self.device = self._get_device(device)
        self.cfg = cfg
        self.batch_size = batch_size
        self.model = AutoEncoder.from_config(cfg).to(self.device)
        self._load_checkpoint(checkpoint_path)

        transform = transforms.Compose([
            transforms.Resize((cfg["encoder"]["image_size"], cfg["encoder"]["image_size"])),
            transforms.ToTensor()
        ])
        self.dataset = LayoutDataset(manifest_path, transform=transform, mode="all", skip_empty=True)
        print(f"[DATASET] Loaded {len(self.dataset)} samples")

    def _get_device(self, device_str):
        if device_str == "auto":
            if torch.cuda.is_available():
                print("[DEVICE] Using CUDA")
                return "cuda"
            elif torch.backends.mps.is_available():
                print("[DEVICE] Using MPS")
                return "mps"
            return "cpu"
        return device_str

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"[MODEL] Loaded checkpoint from {path}")

    # ---------- Tests ----------
    def run_tests(self, outdir: str):
        ensure_dir(outdir)
        print(f"[OUTPUT] Results will be saved to {outdir}")

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        batch = next(iter(loader))
        imgs = batch["layout"].to(self.device)
        with torch.no_grad():
            z = self.model.encoder(imgs)
            recon = self.model.decoder(z)

        # Metrics
        mse = nn.MSELoss()(recon, imgs).item()
        l1 = nn.L1Loss()(recon, imgs).item()
        ssim = self._compute_ssim(recon, imgs)
        latent_var = float(torch.var(z).item())
        latent_mean = float(torch.mean(z).item())
        print(f"[METRICS] MSE={mse:.6f}  L1={l1:.6f}  SSIM={ssim:.6f}")
        print(f"[LATENT] mean={latent_mean:.4f}, var={latent_var:.4f}")

        # Pairwise latent difference
        diffs = []
        for i in range(len(z) - 1):
            diffs.append(torch.mean(torch.abs(z[i] - z[i + 1])).item())
        latent_diff_mean = float(np.mean(diffs))
        print(f"[LATENT] avg pairwise |Δz|={latent_diff_mean:.6f}")

        # Save reconstruction grid
        self._save_reconstruction_grid(imgs, recon, outdir)
        # Latent histogram
        self._plot_latent_histogram(z, outdir)
        # PCA
        self._plot_pca(z, outdir)
        # Interpolation
        self._interpolate(z, outdir)
        # Random latent decode
        self._random_decode(z, outdir)

        metrics = {
            "mse": mse,
            "l1": l1,
            "ssim": ssim,
            "latent_mean": latent_mean,
            "latent_var": latent_var,
            "latent_diff_mean": latent_diff_mean
        }
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("[DONE] Metrics and plots saved")

    # ---------- Helpers ----------
    def _compute_ssim(self, recon, imgs):
        if TORCHMETRICS_AVAILABLE:
            metric = ssim_tm().to(self.device)
            return float(metric(recon, imgs).item())
        else:
            img_np = to_numpy(imgs[0].permute(1, 2, 0))
            rec_np = to_numpy(recon[0].permute(1, 2, 0))
            return float(ssim_sk(img_np, rec_np, channel_axis=2, data_range=1.0))

    def _save_reconstruction_grid(self, imgs, recon, outdir):
        comparison = torch.cat([imgs[:4], recon[:4]], dim=0)
        grid = utils.make_grid(comparison, nrow=4, value_range=(0, 1))
        path = os.path.join(outdir, "reconstruction_grid.png")
        utils.save_image(grid, path)
        print(f"[SAVE] reconstruction grid -> {path}")

    def _plot_latent_histogram(self, z, outdir):
        z_np = to_numpy(z).flatten()
        plt.figure(figsize=(6, 4))
        sns.histplot(z_np, bins=80, kde=True)
        plt.title("Latent Value Distribution")
        plt.tight_layout()
        path = os.path.join(outdir, "latent_hist.png")
        plt.savefig(path)
        plt.close()
        print(f"[SAVE] latent histogram -> {path}")

    def _plot_pca(self, z, outdir):
        z_flat = z.view(z.size(0), -1).cpu().numpy()
        pca = PCA(n_components=min(10, z_flat.shape[1]))
        pca.fit(z_flat)
        var_ratio = pca.explained_variance_ratio_
        plt.figure(figsize=(6, 4))
        sns.barplot(x=np.arange(len(var_ratio)), y=var_ratio)
        plt.title("PCA Explained Variance Ratio (first 10)")
        plt.tight_layout()
        path = os.path.join(outdir, "latent_pca.png")
        plt.savefig(path)
        plt.close()
        print(f"[SAVE] latent PCA plot -> {path}")

    def _interpolate(self, z, outdir):
        if z.size(0) < 2:
            return
        z1, z2 = z[0], z[1]
        alphas = torch.linspace(0, 1, 5).to(self.device)
        imgs = []
        with torch.no_grad():
            for a in alphas:
                zi = (1 - a) * z1 + a * z2
                imgs.append(self.model.decoder(zi.unsqueeze(0))[0])
        grid = utils.make_grid(torch.stack(imgs), nrow=5, value_range=(0, 1))
        path = os.path.join(outdir, "latent_interpolation.png")
        utils.save_image(grid, path)
        print(f"[SAVE] latent interpolation -> {path}")

    def _random_decode(self, z, outdir):
        shape = z.shape
        rand_z = torch.randn_like(z)
        with torch.no_grad():
            rand_img = self.model.decoder(rand_z)
        grid = utils.make_grid(rand_img[:4], nrow=4, value_range=(0, 1))
        path = os.path.join(outdir, "random_latent_decode.png")
        utils.save_image(grid, path)
        print(f"[SAVE] random latent decode -> {path}")


# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Test AutoEncoder for identity mapping behavior")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--layout_manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_suffix = parse_model_name(args.config)
    outdir = os.path.join("tests", f"identity_check_{model_suffix}")
    ensure_dir(outdir)

    print(f"[START] Testing model: {model_suffix}")
    print(f"Results -> {outdir}\n")

    tester = IdentityTester(cfg, args.checkpoint, args.layout_manifest, args.device, args.batch_size)
    tester.run_tests(outdir)


if __name__ == "__main__":
    main()
