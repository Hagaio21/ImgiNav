import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from torchvision.utils import make_grid, save_image

sns.set_style("darkgrid")


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


@torch.no_grad()
def plot_umap(latents, labels, out_path, title_prefix=""):
    """UMAP projection of latents (N, C, H, W) -> (N, 2)."""
    _ensure_dir(out_path)
    lat_flat = latents.reshape(latents.shape[0], -1).cpu().numpy()
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="euclidean")
    emb = reducer.fit_transform(lat_flat)

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
    plt.title(f"UMAP Latent Projection — {title_prefix}" if title_prefix else "UMAP Latent Projection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_latent_hist(latents, out_path, title_prefix=""):
    """Histogram of latent mean and std per channel."""
    _ensure_dir(out_path)
    lat_flat = latents.permute(1, 0, 2, 3).reshape(latents.shape[1], -1)
    mu = lat_flat.mean(dim=1).cpu().numpy()
    std = lat_flat.std(dim=1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sns.histplot(mu, bins=30, ax=axes[0], color="steelblue", kde=True)
    axes[0].set_title(f"Latent Channel Means — {title_prefix}" if title_prefix else "Latent Channel Means")

    sns.histplot(std, bins=30, ax=axes[1], color="seagreen", kde=True)
    axes[1].set_title(f"Latent Channel STDs — {title_prefix}" if title_prefix else "Latent Channel STDs")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def plot_pca_scatter(latents, out_path, title_prefix=""):
    """PCA scatter of first two latent components."""
    _ensure_dir(out_path)
    flat = latents.reshape(latents.shape[0], -1).cpu().numpy()
    pca = PCA(n_components=2)
    emb = pca.fit_transform(flat)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], s=8, alpha=0.7)
    plt.title(f"PCA Latent Scatter — {title_prefix}" if title_prefix else "PCA Latent Scatter")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def save_recon_examples(inputs, recons, out_path, nrow=8, title_prefix=""):
    """Save grid comparison of inputs and reconstructions."""
    _ensure_dir(out_path)
    B = min(inputs.shape[0], nrow)
    inputs = inputs[:B].detach().cpu()
    recons = recons[:B].detach().cpu()

    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
    recons = (recons - recons.min()) / (recons.max() - recons.min() + 1e-8)

    grid = torch.cat([inputs, recons], dim=0)
    grid_img = make_grid(grid, nrow=nrow)
    save_image(grid_img, out_path)

    # optional text-only log
    print(f"Saved reconstruction grid for {title_prefix} -> {out_path}")
