#!/usr/bin/env python3
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from latent_metrics import (
    compute_distribution_metrics,
    compute_geometric_metrics,
    compute_information_metrics,
)
from latent_plots import (
    plot_umap,
    plot_latent_hist,
    plot_pca_scatter,
)


# ----------------------------------------------------------------------
def set_deterministic(seed: int = 0):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------
@torch.no_grad()
def analyze_latent_file(latent_path, output_dir):
    """Analyze one experiment's saved latents.pt."""
    exp_name = os.path.basename(os.path.dirname(latent_path))
    out_dir = os.path.join(output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Analyzing {exp_name}")
    latents = torch.load(latent_path, map_location="cpu")

    # Subsample if extremely large for UMAP
    if latents.shape[0] > 2000:
        step = latents.shape[0] // 2000
        latents_small = latents[::step]
    else:
        latents_small = latents

    # Compute metrics
    dist_metrics = compute_distribution_metrics(latents)
    info_metrics = compute_information_metrics(latents_small)

    # Geometry metrics (approximate, using spatial flatten)
    flat = latents_small.reshape(latents_small.shape[0], -1)
    geom_metrics = compute_geometric_metrics(latents_small, flat)

    # Plots
    plot_latent_hist(latents, os.path.join(out_dir, "latent_hist.png"), exp_name)
    plot_pca_scatter(latents_small, os.path.join(out_dir, "pca_scatter.png"), exp_name)
    plot_umap(latents_small, np.zeros(latents_small.shape[0]), os.path.join(out_dir, "umap.png"), exp_name)

    # Merge all results
    results = {"exp_name": exp_name}
    results.update(dist_metrics)
    results.update(info_metrics)
    results.update(geom_metrics)

    return results


# ----------------------------------------------------------------------
def main(latent_root, output_dir):
    set_deterministic(0)
    os.makedirs(output_dir, exist_ok=True)

    latent_paths = sorted(glob.glob(os.path.join(latent_root, "*/latents.pt")))
    if not latent_paths:
        raise FileNotFoundError(f"No latents.pt files found under {latent_root}")

    print(f"Found {len(latent_paths)} latent sets")

    results = []
    for i, lp in enumerate(latent_paths, 1):
        print(f"[{i}/{len(latent_paths)}] {lp}")
        res = analyze_latent_file(lp, output_dir)
        results.append(res)

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved combined metrics to {csv_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze saved latent embeddings (CPU-only).")
    parser.add_argument("--latent_root", required=True, help="Directory containing <exp>/latents.pt files.")
    parser.add_argument("--output_dir", required=True, help="Directory for CSV and plots.")
    args = parser.parse_args()

    main(args.latent_root, args.output_dir)
