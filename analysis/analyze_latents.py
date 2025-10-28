#!/usr/bin/env python3
"""
Analyze AE/VAE latent spaces for diffusion suitability and identity-risk diagnostics.

Outputs:
  latent_analysis_results/<exp_name>/
      - metrics.json
      - umap_continents.png
      - umap_countries_<type>.png
      - random_decoded.png
      - interpolation.png
  latent_analysis_results/results.csv
  latent_analysis_results/summary_diffusion_fitness.png
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import yaml
from torchvision.utils import make_grid, save_image

from modules.autoencoder import AutoEncoder

sns.set_theme(style="darkgrid")


# ----------------------------------------------------------------------
def set_deterministic(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------
def compute_latent_health_metrics(latents_np):
    flat = latents_np.reshape(latents_np.shape[0], -1)
    var_per_feat = flat.var(axis=0)
    mean_var = var_per_feat.mean()
    zero_var_ratio = np.mean(var_per_feat < 1e-6)

    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    normalized = flat / norms
    sim_matrix = normalized @ normalized.T
    upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    mean_sim, std_sim = upper.mean(), upper.std()

    pca = PCA(n_components=min(50, flat.shape[1]))
    pca.fit(flat)
    var_ratio = pca.explained_variance_ratio_
    pca_var_ratio1 = var_ratio[0]
    pca_var_ratio10 = var_ratio[:10].sum()
    pca_uniformity = 1.0 / (np.std(var_ratio[:10]) + 1e-8)
    isotropy_ratio = var_per_feat.max() / (var_per_feat.min() + 1e-8)

    return dict(
        mean_feature_var=float(mean_var),
        zero_var_ratio=float(zero_var_ratio),
        mean_cosine_sim=float(mean_sim),
        std_cosine_sim=float(std_sim),
        pca_var_ratio1=float(pca_var_ratio1),
        pca_var_ratio10=float(pca_var_ratio10),
        isotropy_ratio=float(isotropy_ratio),
        pca_uniformity=float(pca_uniformity),
    )


# ----------------------------------------------------------------------
def compute_identity_diagnostics(model, latents, room_ids, out_dir, device):
    results = {}
    os.makedirs(out_dir, exist_ok=True)

    # ----- Random latent decoding -----
    z_shape = latents.shape[1:]
    z_rand = torch.randn(16, *z_shape, device=device)
    with torch.no_grad():
        decoded, _ = model.decode(z_rand, from_latent=True)
    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-8)
    save_image(make_grid(decoded.cpu(), nrow=4), os.path.join(out_dir, "random_decoded.png"))
    results["rand_decode_var"] = float(decoded.var().item())

    # ----- Latent interpolation -----
    if len(latents) >= 2:
        idx_a, idx_b = np.random.choice(len(latents), 2, replace=False)
        z_a = torch.tensor(latents[idx_a:idx_a+1], device=device)
        z_b = torch.tensor(latents[idx_b:idx_b+1], device=device)
        alphas = torch.linspace(0, 1, 10, device=device).view(-1, 1, 1, 1)
        z_interp = (1 - alphas) * z_a + alphas * z_b
        with torch.no_grad():
            x_interp, _ = model.decode(z_interp, from_latent=True)
        x_interp = (x_interp - x_interp.min()) / (x_interp.max() - x_interp.min() + 1e-8)
        save_image(make_grid(x_interp.cpu(), nrow=10), os.path.join(out_dir, "interpolation.png"))
        diffs = [torch.mean((x_interp[i+1] - x_interp[i])**2).item() for i in range(9)]
        results["interp_smoothness"] = float(np.mean(diffs))
    else:
        results["interp_smoothness"] = np.nan

    # ----- Seen vs unseen generalization -----
    unique_rooms = np.unique(room_ids)
    split_point = int(0.8 * len(unique_rooms))
    seen_rooms = unique_rooms[:split_point]
    unseen_rooms = unique_rooms[split_point:]

    seen_mask = np.isin(room_ids, seen_rooms)
    unseen_mask = np.isin(room_ids, unseen_rooms)

    def var_mean(mask):
        if not np.any(mask):
            return np.nan
        subset = latents[mask]
        flat = subset.reshape(subset.shape[0], -1)
        return flat.var(axis=0).mean()

    latent_var_seen = var_mean(seen_mask)
    latent_var_unseen = var_mean(unseen_mask)
    results["latent_var_seen"] = float(latent_var_seen)
    results["latent_var_unseen"] = float(latent_var_unseen)

    diff = abs(latent_var_seen - latent_var_unseen)
    r = 0.5 * (1 - results["rand_decode_var"]) + 0.3 * (1 - results["interp_smoothness"]) + 0.2 * diff
    results["identity_risk_score"] = float(max(0.0, min(1.0, r)))

    return results


# ----------------------------------------------------------------------
def analyze_experiment(exp_path, exp_root, output_root, device):
    exp_name = os.path.basename(os.path.dirname(exp_path))
    print(f"[INFO] Analyzing {exp_name}")
    out_dir = os.path.join(output_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    data = torch.load(exp_path, map_location="cpu")
    latents = data["latents"].float().numpy()
    scene_ids = np.array(data["scene_id"])
    types = np.array(data["type"])
    room_ids = np.array(data["room_id"])

    flat = latents.reshape(latents.shape[0], -1)
    metrics = compute_latent_health_metrics(latents)

    try:
        metrics["silhouette_type"] = float(silhouette_score(flat, types, metric="cosine"))
    except Exception:
        metrics["silhouette_type"] = np.nan
    try:
        metrics["silhouette_roomid"] = float(silhouette_score(flat, room_ids, metric="cosine"))
    except Exception:
        metrics["silhouette_roomid"] = np.nan
    try:
        metrics["db_type"] = float(davies_bouldin_score(flat, types))
    except Exception:
        metrics["db_type"] = np.nan

    # ------------------ UMAP visualizations ------------------
    try:
        subset_idx = np.random.choice(len(flat), min(5000, len(flat)), replace=False)
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(flat[subset_idx])
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=types[subset_idx], s=8, palette="tab10")
        plt.legend(loc="best", fontsize="small")
        plt.title(f"{exp_name} — Continents (type)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "umap_continents.png"), dpi=200)
        plt.close()

        # dual palette setup for room_id-based plots
        palette_a = sns.color_palette("husl", 15)
        palette_b = sns.color_palette("coolwarm", 16)
        palette = palette_a + palette_b

        for t in np.unique(types):
            mask = types == t
            if mask.sum() < 100:
                continue
            sub_idx = np.random.choice(np.where(mask)[0], min(3000, mask.sum()), replace=False)
            sub_emb = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(flat[sub_idx])

            unique_rooms = np.unique(room_ids[sub_idx])
            color_map = {rid: palette[i % len(palette)] for i, rid in enumerate(unique_rooms)}
            colors = [color_map[r] for r in room_ids[sub_idx]]

            plt.figure(figsize=(6, 6))
            plt.scatter(sub_emb[:, 0], sub_emb[:, 1], c=colors, s=8)
            plt.title(f"{exp_name} — Countries within {t}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"umap_countries_{t}.png"), dpi=200)
            plt.close()
    except Exception as e:
        print(f"[WARN] UMAP failed for {exp_name}: {e}")

    # ------------------ Room ID legend (always save if possible) ------------------
    try:
        unique_rooms_global = np.unique(room_ids)
        color_map_global = {rid: palette[i % len(palette)] for i, rid in enumerate(unique_rooms_global)}

        fig, ax = plt.subplots(figsize=(4, 0.25 * len(unique_rooms_global)))
        ax.axis("off")
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map_global[r], markersize=6, label=str(r))
            for r in unique_rooms_global
        ]
        ax.legend(
            handles=handles,
            loc="center",
            frameon=False,
            ncol=2,
            title="Room IDs",
            fontsize="small",
            title_fontsize="small"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "room_id_legend.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to save legend for {exp_name}: {e}")

    # ------------------ Identity diagnostics ------------------
    cfg_path = os.path.join(exp_root, exp_name, "output", "autoencoder_config.yaml")
    ckpt_path = os.path.join(exp_root, exp_name, "checkpoints", "ae_latest.pt")
    if os.path.exists(cfg_path) and os.path.exists(ckpt_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model = AutoEncoder.from_config(cfg)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()
        id_metrics = compute_identity_diagnostics(model, latents, room_ids, out_dir, device)
        metrics.update(id_metrics)
        del model
        torch.cuda.empty_cache()
    else:
        print(f"[WARN] Missing model files for {exp_name}, skipping identity checks.")

    # ------------------ Diffusion fitness ------------------
    w1, w2, w3, w4 = 0.3, 0.3, 0.2, 0.2
    isotropy = 1 / (abs(np.log(metrics["isotropy_ratio"])) + 1e-8)
    collapse = metrics["zero_var_ratio"]
    pca_uniformity = metrics["pca_uniformity"]
    silhouette = metrics["silhouette_type"] if not np.isnan(metrics["silhouette_type"]) else 0
    metrics["diffusion_fitness_score"] = (
        w1 * isotropy + w2 * (1 - collapse) + w3 * pca_uniformity + w4 * silhouette
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Completed {exp_name}")
    return dict(exp_name=exp_name, **metrics)


# ----------------------------------------------------------------------
def main(latent_root, exp_root, output_root):
    set_deterministic(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_root, exist_ok=True)

    latent_paths = []
    for exp_name in sorted(os.listdir(latent_root)):
        exp_dir = os.path.join(latent_root, exp_name)
        latents_file = os.path.join(exp_dir, "latents.pt")
        if os.path.isfile(latents_file):
            latent_paths.append(latents_file)

    print(f"[INFO] Found {len(latent_paths)} experiments")

    all_results = []
    for lp in latent_paths:
        res = analyze_experiment(lp, exp_root, output_root, device)
        all_results.append(res)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_root, "results.csv")
    df.to_csv(csv_path, index=False)

    df_sorted = df.sort_values("diffusion_fitness_score", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_sorted, x="diffusion_fitness_score", y="exp_name", palette="viridis")
    plt.xlabel("Diffusion Fitness Score")
    plt.ylabel("Experiment")
    plt.title("Latent Space Diffusion Fitness Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "summary_diffusion_fitness.png"), dpi=250)
    plt.close()

    print("[DONE] Latent analysis complete.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze AE/VAE latents with identity diagnostics.")
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    main(args.latent_root, args.exp_root, args.output_root)
