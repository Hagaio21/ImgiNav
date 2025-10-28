#!/usr/bin/env python3
"""
Analyze layout dataset statistics and diversity.

Computes:
  - Scene/room distribution
  - Empty vs non-empty ratios
  - Image dimension and intensity statistics
  - Duplicate detection via perceptual hash
  - Diversity visualizations
  - Sample grids

Outputs in dataset_analysis/:
  dataset_stats.json
  dataset_summary.csv
  intensity_hist.png
  room_scene_distribution.png
  empty_ratio_pie.png
  random_samples.png
  duplicates_heatmap.png
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import torch
import torchvision.transforms as T


# ----------------------------------------------------------------------
def set_deterministic(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------
def compute_image_stats(path):
    """Return image mean, std, width, height."""
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img) / 255.0
        mean = arr.mean()
        std = arr.std()
        h, w = arr.shape[:2]
        return mean, std, w, h
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ----------------------------------------------------------------------
def perceptual_hash(path, hash_size=8):
    """Compute simple perceptual hash."""
    try:
        img = Image.open(path).convert("L").resize((hash_size, hash_size))
        pixels = np.array(img, dtype=np.float32)
        diff = pixels > pixels.mean()
        return hashlib.sha1(diff.tobytes()).hexdigest()
    except Exception:
        return None


# ----------------------------------------------------------------------
def sample_grid(paths, save_path, n=16):
    """Save grid of n random images."""
    tfm = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    subset = np.random.choice(paths, min(n, len(paths)), replace=False)
    imgs = []
    for p in subset:
        try:
            imgs.append(tfm(Image.open(p).convert("RGB")))
        except Exception:
            continue
    if not imgs:
        return
    grid = make_grid(torch.stack(imgs), nrow=int(np.sqrt(len(imgs))))
    save_image(grid, save_path)


# ----------------------------------------------------------------------
def analyze_dataset(manifest_path, output_dir):
    set_deterministic(0)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} entries from manifest.")

    # ------------------ Basic counts ------------------
    n_total = len(df)
    n_scenes = (df["type"] == "scene").sum()
    n_rooms = (df["type"] == "room").sum()
    n_empty = (df["is_empty"] == True).sum()
    n_nonempty = n_total - n_empty

    # Scene/room per scene distribution
    rooms_per_scene = df[df["type"] == "room"].groupby("scene_id")["room_id"].nunique()

    # ------------------ Image-level statistics ------------------
    means, stds, widths, heights = [], [], [], []
    for p in tqdm(df["layout_path"], desc="Computing image stats"):
        m, s, w, h = compute_image_stats(p)
        means.append(m)
        stds.append(s)
        widths.append(w)
        heights.append(h)

    df["mean_intensity"] = means
    df["std_intensity"] = stds
    df["width"] = widths
    df["height"] = heights
    df["aspect_ratio"] = df["width"] / df["height"]

    # ------------------ Duplicate detection ------------------
    print("Computing perceptual hashes for duplicate detection...")
    df["phash"] = [perceptual_hash(p) for p in tqdm(df["layout_path"], desc="Hashing")]
    dup_counts = df["phash"].value_counts()
    n_unique = dup_counts.shape[0]
    n_duplicates = (dup_counts > 1).sum()

    # ------------------ Summaries ------------------
    summary = dict(
        total_images=int(n_total),
        scenes=int(n_scenes),
        rooms=int(n_rooms),
        empty=int(n_empty),
        non_empty=int(n_nonempty),
        duplicates=int(n_duplicates),
        unique_hashes=int(n_unique),
        avg_width=float(np.nanmean(widths)),
        avg_height=float(np.nanmean(heights)),
        avg_mean_intensity=float(np.nanmean(means)),
        avg_std_intensity=float(np.nanmean(stds)),
        mean_aspect_ratio=float(np.nanmean(df["aspect_ratio"])),
    )

    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)

    # ------------------ Plots ------------------
    plt.figure()
    sns.histplot(df["mean_intensity"].dropna(), bins=50, color="skyblue")
    plt.title("Mean Pixel Intensity Distribution")
    plt.xlabel("Mean intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intensity_hist.png"), dpi=200)
    plt.close()

    plt.figure()
    sns.histplot(df["aspect_ratio"].dropna(), bins=50, color="orange")
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Width / Height")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aspect_ratio_hist.png"), dpi=200)
    plt.close()

    plt.figure()
    sns.histplot(rooms_per_scene, bins=40, color="green")
    plt.title("Rooms per Scene Distribution")
    plt.xlabel("Rooms per scene")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "room_scene_distribution.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.pie([n_empty, n_nonempty], labels=["Empty", "Non-empty"], autopct="%1.1f%%", colors=["lightgray", "lightblue"])
    plt.title("Empty vs Non-Empty Layouts")
    plt.savefig(os.path.join(output_dir, "empty_ratio_pie.png"), dpi=200)
    plt.close()

    # ------------------ Duplicate visualization ------------------
    if n_duplicates > 0:
        dup_pairs = df[df["phash"].isin(dup_counts[dup_counts > 1].index)]
        dup_map = dup_pairs.groupby("phash")["layout_path"].apply(list)
        heat = np.zeros((len(dup_map), len(dup_map)))
        for i, paths_i in enumerate(dup_map):
            for j, paths_j in enumerate(dup_map):
                if i == j:
                    heat[i, j] = 1
                else:
                    heat[i, j] = len(set(paths_i) & set(paths_j)) > 0
        plt.figure(figsize=(6, 5))
        sns.heatmap(heat, cmap="magma", cbar=False)
        plt.title("Duplicate Groups Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "duplicates_heatmap.png"), dpi=200)
        plt.close()

    # ------------------ Sample grids ------------------
    sample_grid(df["layout_path"].tolist(), os.path.join(output_dir, "random_samples.png"), n=16)
    sample_grid(df[df["is_empty"] == False]["layout_path"].tolist(), os.path.join(output_dir, "non_empty_samples.png"), n=16)
    sample_grid(df[df["is_empty"] == True]["layout_path"].tolist(), os.path.join(output_dir, "empty_samples.png"), n=16)

    print("Dataset analysis complete.")
    return summary


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze dataset manifest and image statistics.")
    parser.add_argument("--manifest", required=True, help="Path to dataset CSV manifest.")
    parser.add_argument("--output_dir", required=True, help="Output directory for analysis results.")
    args = parser.parse_args()
    analyze_dataset(args.manifest, args.output_dir)
