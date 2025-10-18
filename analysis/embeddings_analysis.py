"""
analyze_embedding_diversity.py
----------------------------------
Evaluate diversity and similarity of precomputed embeddings (POV, Graph, Layout)
across room and scene datasets.

Usage:
    python analyze_embedding_diversity.py \
        --room_manifest /path/to/room_manifest.csv \
        --scene_manifest /path/to/scene_manifest.csv \
        --out_dir /path/to/output \
        --subsample 5000
"""

import os
import json
import random
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def load_embeddings_from_manifest(df, column_path, key_column="ROOM_ID"):
    """Load embeddings from .pt files grouped by key_column (room or scene)."""
    grouped = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {column_path}"):
        path = row[column_path]
        key = row[key_column]
        if not os.path.exists(path):
            continue
        emb = torch.load(path, map_location="cpu")
        if isinstance(emb, dict) and "embedding" in emb:
            emb = emb["embedding"]
        if isinstance(emb, torch.Tensor):
            emb = emb.flatten().float().numpy()
        else:
            continue
        grouped.setdefault(key, []).append(emb)
    return grouped


def compute_pairwise_stats(grouped_embs, subsample=5000, seed=42):
    """Compute intra-scene and inter-scene similarity and distance."""
    random.seed(seed)
    np.random.seed(seed)

    # Flatten all embeddings and keep track of group labels
    all_embs, labels = [], []
    for k, vlist in grouped_embs.items():
        for v in vlist:
            all_embs.append(v)
            labels.append(k)
    all_embs = np.stack(all_embs)
    labels = np.array(labels)

    # Subsample for efficiency
    if len(all_embs) > subsample:
        idx = np.random.choice(len(all_embs), subsample, replace=False)
        all_embs = all_embs[idx]
        labels = labels[idx]

    # Compute pairwise similarities/distances
    cos_sim = cosine_similarity(all_embs)
    euc_dist = euclidean_distances(all_embs)

    # Mask for intra/inter group
    intra_mask = labels[:, None] == labels[None, :]
    inter_mask = ~intra_mask

    # Upper triangle only (avoid duplicates)
    triu = np.triu_indices_from(cos_sim, k=1)

    intra_cos = cos_sim[triu][intra_mask[triu]]
    inter_cos = cos_sim[triu][inter_mask[triu]]
    intra_euc = euc_dist[triu][intra_mask[triu]]
    inter_euc = euc_dist[triu][inter_mask[triu]]

    return {
        "intra_cos": intra_cos.tolist(),
        "inter_cos": inter_cos.tolist(),
        "intra_euc": intra_euc.tolist(),
        "inter_euc": inter_euc.tolist(),
        "summary": {
            "mean_intra_cos": float(np.mean(intra_cos)),
            "mean_inter_cos": float(np.mean(inter_cos)),
            "mean_intra_euc": float(np.mean(intra_euc)),
            "mean_inter_euc": float(np.mean(inter_euc)),
            "std_intra_cos": float(np.std(intra_cos)),
            "std_inter_cos": float(np.std(inter_cos)),
            "std_intra_euc": float(np.std(intra_euc)),
            "std_inter_euc": float(np.std(inter_euc)),
            "cosine_gap": float(np.mean(intra_cos) - np.mean(inter_cos)),
            "euclidean_gap": float(np.mean(inter_euc) - np.mean(intra_euc)),
        }
    }


def plot_histograms(stats, name, out_dir):
    """Plot histograms for cosine similarity and Euclidean distance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cosine similarity
    plt.figure(figsize=(6, 4))
    plt.hist(stats["intra_cos"], bins=50, alpha=0.6, label="Intra-scene", color="blue")
    plt.hist(stats["inter_cos"], bins=50, alpha=0.6, label="Inter-scene", color="orange")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(f"{name} – Cosine Similarity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.lower()}_cosine_hist.png", dpi=150)
    plt.close()

    # Euclidean distance
    plt.figure(figsize=(6, 4))
    plt.hist(stats["intra_euc"], bins=50, alpha=0.6, label="Intra-scene", color="blue")
    plt.hist(stats["inter_euc"], bins=50, alpha=0.6, label="Inter-scene", color="orange")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.title(f"{name} – Euclidean Distance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.lower()}_euclidean_hist.png", dpi=150)
    plt.close()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze embedding diversity")
    parser.add_argument("--room_manifest", type=str, required=True)
    parser.add_argument("--scene_manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--subsample", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifests...")
    room_df = pd.read_csv(args.room_manifest)
    scene_df = pd.read_csv(args.scene_manifest)

    results = {}

    # --- Room-level embeddings ---
    for col, label in [
        ("POV_EMBEDDING_PATH", "POV"),
        ("ROOM_GRAPH_EMBEDDING_PATH", "RoomGraph"),
        ("ROOM_LAYOUT_EMBEDDING_PATH", "RoomLayout"),
    ]:
        print(f"\nAnalyzing {label} embeddings...")
        grouped = load_embeddings_from_manifest(room_df, col, key_column="ROOM_ID")
        stats = compute_pairwise_stats(grouped, subsample=args.subsample)
        results[label] = stats["summary"]
        plot_histograms(stats, label, out_dir)

    # --- Scene-level embeddings ---
    for col, label in [
        ("SCENE_GRAPH_EMBEDDING_PATH", "SceneGraph"),
        ("SCENE_LAYOUT_EMBEDDING_PATH", "SceneLayout"),
    ]:
        print(f"\nAnalyzing {label} embeddings...")
        grouped = load_embeddings_from_manifest(scene_df, col, key_column="SCENE_ID")
        stats = compute_pairwise_stats(grouped, subsample=args.subsample)
        results[label] = stats["summary"]
        plot_histograms(stats, label, out_dir)

    # Save summary
    summary_path = out_dir / "embedding_diversity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
