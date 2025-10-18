#!/usr/bin/env python3
"""
analyze_alignment_effect_refactored.py
--------------------------------------
Evaluate embedding diversity and alignment quality before and after alignment.

Adds:
- Early subsampling before loading any embeddings
- Combined plots showing before/after/optimal curves

Usage:
    python analysis/analyze_alignment_effect_refactored.py \
        --room_manifest /path/to/room_dataset_with_emb.csv \
        --scene_manifest /path/to/scene_dataset_with_emb.csv \
        --checkpoint /path/to/best.pt \
        --out_dir /path/to/output \
        --subsample 5000
"""

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from modules.alignment import AlignmentMLP


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def safe_load_embedding(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return None
    emb = torch.load(path, map_location="cpu")
    if isinstance(emb, dict) and "embedding" in emb:
        emb = emb["embedding"]
    if isinstance(emb, torch.Tensor):
        return emb.flatten().float()
    return None


def load_embeddings(df, emb_col, key_col):
    grouped = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {emb_col}"):
        emb = safe_load_embedding(row.get(emb_col))
        if emb is None:
            continue
        grouped.setdefault(row[key_col], []).append(emb.numpy())
    return grouped


def compute_stats(embs, subsample=5000):
    all_embs = np.stack(embs)
    if len(all_embs) > subsample:
        idx = np.random.choice(len(all_embs), subsample, replace=False)
        all_embs = all_embs[idx]
    cos_sim = cosine_similarity(all_embs)
    euc_dist = euclidean_distances(all_embs)
    triu = np.triu_indices_from(cos_sim, k=1)
    return {
        "var": float(np.var(all_embs)),
        "mean_cos": float(np.mean(cos_sim[triu])),
        "std_cos": float(np.std(cos_sim[triu])),
        "mean_euc": float(np.mean(euc_dist[triu])),
        "std_euc": float(np.std(euc_dist[triu])),
        "cos_hist": cos_sim[triu],
        "euc_hist": euc_dist[triu],
    }


def plot_comparison(before, after, name, out_dir):
    # Cosine similarity
    plt.figure(figsize=(6, 4))
    plt.hist(before["cos_hist"], bins=50, alpha=0.5, color="blue", label="Before alignment", density=True)
    plt.hist(after["cos_hist"], bins=50, alpha=0.5, color="orange", label="After alignment", density=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Optimal")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(f"{name} – Cosine similarity comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.lower()}_cosine_comparison.png", dpi=150)
    plt.close()

    # Euclidean distance
    plt.figure(figsize=(6, 4))
    plt.hist(before["euc_hist"], bins=50, alpha=0.5, color="blue", label="Before alignment", density=True)
    plt.hist(after["euc_hist"], bins=50, alpha=0.5, color="orange", label="After alignment", density=True)
    plt.axvline(0.0, color="red", linestyle="--", label="Optimal")
    plt.xlabel("Euclidean distance")
    plt.ylabel("Density")
    plt.title(f"{name} – Euclidean distance comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.lower()}_euclidean_comparison.png", dpi=150)
    plt.close()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room_manifest", type=str, required=True)
    parser.add_argument("--scene_manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--subsample", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifests...")
    room_df = pd.read_csv(args.room_manifest)
    scene_df = pd.read_csv(args.scene_manifest)

    # ---------------------------------------------------------
    # Early subsampling by unique ID before loading embeddings
    # ---------------------------------------------------------
    def subsample_rows(df, n):
        if len(df) > n:
            df = df.sample(n, random_state=42)
        return df

    room_df = subsample_rows(room_df, args.subsample)
    scene_df = subsample_rows(scene_df, args.subsample)

    print(f"Subsampled room_df: {len(room_df)} rows")
    print(f"Subsampled scene_df: {len(scene_df)} rows")

    print("Loading alignment checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    align_pov = AlignmentMLP(512, 512)
    align_graph = AlignmentMLP(384, 512)
    align_pov.load_state_dict(ckpt["align_pov"])
    align_graph.load_state_dict(ckpt["align_graph"])
    align_pov.eval()
    align_graph.eval()

    results = {}

    # ============ 1. RAW POV / GRAPH / SCENE GRAPH ============
    print("\n[1] Analyzing raw embedding diversity...")
    pov_raw = load_embeddings(room_df, "POV_EMBEDDING_PATH", "ROOM_ID")
    graph_raw = load_embeddings(room_df, "ROOM_GRAPH_EMBEDDING_PATH", "ROOM_ID")
    scene_graph_raw = load_embeddings(scene_df, "SCENE_GRAPH_EMBEDDING_PATH", "SCENE_ID")

    pov_embs = [x for v in pov_raw.values() for x in v]
    graph_embs = [x for v in graph_raw.values() for x in v]
    scene_graph_embs = [x for v in scene_graph_raw.values() for x in v]

    pov_before = compute_stats(pov_embs, args.subsample)
    graph_before = compute_stats(graph_embs, args.subsample)
    scene_before = compute_stats(scene_graph_embs, args.subsample)

    # ============ 2. APPLY ALIGNMENT ============
    print("\n[2] Applying alignment projections...")

    pov_aligned = [align_pov(torch.tensor(e).unsqueeze(0)).detach().numpy().flatten() for e in tqdm(pov_embs, desc="Aligning POVs")]
    graph_aligned = [align_graph(torch.tensor(e).unsqueeze(0)).detach().numpy().flatten() for e in tqdm(graph_embs, desc="Aligning Room Graphs")]
    scene_graph_aligned = [align_graph(torch.tensor(e).unsqueeze(0)).detach().numpy().flatten() for e in tqdm(scene_graph_embs, desc="Aligning Scene Graphs")]

    pov_after = compute_stats(pov_aligned, args.subsample)
    graph_after = compute_stats(graph_aligned, args.subsample)
    scene_after = compute_stats(scene_graph_aligned, args.subsample)

    # ============ 3. PLOT COMPARISONS ============
    print("\n[3] Plotting before/after/optimal comparisons...")
    plot_comparison(pov_before, pov_after, "POV", out_dir)
    plot_comparison(graph_before, graph_after, "RoomGraph", out_dir)
    plot_comparison(scene_before, scene_after, "SceneGraph", out_dir)

    results["raw_pov"] = {k: v for k, v in pov_before.items() if not isinstance(v, np.ndarray)}
    results["raw_graph"] = {k: v for k, v in graph_before.items() if not isinstance(v, np.ndarray)}
    results["raw_scenegraph"] = {k: v for k, v in scene_before.items() if not isinstance(v, np.ndarray)}
    results["aligned_pov"] = {k: v for k, v in pov_after.items() if not isinstance(v, np.ndarray)}
    results["aligned_graph"] = {k: v for k, v in graph_after.items() if not isinstance(v, np.ndarray)}
    results["aligned_scenegraph"] = {k: v for k, v in scene_after.items() if not isinstance(v, np.ndarray)}

    # ============ 4. ALIGNMENT WITH LAYOUT ============
    print("\n[4] Measuring alignment correlation with layouts...")

    layout_raw = load_embeddings(room_df, "ROOM_LAYOUT_EMBEDDING_PATH", "ROOM_ID")
    layout_embs = [x.flatten() for v in layout_raw.values() for x in v]
    layout_flat = np.stack(layout_embs)
    layout_flat = layout_flat.reshape(len(layout_flat), -1)

    align_layout = AlignmentMLP(64 * 64 * 4, 512)
    align_layout.eval()
    layout_proj = []
    for e in tqdm(layout_flat, desc="Projecting layouts"):
        with torch.no_grad():
            proj = align_layout(torch.tensor(e).unsqueeze(0)).numpy().flatten()
        layout_proj.append(proj)
    layout_proj = np.stack(layout_proj)

    def mean_cosine(a, b):
        sim = cosine_similarity(a, b)
        diag = np.diag(sim)
        return float(np.mean(diag)), float(np.std(diag))

    pov_mean_cos, pov_std_cos = mean_cosine(np.stack(pov_aligned[:len(layout_proj)]),
                                            layout_proj[:len(pov_aligned)])
    graph_mean_cos, graph_std_cos = mean_cosine(np.stack(graph_aligned[:len(layout_proj)]),
                                                layout_proj[:len(graph_aligned)])

    results["alignment_quality"] = {
        "pov_to_layout_mean_cos": pov_mean_cos,
        "pov_to_layout_std_cos": pov_std_cos,
        "graph_to_layout_mean_cos": graph_mean_cos,
        "graph_to_layout_std_cos": graph_std_cos,
    }

    # ============ 5. SAVE ============
    summary_path = out_dir / "alignment_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved full analysis to: {summary_path}")


if __name__ == "__main__":
    main()
