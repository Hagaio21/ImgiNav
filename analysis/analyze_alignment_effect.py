#!/usr/bin/env python3
"""
analyze_alignment_effect_refactored.py
--------------------------------------
Evaluate embedding diversity before and after alignment
with correct scene-room pairing.

Groups embeddings by (scene_id, room_id), computes intra-group cosine and
Euclidean similarities, and compares pre/post alignment distributions
against a desired reference diversity curve.
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
from scipy.stats import norm

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


def load_embeddings(df, emb_col, group_by_scene_only=False, deduplicate=True):
    grouped = {}
    seen_per_group = {}  # Track unique embeddings per group
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {emb_col}"):
        path = row.get(emb_col)
        if not path or not os.path.exists(path):
            continue
            
        scene_id = row.get("SCENE_ID")
        room_id = row.get("ROOM_ID") if not group_by_scene_only else None
        if scene_id is None:
            continue
            
        key = f"{scene_id}_{room_id}" if room_id is not None else scene_id
        
        # For graph embeddings, deduplicate since they're shared across POVs
        if deduplicate and "GRAPH" in emb_col:
            if key not in seen_per_group:
                seen_per_group[key] = set()
            if path in seen_per_group[key]:
                continue  # Skip duplicate graph embeddings
            seen_per_group[key].add(path)
        
        emb = safe_load_embedding(path)
        if emb is None:
            continue
            
        grouped.setdefault(key, []).append(emb.numpy())

    # Debug output
    print(f"  Loaded {sum(len(v) for v in grouped.values())} embeddings in {len(grouped)} groups")
    if grouped:
        sizes = [len(v) for v in grouped.values()]
        print(f"  Group sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
    return grouped

def compute_stats(grouped_embs, subsample=5000):
    cos_all, euc_all = [], []
    for embs in grouped_embs.values():
        if len(embs) < 2:
            continue  # skip single-embedding groups
        arr = np.stack(embs)
        if len(arr) > subsample:
            idx = np.random.choice(len(arr), subsample, replace=False)
            arr = arr[idx]
        cos = cosine_similarity(arr)
        euc = euclidean_distances(arr)
        triu = np.triu_indices_from(cos, k=1)
        cos_all.extend(cos[triu])
        euc_all.extend(euc[triu])

    if len(cos_all) == 0 or len(euc_all) == 0:
        return {
            "cos_hist": np.array([]),
            "euc_hist": np.array([]),
            "mean_cos": 0.0,
            "std_cos": 0.0,
            "mean_euc": 0.0,
            "std_euc": 0.0,
        }

    cos_all, euc_all = np.array(cos_all), np.array(euc_all)
    return {
        "cos_hist": cos_all,
        "euc_hist": euc_all,
        "mean_cos": float(np.mean(cos_all)),
        "std_cos": float(np.std(cos_all)),
        "mean_euc": float(np.mean(euc_all)),
        "std_euc": float(np.std(euc_all)),
    }


def plot_comparison(before, after, name, out_dir):
    bins_cos = np.linspace(0, 1, 60)
    bins_euc = 60

    # desired diversity reference (empirical, adjustable)
    desired_cos_x = np.linspace(0, 1, 200)
    desired_cos_y = norm.pdf(desired_cos_x, 0.85, 0.05)
    desired_cos_y /= desired_cos_y.max()

    # Cosine similarity
    plt.figure(figsize=(6, 4))
    plt.hist(before["cos_hist"], bins=bins_cos, alpha=0.5, color="blue", label="Before alignment", density=True)
    plt.hist(after["cos_hist"], bins=bins_cos, alpha=0.5, color="orange", label="After alignment", density=True)
    plt.plot(desired_cos_x, desired_cos_y * plt.ylim()[1], "k-", linewidth=1.8, label="Desired diversity")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(f"{name} – Cosine similarity comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.lower()}_cosine_comparison.png", dpi=150)
    plt.close()

    # Desired Euclidean distance
    desired_euc_x = np.linspace(0, 20, 200)
    desired_euc_y = norm.pdf(desired_euc_x, 8, 2)
    desired_euc_y /= desired_euc_y.max()

    # Euclidean distance
    plt.figure(figsize=(6, 4))
    plt.hist(before["euc_hist"], bins=bins_euc, alpha=0.5, color="blue", label="Before alignment", density=True)
    plt.hist(after["euc_hist"], bins=bins_euc, alpha=0.5, color="orange", label="After alignment", density=True)
    plt.plot(desired_euc_x, desired_euc_y * plt.ylim()[1], "k-", linewidth=1.8, label="Desired diversity")
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

    # Subsample
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
    align_graph = AlignmentMLP(384, 384)
    align_pov.load_state_dict(ckpt["align_pov"])
    align_graph.load_state_dict(ckpt["align_graph"])
    align_pov.eval()
    align_graph.eval()

    # ============ 1. LOAD EMBEDDINGS ============ #
    print("\n[1] Loading embeddings grouped by (scene, room)...")
    # Load with deduplication for graph embeddings
    pov_raw = load_embeddings(room_df, "POV_EMBEDDING_PATH", group_by_scene_only=False, deduplicate=False)
    graph_raw = load_embeddings(room_df, "ROOM_GRAPH_EMBEDDING_PATH", group_by_scene_only=False, deduplicate=True)
    scene_graph_raw = load_embeddings(scene_df, "SCENE_GRAPH_EMBEDDING_PATH", group_by_scene_only=True, deduplicate=True)


    # ============ 2. COMPUTE BEFORE ============ #
    print("\n[2] Computing before-alignment diversity stats...")
    pov_before = compute_stats(pov_raw, args.subsample)
    graph_before = compute_stats(graph_raw, args.subsample)
    scene_before = compute_stats(scene_graph_raw, args.subsample)

    # ============ 3. APPLY ALIGNMENT ============ #
    print("\n[3] Applying alignment to each embedding...")
    pov_aligned = {}
    graph_aligned = {}
    scene_graph_aligned = {}

    for key, embs in tqdm(pov_raw.items(), desc="Aligning POVs"):
        pov_aligned[key] = [align_pov(torch.tensor(e).unsqueeze(0)).detach().numpy().flatten() for e in embs]

    for key, embs in tqdm(graph_raw.items(), desc="Aligning Room Graphs"):
        graph_aligned[key] = [align_graph(torch.tensor(e).unsqueeze(0)).detach().numpy().flatten() for e in embs]

    for key, embs in tqdm(scene_graph_raw.items(), desc="Aligning Scene Graphs"):
        scene_graph_aligned[key] = [align_graph(torch.tensor(e).unsqueeze(0)).detach().numpy().flatten() for e in embs]

    pov_after = compute_stats(pov_aligned, args.subsample)
    graph_after = compute_stats(graph_aligned, args.subsample)
    scene_after = compute_stats(scene_graph_aligned, args.subsample)

    # ============ 4. PLOT ============ #
    print("\n[4] Plotting before/after vs desired diversity...")
    plot_comparison(pov_before, pov_after, "POV", out_dir)
    plot_comparison(graph_before, graph_after, "RoomGraph", out_dir)
    plot_comparison(scene_before, scene_after, "SceneGraph", out_dir)

    # ============ 5. SAVE SUMMARY ============ #
    summary = {
        "pov_before": {k: v for k, v in pov_before.items() if not isinstance(v, np.ndarray)},
        "pov_after": {k: v for k, v in pov_after.items() if not isinstance(v, np.ndarray)},
        "graph_before": {k: v for k, v in graph_before.items() if not isinstance(v, np.ndarray)},
        "graph_after": {k: v for k, v in graph_after.items() if not isinstance(v, np.ndarray)},
        "scene_before": {k: v for k, v in scene_before.items() if not isinstance(v, np.ndarray)},
        "scene_after": {k: v for k, v in scene_after.items() if not isinstance(v, np.ndarray)},
    }

    with open(out_dir / "alignment_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
