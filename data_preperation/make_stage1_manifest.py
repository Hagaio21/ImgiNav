#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage1_manifest.py

Generate a manifest of scene-level inputs (stage 1).
Optionally split the manifest into multiple shards for HPC parallel jobs.

Each row in the manifest has:
    scene_id, parquet_path, npz_path, scene_info_path

Usage:
    python stage1_manifest.py --scenes-root /path/to/scenes --out manifest.csv
    python stage1_manifest.py --scenes-root /path/to/scenes --out manifest.csv --shards 10
"""

import argparse, csv
from pathlib import Path

def collect_scenes(scenes_root: Path):
    """Traverse scenes_root and return list of dicts with scene info."""
    entries = []
    for scene_dir in sorted(scenes_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        parquet = scene_dir / f"{scene_id}_sem_pointcloud.parquet"
        npz     = scene_dir / f"{scene_id}_sem_pointcloud.npz"
        info    = scene_dir / f"{scene_id}_scene_info.json"

        if not parquet.exists():
            print(f"[warn] missing parquet: {parquet}")
            continue

        entries.append({
            "scene_id": scene_id,
            "parquet_path": str(parquet),
            "npz_path": str(npz) if npz.exists() else "",
            "scene_info_path": str(info) if info.exists() else ""
        })
    return entries

def write_manifest(entries, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id","parquet_path","npz_path","scene_info_path"])
        w.writeheader()
        w.writerows(entries)
    print(f"✔ wrote manifest: {out_path} ({len(entries)} rows)")

def shard_entries(entries, nshards: int):
    """Split entries evenly into nshards lists."""
    shards = [[] for _ in range(nshards)]
    for i, e in enumerate(entries):
        shards[i % nshards].append(e)
    return shards

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True, help="Path to scenes/ directory")
    ap.add_argument("--out", required=True, help="Output manifest CSV (base file)")
    ap.add_argument("--shards", type=int, default=0, help="Number of shards to split into (default: 0 = no sharding)")
    args = ap.parse_args()

    scenes_root = Path(args.scenes_root)
    entries = collect_scenes(scenes_root)

    # Always write the full manifest
    out_path = Path(args.out)
    write_manifest(entries, out_path)

    # Optionally write shard manifests
    if args.shards > 1:
        base = out_path.stem  # filename without extension
        ext = out_path.suffix or ".csv"
        for i, shard in enumerate(shard_entries(entries, args.shards)):
            shard_path = out_path.parent / f"{base}_shard{i:03d}{ext}"
            write_manifest(shard, shard_path)

if __name__ == "__main__":
    main()
