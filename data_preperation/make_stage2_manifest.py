#!/usr/bin/env python3
"""
make_stage2_manifest.py

Scan Stage 2 outputs (room splits) and generate a manifest CSV.

Each row = one room with:
- scene_id
- room_id
- room_parquet
- room_meta

Supports optional sharding for HPC jobs.
"""

import argparse
import csv
from pathlib import Path
import re

def shard_rows(rows, nshards: int):
    """Split rows evenly into nshards lists."""
    shards = [[] for _ in range(nshards)]
    for i, row in enumerate(rows):
        shards[i % nshards].append(row)
    return shards

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True,
                    help="Root directory with scenes/<scene_id>/rooms/")
    ap.add_argument("--out", default="manifest_stage2.csv",
                    help="Output CSV path (default: manifest_stage2.csv)")
    ap.add_argument("--relative", action="store_true",
                    help="Store paths relative to --scenes-root (default: absolute)")
    ap.add_argument("--shards", type=int, default=0,
                    help="Number of shards to split into (default: 0 = no sharding)")
    args = ap.parse_args()

    root = Path(args.scenes_root).resolve()
    rows = []

    for parquet in sorted(root.rglob("rooms/*/*.parquet")):
        fname = parquet.name
        m = re.match(r"(.+?)_(\d+)\.parquet$", fname)
        if not m:
            continue
        scene_id, room_id = m.group(1), int(m.group(2))
        meta = parquet.with_name(f"{scene_id}_{room_id}_meta.json")

        if args.relative:
            parquet_path = parquet.relative_to(root)
            meta_path = meta.relative_to(root) if meta.exists() else ""
        else:
            parquet_path = parquet
            meta_path = meta if meta.exists() else ""

        rows.append({
            "scene_id": scene_id,
            "room_id": room_id,
            "room_parquet": str(parquet_path),
            "room_meta": str(meta_path),
        })

    if not rows:
        print(f"[warn] No room parquets found under {root}")
    else:
        print(f"Found {len(rows)} rooms. Writing manifest(s) to {args.out}")

    # always write the main manifest
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "room_id", "room_parquet", "room_meta"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✔ wrote manifest: {args.out} ({len(rows)} rows)")

    # optionally write shard manifests
    if args.shards > 1:
        base = Path(args.out).stem
        ext = Path(args.out).suffix or ".csv"
        out_dir = Path(args.out).parent
        for i, shard in enumerate(shard_rows(rows, args.shards)):
            shard_path = out_dir / f"{base}_shard{i:03d}{ext}"
            with open(shard_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["scene_id", "room_id", "room_parquet", "room_meta"])
                writer.writeheader()
                writer.writerows(shard)
            print(f"✔ wrote shard {i:03d}: {shard_path} ({len(shard)} rows)")

if __name__ == "__main__":
    main()
