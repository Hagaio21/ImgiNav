#!/usr/bin/env python3
"""
make_graph_manifest.py

Scan graph outputs from build_graphs.py and generate a manifest CSV.

Each row = one graph file with:
- scene_id
- room_id (blank for scene-level)
- type (scene|room)
- graph_path
"""

import argparse
import csv
from pathlib import Path
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True,
                    help="Root directory with scenes/<scene_id>")
    ap.add_argument("--out", default="manifest_graphs.csv",
                    help="Output CSV path (default: manifest_graphs.csv)")
    ap.add_argument("--relative", action="store_true",
                    help="Store paths relative to --scenes-root (default: absolute)")
    args = ap.parse_args()

    root = Path(args.scenes_root).resolve()
    rows = []

    # --- FIXED: Scene-level graphs (only in scene root, not inside rooms/*) ---
    for g in sorted(root.glob("*/*_graph.json")):  # only one level under scene
        if "rooms" in g.parts:  # skip anything inside rooms/
            continue
        parts = g.parts
        try:
            scene_id = [p for p in parts if re.match(r"^[0-9a-fA-F\\-]{36}$", p)][0]
        except Exception:
            continue
        graph_path = g.relative_to(root) if args.relative else g
        rows.append({
            "scene_id": scene_id,
            "room_id": "",
            "type": "scene",
            "graph_path": str(graph_path),
        })

    # Room-level graphs
    for g in sorted(root.rglob("rooms/*/*_graph.json")):
        parts = g.parts
        try:
            scene_id = [p for p in parts if re.match(r"^[0-9a-fA-F\\-]{36}$", p)][0]
            room_idx = parts.index("rooms")
            room_id = parts[room_idx+1]
        except Exception:
            continue
        graph_path = g.relative_to(root) if args.relative else g
        rows.append({
            "scene_id": scene_id,
            "room_id": room_id,
            "type": "room",
            "graph_path": str(graph_path),
        })

    if not rows:
        print(f"[warn] No graphs found under {root}")
    else:
        print(f"Found {len(rows)} graphs (scene+room). Writing manifest to {args.out}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id","room_id","type","graph_path"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
