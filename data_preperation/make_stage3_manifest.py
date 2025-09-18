#!/usr/bin/env python3
"""
make_stage3_manifest.py — corrected version

Generates a manifest CSV with:
- scene_id
- room_id (blank for scene-level)
- type (scene|room)
- meta_path
- layout_path
"""

import argparse
import csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True,
                    help="Root directory with scenes/<scene_id>")
    ap.add_argument("--out", default="manifest_stage3.csv",
                    help="Output CSV path (default: manifest_stage3.csv)")
    ap.add_argument("--relative", action="store_true",
                    help="Store paths relative to --scenes-root (default: absolute)")
    args = ap.parse_args()

    root = Path(args.scenes_root).expanduser().absolute()
    rows = []

    # --- Scene-level layouts ---
    for layout in sorted(root.glob("*/layouts/seg_layout.png")):
        scene_dir = layout.parent.parent  # …/<scene_id>
        scene_id = scene_dir.name

        meta = next(scene_dir.glob("*_scene_meta.json"), None)

        layout_path = layout.relative_to(root) if args.relative else layout
        meta_path = meta.relative_to(root) if (meta and args.relative) else meta

        rows.append({
            "scene_id": scene_id,
            "room_id": "",
            "type": "scene",
            "meta_path": str(meta_path) if meta else "",
            "layout_path": str(layout_path),
        })

    # --- Room-level layouts ---
    for layout in sorted(root.glob("*/rooms/*/layouts/seg_layout.png")):
        room_dir = layout.parent.parent          # …/<scene_id>/rooms/<room_id>
        scene_id = room_dir.parent.parent.name   # ✅ fix: go two levels up
        room_id = room_dir.name

        meta = next(room_dir.glob(f"{scene_id}_{room_id}_meta.json"), None)
        if meta is None:
            meta = next(room_dir.glob("*_meta.json"), None)

        layout_path = layout.relative_to(root) if args.relative else layout
        meta_path = meta.relative_to(root) if (meta and args.relative) else meta

        rows.append({
            "scene_id": scene_id,
            "room_id": room_id,
            "type": "room",
            "meta_path": str(meta_path) if meta else "",
            "layout_path": str(layout_path),
        })

    if not rows:
        print(f"[warn] No layout images found under {root}")
    else:
        scenes = sum(1 for r in rows if r["type"] == "scene")
        rooms = sum(1 for r in rows if r["type"] == "room")
        print(f"Found {len(rows)} layouts → {scenes} scenes, {rooms} rooms. Writing manifest to {args.out}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "room_id", "type", "meta_path", "layout_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
