#!/usr/bin/env python3
"""
make_stage4_manifest.py

Scan Stage 4 outputs (POVs) and generate a manifest CSV.

Each row = one POV file with:
- scene_id
- room_id
- pov_type  (tex / seg / minimap)
- pov_path
"""

import argparse
import csv
from pathlib import Path
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True,
                    help="Root directory with scenes/<scene_id>/rooms/<room_id>/povs/")
    ap.add_argument("--out", default="manifest_stage4.csv",
                    help="Output CSV path (default: manifest_stage4.csv)")
    ap.add_argument("--relative", action="store_true",
                    help="Store paths relative to --scenes-root (default: absolute)")
    args = ap.parse_args()

    root = Path(args.scenes_root).resolve()
    rows = []

    for img in sorted(root.rglob("rooms/*/povs/**/*.png")):
        parts = img.parts
        try:
            scene_id = [p for p in parts if re.match(r"^[0-9a-fA-F\-]{36}$", p)][0]
            room_idx = parts.index("rooms")
            room_id = int(parts[room_idx + 1])
        except Exception:
            continue

        if "tex" in parts:
            pov_type = "tex"
        elif "seg" in parts:
            pov_type = "seg"
        elif img.name == "minimap.png":
            pov_type = "minimap"
        else:
            pov_type = "other"

        pov_path = img.relative_to(root) if args.relative else img
        rows.append({
            "scene_id": scene_id,
            "room_id": room_id,
            "pov_type": pov_type,
            "pov_path": str(pov_path),
        })

    if not rows:
        print(f"[warn] No POV images found under {root}")
    else:
        print(f"Found {len(rows)} POVs. Writing manifest to {args.out}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "room_id", "pov_type", "pov_path"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
