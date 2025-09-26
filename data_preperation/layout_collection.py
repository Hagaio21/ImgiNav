#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def analyze_layout(img_path: Path, white_vals, gray_vals) -> dict:
    """Analyze one layout file and return row dict."""
    stem = img_path.stem
    parts = stem.split("_")
    scene_id = parts[0]

    if "_scene_" in stem:
        layout_type = "scene"
        room_id = "scene"
    else:
        layout_type = "room"
        room_id = parts[1]

    im = Image.open(img_path).convert("RGB")
    colors = {tuple(rgb) for count, rgb in im.getcolors(maxcolors=1_000_000)}

    is_empty = colors.issubset(white_vals | gray_vals) and len(colors) <= 2

    return {
        "scene_id": scene_id,
        "type": layout_type,
        "room_id": room_id,
        "layout_path": str(img_path.resolve()),
        "is_empty": str(is_empty).lower(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing layout PNGs")
    ap.add_argument("--out", default="layouts.csv", help="Output CSV path")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of parallel workers (default: all cores)")
    args = ap.parse_args()

    root = Path(args.root)

    # Adjust once you confirm actual RGB values
    white_vals = {(240, 240, 240), (255, 255, 255)}
    gray_vals = {(200, 200, 200), (211, 211, 211)}

    # Explicit search patterns
    print("[INFO] Scanning for scene layouts...", flush=True)
    scene_files = list(root.rglob("*/layouts/*_scene_layout.png"))
    print(f"[INFO] Found {len(scene_files)} scene layouts", flush=True)

    print("[INFO] Scanning for room layouts...", flush=True)
    room_files = list(root.rglob("*/rooms/*/layouts/*_room_*_layout.png"))
    print(f"[INFO] Found {len(room_files)} room layouts", flush=True)

    img_files = scene_files + room_files
    total = len(img_files)
    print(f"[INFO] Total files to process: {total}", flush=True)
    print(f"[INFO] Using {args.workers} workers", flush=True)

    rows = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_to_file = {ex.submit(analyze_layout, p, white_vals, gray_vals): p for p in img_files}
        for future in as_completed(future_to_file):
            row = future.result()
            rows.append(row)
            completed += 1
            if completed % 500 == 0 or completed == total:
                print(f"[PROGRESS] {completed}/{total} files ({100*completed/total:.1f}%)", flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "type", "room_id", "layout_path", "is_empty"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
