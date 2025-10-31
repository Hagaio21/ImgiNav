#!/usr/bin/env python3

import argparse
import csv
import os
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import re


def collect_all(data_root: Path, output_csv: Path):
    """
    Recursively scan a dataset root and collect scene, room, and POV entries
    into a unified manifest CSV. Works for both flat and nested scene structures.
    """
    import csv
    from tqdm import tqdm

    manifest_rows = []

    scene_dirs = [p for p in data_root.iterdir() if p.is_dir()]

    print(f"[INFO] Scanning {len(scene_dirs)} scenes under {data_root}")
    for scene_path in tqdm(scene_dirs, desc="Scenes"):
        scene_id = scene_path.name

        # ----------- Scene-level -----------
        scene_graph = next(scene_path.glob(f"{scene_id}_scene_graph.json"), None)
        layout_img = next(scene_path.glob(f"layouts/{scene_id}_scene_layout.png"), None)

        manifest_rows.append({
            "sample_id": f"{scene_id}_scene",
            "sample_type": "scene",
            "scene_id": scene_id,
            "room_id": "0000",
            "pov_type": "",
            "viewpoint": "",
            "pov_image": "",
            "pov_embedding": "",
            "graph_text": str(scene_graph) if scene_graph and scene_graph.exists() else "",
            "graph_embedding": "",
            "layout_image": str(layout_img) if layout_img and layout_img.exists() else "",
            "layout_embedding": "",
            "is_empty": 0,
        })

        # ----------- Room-level -----------
        rooms_root = scene_path / "rooms"
        if not rooms_root.exists():
            continue

        for room_dir in sorted(p for p in rooms_root.iterdir() if p.is_dir()):
            parquet_file = next(room_dir.glob(f"{scene_id}_*.parquet"), None)
            if parquet_file:
                # extract room id from filename like <scene_id>_<room>.parquet
                match = re.search(r"_(\d+)\.parquet$", parquet_file.name)
                room_id = match.group(1).zfill(4) if match else "0000"
            else:
                # fallback to folder name if no parquet
                match = re.search(r"\d+", room_dir.name)
                room_id = match.group(0).zfill(4) if match else "0000"

            layout_img = None
            layout_patterns = [
                f"layouts/{scene_id}_{room_id}_*layout.png",
                f"**/layouts/{scene_id}_{room_id}_*layout.png",
                f"**/{scene_id}_{room_id}_room_seg_layout.png",
                f"**/{scene_id}_{room_id}_room_layout.png",
                f"**/{scene_id}_{room_id}_layout.png",
                f"rooms/{room_id}/layouts/{scene_id}_sem_room_seg_layout.png",
                f"rooms/{room_id}/layouts/*_sem_room_seg_layout.png"
            ]
            for pat in layout_patterns:
                found = next(room_dir.glob(pat), None)
                if found and "graph_vis" not in found.name.lower():
                    layout_img = found
                    break
            graph_json = next(room_dir.glob(f"**/{scene_id}_{room_id}_graph.json"), None)
            graph_pt = next(room_dir.glob(f"**/{scene_id}_{room_id}_graph.pt"), None)
            layout_emb = next(room_dir.glob(f"**/{scene_id}_{room_id}_layout_emb.pt"), None)

            manifest_rows.append({
                "sample_id": f"{scene_id}_{room_id}",
                "sample_type": "room",
                "scene_id": scene_id,
                "room_id": room_id,
                "pov_type": "",
                "viewpoint": "",
                "pov_image": "",
                "pov_embedding": "",
                "graph_text": str(graph_json) if graph_json and graph_json.exists() else "",
                "graph_embedding": str(graph_pt) if graph_pt and graph_pt.exists() else "",
                "layout_image": str(layout_img) if layout_img and layout_img.exists() else "",
                "layout_embedding": str(layout_emb) if layout_emb and layout_emb.exists() else "",
                "is_empty": 0,
            })

            # ----------- POV-level -----------
            pov_root = room_dir / "povs"
            if not pov_root.exists():
                continue

            for tex_path in sorted(pov_root.glob("**/*_pov_tex.png")):
                base = tex_path.stem.replace("_pov_tex", "")
                match = re.search(r"_(\d+)_v(\d+)", base)
                if not match:
                    continue
                room_id, view_id = match.groups()
                view_id = f"v{view_id.zfill(2)}"

                seg_img = next(pov_root.glob(f"**/{scene_id}_{room_id}_{view_id}_pov_seg.png"), None)
                tex_emb = next(pov_root.glob(f"**/{scene_id}_{room_id}_{view_id}_pov_tex.pt"), None)
                seg_emb = next(pov_root.glob(f"**/{scene_id}_{room_id}_{view_id}_pov_seg.pt"), None)

                manifest_rows.append({
                    "sample_id": f"{scene_id}_{room_id}_{view_id}",
                    "sample_type": "pov",
                    "scene_id": scene_id,
                    "room_id": room_id,
                    "pov_type": "tex",
                    "viewpoint": view_id,
                    "pov_image": str(tex_path),
                    "pov_embedding": str(tex_emb) if tex_emb and tex_emb.exists() else "",
                    "graph_text": "",
                    "graph_embedding": str(seg_emb) if seg_emb and seg_emb.exists() else "",
                    "layout_image": str(seg_img) if seg_img and seg_img.exists() else "",
                    "layout_embedding": "",
                    "is_empty": 0,
                })

    # ----------- Write CSV -----------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "sample_type", "scene_id", "room_id",
            "pov_type", "viewpoint",
            "pov_image", "pov_embedding",
            "graph_text", "graph_embedding",
            "layout_image", "layout_embedding",
            "is_empty",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[INFO] Wrote manifest → {output_csv} ({len(manifest_rows)} entries)")




def collect_graphs(root_dir: Path, output_path: Path):
    graphs = []
    
    print("Scanning for graph files...")
    
    for graph_path in root_dir.rglob("*_graph.json"):
        filename = graph_path.stem
        
        if filename.endswith("_scene_graph"):
            scene_id = filename.replace("_scene_graph", "")
            graph_type = "scene"
            room_id = "scene"
        else:
            parts = filename.replace("_graph", "").split("_")
            if len(parts) >= 2:
                room_id = parts[-1]
                scene_id = "_".join(parts[:-1])
                graph_type = "room"
            else:
                continue
        
        if graph_type == "scene":
            layout_filename = f"{scene_id}_scene_layout.png"
        else:
            layout_filename = f"{scene_id}_{room_id}_room_seg_layout.png"
        
        layout_path = graph_path.parent / layout_filename
        
        graphs.append({
            'scene_id': scene_id,
            'type': graph_type,
            'room_id': room_id,
            'layout_path': str(layout_path) if layout_path.exists() else '',
            'graph_path': str(graph_path),
            'is_empty': 'false'
        })
    
    if not graphs:
        print("[error] No graphs found")
        return
    
    scene_count = sum(1 for g in graphs if g['type'] == 'scene')
    room_count = sum(1 for g in graphs if g['type'] == 'room')
    
    from common.utils import safe_mkdir
    safe_mkdir(output_path.parent)
    
    from common.file_io import create_manifest
    fieldnames = ['scene_id', 'type', 'room_id', 'layout_path', 'graph_path', 'is_empty']
    create_manifest(graphs, output_path, fieldnames)
    
    print(f"\nTotal graphs: {len(graphs)}")
    print(f"Scene graphs: {scene_count}")
    print(f"Room graphs:  {room_count}")
    print(f"\nManifest: {output_path}")


def analyze_layout(img_path: Path, white_vals: set, gray_vals: set) -> dict:
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


def collect_layouts(root: Path, output_path: Path, workers: int = None):
    if workers is None:
        workers = multiprocessing.cpu_count()

    white_vals = {(240, 240, 240), (255, 255, 255)}
    gray_vals = {(200, 200, 200), (211, 211, 211)}

    print("[INFO] Scanning for scene layouts...", flush=True)
    scene_files = list(root.rglob("*/layouts/*_scene_layout.png"))
    print(f"[INFO] Found {len(scene_files)} scene layouts", flush=True)

    print("[INFO] Scanning for room layouts...", flush=True)
    room_files = list(root.rglob("*/rooms/*/layouts/*_room_*_layout.png"))
    print(f"[INFO] Found {len(room_files)} room layouts", flush=True)

    img_files = scene_files + room_files
    total = len(img_files)
    print(f"[INFO] Total files to process: {total}", flush=True)
    print(f"[INFO] Using {workers} workers", flush=True)

    rows = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_file = {ex.submit(analyze_layout, p, white_vals, gray_vals): p for p in img_files}
        for future in as_completed(future_to_file):
            row = future.result()
            rows.append(row)
            completed += 1
            if completed % 500 == 0 or completed == total:
                print(f"[PROGRESS] {completed}/{total} files ({100*completed/total:.1f}%)", flush=True)

    from common.file_io import create_manifest
    fieldnames = ["scene_id", "type", "room_id", "layout_path", "is_empty"]
    create_manifest(rows, output_path, fieldnames)

    print(f"[INFO] Wrote {len(rows)} rows to {output_path}", flush=True)

def load_empty_map(layouts_csv: Path) -> Dict[Tuple[str, str], int]:
    empty_map = {}
    if not layouts_csv.exists():
        return empty_map
    from common.file_io import read_manifest
    rows = read_manifest(layouts_csv)
    for row in rows:
            sid = row.get("scene_id", "")
            rid = row.get("room_id", "")
            try:
                empty = int(row.get("is_empty", "0"))
            except ValueError:
                empty = 0
            empty_map[(sid, rid)] = empty
    return empty_map


def collect_povs(root: Path, out_csv: Path, empty_map: Dict[Tuple[str, str], int]):
    rows = []
    for scene_id in os.listdir(root):
        scene_dir = Path(root) / scene_id
        if not scene_dir.is_dir():
            continue
        rooms_dir = scene_dir / "rooms"
        if not rooms_dir.exists():
            continue

        for room_id in os.listdir(rooms_dir):
            povs_dir = rooms_dir / room_id / "povs"
            if not povs_dir.exists():
                continue
            for pov_type in ("seg", "tex"):
                tdir = povs_dir / pov_type
                if not tdir.exists():
                    continue
                for f in tdir.glob("*.png"):
                    is_empty = empty_map.get((scene_id, room_id), 0)
                    rows.append([scene_id, room_id, pov_type, str(f.resolve()), is_empty])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "room_id", "type", "pov_path", "is_empty"])
        writer.writerows(rows)


def collect_dataset_files(root: Path, out_csv: Path):
    rows = []

    for scene_id in os.listdir(root):
        scene_dir = Path(root) / scene_id
        if not scene_dir.is_dir():
            continue

        # Scene-level files
        for f in scene_dir.iterdir():
            if not f.is_file():
                continue

            if f.name.endswith("_scene_info.json"):
                cat = "scene_info"
            elif f.name.endswith("_sem_pointcloud.parquet"):
                cat = "scene_parquet"
            elif f.name.endswith("_scene_layout.png"):
                cat = "scene_layout"
            else:
                cat = "other"
            rows.append([scene_id, "", cat, str(f.resolve())])

        # Rooms
        rooms_dir = scene_dir / "rooms"
        if not rooms_dir.exists():
            continue

        for room_id in os.listdir(rooms_dir):
            room_dir = rooms_dir / room_id
            if not room_dir.is_dir():
                continue

            for f in room_dir.iterdir():
                if not f.is_file():
                    continue
                if f.name.endswith(".parquet"):
                    cat = "room_parquet"
                elif f.name.endswith("_meta.json"):
                    cat = "room_meta"
                else:
                    cat = "other"
                rows.append([scene_id, room_id, cat, str(f.resolve())])

            # Room layouts
            layouts_dir = room_dir / "layouts"
            if layouts_dir.exists():
                for f in layouts_dir.iterdir():
                    if f.name.endswith("_room_seg_layout.png"):
                        cat = "room_layout_seg"
                    else:
                        cat = "other"
                    rows.append([scene_id, room_id, cat, str(f.resolve())])

            # POVs
            povs_dir = room_dir / "povs"
            if povs_dir.exists():
                for f in povs_dir.iterdir():
                    if not f.is_file():
                        continue
                    if f.name.endswith("_pov_meta.json"):
                        cat = "pov_meta"
                    elif f.name.endswith("_minimap.png"):
                        cat = "pov_minimap"
                    else:
                        cat = "other"
                    rows.append([scene_id, room_id, cat, str(f.resolve())])

                for pov_type in ("seg", "tex"):
                    tdir = povs_dir / pov_type
                    if not tdir.exists():
                        continue
                    for f in tdir.iterdir():
                        if not f.name.endswith(".png"):
                            cat = "other"
                        else:
                            cat = "pov_seg" if pov_type == "seg" else "pov_tex"
                        rows.append([scene_id, room_id, cat, str(f.resolve())])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "room_id", "category", "file_path"])
        writer.writerows(rows)


def collect_dataset(root: Path, out_dir: Path, layouts_csv: Path):
    os.makedirs(out_dir, exist_ok=True)
    empty_map = load_empty_map(layouts_csv)
    
    povs_csv = out_dir / "povs.csv"
    data_csv = out_dir / "data.csv"
    
    collect_povs(root, povs_csv, empty_map)
    collect_dataset_files(root, data_csv)


def main():
    import time
    parser = argparse.ArgumentParser(
        description="Dataset collection utility: builds manifests for scenes, rooms, layouts, graphs, or POVs."
    )
    parser.add_argument(
        "--type", required=True,
        choices=["all", "graphs", "layouts", "dataset", "povs"],
        help="Collection type to run. "
             "'all' = full manifest, 'graphs' = collect graph files, "
             "'layouts' = collect layout images, 'dataset' = collect dataset file structure, 'povs' = collect POV metadata."
    )

    # Common arguments
    parser.add_argument("--root", help="Dataset root directory (for graphs, layouts, dataset, or povs)")
    parser.add_argument("--data_root", help="Dataset root directory (for 'all' mode only)")
    parser.add_argument("--output", help="Output manifest CSV path")
    parser.add_argument("--out", help="Output directory (for 'dataset' type)")

    # Type-specific
    parser.add_argument("--layouts", help="Path to layouts.csv (for dataset or povs mode)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                       help="Number of parallel workers (for 'layouts' type)")

    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print(f"[INFO] Starting collection")
    print(f"[INFO] Mode       : {args.type}")
    print(f"[INFO] Data root  : {args.data_root or args.root}")
    print(f"[INFO] Output     : {args.output or args.out}")
    print("=" * 70)

    try:
        if args.type == "all":
            if not args.data_root or not args.output:
                parser.error("--type all requires --data_root and --output")
            print("[STEP] Collecting complete dataset manifest...")
            collect_all(Path(args.data_root), Path(args.output))

        elif args.type == "graphs":
            if not args.root or not args.output:
                parser.error("--type graphs requires --root and --output")
            print("[STEP] Collecting graph metadata...")
            collect_graphs(Path(args.root), Path(args.output))

        elif args.type == "layouts":
            if not args.root or not args.output:
                parser.error("--type layouts requires --root and --output")
            print(f"[STEP] Scanning layouts with {args.workers} workers...")
            collect_layouts(Path(args.root), Path(args.output), args.workers)

        elif args.type == "dataset":
            if not args.root or not args.out or not args.layouts:
                parser.error("--type dataset requires --root, --out, and --layouts")
            print("[STEP] Collecting dataset structure and layout metadata...")
            collect_dataset(Path(args.root), Path(args.out), Path(args.layouts))

        elif args.type == "povs":
            if not args.root or not args.output or not args.layouts:
                parser.error("--type povs requires --root, --output, and --layouts")
            print("[STEP] Collecting POV metadata and linking layouts...")
            empty_map = load_empty_map(Path(args.layouts))
            collect_povs(Path(args.root), Path(args.output), empty_map)

        else:
            parser.error(f"Unknown collection type: {args.type}")

    except Exception as e:
        import traceback
        print("\n[ERROR] Collection failed:")
        print(traceback.format_exc())
        raise e

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[DONE] Completed collection type '{args.type}' in {elapsed:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()


