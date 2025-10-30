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


 

def collect_all(data_root: Path, output_manifest: Path):
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    manifest_rows = []
    fieldnames = [
        "sample_id", "sample_type", "scene_id", "room_id",
        "pov_type", "viewpoint", "pov_image", "pov_embedding",
        "graph_text", "graph_embedding", "layout_image", "layout_embedding",
        "is_empty",
    ]

    n_scenes = n_rooms = n_povs = 0
    n_missing_layout = n_missing_graph = 0
    n_valid_scene = n_valid_room = 0
    n_skipped_room = n_skipped_scene = 0

    scene_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {data_root}")

    for scene_dir in tqdm(scene_dirs, desc="Scanning scenes"):
        scene_id = scene_dir.name
        n_scenes += 1
        rooms_root = scene_dir / "rooms"
        layouts_root = scene_dir / "layouts"
        scene_layout_path = layouts_root / f"{scene_id}_scene_layout.png"
        # Try .json first (from build_graphs), fallback to .txt (from create_graph_text_files)
        scene_graph_path = scene_dir / f"{scene_id}_scene_graph.json"
        if not scene_graph_path.exists():
            scene_graph_path = scene_dir / f"{scene_id}_scene_graph.txt"

        # Scene-level sample
        scene_ok = True
        if not scene_layout_path.exists():
            n_missing_layout += 1
            scene_ok = False
        if not scene_graph_path.exists():
            n_missing_graph += 1
            scene_ok = False

        if scene_ok:
            n_valid_scene += 1
            layout_emb = scene_layout_path.with_name(f"{scene_id}_scene_layout_emb.pt")
            manifest_rows.append({
                "sample_id": f"{scene_id}_scene",
                "sample_type": "scene",
                "scene_id": scene_id,
                "room_id": "",
                "pov_type": "",
                "viewpoint": "",
                "pov_image": "",
                "pov_embedding": "",
                "graph_text": str(scene_graph_path.resolve()),
                "graph_embedding": str(scene_graph_path.with_suffix(".pt").resolve())
                    if scene_graph_path.with_suffix(".pt").exists() else "",
                "layout_image": str(scene_layout_path.resolve()),
                "layout_embedding": str(layout_emb.resolve()) if layout_emb.exists() else "",
                "is_empty": 0,
            })
        else:
            n_skipped_scene += 1

        # Room-level samples
        if not rooms_root.exists():
            continue

        for room_dir in sorted(rooms_root.iterdir()):
            if not room_dir.is_dir():
                continue

            room_id = room_dir.name
            try:
                room_id_int = int(room_id)
            except ValueError:
                room_id_int = room_id

            n_rooms += 1
            layout_path = room_dir / "layouts" / f"{scene_id}_{room_id}_room_seg_layout.png"
            # Try .json first (from build_graphs), fallback to .txt (from create_graph_text_files)
            graph_path = room_dir / "layouts" / f"{scene_id}_{room_id}_graph.json"
            if not graph_path.exists():
                graph_path = room_dir / "layouts" / f"{scene_id}_{room_id}_graph.txt"

            if not layout_path.exists() or not graph_path.exists():
                n_skipped_room += 1
                if not layout_path.exists():
                    n_missing_layout += 1
                if not graph_path.exists():
                    n_missing_graph += 1
                continue

            n_valid_room += 1
            layout_emb = layout_path.with_name(f"{scene_id}_{room_id}_layout_emb.pt")
            graph_emb = graph_path.with_suffix(".pt")

            # Find POVs
            povs_found = False
            for pov_type in ["seg", "tex"]:
                pov_dir = room_dir / "povs" / pov_type
                if not pov_dir.exists():
                    continue

                for pov_img in sorted(pov_dir.glob(f"{scene_id}_{room_id}_v*_pov_{pov_type}.png")):
                    povs_found = True
                    n_povs += 1
                    fname = pov_img.stem
                    parts = fname.split("_")
                    view_token = [p for p in parts if p.startswith("v")]
                    viewpoint = view_token[0] if view_token else "v00"
                    sample_id = f"{scene_id}_{room_id}_{pov_type}_{viewpoint}"

                    pov_emb = pov_img.with_suffix(".pt")

                    manifest_rows.append({
                        "sample_id": sample_id,
                        "sample_type": "room",
                        "scene_id": scene_id,
                        "room_id": room_id_int,
                        "pov_type": pov_type,
                        "viewpoint": viewpoint,
                        "pov_image": str(pov_img.resolve()),
                        "pov_embedding": str(pov_emb.resolve()) if pov_emb.exists() else "",
                        "graph_text": str(graph_path.resolve()),
                        "graph_embedding": str(graph_emb.resolve()) if graph_emb.exists() else "",
                        "layout_image": str(layout_path.resolve()),
                        "layout_embedding": str(layout_emb.resolve()) if layout_emb.exists() else "",
                        "is_empty": 0,
                    })
            
            # If no POVs found, still add a room-level entry with empty POV fields
            if not povs_found:
                sample_id = f"{scene_id}_{room_id}_room"
                manifest_rows.append({
                    "sample_id": sample_id,
                    "sample_type": "room",
                    "scene_id": scene_id,
                    "room_id": room_id_int,
                    "pov_type": "",
                    "viewpoint": "",
                    "pov_image": "",
                    "pov_embedding": "",
                    "graph_text": str(graph_path.resolve()),
                    "graph_embedding": str(graph_emb.resolve()) if graph_emb.exists() else "",
                    "layout_image": str(layout_path.resolve()),
                    "layout_embedding": str(layout_emb.resolve()) if layout_emb.exists() else "",
                    "is_empty": 0,
                })

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows, columns=fieldnames).to_csv(output_manifest, index=False)

    print("\n[SUMMARY]")
    print(f"Scenes total: {n_scenes}")
    print(f"Rooms total:  {n_rooms}")
    print(f"POVs total:   {n_povs}")
    print(f"Valid scenes: {n_valid_scene}, Skipped scenes: {n_skipped_scene}")
    print(f"Valid rooms:  {n_valid_room}, Skipped rooms:  {n_skipped_room}")
    print(f"Missing graphs:  {n_missing_graph}")
    print(f"Missing layouts: {n_missing_layout}")
    print(f"Final manifest entries: {len(manifest_rows)}")
    print(f"[INFO] Wrote manifest → {output_manifest}")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True,
                       choices=["all", "graphs", "layouts", "dataset", "povs"],
                       help="Collection type")
    
    # Common arguments
    parser.add_argument("--root", help="Dataset root directory (for graphs, layouts, dataset)")
    parser.add_argument("--data_root", help="Dataset root directory (for 'all' type)")
    parser.add_argument("--output", help="Output manifest CSV path")
    parser.add_argument("--out", help="Output directory (for 'dataset' type)")
    
    # Type-specific arguments
    parser.add_argument("--layouts", help="Path to layouts.csv (for dataset type, provides is_empty info)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                       help="Number of parallel workers (for layouts type)")
    
    args = parser.parse_args()
    
    if args.type == "all":
        if not args.data_root or not args.output:
            parser.error("--type all requires --data_root and --output")
        collect_all(Path(args.data_root), Path(args.output))
    
    elif args.type == "graphs":
        if not args.root or not args.output:
            parser.error("--type graphs requires --root and --output")
        collect_graphs(Path(args.root), Path(args.output))
    
    elif args.type == "layouts":
        if not args.root or not args.output:
            parser.error("--type layouts requires --root and --output")
        collect_layouts(Path(args.root), Path(args.output), args.workers)
    
    elif args.type == "dataset":
        if not args.root or not args.out or not args.layouts:
            parser.error("--type dataset requires --root, --out, and --layouts")
        collect_dataset(Path(args.root), Path(args.out), Path(args.layouts))
    
    elif args.type == "povs":
        if not args.root or not args.output or not args.layouts:
            parser.error("--type povs requires --root, --output, and --layouts")
        empty_map = load_empty_map(Path(args.layouts))
        collect_povs(Path(args.root), Path(args.output), empty_map)


if __name__ == "__main__":
    main()

