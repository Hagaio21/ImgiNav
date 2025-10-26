#!/usr/bin/env python3
"""
build_graphs.py

Constructs scene-level and room-level graphs from parquet point cloud data.

Inputs:
  - semantic_maps.json (must be located in dataset root or parent folder)
  - <scene_id>/<scene_id>_scene_info.json
  - <scene_id>/rooms/<room_id>/<scene_id>_<room_id>.parquet
  - <scene_id>/rooms/<room_id>/<scene_id>_<room_id>_meta.json

Outputs:
  - <scene_id>/<scene_id>_graph.json
  - <scene_id>/rooms/<room_id>/<scene_id>_<room_id>_graph.json
"""

import argparse, sys, json, re, csv
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

# ----------------- helpers -----------------

def find_semantic_maps_json(start: Path) -> Optional[Path]:
    for p in [start, *start.parents]:
        cand = p / "semantic_maps.json"
        if cand.exists():
            return cand
    return None

def load_semantic_maps(maps_path: Path) -> Dict:
    return json.loads(maps_path.read_text(encoding="utf-8"))

def center_and_bbox(df: pd.DataFrame):
    coords = df[["x","y","z"]].to_numpy(dtype=np.float64)
    if coords.shape[0] == 0:
        return [0,0,0], [[0,0,0],[0,0,0]]
    center = coords.mean(axis=0).tolist()
    mins = coords.min(axis=0).tolist()
    maxs = coords.max(axis=0).tolist()
    return center, [mins, maxs]

def bbox_overlap(b1, b2, tol=0.0):
    """Check if 2 axis-aligned bounding boxes overlap with tolerance."""
    for i in range(3):
        if b1[1][i] < b2[0][i] - tol: return False
        if b2[1][i] < b1[0][i] - tol: return False
    return True

def xy_overlap(b1, b2, tol=0.0):
    """Check XY overlap only (ignore Z)."""
    for i in [0,1]:
        if b1[1][i] < b2[0][i] - tol: return False
        if b2[1][i] < b1[0][i] - tol: return False
    return True

def objects_from_parquet(parquet_path: Path, maps: Dict) -> List[Dict]:
    df = pd.read_parquet(parquet_path)
    objs = []
    for oid, g in df.groupby("label_id"):
        center, bbox = center_and_bbox(g)
        label = None
        for k,v in maps.get("label2id", {}).items():
            if v == oid:
                label = k
                break
        objs.append({
            "object_id": f"{parquet_path.stem}_{oid}",
            "label_id": int(oid),
            "label": label or "unknown",
            "center": center,
            "bbox": bbox
        })
    return objs

def build_room_graph(scene_id: str, room_id: str, room_dir: Path, maps: Dict,
                     near_thresh: float, overlap_tol: float, above_gap: float) -> Dict:
    parquet_files = list(room_dir.glob(f"{scene_id}_{room_id}.parquet"))
    if not parquet_files:
        return {}
    parquet_path = parquet_files[0]
    objects = objects_from_parquet(parquet_path, maps)

    edges = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if j <= i: continue
            ca, cb = np.array(a["center"]), np.array(b["center"])
            ba, bb = a["bbox"], b["bbox"]

            # near
            d = np.linalg.norm(ca - cb)
            if d < near_thresh:
                edges.append({
                    "obj_a": a["object_id"],
                    "obj_b": b["object_id"],
                    "relation": "near",
                    "distance": float(d)
                })

            # overlap
            if bbox_overlap(ba, bb, tol=overlap_tol):
                edges.append({
                    "obj_a": a["object_id"],
                    "obj_b": b["object_id"],
                    "relation": "overlap"
                })

            # vertical relations
            if xy_overlap(ba, bb, tol=overlap_tol):
                if ba[0][2] >= bb[1][2] - overlap_tol:
                    dz = ba[0][2] - bb[1][2]
                    if abs(dz) <= above_gap:
                        edges.append({"obj_a": a["object_id"], "obj_b": b["object_id"], "relation": "on_top_of"})
                    elif dz > above_gap:
                        edges.append({"obj_a": a["object_id"], "obj_b": b["object_id"], "relation": "above"})
                if bb[0][2] >= ba[1][2] - overlap_tol:
                    dz = bb[0][2] - ba[1][2]
                    if abs(dz) <= above_gap:
                        edges.append({"obj_a": b["object_id"], "obj_b": a["object_id"], "relation": "on_top_of"})
                    elif dz > above_gap:
                        edges.append({"obj_a": b["object_id"], "obj_b": a["object_id"], "relation": "above"})

            # directional (simple global x/y comparison)
            if abs(ca[0]-cb[0]) > abs(ca[1]-cb[1]):
                if ca[0] < cb[0]:
                    edges.append({"obj_a": a["object_id"], "obj_b": b["object_id"], "relation": "left_of"})
                else:
                    edges.append({"obj_a": a["object_id"], "obj_b": b["object_id"], "relation": "right_of"})
            else:
                if ca[1] < cb[1]:
                    edges.append({"obj_a": a["object_id"], "obj_b": b["object_id"], "relation": "front_of"})
                else:
                    edges.append({"obj_a": a["object_id"], "obj_b": b["object_id"], "relation": "behind"})

    graph = {
        "scene_id": scene_id,
        "room_id": room_id,
        "objects": objects,
        "edges": edges
    }
    outp = room_dir / f"{scene_id}_{room_id}_graph.json"
    outp.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"  ↳ wrote room graph: {outp}")
    return graph

def build_scene_graph(scene_id: str, scene_dir: Path, maps: Dict,
                      adjacent_thresh: float, overlap_tol: float) -> Dict:
    room_graphs = []
    rooms_root = scene_dir / "rooms"
    if not rooms_root.exists():
        return {}

    for room_id_dir in sorted(rooms_root.iterdir()):
        if not room_id_dir.is_dir():
            continue
        room_id = room_id_dir.name
        rg_path = room_id_dir / f"{scene_id}_{room_id}_graph.json"
        if not rg_path.exists():
            continue
        try:
            if rg_path.stat().st_size == 0:
                print(f"[warn] empty room graph file skipped: {rg_path}")
                continue
            rg = json.loads(rg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] failed to load room graph {rg_path}: {e}")
            continue

        if rg and rg.get("objects"):
            all_centers = np.array([o["center"] for o in rg["objects"]])
            center = all_centers.mean(axis=0).tolist()
            # compute room bbox
            mins = np.min([o["bbox"][0] for o in rg["objects"]], axis=0).tolist()
            maxs = np.max([o["bbox"][1] for o in rg["objects"]], axis=0).tolist()
            room_graphs.append({
                "room_id": room_id,
                "center": center,
                "bbox": [mins, maxs]
            })

    edges = []
    for i, a in enumerate(room_graphs):
        for j, b in enumerate(room_graphs):
            if j <= i: continue
            ca, cb = np.array(a["center"]), np.array(b["center"])
            d = np.linalg.norm(ca - cb)

            # adjacent
            if d < adjacent_thresh:
                edges.append({"room_a": a["room_id"], "room_b": b["room_id"], "relation": "adjacent", "distance": float(d)})

            # overlap / connected
            if bbox_overlap(a["bbox"], b["bbox"], tol=overlap_tol):
                edges.append({"room_a": a["room_id"], "room_b": b["room_id"], "relation": "overlap"})
            else:
                # check if touching (faces within tol)
                touching = any(abs(a["bbox"][1][k]-b["bbox"][0][k]) <= overlap_tol or
                               abs(b["bbox"][1][k]-a["bbox"][0][k]) <= overlap_tol for k in range(3))
                if touching:
                    edges.append({"room_a": a["room_id"], "room_b": b["room_id"], "relation": "connected"})

            # directional
            if abs(ca[0]-cb[0]) > abs(ca[1]-cb[1]):
                if ca[0] < cb[0]:
                    edges.append({"room_a": a["room_id"], "room_b": b["room_id"], "relation": "left_of"})
                else:
                    edges.append({"room_a": a["room_id"], "room_b": b["room_id"], "relation": "right_of"})
            else:
                if ca[1] < cb[1]:
                    edges.append({"room_a": a["room_id"], "room_b": b["room_id"], "relation": "front_of"})
                else:
                    edges.append({"room_a": a["room_id"], "room_b": b["room_id"], "relation": "behind"})

    graph = {
        "scene_id": scene_id,
        "rooms": room_graphs,
        "edges": edges
    }
    outp = scene_dir / f"{scene_id}_graph.json"
    outp.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"✔ wrote scene graph: {outp}")
    return graph

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root folder containing scenes/<scene_id>")
    ap.add_argument("--manifest", type=str,
        help="Optional manifest CSV listing files to process (overrides auto discovery)")

    # thresholds
    ap.add_argument("--near-thresh", type=float, default=2.0)
    ap.add_argument("--adjacent-thresh", type=float, default=5.0)
    ap.add_argument("--overlap-tol", type=float, default=0.1)
    ap.add_argument("--above-gap", type=float, default=0.5)

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    maps_path = find_semantic_maps_json(in_dir)
    if maps_path is None:
        print("semantic_maps.json not found.", file=sys.stderr)
        sys.exit(2)
    maps = load_semantic_maps(maps_path)

    scenes = []
    if args.manifest:
        with open(args.manifest, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "scene_id" in row and row["scene_id"]:
                    scenes.append(row["scene_id"])
    else:
        for p in in_dir.rglob("*_scene_info.json"):
            scenes.append(p.stem.replace("_scene_info",""))

    if not scenes:
        print("No scenes found.", file=sys.stderr)
        sys.exit(2)

    for sid in scenes:
        scene_dir = in_dir / sid
        if not scene_dir.exists():
            continue
        print(f"Processing scene {sid} ...")

        # build room graphs
        rooms_root = scene_dir / "rooms"
        if rooms_root.exists():
            for room_id_dir in sorted(rooms_root.iterdir()):
                if not room_id_dir.is_dir():
                    continue
                build_room_graph(
                    sid, room_id_dir.name, room_id_dir, maps,
                    near_thresh=args.near_thresh,
                    overlap_tol=args.overlap_tol,
                    above_gap=args.above_gap
                )

        # build scene graph
        build_scene_graph(
            sid, scene_dir, maps,
            adjacent_thresh=args.adjacent_thresh,
            overlap_tol=args.overlap_tol
        )

if __name__ == "__main__":
    main()
