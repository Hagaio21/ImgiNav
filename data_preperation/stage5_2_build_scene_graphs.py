#!/usr/bin/env python3
"""
stage5_build_and_visualize_scene_graphs.py

Build scene graphs from point clouds and visualize them on scene layouts.
"""

import argparse
import json
import csv
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def load_taxonomy(tax_path: Path):
    """Load room type mappings from taxonomy."""
    tax = json.loads(tax_path.read_text(encoding="utf-8"))
    return tax.get("id2room", {}), tax.get("room2id", {})


def find_scene_pointclouds(root: Path, manifest: Path = None):
    """Find scene-level point cloud files."""
    scenes = []
    if manifest and manifest.exists():
        with open(manifest, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                # Accept your existing scene_list.csv format
                scene_id = row["scene_id"]
                pc_path = Path(row["parquet_file_path"])
                if pc_path.exists():
                    scenes.append((scene_id, pc_path))

    else:
        for pc_path in root.rglob("*_sem_pointcloud.parquet"):
            scene_id = pc_path.stem.replace("_sem_pointcloud", "")
            scenes.append((scene_id, pc_path))
    return scenes


def extract_room_data(pc_path: Path, id2room: dict):
    """Extract room information from scene point cloud."""
    df = pd.read_parquet(pc_path)
    
    if "room_id" not in df.columns:
        print(f"[warn] No room_id column in {pc_path}", flush=True)
        return []
    
    rooms = []
    xyz = df[["x", "y", "z"]].to_numpy()
    floor_label_ids = [4002, 2052]
    has_labels = "label_id" in df.columns
    
    for room_id, group in df.groupby("room_id"):
        room_id_int = int(room_id)
        room_type = id2room.get(str(room_id_int), f"Room_{room_id_int}")
        
        indices = group.index.to_numpy()
        room_xyz = xyz[indices]
        
        if len(room_xyz) < 10:
            continue
        
        centroid_xyz = room_xyz.mean(axis=0)
        centroid_xy = centroid_xyz[:2].tolist()
        floor_centroid_xy = centroid_xy
        
        if has_labels:
            room_labels = df.iloc[indices]["label_id"].to_numpy()
            floor_mask = np.isin(room_labels, floor_label_ids)
            if floor_mask.sum() > 0:
                floor_xyz = room_xyz[floor_mask]
                floor_centroid_xy = floor_xyz.mean(axis=0)[:2].tolist()
        
        rooms.append({
            "room_id": room_id_int,
            "room_type": room_type,
            "centroid_xy": centroid_xy,
            "floor_centroid_xy": floor_centroid_xy,
            "centroid_xyz": centroid_xyz.tolist(),
            "points": room_xyz,
            "bbox": {
                "min": room_xyz.min(axis=0).tolist(),
                "max": room_xyz.max(axis=0).tolist()
            }
        })
    
    return rooms


def check_adjacency_3d(points_a, points_b, threshold=0.3):
    """Check if two rooms are adjacent based on 3D proximity."""
    max_sample = 1000
    if len(points_a) > max_sample:
        points_a = points_a[np.random.choice(len(points_a), max_sample, replace=False)]
    if len(points_b) > max_sample:
        points_b = points_b[np.random.choice(len(points_b), max_sample, replace=False)]
    
    distances = cdist(points_a, points_b, metric='euclidean')
    return distances.min() < threshold


def compute_scene_center(room_centers):
    """Compute center of all room centers."""
    if len(room_centers) == 0:
        return np.array([0.0, 0.0])
    return np.array(room_centers).mean(axis=0)


def angle_from_center(center, point):
    """Calculate angle from center to point."""
    v = point - center
    return np.arctan2(v[1], v[0])


def build_scene_graph(scene_id, pc_path, id2room, dataset_root):
    """Build scene graph from point cloud."""
    print(f"\nProcessing: {scene_id}", flush=True)
    
    rooms = extract_room_data(pc_path, id2room)
    if len(rooms) == 0:
        print(f"[warn] No rooms found", flush=True)
        return None
    
    print(f"  Found {len(rooms)} rooms", flush=True)
    
    scene_center = compute_scene_center([r["floor_centroid_xy"] for r in rooms])
    
    # Build nodes
    nodes = []
    for r in rooms:
        nodes.append({
            "id": f"room_{r['room_id']}",
            "room_id": r["room_id"],
            "room_type": r["room_type"],
            "centroid_xy": r["centroid_xy"],
            "floor_centroid_xy": r["floor_centroid_xy"],
            "centroid_xyz": r["centroid_xyz"],
            "bbox": r["bbox"]
        })
    
    # Build edges
    edges = []
    for i, a in enumerate(rooms):
        for j, b in enumerate(rooms):
            if j <= i:
                continue
            
            is_adjacent = check_adjacency_3d(a["points"], b["points"])
            dist_rel = "adjacent" if is_adjacent else None
            
            ang_a = angle_from_center(scene_center, a["centroid_xy"])
            ang_b = angle_from_center(scene_center, b["centroid_xy"])
            d_ang = np.rad2deg((ang_b - ang_a + np.pi*2) % (np.pi*2))
            
            if d_ang < 45 or d_ang > 315:
                dir_a_to_b, dir_b_to_a = "front_of", "behind"
            elif 45 <= d_ang < 135:
                dir_a_to_b, dir_b_to_a = "right_of", "left_of"
            elif 135 <= d_ang < 225:
                dir_a_to_b, dir_b_to_a = "behind", "front_of"
            else:
                dir_a_to_b, dir_b_to_a = "left_of", "right_of"
            
            edges.append({
                "room_a": f"room_{a['room_id']}",
                "room_b": f"room_{b['room_id']}",
                "distance_relation": dist_rel,
                "direction_relation": dir_a_to_b
            })
            edges.append({
                "room_a": f"room_{b['room_id']}",
                "room_b": f"room_{a['room_id']}",
                "distance_relation": dist_rel,
                "direction_relation": dir_b_to_a
            })
    
    scene_graph = {
        "scene_id": scene_id,
        "scene_center": scene_center.tolist(),
        "nodes": nodes,
        "edges": edges
    }
    
    # Save
    output_json = pc_path.parent / f"{scene_id}_scene_graph.json"
    output_json.write_text(json.dumps(scene_graph, indent=2), encoding="utf-8")
    print(f"  ✔ Saved scene graph", flush=True)
    
    return scene_graph


def visualize_scene_graph(scene_id, dataset_root):
    """Create visualization of scene graph on scene layout."""
    scene_dir = dataset_root / scene_id
    
    # Load files
    scene_graph_path = scene_dir / f"{scene_id}_scene_graph.json"
    scene_info_path = scene_dir / f"{scene_id}_scene_info.json"
    layout_path = scene_dir / "layouts" / f"{scene_id}_scene_layout.png"
    
    if not all([scene_graph_path.exists(), scene_info_path.exists(), layout_path.exists()]):
        print(f"  [skip] Missing files for visualization", flush=True)
        return
    
    scene_graph = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    
    if "origin_world" not in scene_info:
        print(f"  [skip] No coordinate frame in scene_info", flush=True)
        return
    
    # Load coordinate frame
    origin = np.array(scene_info["origin_world"], dtype=np.float64)
    u = np.array(scene_info["u_world"], dtype=np.float64)
    v = np.array(scene_info["v_world"], dtype=np.float64)
    n = np.array(scene_info["n_world"], dtype=np.float64)
    
    # Transform function
    def world_to_uv(xyz):
        R = np.column_stack([u, v, n])
        local = (xyz - origin) @ R
        return local[:, :2]
    
    # Load image
    img = cv2.imread(str(layout_path))
    if img is None:
        print(f"  [skip] Cannot read layout", flush=True)
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    vis = cv2.addWeighted(img, 0.3, np.full_like(img, 255), 0.7, 0)
    
    nodes = scene_graph["nodes"]
    edges = scene_graph["edges"]
    scene_center = np.array(scene_graph["scene_center"], dtype=np.float64)
    
    if len(nodes) == 0:
        return
    
    # Transform node positions
    node_xyz = np.array([[n["floor_centroid_xy"][0], n["floor_centroid_xy"][1], n["centroid_xyz"][2]] 
                         for n in nodes], dtype=np.float64)
    node_uv = world_to_uv(node_xyz)
    
    u_min, u_max = node_uv[:, 0].min(), node_uv[:, 0].max()
    v_min, v_max = node_uv[:, 1].min(), node_uv[:, 1].max()
    
    span = max(u_max - u_min, v_max - v_min, 1e-6)
    margin = 10
    scale = (min(w, h) - 2 * margin) / span
    
    def uv_to_img(uv):
        u_pix = (uv[0] - u_min) * scale + margin
        v_pix = (uv[1] - v_min) * scale + margin
        return (int(np.clip(u_pix, 0, w - 1)), int(np.clip((h - 1) - v_pix, 0, h - 1)))
    
    node_positions = {nodes[i]["id"]: uv_to_img(node_uv[i]) for i in range(len(nodes))}
    
    scene_center_xyz = np.array([scene_center[0], scene_center[1], node_xyz[:, 2].mean()])
    scene_center_img = uv_to_img(world_to_uv(scene_center_xyz.reshape(1, 3))[0])
    
    # Draw edges
    drawn = set()
    for e in edges:
        if e.get("distance_relation") != "adjacent":
            continue
        key = tuple(sorted([e["room_a"], e["room_b"]]))
        if key in drawn:
            continue
        drawn.add(key)
        
        pa = node_positions.get(e["room_a"])
        pb = node_positions.get(e["room_b"])
        if pa and pb:
            cv2.line(vis, pa, pb, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Draw center
    cv2.circle(vis, scene_center_img, 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, scene_center_img, 8, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw nodes
    for node in nodes:
        pos = node_positions[node["id"]]
        
        if 0 <= pos[0] < w and 0 <= pos[1] < h:
            color = tuple(int(c) for c in img[pos[1], pos[0]])
            if color == (255, 255, 255):
                np.random.seed(hash(node["room_type"]) % 2**32)
                color = tuple(np.random.randint(50, 255, 3).tolist())
        else:
            np.random.seed(hash(node["room_type"]) % 2**32)
            color = tuple(np.random.randint(50, 255, 3).tolist())
        
        cv2.circle(vis, pos, 12, color, -1, cv2.LINE_AA)
        cv2.circle(vis, pos, 12, (0, 0, 0), 2, cv2.LINE_AA)
        
        label = node["room_type"]
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = pos[0] - tw // 2, pos[1] + 25
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (tx - 3, ty - th - 3), (tx + tw + 3, ty + baseline + 1), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save
    output_path = scene_dir / "layouts" / f"{scene_id}_scene_graph_vis.png"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  ✔ Saved visualization", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--adjacency_thresh", type=float, default=0.3)
    args = parser.parse_args()
    
    id2room, _ = load_taxonomy(Path(args.taxonomy))
    dataset_root = Path(args.in_dir)
    manifest_path = Path(args.manifest) if args.manifest else None
    
    scenes = find_scene_pointclouds(dataset_root, manifest_path)
    print(f"Found {len(scenes)} scenes\n", flush=True)
    
    for scene_id, pc_path in scenes:
        try:
            build_scene_graph(scene_id, pc_path, id2room, dataset_root)
            visualize_scene_graph(scene_id, dataset_root)
        except Exception as e:
            print(f"[error] {scene_id}: {e}", flush=True)


if __name__ == "__main__":
    main()