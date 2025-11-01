#!/usr/bin/env python3

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from utils.geometry_utils import angle_from_center, compute_directional_relations
from utils.text_utils import graph2text
from common.utils import write_json


# =============================================================================
# File Discovery - Imported from utils.file_discovery
# =============================================================================
from utils.file_discovery import find_layouts, find_scene_pointclouds
from common.taxonomy import Taxonomy


# =============================================================================
# Room Graph Building (from layouts)
# =============================================================================

def compute_room_center(img: np.ndarray) -> np.ndarray:
    mask_room = ~(img == 255).all(axis=2)
    ys, xs = np.nonzero(mask_room)
    if len(xs) == 0:
        return np.array([img.shape[1] / 2, img.shape[0] / 2])
    return np.array([xs.mean(), ys.mean()])


def extract_color_regions(img: np.ndarray, color_to_label: Dict) -> Dict:
    h, w = img.shape[:2]
    color_masks = defaultdict(list)
    
    # Get unique colors in the image
    unique_colors = np.unique(img.reshape(-1, 3), axis=0)
    
    print(f"  Found {len(unique_colors)} unique colors in image", flush=True)
    
    # For each unique color, check if it matches a taxonomy color
    for color in unique_colors:
        color_t = tuple(int(c) for c in color)
        
        # Skip white background
        if color_t == (255, 255, 255):
            continue
        
        # Check if this exact color is in taxonomy
        if color_t in color_to_label:
            # Find all pixels with this color
            mask = np.all(img == color, axis=-1)
            ys, xs = np.nonzero(mask)
            points = list(zip(xs, ys))
            color_masks[color_t] = points
            print(f"  Matched color {color_t} ({color_to_label[color_t]['label']}): {len(points)} pixels", flush=True)
    
    return color_masks


def find_color_clusters(points: List, linkage_thresh: int = 15) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    pts = np.array(points)
    if len(pts) == 1:
        return [pts]
    
    # DBSCAN parameters
    clustering = DBSCAN(eps=linkage_thresh, min_samples=5).fit(pts)
    labels = clustering.labels_
    
    # Group points by cluster (-1 is noise, which we ignore)
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_mask = labels == label
        clusters.append(pts[cluster_mask])
    
    return clusters


def visualize_room_graph(img: np.ndarray, room_center: np.ndarray, 
                        nodes: List[Dict], edges: List[Dict], out_path: Path):
    h, w = img.shape[:2]
    
    # Create visualization on white background
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Draw the original image with transparency
    alpha = 0.3
    vis = cv2.addWeighted(img, alpha, vis, 1 - alpha, 0)
    
    # Define color scheme
    EDGE_COLOR = (0, 0, 0)
    NODE_OUTLINE = (0, 0, 0)
    TEXT_COLOR = (0, 0, 0)
    TEXT_BG_COLOR = (255, 255, 255)
    
    # Draw edges first (behind nodes)
    for e in edges:
        if e["distance_relation"] is None:
            continue
        
        a = next((n for n in nodes if n["id"] == e["obj_a"]), None)
        b = next((n for n in nodes if n["id"] == e["obj_b"]), None)
        if not a or not b:
            continue
        
        ca, cb = np.array(a["center"], int), np.array(b["center"], int)
        cv2.line(vis, tuple(ca), tuple(cb), EDGE_COLOR, 1, cv2.LINE_AA)
        
        # Label format: (distance_relation, direction_relation)
        mid = ((ca + cb) / 2).astype(int)
        text = f"({e['distance_relation']}, {e['direction_relation']})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(vis, (mid[0] - 2, mid[1] - th - 2), 
                     (mid[0] + tw + 2, mid[1] + 2), TEXT_BG_COLOR, -1)
        cv2.putText(vis, text, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.3, TEXT_COLOR, 1, cv2.LINE_AA)
    
    # Draw room center
    rcx, rcy = map(int, room_center)
    cv2.circle(vis, (rcx, rcy), 12, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, (rcx, rcy), 12, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw nodes
    for n in nodes:
        cx, cy = map(int, n["center"])
        node_color = n.get("color", (128, 128, 128))
        
        cv2.circle(vis, (cx, cy), 8, node_color, -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 8, NODE_OUTLINE, 2, cv2.LINE_AA)
        
        # Add label below node
        label = n["label"]
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_x = cx - tw // 2
        text_y = cy + 18
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (text_x - 2, text_y - th - 2), 
                     (text_x + tw + 2, text_y + baseline), TEXT_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    
    # Convert back to BGR for saving with OpenCV
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), vis_bgr)
    print(f"  ↳ saved visualization {out_path}", flush=True)


def build_room_graph_from_layout(scene_id: str, room_id: str, layout_path: Path, 
                                 color_to_label: Dict, taxonomy=None):
    img = cv2.imread(str(layout_path))
    if img is None:
        print(f"[warn] cannot read {layout_path}", flush=True)
        return None
    
    # Convert BGR to RGB (OpenCV loads as BGR, taxonomy is RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    room_center = compute_room_center(img)
    near_thresh = 0.05 * w
    by_thresh = 0.02 * w
    
    # Extract color regions
    color_regions = extract_color_regions(img, color_to_label)
    print(f"  Found {len(color_regions)} distinct taxonomy colors", flush=True)
    
    nodes = []
    node_id = 0
    node_clusters = {}
    
    for tax_color, points in color_regions.items():
        label_info = color_to_label[tax_color]
        clusters = find_color_clusters(points, linkage_thresh=15)
        print(f"  Color {tax_color} ({label_info['label']}): {len(clusters)} clusters from {len(points)} pixels", flush=True)
        
        for ci, cluster in enumerate(clusters):
            centroid = cluster.mean(axis=0)
            node_key = f"{label_info['label']}_{node_id}"
            nodes.append({
                "id": node_key,
                "label": label_info["label"],
                "label_id": label_info["label_id"],
                "center": centroid.tolist(),
                "color": tax_color
            })
            node_clusters[node_key] = cluster
            node_id += 1
    
    print(f"  Total nodes created: {len(nodes)}", flush=True)
    
    # Build edges
    edges = []
    for i, a in enumerate(nodes):
        ca = np.array(a["center"])
        la = a["label"].lower()
        cluster_a = node_clusters[a["id"]]
        
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            cb = np.array(b["center"])
            lb = b["label"].lower()
            cluster_b = node_clusters[b["id"]]
            
            # Compute minimum distance between clusters
            distances = cdist(cluster_a, cluster_b, metric='euclidean')
            min_cluster_distance = distances.min()
            del distances
            
            # Determine proximity relation
            prox_a_to_b = None
            prox_b_to_a = None
            if "structure" in (la, lb):
                if min_cluster_distance < by_thresh:
                    prox_a_to_b = prox_b_to_a = "by"
            else:
                if min_cluster_distance < near_thresh:
                    prox_a_to_b = prox_b_to_a = "near"
            
            # Directional relations
            ang_a = angle_from_center(room_center, ca)
            ang_b = angle_from_center(room_center, cb)
            dir_a_to_b, dir_b_to_a = compute_directional_relations(ang_a, ang_b)
            
            edges.append({
                "obj_a": a["id"], 
                "obj_b": b["id"], 
                "distance_relation": prox_a_to_b,
                "direction_relation": dir_a_to_b
            })
            edges.append({
                "obj_a": b["id"], 
                "obj_b": a["id"], 
                "distance_relation": prox_b_to_a,
                "direction_relation": dir_b_to_a
            })
    
    graph = {
        "scene_id": scene_id,
        "room_id": room_id,
        "room_center": room_center.tolist(),
        "nodes": nodes,
        "edges": edges
    }
    
    out_json = layout_path.with_name(f"{scene_id}_{room_id}_graph.json")
    out_txt = layout_path.with_name(f"{scene_id}_{room_id}_graph.txt")
    out_vis = layout_path.with_name(f"{scene_id}_{room_id}_graph_vis.png")
    write_json(graph, out_json)
    
    # Generate text version if taxonomy is available
    if taxonomy is not None:
        try:
            text = graph2text(out_json, taxonomy)
            if text:
                out_txt.write_text(text, encoding="utf-8")
                print(f"✔ wrote {out_txt}", flush=True)
        except Exception as e:
            print(f"  [warn] Failed to generate text for {out_json}: {e}", flush=True)
    
    visualize_room_graph(img, room_center, nodes, edges, out_vis)
    print(f"✔ wrote {out_json}", flush=True)
    
    # Clean up
    del img, color_regions, node_clusters, nodes, edges, graph
    gc.collect()
    
    return None


# =============================================================================
# Scene Graph Building (from point clouds)
# =============================================================================

def extract_room_data(pc_path: Path, id2room: Dict) -> List[Dict]:
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


def check_adjacency_3d(points_a: np.ndarray, points_b: np.ndarray, threshold: float = 0.3) -> bool:
    max_sample = 1000
    if len(points_a) > max_sample:
        points_a = points_a[np.random.choice(len(points_a), max_sample, replace=False)]
    if len(points_b) > max_sample:
        points_b = points_b[np.random.choice(len(points_b), max_sample, replace=False)]
    
    distances = cdist(points_a, points_b, metric='euclidean')
    return distances.min() < threshold


def compute_scene_center(room_centers: List) -> np.ndarray:
    if len(room_centers) == 0:
        return np.array([0.0, 0.0])
    return np.array(room_centers).mean(axis=0)


def build_scene_graph_from_pointcloud(scene_id: str, pc_path: Path, id2room: Dict, 
                                     dataset_root: Path, adjacency_thresh: float = 0.3,
                                     taxonomy=None):
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
            
            is_adjacent = check_adjacency_3d(a["points"], b["points"], adjacency_thresh)
            dist_rel = "adjacent" if is_adjacent else None
            
            ang_a = angle_from_center(scene_center, a["centroid_xy"])
            ang_b = angle_from_center(scene_center, b["centroid_xy"])
            dir_a_to_b, dir_b_to_a = compute_directional_relations(ang_a, ang_b)
            
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
    output_txt = pc_path.parent / f"{scene_id}_scene_graph.txt"
    write_json(scene_graph, output_json)
    
    # Generate text version if taxonomy is available
    if taxonomy is not None:
        try:
            text = graph2text(output_json, taxonomy)
            if text:
                output_txt.write_text(text, encoding="utf-8")
                print(f"  ✔ Saved scene graph text", flush=True)
        except Exception as e:
            print(f"  [warn] Failed to generate text for {output_json}: {e}", flush=True)
    
    print(f"  ✔ Saved scene graph", flush=True)
    
    return scene_graph


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified graph building for room and scene graphs"
    )
    
    parser.add_argument(
        "--type",
        required=True,
        choices=["room", "scene"],
        help="Graph type: 'room' builds from layouts, 'scene' builds from point clouds"
    )
    parser.add_argument(
        "--in_dir",
        help="Dataset root (required for room type)"
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Path to taxonomy.json"
    )
    parser.add_argument(
        "--manifest",
        help="Optional manifest CSV"
    )
    parser.add_argument(
        "--layout",
        help="Optional single layout path for testing (room type only)"
    )
    parser.add_argument(
        "--adjacency_thresh",
        type=float,
        default=0.3,
        help="Adjacency threshold for scene graphs (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    if args.type == "room":
        tax = Taxonomy(Path(args.taxonomy))
        color_to_label = tax.get_color_to_label_dict()
        print(f"Loaded {len(color_to_label)} taxonomy colors (super categories + wall)", flush=True)
        
        if args.layout:
            layout_path = Path(args.layout)
            parts = layout_path.stem.split("_")
            if len(parts) >= 3:
                sid, rid = parts[0], parts[1]
            else:
                sid, rid = "unknown_scene", "unknown_room"
            build_room_graph_from_layout(sid, rid, layout_path, color_to_label, taxonomy=tax)
            return
        
        # Find layouts from manifest or directory
        layouts = find_layouts(
            Path(args.in_dir) if args.in_dir else None,
            Path(args.manifest) if args.manifest else None
        )
        if not layouts:
            print("No layouts found.", flush=True)
            return
        
        for sid, rid, layout_path in layouts:
            build_room_graph_from_layout(sid, rid, layout_path, color_to_label, taxonomy=tax)
            gc.collect()
    
    elif args.type == "scene":
        taxonomy = Taxonomy(Path(args.taxonomy))
        id2room, _ = taxonomy.get_room_mappings()
        
        if not args.in_dir:
            parser.error("--in_dir is required for scene type")
        
        dataset_root = Path(args.in_dir)
        manifest_path = Path(args.manifest) if args.manifest else None
        
        scenes = find_scene_pointclouds(dataset_root, manifest_path)
        print(f"Found {len(scenes)} scenes\n", flush=True)
        
        for scene_id, pc_path in scenes:
            try:
                build_scene_graph_from_pointcloud(
                    scene_id, pc_path, id2room, dataset_root,
                    adjacency_thresh=args.adjacency_thresh,
                    taxonomy=taxonomy
                )
            except Exception as e:
                print(f"[error] {scene_id}: {e}", flush=True)


if __name__ == "__main__":
    main()

