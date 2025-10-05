#!/usr/bin/env python3
"""
build_room_graphs_from_layouts.py

Creates room-level graphs from color-segmented layouts.
Each color blob (clustered by minimal linkage) becomes a node.
"""

import argparse, json, csv
from pathlib import Path
import cv2
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from collections import defaultdict

# ------------------------------------------------------------
def load_taxonomy(tax_path: Path):
    tax = json.loads(tax_path.read_text(encoding="utf-8"))
    color_to_label = {}
    
    # Load super-category colors (1000-1999) and wall category (2053)
    for sid, rgb in tax.get("id2color", {}).items():
        sid_int = int(sid)
        
        if 1000 <= sid_int <= 1999:
            # Super categories
            super_label = tax["id2super"].get(sid, f"Unknown_{sid}")
            color_to_label[tuple(rgb)] = {
                "label_id": sid_int,
                "label": super_label
            }
        elif sid_int == 2053:
            # Wall - get category name first, then map to super
            category_name = tax["id2category"].get(sid, "wall")
            super_label = tax.get("category2super", {}).get(category_name, "Structure")
            color_to_label[tuple(rgb)] = {
                "label_id": sid_int,
                "label": super_label
            }
    
    print(f"Loaded {len(color_to_label)} taxonomy colors (super categories + wall)", flush=True)
    return color_to_label

def find_layouts(root: Path | None, manifest: Path | None):
    layouts = []
    if manifest and manifest.exists():
        with open(manifest, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                # Skip scene-level layouts - only process individual rooms
                if row["room_id"] == "scene":
                    continue
                    
                # Use the layout_path from CSV directly
                layout_path = Path(row["layout_path"])
                if layout_path.exists():
                    sid = row["scene_id"]
                    rid = row["room_id"]
                    layouts.append((sid, rid, layout_path))
    elif root:
        for p in root.rglob("*_room_seg_layout.png"):
            parts = p.stem.split("_")
            # Skip scene layouts based on filename pattern
            if len(parts) >= 3 and parts[1] == "scene":
                continue
            if len(parts) >= 3:
                sid, rid = parts[0], parts[1]
                layouts.append((sid, rid, p))
    return layouts

# ------------------------------------------------------------
def compute_room_center(img):
    mask_room = ~(img == 255).all(axis=2)
    ys, xs = np.nonzero(mask_room)
    if len(xs) == 0:
        return np.array([img.shape[1] / 2, img.shape[0] / 2])
    return np.array([xs.mean(), ys.mean()])

def angle_from_center(center, point):
    v = point - center
    return np.arctan2(v[1], v[0])

# ------------------------------------------------------------
# No longer needed - replaced by vectorized approach

def extract_color_regions(img, color_to_label):
    """
    Extract regions by exact color match to taxonomy colors.
    Returns dict mapping taxonomy_color -> list of pixel coordinates
    """
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
        else:
            print(f"  Skipping unknown color {color_t}", flush=True)
    
    return color_masks

def find_color_clusters(points, linkage_thresh=15):
    """Cluster points using DBSCAN (handles sparse clouds well)."""
    if len(points) == 0:
        return []
    pts = np.array(points)
    if len(pts) == 1:
        return [pts]
    
    from sklearn.cluster import DBSCAN
    
    # DBSCAN parameters:
    # eps = maximum distance between points in same cluster
    # min_samples = minimum points to form a cluster
    clustering = DBSCAN(eps=linkage_thresh, min_samples=5).fit(pts)
    
    labels = clustering.labels_
    
    # Group points by cluster (-1 is noise, which we'll ignore or treat separately)
    clusters = []
    for label in set(labels):
        if label == -1:
            # Noise points - you can choose to ignore or keep as individual clusters
            continue
        cluster_mask = labels == label
        clusters.append(pts[cluster_mask])
    
    return clusters

# ------------------------------------------------------------
def visualize(img, room_center, nodes, edges, out_path):
    """Create a clear visualization with colored nodes and black edges."""
    h, w = img.shape[:2]
    
    # Create visualization on white background
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Draw the original image with transparency
    alpha = 0.3
    vis = cv2.addWeighted(img, alpha, vis, 1 - alpha, 0)
    
    # Define color scheme
    EDGE_COLOR = (0, 0, 0)           # Black for edges
    NODE_OUTLINE = (0, 0, 0)         # Black outline for nodes
    ROOM_CENTER_COLOR = (0, 200, 0)  # Green for room center
    TEXT_COLOR = (0, 0, 0)           # Black text
    TEXT_BG_COLOR = (255, 255, 255)  # White background for text
    
    # Draw edges first (behind nodes)
    # Only visualize edges that have a distance relation (by/near)
    for e in edges:
        # Skip if no distance relation
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
    
    # Draw room center as a larger black circle
    rcx, rcy = map(int, room_center)
    # Draw filled black circle (larger than nodes)
    cv2.circle(vis, (rcx, rcy), 12, (0, 0, 0), -1, cv2.LINE_AA)
    # Optional: draw white outline for visibility
    cv2.circle(vis, (rcx, rcy), 12, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw nodes with cluster colors
    for n in nodes:
        cx, cy = map(int, n["center"])
        
        # Use the stored taxonomy color instead of sampling from image
        node_color = n.get("color", (128, 128, 128))  # Default gray if color missing
        
        # Draw filled circle with cluster color
        cv2.circle(vis, (cx, cy), 8, node_color, -1, cv2.LINE_AA)
        # Draw black outline
        cv2.circle(vis, (cx, cy), 8, NODE_OUTLINE, 2, cv2.LINE_AA)
        
        # Add label BELOW the node
        label = n["label"]
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        
        # Position text below node (centered)
        text_x = cx - tw // 2
        text_y = cy + 18  # Below the node
        
        # Draw semi-transparent background for text
        overlay = vis.copy()
        cv2.rectangle(overlay, (text_x - 2, text_y - th - 2), 
                     (text_x + tw + 2, text_y + baseline), TEXT_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
        
        # Draw text
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    
    # Convert back to BGR for saving with OpenCV
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), vis_bgr)
    print(f"  ↳ saved visualization {out_path}", flush=True)

# ------------------------------------------------------------
def build_room_graph(scene_id, room_id, layout_path, color_to_label):
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

    # Extract color regions with exact matching
    color_regions = extract_color_regions(img, color_to_label)
    print(f"  Found {len(color_regions)} distinct taxonomy colors", flush=True)

    nodes = []
    node_id = 0

    # Store cluster points with nodes for distance calculations
    node_clusters = {}
    node_id = 0

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
            # Store cluster points for distance calculation
            node_clusters[node_key] = cluster
            node_id += 1

    print(f"  Total nodes created: {len(nodes)}", flush=True)

    from scipy.spatial.distance import cdist
    
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
            
            # Compute true minimum distance between clusters using cdist
            # This computes distance from every point in cluster_a to every point in cluster_b
            distances = cdist(cluster_a, cluster_b, metric='euclidean')
            min_cluster_distance = distances.min()

            # Determine proximity relation based on minimum cluster distance
            prox_a_to_b = None
            prox_b_to_a = None
            if "structure" in (la, lb):
                if min_cluster_distance < by_thresh:
                    prox_a_to_b = prox_b_to_a = "by"
            else:
                if min_cluster_distance < near_thresh:
                    prox_a_to_b = prox_b_to_a = "near"

            # Directional relations based on centroids
            ang_a = angle_from_center(room_center, ca)
            ang_b = angle_from_center(room_center, cb)
            d_ang = np.rad2deg((ang_b - ang_a + np.pi*2) % (np.pi*2))
            
            # Direction from A to B
            if d_ang < 45 or d_ang > 315:
                dir_a_to_b = "front_of"
                dir_b_to_a = "behind"
            elif 45 <= d_ang < 135:
                dir_a_to_b = "right_of"
                dir_b_to_a = "left_of"
            elif 135 <= d_ang < 225:
                dir_a_to_b = "behind"
                dir_b_to_a = "front_of"
            else:
                dir_a_to_b = "left_of"
                dir_b_to_a = "right_of"

            # Create edges
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
    out_vis = layout_path.with_name(f"{scene_id}_{room_id}_graph_vis.png")
    out_json.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    visualize(img, room_center, nodes, edges, out_vis)
    print(f"✔ wrote {out_json}", flush=True)
    return graph

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", help="Dataset root containing scenes/<scene_id>/rooms/<room_id>")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    ap.add_argument("--manifest", help="Optional manifest CSV listing scene_id,room_id,layout_path")
    ap.add_argument("--layout", help="Optional single layout path for testing")
    args = ap.parse_args()

    color_to_label = load_taxonomy(Path(args.taxonomy))

    if args.layout:
        layout_path = Path(args.layout)
        parts = layout_path.stem.split("_")
        if len(parts) >= 3:
            sid, rid = parts[0], parts[1]
        else:
            sid, rid = "unknown_scene", "unknown_room"
        build_room_graph(sid, rid, layout_path, color_to_label)
        return

    # Find layouts from manifest or directory
    layouts = find_layouts(Path(args.in_dir) if args.in_dir else None, 
                          Path(args.manifest) if args.manifest else None)
    if not layouts:
        print("No layouts found.", flush=True)
        return

    for sid, rid, layout_path in layouts:
        build_room_graph(sid, rid, layout_path, color_to_label)

if __name__ == "__main__":
    main()