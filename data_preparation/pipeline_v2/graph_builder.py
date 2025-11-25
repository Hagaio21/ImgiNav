#!/usr/bin/env python3
"""
Graph builder for pipeline v2.
Completely new implementation with a different approach.
Builds room graphs from segmentation layout images using connected components and bounding boxes.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from data_preparation.utils.text_utils import graph2text
from common.taxonomy import Taxonomy
from common.utils import write_json


def find_connected_components(img: np.ndarray, color: Tuple[int, int, int]) -> List[Dict]:
    """
    Find connected components (object instances) for a given color using OpenCV.
    
    Args:
        img: RGB image array (H, W, 3)
        color: (R, G, B) tuple to find
        
    Returns:
        List of component dicts with 'mask', 'bbox', 'centroid', 'area'
    """
    # Create binary mask for this color
    mask = np.all(img == color, axis=-1).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    components = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10:  # Filter tiny components
            continue
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        components.append({
            'mask': (labels == i).astype(np.uint8),
            'bbox': (x, y, w, h),
            'centroid': (int(centroids[i][0]), int(centroids[i][1])),
            'area': area
        })
    
    return components


def compute_bbox_distance(bbox_a: Tuple[int, int, int, int], 
                          bbox_b: Tuple[int, int, int, int]) -> float:
    """
    Compute minimum distance between two bounding boxes.
    
    Args:
        bbox_a: (x, y, w, h) for box A
        bbox_b: (x, y, w, h) for box B
        
    Returns:
        Minimum distance between boxes (0 if overlapping)
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    
    # Box A boundaries
    a_left, a_right = x1, x1 + w1
    a_top, a_bottom = y1, y1 + h1
    
    # Box B boundaries
    b_left, b_right = x2, x2 + w2
    b_top, b_bottom = y2, y2 + h2
    
    # Check if boxes overlap
    if not (a_right < b_left or b_right < a_left or a_bottom < b_top or b_bottom < a_top):
        return 0.0
    
    # Compute minimum distance
    dx = max(0, max(a_left - b_right, b_left - a_right))
    dy = max(0, max(a_top - b_bottom, b_top - a_bottom))
    
    return np.sqrt(dx * dx + dy * dy)


def compute_spatial_relation(centroid_a: Tuple[int, int], 
                            centroid_b: Tuple[int, int],
                            room_center: Tuple[int, int]) -> Tuple[str, str]:
    """
    Compute directional relation between two objects based on their centroids.
    
    Args:
        centroid_a: (x, y) of object A
        centroid_b: (x, y) of object B
        room_center: (x, y) of room center
        
    Returns:
        Tuple of (direction from A to B, direction from B to A)
    """
    # Vector from room center to each object
    vec_a = np.array([centroid_a[0] - room_center[0], centroid_a[1] - room_center[1]])
    vec_b = np.array([centroid_b[0] - room_center[0], centroid_b[1] - room_center[1]])
    
    # Compute angle from room center
    angle_a = np.arctan2(vec_a[1], vec_a[0])
    angle_b = np.arctan2(vec_b[1], vec_b[0])
    
    # Relative angle
    rel_angle = (angle_b - angle_a) % (2 * np.pi)
    
    # Map to directions
    if rel_angle < np.pi / 8 or rel_angle >= 15 * np.pi / 8:
        dir_a_to_b = "to the right of"
        dir_b_to_a = "to the left of"
    elif rel_angle < 3 * np.pi / 8:
        dir_a_to_b = "to the bottom-right of"
        dir_b_to_a = "to the top-left of"
    elif rel_angle < 5 * np.pi / 8:
        dir_a_to_b = "below"
        dir_b_to_a = "above"
    elif rel_angle < 7 * np.pi / 8:
        dir_a_to_b = "to the bottom-left of"
        dir_b_to_a = "to the top-right of"
    elif rel_angle < 9 * np.pi / 8:
        dir_a_to_b = "to the left of"
        dir_b_to_a = "to the right of"
    elif rel_angle < 11 * np.pi / 8:
        dir_a_to_b = "to the top-left of"
        dir_b_to_a = "to the bottom-right of"
    elif rel_angle < 13 * np.pi / 8:
        dir_a_to_b = "above"
        dir_b_to_a = "below"
    else:
        dir_a_to_b = "to the top-right of"
        dir_b_to_a = "to the bottom-left of"
    
    return dir_a_to_b, dir_b_to_a


def compute_room_center_from_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """
    Compute room center as the center of all object bounding boxes.
    
    Args:
        bboxes: List of (x, y, w, h) bounding boxes
        
    Returns:
        (x, y) room center
    """
    if not bboxes:
        return (128, 128)  # Default center for 256x256 image
    
    centers = []
    for x, y, w, h in bboxes:
        centers.append((x + w / 2, y + h / 2))
    
    center_x = int(np.mean([c[0] for c in centers]))
    center_y = int(np.mean([c[1] for c in centers]))
    return (center_x, center_y)


def visualize_graph(img: np.ndarray, room_center: Tuple[int, int],
                   nodes: List[Dict], edges: List[Dict], out_path: Path):
    """
    Create visualization of graph overlaid on layout image.
    
    Args:
        img: Original segmentation layout image (RGB)
        room_center: (x, y) room center
        nodes: List of node dicts
        edges: List of edge dicts
        out_path: Output path for visualization PNG
    """
    h, w = img.shape[:2]
    
    # Create visualization on white background
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Draw the original image with transparency
    alpha = 0.3
    vis = cv2.addWeighted(img, alpha, vis, 1 - alpha, 0)
    
    # Draw edges first
    for e in edges:
        if e.get("distance_relation") is None:
            continue
        
        a = next((n for n in nodes if n["id"] == e["obj_a"]), None)
        b = next((n for n in nodes if n["id"] == e["obj_b"]), None)
        if not a or not b:
            continue
        
        ca = tuple(map(int, a["center"]))
        cb = tuple(map(int, b["center"]))
        cv2.line(vis, ca, cb, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Label edge
        mid = ((np.array(ca) + np.array(cb)) / 2).astype(int)
        text = f"({e.get('distance_relation', '')}, {e.get('direction_relation', '')})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(vis, (mid[0] - 2, mid[1] - th - 2), 
                     (mid[0] + tw + 2, mid[1] + 2), (255, 255, 255), -1)
        cv2.putText(vis, text, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.3, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Draw room center
    cv2.circle(vis, room_center, 12, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, room_center, 12, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw nodes with bounding boxes
    for n in nodes:
        center = tuple(map(int, n["center"]))
        bbox = n.get("bbox")
        node_color = tuple(map(int, n.get("color", (128, 128, 128))))
        
        # Draw bounding box if available
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), node_color, 2)
        
        # Draw center point
        cv2.circle(vis, center, 6, node_color, -1, cv2.LINE_AA)
        cv2.circle(vis, center, 6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Add label
        label = n["label"]
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_x = center[0] - tw // 2
        text_y = center[1] + 20
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (text_x - 2, text_y - th - 2), 
                     (text_x + tw + 2, text_y + baseline), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Convert back to BGR for saving
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), vis_bgr)
    print(f"  ↳ saved visualization {out_path}", flush=True)


def build_room_graph_from_layout(scene_id: str, room_name: str, layout_path: Path,
                                 taxonomy: Taxonomy, output_dir: Path):
    """
    Build room graph from segmentation layout image using a new approach:
    - Connected components for object detection
    - Bounding boxes for spatial relationships
    - No clustering, no point clouds
    
    Args:
        scene_id: Scene identifier
        room_name: Room name (e.g., "scene", "bedroom", "livingroom")
        layout_path: Path to segmentation layout image
        taxonomy: Taxonomy object
        output_dir: Directory to save graph files
    """
    # Load segmentation image
    img = cv2.imread(str(layout_path))
    if img is None:
        print(f"[warn] cannot read {layout_path}", flush=True)
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Get color to label mapping from taxonomy
    color_to_label = taxonomy.get_color_to_label_dict()
    
    # Find all unique colors in image (excluding white background)
    unique_colors = np.unique(img.reshape(-1, 3), axis=0)
    
    # Build nodes from connected components
    nodes = []
    node_id = 0
    all_bboxes = []
    
    for color in unique_colors:
        color_t = tuple(int(c) for c in color)
        
        # Skip white background
        if color_t == (255, 255, 255):
            continue
        
        # Check if this color is in taxonomy
        if color_t not in color_to_label:
            continue
        
        label_info = color_to_label[color_t]
        
        # Find connected components for this color
        components = find_connected_components(img, color_t)
        
        print(f"  Color {color_t} ({label_info['label']}): {len(components)} instances", flush=True)
        
        for comp in components:
            node_key = f"{label_info['label']}_{node_id}"
            centroid = comp['centroid']
            bbox = comp['bbox']
            
            nodes.append({
                "id": node_key,
                "label": label_info["label"],
                "label_id": label_info["label_id"],
                "center": list(centroid),
                "bbox": bbox,
                "area": int(comp['area']),
                "color": color_t
            })
            
            all_bboxes.append(bbox)
            node_id += 1
    
    print(f"  Total nodes created: {len(nodes)}", flush=True)
    
    if len(nodes) == 0:
        print(f"  [warn] No objects found in layout", flush=True)
        return None
    
    # Compute room center from bounding boxes
    room_center = compute_room_center_from_bboxes(all_bboxes)
    
    # Build edges based on bounding box distances
    edges = []
    image_diagonal = np.sqrt(h * h + w * w)
    near_threshold = 0.1 * image_diagonal  # 10% of image diagonal
    by_threshold = 0.05 * image_diagonal   # 5% of image diagonal
    
    for i, a in enumerate(nodes):
        bbox_a = a["bbox"]
        center_a = tuple(a["center"])
        label_a = a["label"].lower()
        
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            
            bbox_b = b["bbox"]
            center_b = tuple(b["center"])
            label_b = b["label"].lower()
            
            # Compute distance between bounding boxes
            distance = compute_bbox_distance(bbox_a, bbox_b)
            
            # Determine proximity relation
            distance_relation = None
            if "structure" in label_a or "structure" in label_b:
                if distance < by_threshold:
                    distance_relation = "by"
            else:
                if distance < near_threshold:
                    distance_relation = "near"
            
            # Compute directional relation
            dir_a_to_b, dir_b_to_a = compute_spatial_relation(
                center_a, center_b, room_center
            )
            
            edges.append({
                "obj_a": a["id"],
                "obj_b": b["id"],
                "distance_relation": distance_relation,
                "direction_relation": dir_a_to_b
            })
            edges.append({
                "obj_a": b["id"],
                "obj_b": a["id"],
                "distance_relation": distance_relation,
                "direction_relation": dir_b_to_a
            })
    
    # Build graph structure
    graph = {
        "scene_id": scene_id,
        "room_name": room_name,
        "room_center": list(room_center),
        "nodes": nodes,
        "edges": edges
    }
    
    # Generate output filenames using room_name
    safe_room_name = room_name.replace(" ", "_").replace("/", "_").lower()
    graph_json = output_dir / f"{scene_id}_{safe_room_name}_graph.json"
    graph_txt = output_dir / f"{scene_id}_{safe_room_name}_graph.txt"
    graph_vis = output_dir / f"{scene_id}_{safe_room_name}_graph_vis.png"
    
    # Save JSON
    write_json(graph, graph_json)
    print(f"✔ wrote {graph_json}", flush=True)
    
    # Generate text version
    try:
        text = graph2text(graph_json, taxonomy)
        if text:
            graph_txt.write_text(text, encoding="utf-8")
            print(f"✔ wrote {graph_txt}", flush=True)
    except Exception as e:
        print(f"  [warn] Failed to generate text for {graph_json}: {e}", flush=True)
    
    # Generate visualization
    visualize_graph(img, room_center, nodes, edges, graph_vis)
    
    return graph
