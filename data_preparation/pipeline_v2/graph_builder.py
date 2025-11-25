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




def build_room_graph_from_metadata(scene_id: str, room_name: str, room_metadata: Dict,
                                   taxonomy: Taxonomy, output_dir: Path):
    """
    Build room graph from metadata (furniture positions, room center, etc.).
    This is faster and doesn't require layout images.
    
    Args:
        scene_id: Scene identifier
        room_name: Room name (e.g., "scene", "bedroom", "livingroom")
        room_metadata: Room metadata dict from scene metadata JSON
        taxonomy: Taxonomy object
        output_dir: Directory to save graph files
    """
    if 'furniture' not in room_metadata or len(room_metadata['furniture']) == 0:
        print(f"  [warn] No furniture in room {room_name}", flush=True)
        return None
    
    # Get room bounds for coordinate normalization
    room_bounds = room_metadata.get('bounds', {})
    if 'min' not in room_bounds or 'max' not in room_bounds:
        print(f"  [warn] No bounds for room {room_name}", flush=True)
        return None
    
    room_min = np.array(room_bounds['min'])
    room_max = np.array(room_bounds['max'])
    room_center_3d = np.array(room_metadata.get('center', (room_min + room_max) / 2))
    room_size = room_max - room_min
    
    # Project 3D positions to 2D (top-down view: use X and Z, ignore Y/height)
    # For layout view, we use X and Z coordinates
    up_axis = 1  # Y is typically up in 3D-FRONT
    horizontal_axes = [0, 2]  # X and Z
    
    # Build nodes from furniture metadata
    nodes = []
    node_id = 0
    all_positions_2d = []
    
    for furniture in room_metadata['furniture']:
        position_3d = np.array(furniture['position'])
        position_2d = position_3d[horizontal_axes]  # [x, z] coordinates
        
        # Normalize to image coordinates (0-255 for 256x256 image)
        # Map room bounds to image coordinates
        normalized_x = ((position_2d[0] - room_min[horizontal_axes[0]]) / room_size[horizontal_axes[0]]) * 255
        normalized_z = ((position_2d[1] - room_min[horizontal_axes[1]]) / room_size[horizontal_axes[1]]) * 255
        
        # Create bounding box from furniture bounds
        furniture_bounds = furniture.get('bounds', {})
        if 'min' in furniture_bounds and 'max' in furniture_bounds:
            bbox_min = np.array(furniture_bounds['min'])[horizontal_axes]
            bbox_max = np.array(furniture_bounds['max'])[horizontal_axes]
            
            # Normalize bbox to image coordinates
            bbox_x = ((bbox_min[0] - room_min[horizontal_axes[0]]) / room_size[horizontal_axes[0]]) * 255
            bbox_z = ((bbox_min[1] - room_min[horizontal_axes[1]]) / room_size[horizontal_axes[1]]) * 255
            bbox_w = ((bbox_max[0] - bbox_min[0]) / room_size[horizontal_axes[0]]) * 255
            bbox_h = ((bbox_max[1] - bbox_min[1]) / room_size[horizontal_axes[1]]) * 255
            
            bbox = (int(bbox_x), int(bbox_z), int(bbox_w), int(bbox_h))
        else:
            # Fallback: create small bbox around position
            bbox_size = 10  # pixels
            bbox = (int(normalized_x - bbox_size/2), int(normalized_z - bbox_size/2), bbox_size, bbox_size)
        
        category_id = furniture.get('category_id', 0)
        category_name = furniture.get('category_name', 'unknown')
        label = furniture.get('label', 'unknown')
        
        # Get label_id from taxonomy
        label_id = taxonomy.data.get('label2id', {}).get(label, 0)
        
        node_key = f"{label}_{node_id}"
        nodes.append({
            "id": node_key,
            "label": label,
            "label_id": label_id,
            "center": [int(normalized_x), int(normalized_z)],
            "bbox": bbox,
            "area": int(bbox[2] * bbox[3]),  # width * height
            "position_3d": position_3d.tolist(),
            "category_id": category_id,
            "category_name": category_name
        })
        
        all_positions_2d.append([normalized_x, normalized_z])
        node_id += 1
    
    print(f"  Total nodes created: {len(nodes)}", flush=True)
    
    if len(nodes) == 0:
        print(f"  [warn] No objects found in room metadata", flush=True)
        return None
    
    # Compute room center in 2D
    if all_positions_2d:
        room_center_2d = np.array(all_positions_2d).mean(axis=0)
    else:
        room_center_2d = np.array([128, 128])  # Center of 256x256 image
    
    # Build edges based on 2D distances
    edges = []
    image_diagonal = np.sqrt(256 * 256 + 256 * 256)  # 256x256 image
    near_threshold = 0.1 * image_diagonal  # 10% of image diagonal
    by_threshold = 0.05 * image_diagonal   # 5% of image diagonal
    
    for i, a in enumerate(nodes):
        center_a = np.array(a["center"])
        label_a = a["label"].lower()
        
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            
            center_b = np.array(b["center"])
            label_b = b["label"].lower()
            
            # Compute 2D distance
            dist_2d = np.linalg.norm(center_b - center_a)
            
            # Determine relationship
            relation = None
            if dist_2d < near_threshold:
                relation = "near"
            elif dist_2d < by_threshold:
                relation = "by"
            
            if relation:
                # Compute angle from room center
                vec_a = center_a - room_center_2d
                vec_b = center_b - room_center_2d
                
                angle_a = np.arctan2(vec_a[1], vec_a[0])
                angle_b = np.arctan2(vec_b[1], vec_b[0])
                angle_diff = angle_b - angle_a
                
                # Normalize angle to [-pi, pi]
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # Determine directional relation
                if abs(angle_diff) < np.pi / 6:  # ~30 degrees
                    direction = "right_of" if angle_diff > 0 else "left_of"
                elif abs(angle_diff - np.pi / 2) < np.pi / 6:
                    direction = "behind"
                elif abs(angle_diff + np.pi / 2) < np.pi / 6:
                    direction = "in_front_of"
                else:
                    direction = relation
                
                edges.append({
                    "source": a["id"],
                    "target": b["id"],
                    "relation": direction,
                    "distance": float(dist_2d)
                })
    
    # Create graph structure
    graph = {
        "scene_id": scene_id,
        "room_name": room_name,
        "room_center": room_center_2d.tolist(),
        "nodes": nodes,
        "edges": edges
    }
    
    # Save graph files
    safe_room_name = room_name.replace(" ", "_").replace("/", "_").lower()
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    graph_json = graphs_dir / f"{scene_id}_{safe_room_name}_graph.json"
    graph_txt = graphs_dir / f"{scene_id}_{safe_room_name}_graph.txt"
    
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
    
    return graph


def build_scene_graph_from_metadata(scene_id: str, scene_metadata: Dict,
                                    taxonomy: Taxonomy, output_dir: Path):
    """
    Build scene-level graph from metadata (all furniture positions across all rooms).
    This is faster and doesn't require layout images.
    
    Args:
        scene_id: Scene identifier
        scene_metadata: Full scene metadata dict from scene metadata JSON
        taxonomy: Taxonomy object
        output_dir: Directory to save graph files
    """
    # Collect all furniture from all rooms
    all_furniture = []
    scene_bounds = scene_metadata.get('bounds', {})
    
    if 'min' not in scene_bounds or 'max' not in scene_bounds:
        print(f"  [warn] No bounds for scene", flush=True)
        return None
    
    scene_min = np.array(scene_bounds['min'])
    scene_max = np.array(scene_bounds['max'])
    scene_center_3d = np.array(scene_metadata.get('center', (scene_min + scene_max) / 2))
    scene_size = scene_max - scene_min
    
    # Collect furniture from all rooms
    for room_name, room_info in scene_metadata.get('rooms', {}).items():
        room_furniture = room_info.get('furniture', [])
        for furniture in room_furniture:
            # Add room name to furniture info
            furniture_with_room = furniture.copy()
            furniture_with_room['room_name'] = room_name
            all_furniture.append(furniture_with_room)
    
    if len(all_furniture) == 0:
        print(f"  [warn] No furniture in scene", flush=True)
        return None
    
    # Project 3D positions to 2D (top-down view: use X and Z, ignore Y/height)
    up_axis = 1  # Y is typically up in 3D-FRONT
    horizontal_axes = [0, 2]  # X and Z
    
    # Build nodes from furniture metadata
    nodes = []
    node_id = 0
    all_positions_2d = []
    
    for furniture in all_furniture:
        position_3d = np.array(furniture['position'])
        position_2d = position_3d[horizontal_axes]  # [x, z] coordinates
        
        # Normalize to image coordinates (0-255 for 256x256 image)
        normalized_x = ((position_2d[0] - scene_min[horizontal_axes[0]]) / scene_size[horizontal_axes[0]]) * 255
        normalized_z = ((position_2d[1] - scene_min[horizontal_axes[1]]) / scene_size[horizontal_axes[1]]) * 255
        
        # Create bounding box from furniture bounds
        furniture_bounds = furniture.get('bounds', {})
        if 'min' in furniture_bounds and 'max' in furniture_bounds:
            bbox_min = np.array(furniture_bounds['min'])[horizontal_axes]
            bbox_max = np.array(furniture_bounds['max'])[horizontal_axes]
            
            # Normalize bbox to image coordinates
            bbox_x = ((bbox_min[0] - scene_min[horizontal_axes[0]]) / scene_size[horizontal_axes[0]]) * 255
            bbox_z = ((bbox_min[1] - scene_min[horizontal_axes[1]]) / scene_size[horizontal_axes[1]]) * 255
            bbox_w = ((bbox_max[0] - bbox_min[0]) / scene_size[horizontal_axes[0]]) * 255
            bbox_h = ((bbox_max[1] - bbox_min[1]) / scene_size[horizontal_axes[1]]) * 255
            
            bbox = (int(bbox_x), int(bbox_z), int(bbox_w), int(bbox_h))
        else:
            # Fallback: create small bbox around position
            bbox_size = 10  # pixels
            bbox = (int(normalized_x - bbox_size/2), int(normalized_z - bbox_size/2), bbox_size, bbox_size)
        
        category_id = furniture.get('category_id', 0)
        category_name = furniture.get('category_name', 'unknown')
        label = furniture.get('label', 'unknown')
        room_name = furniture.get('room_name', 'unknown')
        
        # Get label_id from taxonomy
        label_id = taxonomy.data.get('label2id', {}).get(label, 0)
        
        node_key = f"{label}_{node_id}"
        nodes.append({
            "id": node_key,
            "label": label,
            "label_id": label_id,
            "center": [int(normalized_x), int(normalized_z)],
            "bbox": bbox,
            "area": int(bbox[2] * bbox[3]),  # width * height
            "position_3d": position_3d.tolist(),
            "category_id": category_id,
            "category_name": category_name,
            "room_name": room_name
        })
        
        all_positions_2d.append([normalized_x, normalized_z])
        node_id += 1
    
    print(f"  Total nodes created: {len(nodes)}", flush=True)
    
    if len(nodes) == 0:
        print(f"  [warn] No objects found in scene metadata", flush=True)
        return None
    
    # Compute scene center in 2D
    if all_positions_2d:
        scene_center_2d = np.array(all_positions_2d).mean(axis=0)
    else:
        scene_center_2d = np.array([128, 128])  # Center of 256x256 image
    
    # Build edges based on 2D distances
    edges = []
    image_diagonal = np.sqrt(256 * 256 + 256 * 256)  # 256x256 image
    near_threshold = 0.1 * image_diagonal  # 10% of image diagonal
    by_threshold = 0.05 * image_diagonal   # 5% of image diagonal
    
    for i, a in enumerate(nodes):
        center_a = np.array(a["center"])
        label_a = a["label"].lower()
        room_a = a.get("room_name", "unknown")
        
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            
            center_b = np.array(b["center"])
            label_b = b["label"].lower()
            room_b = b.get("room_name", "unknown")
            
            # Compute 2D distance
            dist_2d = np.linalg.norm(center_b - center_a)
            
            # Determine relationship
            relation = None
            if dist_2d < near_threshold:
                relation = "near"
            elif dist_2d < by_threshold:
                relation = "by"
            
            if relation:
                # Compute angle from scene center
                vec_a = center_a - scene_center_2d
                vec_b = center_b - scene_center_2d
                
                angle_a = np.arctan2(vec_a[1], vec_a[0])
                angle_b = np.arctan2(vec_b[1], vec_b[0])
                angle_diff = angle_b - angle_a
                
                # Normalize angle to [-pi, pi]
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # Determine directional relation
                if abs(angle_diff) < np.pi / 6:  # ~30 degrees
                    direction = "right_of" if angle_diff > 0 else "left_of"
                elif abs(angle_diff - np.pi / 2) < np.pi / 6:
                    direction = "behind"
                elif abs(angle_diff + np.pi / 2) < np.pi / 6:
                    direction = "in_front_of"
                else:
                    direction = relation
                
                # Add room information to edge if objects are in different rooms
                edge_data = {
                    "source": a["id"],
                    "target": b["id"],
                    "relation": direction,
                    "distance": float(dist_2d)
                }
                
                if room_a != room_b:
                    edge_data["cross_room"] = True
                    edge_data["source_room"] = room_a
                    edge_data["target_room"] = room_b
                
                edges.append(edge_data)
    
    # Create graph structure
    graph = {
        "scene_id": scene_id,
        "room_name": "scene",
        "room_center": scene_center_2d.tolist(),
        "nodes": nodes,
        "edges": edges
    }
    
    # Save graph files
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    graph_json = graphs_dir / f"{scene_id}_scene_graph.json"
    graph_txt = graphs_dir / f"{scene_id}_scene_graph.txt"
    
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
    
    return graph
