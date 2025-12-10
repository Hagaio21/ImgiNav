#!/usr/bin/env python3
"""
Evaluation metrics for floorplan generation.

Workflow:
1. Clean generated image by snapping pixels to closest taxonomy colors
2. Extract color blobs (connected components) per class
3. Compute metrics:
   - Class presence: precision, recall, F1 for which classes appear
   - Per-class count: exact match, count accuracy per class
   - Per-class blob IoU: spatial accuracy per class
   - Wall pixel IoU: structural accuracy for walls
   - Weighted pixel accuracy: overall correctness with background de-weighted
   - Path metrics: A* path comparison for navigation evaluation
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from heapq import heappush, heappop
from scipy import ndimage
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Image Cleaning (snap to taxonomy colors)
# =============================================================================

def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy and build color mappings."""
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)
    
    categories = taxonomy.get("categories", [])
    category_to_color = taxonomy.get("category_to_color", {})
    
    # Build mappings
    color_to_name = {}
    name_to_color = {}
    name_to_id = {"Background": 0}
    id_to_name = {0: "Background"}
    
    for idx, cat in enumerate(categories):
        class_id = idx + 1
        name_to_id[cat] = class_id
        id_to_name[class_id] = cat
        
        if cat in category_to_color:
            color = tuple(category_to_color[cat])
            color_to_name[color] = cat
            name_to_color[cat] = color
    
    # Background is white
    color_to_name[(255, 255, 255)] = "Background"
    name_to_color["Background"] = (255, 255, 255)
    
    return {
        "categories": categories,
        "color_to_name": color_to_name,
        "name_to_color": name_to_color,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "num_classes": len(categories) + 1
    }


def clean_image(
    image: np.ndarray,
    taxonomy_info: Dict,
    white_threshold: int = 250
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean image by snapping each pixel to closest taxonomy color.
    
    Args:
        image: RGB image (H, W, 3) with values 0-255
        taxonomy_info: Taxonomy info dict from load_taxonomy()
        white_threshold: Pixels with all RGB > threshold are background
    
    Returns:
        cleaned_rgb: Cleaned RGB image (H, W, 3)
        class_map: Class ID map (H, W)
    """
    h, w = image.shape[:2]
    name_to_color = taxonomy_info["name_to_color"]
    name_to_id = taxonomy_info["name_to_id"]
    
    # Pre-compute category colors as arrays
    category_colors = {}
    for name, color in name_to_color.items():
        category_colors[name] = np.array(color, dtype=np.float32)
    
    # Identify background pixels (white)
    is_white = np.all(image > white_threshold, axis=2)
    
    # Initialize outputs
    cleaned_rgb = np.full_like(image, 255, dtype=np.uint8)
    class_map = np.zeros((h, w), dtype=np.int32)
    
    # Process non-white pixels
    non_white_mask = ~is_white
    if np.any(non_white_mask):
        y_coords, x_coords = np.where(non_white_mask)
        pixels = image[y_coords, x_coords].astype(np.float32)
        
        # Find closest category for each pixel
        best_names = np.full(len(pixels), "Background", dtype=object)
        best_dist = np.full(len(pixels), np.inf, dtype=np.float32)
        
        for name, color in category_colors.items():
            if name == "Background":
                continue
            dists = np.linalg.norm(pixels - color, axis=1)
            closer = dists < best_dist
            best_names[closer] = name
            best_dist[closer] = dists[closer]
        
        # Assign cleaned colors and class IDs
        for name in category_colors.keys():
            mask = (best_names == name)
            if np.any(mask):
                color = category_colors[name].astype(np.uint8)
                class_id = name_to_id.get(name, 0)
                matched_y = y_coords[mask]
                matched_x = x_coords[mask]
                cleaned_rgb[matched_y, matched_x] = color
                class_map[matched_y, matched_x] = class_id
    
    return cleaned_rgb, class_map


# =============================================================================
# Object Extraction (color blobs as connected components)
# =============================================================================

def extract_blobs(
    class_map: np.ndarray,
    exclude_classes: Optional[List[int]] = None,
    min_pixels: int = 10
) -> Dict[int, List[Dict]]:
    """
    Extract blobs (connected components) for each class.
    
    Args:
        class_map: Class ID map (H, W)
        exclude_classes: Class IDs to exclude (e.g., 0=background)
        min_pixels: Minimum pixels to count as a blob
    
    Returns:
        Dictionary mapping class_id -> list of blob dicts
        Each blob dict has: centroid, bbox, pixel_count, mask
    """
    if exclude_classes is None:
        exclude_classes = [0]
    
    blobs_by_class = defaultdict(list)
    unique_classes = np.unique(class_map)
    
    for class_id in unique_classes:
        if class_id in exclude_classes:
            continue
        
        mask = (class_map == class_id)
        labeled, num_features = ndimage.label(mask)
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            pixel_count = component.sum()
            
            if pixel_count < min_pixels:
                continue
            
            y_coords, x_coords = np.where(component)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))
            bbox = {
                'min_x': int(x_coords.min()),
                'min_y': int(y_coords.min()),
                'max_x': int(x_coords.max()),
                'max_y': int(y_coords.max())
            }
            
            blobs_by_class[int(class_id)].append({
                'centroid': centroid,
                'bbox': bbox,
                'pixel_count': int(pixel_count),
                'mask': component  # Store mask for pixel-level IoU
            })
    
    return dict(blobs_by_class)


# =============================================================================
# Class Presence Metrics
# =============================================================================

def compute_class_presence_metrics(
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]],
    id_to_name: Dict[int, str]
) -> Dict[str, Any]:
    """
    Compute class presence metrics.
    Measures if generated image has the same classes present as target.
    """
    pred_classes = set(pred_blobs.keys())
    target_classes = set(target_blobs.keys())
    
    common = pred_classes & target_classes
    missing = target_classes - pred_classes
    extra = pred_classes - target_classes
    
    # Metrics
    precision = len(common) / len(pred_classes) if len(pred_classes) > 0 else 1.0
    recall = len(common) / len(target_classes) if len(target_classes) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = len(common) / len(pred_classes | target_classes) if len(pred_classes | target_classes) > 0 else 1.0
    
    return {
        "class_precision": float(precision),
        "class_recall": float(recall),
        "class_f1": float(f1),
        "class_jaccard": float(jaccard),
        "num_common_classes": len(common),
        "num_missing_classes": len(missing),
        "num_extra_classes": len(extra),
        "common_classes": [id_to_name.get(c, str(c)) for c in common],
        "missing_classes": [id_to_name.get(c, str(c)) for c in missing],
        "extra_classes": [id_to_name.get(c, str(c)) for c in extra]
    }


# =============================================================================
# Per-Class Count Metrics
# =============================================================================

def compute_count_metrics(
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]],
    id_to_name: Dict[int, str]
) -> Dict[str, Any]:
    """
    Compute per-class object count metrics.
    
    Returns:
        - Per-class: exact match, count difference, count ratio
        - Aggregate: mean exact match rate, total count accuracy
    """
    all_classes = set(pred_blobs.keys()) | set(target_blobs.keys())
    
    per_class = {}
    exact_matches = 0
    total_classes = len(all_classes)
    total_count_error = 0
    total_target_objects = 0
    total_pred_objects = 0
    
    for class_id in all_classes:
        target_count = len(target_blobs.get(class_id, []))
        pred_count = len(pred_blobs.get(class_id, []))
        
        total_target_objects += target_count
        total_pred_objects += pred_count
        
        # Exact match
        exact_match = 1 if pred_count == target_count else 0
        exact_matches += exact_match
        
        # Count error
        count_diff = pred_count - target_count
        count_error = abs(count_diff)
        total_count_error += count_error
        
        # Count ratio (how close is pred to target)
        if target_count > 0:
            count_ratio = min(pred_count, target_count) / max(pred_count, target_count) if max(pred_count, target_count) > 0 else 1.0
        else:
            count_ratio = 1.0 if pred_count == 0 else 0.0
        
        class_name = id_to_name.get(class_id, str(class_id))
        per_class[class_name] = {
            "target_count": target_count,
            "pred_count": pred_count,
            "count_diff": count_diff,
            "exact_match": exact_match,
            "count_ratio": float(count_ratio)
        }
    
    # Aggregate metrics
    exact_match_rate = exact_matches / total_classes if total_classes > 0 else 1.0
    
    # Total count accuracy: penalize both over and under counting
    max_objects = max(total_target_objects, total_pred_objects, 1)
    total_count_accuracy = 1.0 - (total_count_error / (2 * max_objects))
    total_count_accuracy = max(0.0, total_count_accuracy)
    
    return {
        "exact_match_rate": float(exact_match_rate),
        "total_count_accuracy": float(total_count_accuracy),
        "total_target_objects": total_target_objects,
        "total_pred_objects": total_pred_objects,
        "total_count_error": total_count_error,
        "per_class": per_class
    }


# =============================================================================
# Blob Matching and IoU
# =============================================================================

def bbox_iou(bbox1: Dict, bbox2: Dict) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(bbox1['min_x'], bbox2['min_x'])
    y1 = max(bbox1['min_y'], bbox2['min_y'])
    x2 = min(bbox1['max_x'], bbox2['max_x'])
    y2 = min(bbox1['max_y'], bbox2['max_y'])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1['max_x'] - bbox1['min_x']) * (bbox1['max_y'] - bbox1['min_y'])
    area2 = (bbox2['max_x'] - bbox2['min_x']) * (bbox2['max_y'] - bbox2['min_y'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute pixel-level IoU between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Compute L2 distance between two centroids."""
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def match_blobs_greedy(
    pred_blobs: List[Dict],
    target_blobs: List[Dict],
    max_distance: float = None
) -> List[Tuple[int, int, float]]:
    """
    Match predicted blobs to target blobs using greedy matching by min centroid distance.
    
    Args:
        pred_blobs: List of predicted blob dicts
        target_blobs: List of target blob dicts
        max_distance: Maximum distance for valid match (None = no limit)
    
    Returns:
        List of (pred_idx, target_idx, distance) tuples
    """
    if len(pred_blobs) == 0 or len(target_blobs) == 0:
        return []
    
    distances = []
    for i, pred in enumerate(pred_blobs):
        for j, target in enumerate(target_blobs):
            dist = centroid_distance(pred['centroid'], target['centroid'])
            if max_distance is None or dist <= max_distance:
                distances.append((dist, i, j))
    
    distances.sort()
    matched_pred = set()
    matched_target = set()
    matches = []
    
    for dist, i, j in distances:
        if i not in matched_pred and j not in matched_target:
            matches.append((i, j, dist))
            matched_pred.add(i)
            matched_target.add(j)
    
    return matches


def compute_per_class_blob_metrics(
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]],
    id_to_name: Dict[int, str],
    image_size: int = 256
) -> Dict[str, Any]:
    """
    Compute per-class blob matching metrics.
    
    For each class:
        - detection_precision: matched / predicted (are predictions valid?)
        - detection_recall: matched / target (did we find them?)
        - avg_bbox_iou: average bounding box IoU for matched pairs
        - avg_centroid_dist: average centroid distance for matched pairs
    """
    all_classes = set(pred_blobs.keys()) | set(target_blobs.keys())
    
    per_class = {}
    
    # Aggregates
    total_matched = 0
    total_target = 0
    total_pred = 0
    total_bbox_iou = 0.0
    total_centroid_dist = 0.0
    
    for class_id in all_classes:
        pred_list = pred_blobs.get(class_id, [])
        target_list = target_blobs.get(class_id, [])
        
        n_pred = len(pred_list)
        n_target = len(target_list)
        
        total_pred += n_pred
        total_target += n_target
        
        # Match blobs
        matches = match_blobs_greedy(pred_list, target_list)
        n_matched = len(matches)
        total_matched += n_matched
        
        # Compute IoU and distance for matches
        class_bbox_iou = 0.0
        class_centroid_dist = 0.0
        
        for pred_idx, target_idx, dist in matches:
            iou = bbox_iou(pred_list[pred_idx]['bbox'], target_list[target_idx]['bbox'])
            class_bbox_iou += iou
            class_centroid_dist += dist
            total_bbox_iou += iou
            total_centroid_dist += dist
        
        # Per-class metrics
        detection_precision = n_matched / n_pred if n_pred > 0 else 1.0
        detection_recall = n_matched / n_target if n_target > 0 else 1.0
        detection_f1 = 2 * detection_precision * detection_recall / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0.0
        avg_bbox_iou = class_bbox_iou / n_matched if n_matched > 0 else 0.0
        avg_centroid_dist = class_centroid_dist / n_matched if n_matched > 0 else 0.0
        
        class_name = id_to_name.get(class_id, str(class_id))
        per_class[class_name] = {
            "n_target": n_target,
            "n_pred": n_pred,
            "n_matched": n_matched,
            "detection_precision": float(detection_precision),
            "detection_recall": float(detection_recall),
            "detection_f1": float(detection_f1),
            "avg_bbox_iou": float(avg_bbox_iou),
            "avg_centroid_dist": float(avg_centroid_dist),
            "normalized_centroid_dist": float(avg_centroid_dist / image_size) if image_size > 0 else 0.0
        }
    
    # Aggregate metrics
    aggregate_precision = total_matched / total_pred if total_pred > 0 else 1.0
    aggregate_recall = total_matched / total_target if total_target > 0 else 1.0
    aggregate_f1 = 2 * aggregate_precision * aggregate_recall / (aggregate_precision + aggregate_recall) if (aggregate_precision + aggregate_recall) > 0 else 0.0
    mean_bbox_iou = total_bbox_iou / total_matched if total_matched > 0 else 0.0
    mean_centroid_dist = total_centroid_dist / total_matched if total_matched > 0 else 0.0
    
    return {
        "detection_precision": float(aggregate_precision),
        "detection_recall": float(aggregate_recall),
        "detection_f1": float(aggregate_f1),
        "mean_bbox_iou": float(mean_bbox_iou),
        "mean_centroid_dist": float(mean_centroid_dist),
        "normalized_centroid_dist": float(mean_centroid_dist / image_size) if image_size > 0 else 0.0,
        "total_matched": total_matched,
        "total_target": total_target,
        "total_pred": total_pred,
        "per_class": per_class
    }


# =============================================================================
# Pixel-Level Metrics (for walls, floor, background)
# =============================================================================

def compute_pixel_metrics(
    pred_class_map: np.ndarray,
    target_class_map: np.ndarray,
    id_to_name: Dict[int, str],
    background_weight: float = 0.1,
    structural_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute pixel-level metrics.
    
    Args:
        pred_class_map: Predicted class map (H, W)
        target_class_map: Target class map (H, W)
        id_to_name: Mapping from class ID to name
        background_weight: Weight for background pixels (default 0.1 to de-emphasize)
        structural_classes: List of class names to compute separate pixel IoU (e.g., ["Wall", "Floor"])
    
    Returns:
        - weighted_pixel_accuracy: Overall accuracy with background de-weighted
        - per_class_pixel_iou: IoU for each class
        - structural_iou: Separate IoU for walls/floor
    """
    if structural_classes is None:
        structural_classes = ["Wall", "Floor"]
    
    name_to_id = {v: k for k, v in id_to_name.items()}
    
    h, w = pred_class_map.shape
    total_pixels = h * w
    
    # Per-class pixel IoU
    all_classes = set(np.unique(pred_class_map)) | set(np.unique(target_class_map))
    per_class_iou = {}
    
    for class_id in all_classes:
        pred_mask = (pred_class_map == class_id)
        target_mask = (target_class_map == class_id)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        iou = float(intersection / union) if union > 0 else 0.0
        class_name = id_to_name.get(class_id, str(class_id))
        per_class_iou[class_name] = iou
    
    # Weighted pixel accuracy
    correct = (pred_class_map == target_class_map)
    
    # Create weight map: background gets lower weight
    weights = np.ones((h, w), dtype=np.float32)
    background_mask = (target_class_map == 0)
    weights[background_mask] = background_weight
    
    weighted_correct = (correct * weights).sum()
    weighted_total = weights.sum()
    weighted_accuracy = float(weighted_correct / weighted_total) if weighted_total > 0 else 0.0
    
    # Unweighted accuracy for reference
    unweighted_accuracy = float(correct.sum() / total_pixels)
    
    # Object-only accuracy (exclude background)
    non_background = (target_class_map != 0)
    if non_background.sum() > 0:
        object_accuracy = float(correct[non_background].sum() / non_background.sum())
    else:
        object_accuracy = 1.0
    
    # Structural class IoU (walls, floor)
    structural_iou = {}
    for class_name in structural_classes:
        class_id = name_to_id.get(class_name)
        if class_id is not None:
            pred_mask = (pred_class_map == class_id)
            target_mask = (target_class_map == class_id)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            iou = float(intersection / union) if union > 0 else 0.0
            structural_iou[class_name] = iou
    
    # Mean IoU (excluding background)
    non_bg_ious = [iou for name, iou in per_class_iou.items() if name != "Background"]
    mean_iou = float(np.mean(non_bg_ious)) if non_bg_ious else 0.0
    
    return {
        "weighted_pixel_accuracy": weighted_accuracy,
        "unweighted_pixel_accuracy": unweighted_accuracy,
        "object_pixel_accuracy": object_accuracy,
        "mean_iou": mean_iou,
        "structural_iou": structural_iou,
        "per_class_iou": per_class_iou
    }


# =============================================================================
# Path-based Metrics (A* comparison)
# =============================================================================

def astar_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding. Grid: 1=traversable, 0=obstacle.
    start/goal: (x, y) positions
    Returns: Path as list of (x, y) or None
    """
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal
    
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if grid[sy, sx] == 0 or grid[gy, gx] == 0:
        return None
    
    open_set = [(0, sx, sy)]
    came_from = {}
    g_score = np.full((h, w), np.inf, dtype=np.float32)
    g_score[sy, sx] = 0
    visited = np.zeros((h, w), dtype=bool)
    
    # 8-connectivity
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]
    
    while open_set:
        _, cx, cy = heappop(open_set)
        
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        
        if cx == gx and cy == gy:
            # Reconstruct path
            path = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path.append((cx, cy))
            return path[::-1]
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 1 and not visited[ny, nx]:
                # Diagonal cost is sqrt(2), straight cost is 1
                move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                tentative_g = g_score[cy, cx] + move_cost
                
                if tentative_g < g_score[ny, nx]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[ny, nx] = tentative_g
                    h_score = abs(nx - gx) + abs(ny - gy)
                    f_score = tentative_g + h_score
                    heappush(open_set, (f_score, nx, ny))
    
    return None


def compute_path_overlap(path1: List[Tuple[int, int]], path2: List[Tuple[int, int]], tolerance: int = 2) -> float:
    """
    Compute overlap between two paths.
    A point in path1 is considered overlapping if within tolerance pixels of any point in path2.
    """
    if len(path1) == 0 or len(path2) == 0:
        return 0.0
    
    path2_set = set()
    for x, y in path2:
        for dx in range(-tolerance, tolerance + 1):
            for dy in range(-tolerance, tolerance + 1):
                path2_set.add((x + dx, y + dy))
    
    overlap_count = sum(1 for p in path1 if p in path2_set)
    return overlap_count / len(path1)


def compute_path_metrics(
    pred_class_map: np.ndarray,
    target_class_map: np.ndarray,
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]],
    traversable_ids: List[int],
    camera_position: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Compute path-based metrics using A* pathfinding.
    
    Finds paths from camera position (or center) to each matched object.
    Compares path existence and similarity between target and predicted layouts.
    
    Args:
        pred_class_map: Predicted class map
        target_class_map: Target class map
        pred_blobs: Predicted blobs by class
        target_blobs: Target blobs by class
        traversable_ids: Class IDs that are traversable
        camera_position: Optional camera (x, y) position. If None, uses center of floor.
    
    Returns:
        Path comparison metrics
    """
    h, w = pred_class_map.shape
    
    # Create traversability grids
    pred_grid = np.isin(pred_class_map, traversable_ids).astype(np.uint8)
    target_grid = np.isin(target_class_map, traversable_ids).astype(np.uint8)
    
    # Find start position
    if camera_position is not None:
        start_position = camera_position
    else:
        # Use center of traversable area in target
        traversable_coords = np.where(target_grid == 1)
        if len(traversable_coords[0]) > 0:
            start_y = int(np.mean(traversable_coords[0]))
            start_x = int(np.mean(traversable_coords[1]))
            start_position = (start_x, start_y)
        else:
            start_position = (w // 2, h // 2)
    
    # Match objects across all classes
    all_classes = set(pred_blobs.keys()) | set(target_blobs.keys())
    
    both_valid = 0
    target_only = 0
    pred_only = 0
    neither = 0
    total_matched = 0
    overlaps = []
    length_ratios = []
    per_object_results = []
    
    for class_id in all_classes:
        pred_list = pred_blobs.get(class_id, [])
        target_list = target_blobs.get(class_id, [])
        
        matches = match_blobs_greedy(pred_list, target_list)
        
        for pred_idx, target_idx, _ in matches:
            total_matched += 1
            
            # Get goals (centroids)
            pred_centroid = pred_list[pred_idx]['centroid']
            target_centroid = target_list[target_idx]['centroid']
            
            # Find nearest traversable cell for goals
            pred_goal = (int(pred_centroid[0]), int(pred_centroid[1]))
            target_goal = (int(target_centroid[0]), int(target_centroid[1]))
            
            # Clamp to valid range
            pred_goal = (max(0, min(w-1, pred_goal[0])), max(0, min(h-1, pred_goal[1])))
            target_goal = (max(0, min(w-1, target_goal[0])), max(0, min(h-1, target_goal[1])))
            
            # Check if goals are on traversable cells, if not find nearest
            if pred_grid[pred_goal[1], pred_goal[0]] == 0:
                pred_goal = None
            if target_grid[target_goal[1], target_goal[0]] == 0:
                target_goal = None
            
            if pred_goal is None and target_goal is None:
                neither += 1
                continue
            
            # Compute paths
            target_path = astar_path(target_grid, start_position, target_goal) if target_goal else None
            pred_path = astar_path(pred_grid, start_position, pred_goal) if pred_goal else None
            
            target_exists = target_path is not None
            pred_exists = pred_path is not None
            
            result = {
                "class_id": class_id,
                "target_reachable": target_exists,
                "pred_reachable": pred_exists
            }
            
            if target_exists and pred_exists:
                both_valid += 1
                overlap = compute_path_overlap(pred_path, target_path)
                overlaps.append(overlap)
                if len(target_path) > 0:
                    ratio = len(pred_path) / len(target_path)
                    length_ratios.append(ratio)
                    result["path_length_ratio"] = ratio
                result["path_overlap"] = overlap
            elif target_exists:
                target_only += 1
            elif pred_exists:
                pred_only += 1
            else:
                neither += 1
            
            per_object_results.append(result)
    
    total = total_matched if total_matched > 0 else 1
    
    # Reachability agreement: fraction where both agree (both reachable or neither)
    agreement = (both_valid + neither) / total
    
    # Reachability preservation: of objects reachable in target, how many reachable in pred?
    target_reachable_total = both_valid + target_only
    reachability_preservation = both_valid / target_reachable_total if target_reachable_total > 0 else 1.0
    
    return {
        "path_agreement_rate": float(agreement),
        "reachability_preservation": float(reachability_preservation),
        "path_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "path_overlap_std": float(np.std(overlaps)) if overlaps else 0.0,
        "path_length_ratio_mean": float(np.mean(length_ratios)) if length_ratios else 1.0,
        "path_length_ratio_std": float(np.std(length_ratios)) if length_ratios else 0.0,
        "matched_objects": total_matched,
        "both_reachable": both_valid,
        "target_only_reachable": target_only,
        "pred_only_reachable": pred_only,
        "neither_reachable": neither,
        "per_object": per_object_results
    }


# =============================================================================
# Main Evaluator Class
# =============================================================================

class FloorplanEvaluator:
    """
    Complete evaluator for floorplan generation with discriminative metrics.
    
    Metrics computed:
    1. Class presence: precision, recall, F1 for which classes appear
    2. Per-class count: exact match rate, count accuracy per class
    3. Per-class blob detection: precision, recall, IoU per class
    4. Pixel metrics: weighted accuracy, structural IoU (walls), mean IoU
    5. Path metrics: reachability preservation, path overlap
    
    Usage:
        evaluator = FloorplanEvaluator(taxonomy_path)
        metrics = evaluator.evaluate(pred_rgb, target_rgb)
    """
    
    def __init__(
        self,
        taxonomy_path: Path,
        traversable_names: Optional[List[str]] = None,
        structural_names: Optional[List[str]] = None,
        exclude_from_blobs: Optional[List[str]] = None,
        min_blob_pixels: int = 10,
        background_weight: float = 0.1
    ):
        self.taxonomy_info = load_taxonomy(taxonomy_path)
        self.min_blob_pixels = min_blob_pixels
        self.background_weight = background_weight
        
        # Traversable classes for pathfinding
        if traversable_names is None:
            traversable_names = ["Floor", "Door", "Window"]
        self.traversable_ids = []
        for name in traversable_names:
            class_id = self.taxonomy_info["name_to_id"].get(name)
            if class_id is not None:
                self.traversable_ids.append(class_id)
        
        # Structural classes for pixel IoU
        if structural_names is None:
            structural_names = ["Wall", "Floor"]
        self.structural_names = structural_names
        
        # Excluded classes for blob detection
        if exclude_from_blobs is None:
            exclude_from_blobs = ["Background", "Floor", "Wall"]
        self.exclude_blob_ids = []
        for name in exclude_from_blobs:
            class_id = self.taxonomy_info["name_to_id"].get(name)
            if class_id is not None:
                self.exclude_blob_ids.append(class_id)
    
    def clean_and_extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Clean image and extract blobs."""
        cleaned_rgb, class_map = clean_image(image, self.taxonomy_info)
        blobs = extract_blobs(class_map, self.exclude_blob_ids, self.min_blob_pixels)
        return cleaned_rgb, class_map, blobs
    
    def evaluate(
        self,
        pred_rgb: np.ndarray,
        target_rgb: np.ndarray,
        compute_paths: bool = True,
        camera_position: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate predicted image against target.
        
        Args:
            pred_rgb: Predicted RGB image (H, W, 3)
            target_rgb: Target RGB image (H, W, 3)
            compute_paths: Whether to compute path metrics
            camera_position: Optional (x, y) camera position for path metrics
        
        Returns:
            Dictionary with all metrics
        """
        pred_cleaned, pred_class_map, pred_blobs = self.clean_and_extract(pred_rgb)
        target_cleaned, target_class_map, target_blobs = self.clean_and_extract(target_rgb)
        
        image_size = pred_rgb.shape[0]
        id_to_name = self.taxonomy_info["id_to_name"]
        
        results = {}
        
        # 1. Class presence metrics
        class_presence = compute_class_presence_metrics(pred_blobs, target_blobs, id_to_name)
        results["class_presence"] = class_presence
        
        # 2. Count metrics
        count_metrics = compute_count_metrics(pred_blobs, target_blobs, id_to_name)
        results["counts"] = count_metrics
        
        # 3. Blob detection/matching metrics
        blob_metrics = compute_per_class_blob_metrics(pred_blobs, target_blobs, id_to_name, image_size)
        results["blobs"] = blob_metrics
        
        # 4. Pixel-level metrics
        pixel_metrics = compute_pixel_metrics(
            pred_class_map, target_class_map, id_to_name,
            background_weight=self.background_weight,
            structural_classes=self.structural_names
        )
        results["pixels"] = pixel_metrics
        
        # 5. Path metrics
        if compute_paths:
            path_metrics = compute_path_metrics(
                pred_class_map, target_class_map,
                pred_blobs, target_blobs,
                self.traversable_ids,
                camera_position=camera_position
            )
            results["paths"] = path_metrics
        
        # Summary: key metrics for comparison
        results["summary"] = {
            # Class presence
            "class_f1": class_presence["class_f1"],
            "class_recall": class_presence["class_recall"],
            
            # Counts
            "count_exact_match_rate": count_metrics["exact_match_rate"],
            "count_accuracy": count_metrics["total_count_accuracy"],
            
            # Blob detection
            "detection_f1": blob_metrics["detection_f1"],
            "detection_recall": blob_metrics["detection_recall"],
            "mean_bbox_iou": blob_metrics["mean_bbox_iou"],
            
            # Pixels
            "weighted_pixel_accuracy": pixel_metrics["weighted_pixel_accuracy"],
            "object_pixel_accuracy": pixel_metrics["object_pixel_accuracy"],
            "mean_iou": pixel_metrics["mean_iou"],
            "wall_iou": pixel_metrics["structural_iou"].get("Wall", 0.0),
            "floor_iou": pixel_metrics["structural_iou"].get("Floor", 0.0),
        }
        
        if compute_paths:
            results["summary"]["reachability_preservation"] = path_metrics["reachability_preservation"]
            results["summary"]["path_overlap"] = path_metrics["path_overlap_mean"]
            results["summary"]["path_length_ratio"] = path_metrics["path_length_ratio_mean"]
        
        # Store cleaned images for visualization
        results["_cleaned_pred"] = pred_cleaned
        results["_cleaned_target"] = target_cleaned
        
        return results
    
    def evaluate_batch(
        self,
        pred_images: List[np.ndarray],
        target_images: List[np.ndarray],
        compute_paths: bool = True
    ) -> Dict[str, Any]:
        """Evaluate batch and compute aggregate metrics."""
        all_metrics = []
        all_per_class = defaultdict(lambda: defaultdict(list))
        
        for pred, target in zip(pred_images, target_images):
            metrics = self.evaluate(pred, target, compute_paths)
            all_metrics.append(metrics["summary"])
            
            # Collect per-class metrics
            for class_name, class_metrics in metrics["blobs"]["per_class"].items():
                for metric_name, value in class_metrics.items():
                    if isinstance(value, (int, float)):
                        all_per_class[class_name][metric_name].append(value)
        
        # Aggregate summary metrics
        aggregated = {}
        keys = all_metrics[0].keys()
        
        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
        
        # Aggregate per-class metrics
        per_class_aggregated = {}
        for class_name, class_metrics in all_per_class.items():
            per_class_aggregated[class_name] = {}
            for metric_name, values in class_metrics.items():
                per_class_aggregated[class_name][f"{metric_name}_mean"] = float(np.mean(values))
                per_class_aggregated[class_name][f"{metric_name}_std"] = float(np.std(values))
        
        aggregated["per_class"] = per_class_aggregated
        aggregated["num_samples"] = len(all_metrics)
        
        return aggregated


# =============================================================================
# Tensor conversion utilities
# =============================================================================

def tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy RGB image (H, W, 3) with values 0-255."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        if tensor.shape[0] in [1, 3, 4]:
            tensor = tensor.permute(1, 2, 0)
        
        if tensor.shape[2] > 3:
            tensor = tensor[:, :, :3]
        
        arr = tensor.numpy()
        
        # Handle [-1, 1] range
        if arr.min() < 0:
            arr = (arr + 1) / 2
        
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr
    
    return tensor


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    evaluator = FloorplanEvaluator(args.taxonomy)
    
    pred_img = np.array(Image.open(args.pred).convert("RGB"))
    target_img = np.array(Image.open(args.target).convert("RGB"))
    
    metrics = evaluator.evaluate(pred_img, target_img)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n[Class Presence]")
    print(f"  Class F1: {metrics['summary']['class_f1']:.3f}")
    print(f"  Class Recall: {metrics['summary']['class_recall']:.3f}")
    print(f"  Missing: {metrics['class_presence']['missing_classes']}")
    print(f"  Extra: {metrics['class_presence']['extra_classes']}")
    
    print("\n[Object Counts]")
    print(f"  Exact Match Rate: {metrics['summary']['count_exact_match_rate']:.3f}")
    print(f"  Count Accuracy: {metrics['summary']['count_accuracy']:.3f}")
    
    print("\n[Blob Detection]")
    print(f"  Detection F1: {metrics['summary']['detection_f1']:.3f}")
    print(f"  Detection Recall: {metrics['summary']['detection_recall']:.3f}")
    print(f"  Mean BBox IoU: {metrics['summary']['mean_bbox_iou']:.3f}")
    
    print("\n[Pixel Metrics]")
    print(f"  Weighted Pixel Accuracy: {metrics['summary']['weighted_pixel_accuracy']:.3f}")
    print(f"  Object Pixel Accuracy: {metrics['summary']['object_pixel_accuracy']:.3f}")
    print(f"  Mean IoU: {metrics['summary']['mean_iou']:.3f}")
    print(f"  Wall IoU: {metrics['summary']['wall_iou']:.3f}")
    print(f"  Floor IoU: {metrics['summary']['floor_iou']:.3f}")
    
    if "reachability_preservation" in metrics['summary']:
        print("\n[Path Metrics]")
        print(f"  Reachability Preservation: {metrics['summary']['reachability_preservation']:.3f}")
        print(f"  Path Overlap: {metrics['summary']['path_overlap']:.3f}")
        print(f"  Path Length Ratio: {metrics['summary']['path_length_ratio']:.3f}")
    
    print("\n[Per-Class Detection]")
    for class_name, class_metrics in metrics['blobs']['per_class'].items():
        if class_metrics['n_target'] > 0 or class_metrics['n_pred'] > 0:
            print(f"  {class_name}: F1={class_metrics['detection_f1']:.2f}, IoU={class_metrics['avg_bbox_iou']:.2f}, T={class_metrics['n_target']}, P={class_metrics['n_pred']}")
    
    print("=" * 60)
    
    if args.output:
        # Remove non-serializable items
        output_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        with open(args.output, "w") as f:
            json.dump(output_metrics, f, indent=2)
        print(f"\nSaved to {args.output}")