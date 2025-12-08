#!/usr/bin/env python3
"""
Evaluation metrics for floorplan generation.

Workflow:
1. Clean generated image by snapping pixels to closest taxonomy colors
2. Extract color blobs (connected components) per class
3. Compute metrics:
   - Palette matching: Does generated have same classes present as target?
   - Object count: Number of blobs per class
   - BBox IoU: IoU between matched blob bounding boxes
   - Centroid L1: Distance between matched blob centroids
   - Path similarity: Compare A* paths between target and generated
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
        Each blob dict has: centroid, bbox, pixel_count
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
                'pixel_count': int(pixel_count)
            })
    
    return dict(blobs_by_class)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_palette_metrics(
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]],
    id_to_name: Dict[int, str]
) -> Dict[str, Any]:
    """
    Compute palette matching metrics.
    Checks if generated image has same classes present as target.
    """
    pred_classes = set(pred_blobs.keys())
    target_classes = set(target_blobs.keys())
    
    common = pred_classes & target_classes
    missing = target_classes - pred_classes
    extra = pred_classes - target_classes
    
    union = pred_classes | target_classes
    jaccard = len(common) / len(union) if len(union) > 0 else 1.0
    recall = len(common) / len(target_classes) if len(target_classes) > 0 else 1.0
    precision = len(common) / len(pred_classes) if len(pred_classes) > 0 else 1.0
    
    return {
        "palette_jaccard": float(jaccard),
        "palette_precision": float(precision),
        "palette_recall": float(recall),
        "common_classes": [id_to_name.get(c, str(c)) for c in common],
        "missing_classes": [id_to_name.get(c, str(c)) for c in missing],
        "extra_classes": [id_to_name.get(c, str(c)) for c in extra]
    }


def compute_object_count_metrics(
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]]
) -> Dict[str, Any]:
    """Compute object count metrics per class."""
    all_classes = set(pred_blobs.keys()) | set(target_blobs.keys())
    
    total_target = 0
    total_pred = 0
    count_diff_sum = 0
    per_class = {}
    
    for class_id in all_classes:
        target_count = len(target_blobs.get(class_id, []))
        pred_count = len(pred_blobs.get(class_id, []))
        
        total_target += target_count
        total_pred += pred_count
        count_diff_sum += abs(target_count - pred_count)
        
        per_class[class_id] = {
            "target": target_count,
            "pred": pred_count,
            "diff": pred_count - target_count
        }
    
    max_count = max(total_target, total_pred, 1)
    count_accuracy = 1.0 - (count_diff_sum / (2 * max_count))
    count_accuracy = max(0.0, count_accuracy)
    
    return {
        "object_count_accuracy": float(count_accuracy),
        "total_target_objects": total_target,
        "total_pred_objects": total_pred,
        "per_class_counts": per_class
    }


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


def centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Compute L1 distance between two centroids."""
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])


def match_blobs(
    pred_blobs: List[Dict],
    target_blobs: List[Dict]
) -> List[Tuple[int, int, float]]:
    """
    Match predicted blobs to target blobs using greedy matching by min centroid distance.
    Returns: List of (pred_idx, target_idx, distance) tuples
    """
    if len(pred_blobs) == 0 or len(target_blobs) == 0:
        return []
    
    distances = []
    for i, pred in enumerate(pred_blobs):
        for j, target in enumerate(target_blobs):
            dist = centroid_distance(pred['centroid'], target['centroid'])
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


def compute_blob_matching_metrics(
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]],
    image_size: int = 256
) -> Dict[str, Any]:
    """Compute blob matching metrics (BBox IoU, Centroid L1)."""
    all_classes = set(pred_blobs.keys()) | set(target_blobs.keys())
    
    total_iou = 0.0
    total_centroid_dist = 0.0
    total_matches = 0
    per_class_metrics = {}
    
    for class_id in all_classes:
        pred_list = pred_blobs.get(class_id, [])
        target_list = target_blobs.get(class_id, [])
        
        matches = match_blobs(pred_list, target_list)
        
        class_iou = 0.0
        class_dist = 0.0
        
        for pred_idx, target_idx, dist in matches:
            iou = bbox_iou(pred_list[pred_idx]['bbox'], target_list[target_idx]['bbox'])
            class_iou += iou
            class_dist += dist
        
        num_matches = len(matches)
        total_matches += num_matches
        total_iou += class_iou
        total_centroid_dist += class_dist
        
        per_class_metrics[class_id] = {
            "matches": num_matches,
            "avg_iou": class_iou / num_matches if num_matches > 0 else 0.0,
            "avg_centroid_dist": class_dist / num_matches if num_matches > 0 else 0.0
        }
    
    normalized_dist = total_centroid_dist / (total_matches * image_size) if total_matches > 0 else 1.0
    
    return {
        "mean_bbox_iou": total_iou / total_matches if total_matches > 0 else 0.0,
        "mean_centroid_l1": total_centroid_dist / total_matches if total_matches > 0 else float('inf'),
        "normalized_centroid_l1": normalized_dist,
        "total_matches": total_matches,
        "per_class": per_class_metrics
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
    
    while open_set:
        _, cx, cy = heappop(open_set)
        
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        
        if cx == gx and cy == gy:
            path = []
            current = (cx, cy)
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 1 and not visited[ny, nx]:
                tentative_g = g_score[cy, cx] + 1
                if tentative_g < g_score[ny, nx]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[ny, nx] = tentative_g
                    f = tentative_g + abs(nx - gx) + abs(ny - gy)
                    heappush(open_set, (f, nx, ny))
    
    return None


def create_traversability_grid(
    class_map: np.ndarray,
    traversable_ids: List[int],
    resolution: Optional[int] = None
) -> np.ndarray:
    """Create binary traversability grid from class map."""
    grid = np.zeros_like(class_map, dtype=np.int32)
    for class_id in traversable_ids:
        grid[class_map == class_id] = 1
    
    if resolution is not None and resolution != class_map.shape[0]:
        grid_img = Image.fromarray((grid * 255).astype(np.uint8))
        grid_resized = grid_img.resize((resolution, resolution), Image.NEAREST)
        grid = (np.array(grid_resized) > 127).astype(np.int32)
    
    return grid


def find_nearest_walkable(
    grid: np.ndarray,
    position: Tuple[int, int],
    max_search_radius: int = 10
) -> Optional[Tuple[int, int]]:
    """
    Find nearest walkable cell to a position.
    Used to find path goals near object centroids.
    
    Args:
        grid: Traversability grid (1=walkable, 0=obstacle)
        position: Target position (x, y)
        max_search_radius: Max distance to search
    
    Returns:
        (x, y) of nearest walkable cell, or None if not found
    """
    h, w = grid.shape
    x, y = int(position[0]), int(position[1])
    
    # Check if position itself is walkable
    if 0 <= x < w and 0 <= y < h and grid[y, x] == 1:
        return (x, y)
    
    # Search in expanding squares
    for radius in range(1, max_search_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue  # Only check perimeter
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 1:
                    return (nx, ny)
    
    return None


def compute_path_overlap(path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]) -> float:
    """
    Compute overlap ratio between two paths.
    Returns fraction of path1 cells within 1 cell of any path2 cell.
    """
    if not path1 or not path2:
        return 0.0
    
    path2_expanded = set()
    for x, y in path2:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                path2_expanded.add((x + dx, y + dy))
    
    overlap = sum(1 for p in path1 if p in path2_expanded)
    return overlap / len(path1)


def compute_path_metrics(
    pred_class_map: np.ndarray,
    target_class_map: np.ndarray,
    pred_blobs: Dict[int, List[Dict]],
    target_blobs: Dict[int, List[Dict]],
    traversable_ids: List[int],
    start_position: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Compute path-based metrics.
    
    For each object class that exists in BOTH target and prediction,
    compare A* paths from start to matched object centroids.
    
    Args:
        pred_class_map: Predicted class map (H, W)
        target_class_map: Target class map (H, W)
        pred_blobs: Detected objects in prediction {class_id: [blob_info, ...]}
        target_blobs: Detected objects in target {class_id: [blob_info, ...]}
        traversable_ids: Class IDs that are walkable (Floor, Door, Window)
        start_position: Starting position (x, y). Default: center-bottom (door position)
    """
    h, w = target_class_map.shape
    
    pred_grid = create_traversability_grid(pred_class_map, traversable_ids)
    target_grid = create_traversability_grid(target_class_map, traversable_ids)
    
    # Default start: center-bottom (typical door/POV position)
    if start_position is None:
        start_position = (w // 2, h - 2)
    
    # Find matched classes (exist in both target and pred, excluding traversable)
    target_classes = set(target_blobs.keys()) - set(traversable_ids)
    pred_classes = set(pred_blobs.keys()) - set(traversable_ids)
    matched_classes = target_classes & pred_classes
    
    if len(matched_classes) == 0:
        return {
            "path_agreement_rate": 0.0,
            "path_overlap_mean": 0.0,
            "path_length_ratio_mean": 0.0,
            "matched_objects": 0,
            "both_reachable": 0,
            "target_only_reachable": 0,
            "pred_only_reachable": 0,
            "neither_reachable": 0
        }
    
    both_valid = 0
    target_only = 0
    pred_only = 0
    neither = 0
    overlaps = []
    length_ratios = []
    total_matched = 0
    
    for class_id in matched_classes:
        # For each matched class, pair up blobs by proximity
        t_blobs = target_blobs[class_id]
        p_blobs = pred_blobs[class_id]
        
        # Simple greedy matching: for each target blob, find closest pred blob
        used_pred = set()
        for t_blob in t_blobs:
            t_centroid = t_blob["centroid"]
            
            # Find closest unused pred blob
            best_p_idx = None
            best_dist = float('inf')
            for p_idx, p_blob in enumerate(p_blobs):
                if p_idx in used_pred:
                    continue
                p_centroid = p_blob["centroid"]
                dist = abs(t_centroid[0] - p_centroid[0]) + abs(t_centroid[1] - p_centroid[1])
                if dist < best_dist:
                    best_dist = dist
                    best_p_idx = p_idx
            
            if best_p_idx is None:
                continue
            
            used_pred.add(best_p_idx)
            p_blob = p_blobs[best_p_idx]
            total_matched += 1
            
            # Find walkable goals near each centroid
            target_goal = find_nearest_walkable(target_grid, t_blob["centroid"])
            pred_goal = find_nearest_walkable(pred_grid, p_blob["centroid"])
            
            if target_goal is None or pred_goal is None:
                neither += 1
                continue
            
            # Compute paths
            target_path = astar_path(target_grid, start_position, target_goal)
            pred_path = astar_path(pred_grid, start_position, pred_goal)
            
            target_exists = target_path is not None
            pred_exists = pred_path is not None
            
            if target_exists and pred_exists:
                both_valid += 1
                overlap = compute_path_overlap(pred_path, target_path)
                overlaps.append(overlap)
                if len(target_path) > 0:
                    length_ratios.append(len(pred_path) / len(target_path))
            elif target_exists:
                target_only += 1
            elif pred_exists:
                pred_only += 1
            else:
                neither += 1
    
    total = total_matched if total_matched > 0 else 1
    agreement = (both_valid + neither) / total
    
    return {
        "path_agreement_rate": float(agreement),
        "path_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "path_length_ratio_mean": float(np.mean(length_ratios)) if length_ratios else 0.0,
        "matched_objects": total_matched,
        "both_reachable": both_valid,
        "target_only_reachable": target_only,
        "pred_only_reachable": pred_only,
        "neither_reachable": neither
    }


# =============================================================================
# Main Evaluator Class
# =============================================================================

class FloorplanEvaluator:
    """
    Complete evaluator for floorplan generation.
    
    Usage:
        evaluator = FloorplanEvaluator(taxonomy_path)
        metrics = evaluator.evaluate(pred_rgb, target_rgb)
    """
    
    def __init__(
        self,
        taxonomy_path: Path,
        traversable_names: Optional[List[str]] = None,
        exclude_names: Optional[List[str]] = None,
        min_blob_pixels: int = 10
    ):
        self.taxonomy_info = load_taxonomy(taxonomy_path)
        self.min_blob_pixels = min_blob_pixels
        
        # Traversable classes for pathfinding (can walk on these)
        if traversable_names is None:
            traversable_names = ["Floor", "Door", "Window"]
        self.traversable_ids = []
        for name in traversable_names:
            class_id = self.taxonomy_info["name_to_id"].get(name)
            if class_id is not None:
                self.traversable_ids.append(class_id)
        
        # Excluded classes for blob detection  
        if exclude_names is None:
            exclude_names = ["Background", "Floor", "Wall"]
        self.exclude_ids = []
        for name in exclude_names:
            class_id = self.taxonomy_info["name_to_id"].get(name)
            if class_id is not None:
                self.exclude_ids.append(class_id)
    
    def clean_and_extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Clean image and extract blobs."""
        cleaned_rgb, class_map = clean_image(image, self.taxonomy_info)
        blobs = extract_blobs(class_map, self.exclude_ids, self.min_blob_pixels)
        return cleaned_rgb, class_map, blobs
    
    def evaluate(
        self,
        pred_rgb: np.ndarray,
        target_rgb: np.ndarray,
        compute_paths: bool = True
    ) -> Dict[str, Any]:
        """Evaluate predicted image against target."""
        pred_cleaned, pred_class_map, pred_blobs = self.clean_and_extract(pred_rgb)
        target_cleaned, target_class_map, target_blobs = self.clean_and_extract(target_rgb)
        
        results = {}
        
        # Palette metrics
        palette = compute_palette_metrics(
            pred_blobs, target_blobs,
            self.taxonomy_info["id_to_name"]
        )
        results["palette"] = palette
        
        # Object count metrics
        counts = compute_object_count_metrics(pred_blobs, target_blobs)
        results["object_counts"] = counts
        
        # Blob matching metrics
        image_size = pred_rgb.shape[0]
        matching = compute_blob_matching_metrics(pred_blobs, target_blobs, image_size)
        results["blob_matching"] = matching
        
        # Path metrics
        if compute_paths:
            paths = compute_path_metrics(
                pred_class_map, target_class_map,
                pred_blobs, target_blobs,
                self.traversable_ids
            )
            results["paths"] = paths
        
        # Summary scores
        results["summary"] = {
            "palette_jaccard": palette["palette_jaccard"],
            "object_count_accuracy": counts["object_count_accuracy"],
            "mean_bbox_iou": matching["mean_bbox_iou"],
            "normalized_centroid_l1": matching["normalized_centroid_l1"],
        }
        if compute_paths:
            results["summary"]["path_agreement"] = results["paths"]["path_agreement_rate"]
            results["summary"]["path_overlap"] = results["paths"]["path_overlap_mean"]
            # Object reachability: fraction of matched objects reachable in both
            matched = results["paths"]["matched_objects"]
            if matched > 0:
                reachable = results["paths"]["both_reachable"]
                results["summary"]["object_reachability"] = reachable / matched
            else:
                results["summary"]["object_reachability"] = 0.0
        
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
        
        for pred, target in zip(pred_images, target_images):
            metrics = self.evaluate(pred, target, compute_paths)
            all_metrics.append(metrics["summary"])
        
        aggregated = {}
        keys = all_metrics[0].keys()
        
        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        
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
        
        if arr.min() < 0:
            arr = (arr + 1) / 2
        
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr
    
    return tensor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    
    evaluator = FloorplanEvaluator(args.taxonomy)
    
    pred_img = np.array(Image.open(args.pred).convert("RGB"))
    target_img = np.array(Image.open(args.target).convert("RGB"))
    
    metrics = evaluator.evaluate(pred_img, target_img)
    
    print("\n=== Evaluation Results ===")
    print(f"Palette Jaccard: {metrics['summary']['palette_jaccard']:.3f}")
    print(f"Object Count Accuracy: {metrics['summary']['object_count_accuracy']:.3f}")
    print(f"Mean BBox IoU: {metrics['summary']['mean_bbox_iou']:.3f}")
    print(f"Normalized Centroid L1: {metrics['summary']['normalized_centroid_l1']:.3f}")
    if "path_agreement" in metrics['summary']:
        print(f"Path Agreement: {metrics['summary']['path_agreement']:.3f}")
        print(f"Path Overlap: {metrics['summary']['path_overlap']:.3f}")