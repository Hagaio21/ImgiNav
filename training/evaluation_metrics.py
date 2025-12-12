#!/usr/bin/env python3
"""
Evaluation metrics for floorplan generation with supercategory-level analysis.

Metrics computed at SUPERCATEGORY level:
1. Structure: Floor IoU, Wall IoU (separate)
2. Openings: Combined Door + Window IoU
3. Furniture supercategories: Bed, Cabinet/Shelf/Desk, Chair, Lighting, Pier/Stool, Sofa, Table, Others
   - Per-supercategory IoU
   - Presence accuracy (correct prediction of supercategory existence)
   - Count accuracy (correct number of instances)
   - L1 distance from camera (for matched instances)

Visualization:
- 80° FOV beam on target image (POV conditioning)
- Legend with supercategory names from taxonomy
- Camera position marker
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy import ndimage
from dataclasses import dataclass, field
import math
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NormalizedObject:
    """Object representation with normalized coordinates and camera-relative position."""
    class_id: int
    class_name: str
    supercategory: str
    centroid: Tuple[float, float]
    bbox: Dict[str, int]
    pixel_count: int
    mask: np.ndarray
    centroid_norm: Tuple[float, float]
    bbox_norm: Dict[str, float]
    density: float
    # Camera-relative metrics
    camera_distance: float = 0.0  # L1 distance from camera (normalized)
    camera_angle: float = 0.0  # Angle from camera (radians)
    camera_x_offset: float = 0.0  # Horizontal offset from camera (normalized)
    camera_y_offset: float = 0.0  # Vertical offset from camera (normalized)


# =============================================================================
# Taxonomy Loading and Mappings
# =============================================================================

def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy and build color/supercategory mappings."""
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)
    
    categories = taxonomy.get("categories", [])
    supercategories = taxonomy.get("supercategories", [])
    category_to_super = taxonomy.get("category_to_super", {})
    category_to_color = taxonomy.get("category_to_color", {})
    supercategory_to_color = taxonomy.get("supercategory_to_color", {})
    color_to_category = taxonomy.get("color_to_category", {})
    
    # Build mappings
    color_to_name = {}
    name_to_color = {}
    name_to_id = {"Background": 0}
    id_to_name = {0: "Background"}
    name_to_super = {}
    
    for idx, cat in enumerate(categories):
        class_id = idx + 1
        name_to_id[cat] = class_id
        id_to_name[class_id] = cat
        
        if cat in category_to_color:
            color = tuple(category_to_color[cat])
            color_to_name[color] = cat
            name_to_color[cat] = color
        
        # Map category to supercategory
        if cat in category_to_super:
            name_to_super[cat] = category_to_super[cat]
        else:
            name_to_super[cat] = "Others"
    
    # Background and structural classes
    color_to_name[(255, 255, 255)] = "Background"
    name_to_color["Background"] = (255, 255, 255)
    name_to_super["Background"] = "Background"
    
    # Convert supercategory colors
    super_to_color = {}
    for super_name, color in supercategory_to_color.items():
        super_to_color[super_name] = tuple(color)
    
    return {
        "categories": categories,
        "supercategories": supercategories,
        "color_to_name": color_to_name,
        "name_to_color": name_to_color,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "name_to_super": name_to_super,
        "super_to_color": super_to_color,
        "category_to_super": category_to_super,
        "num_classes": len(categories) + 1
    }


def get_supercategory(class_name: str, taxonomy_info: Dict) -> str:
    """Get supercategory for a class name."""
    return taxonomy_info["name_to_super"].get(class_name, "Others")


# =============================================================================
# Image Cleaning
# =============================================================================

def clean_image(
    image: np.ndarray,
    taxonomy_info: Dict,
    white_threshold: int = 250
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean image by snapping each pixel to closest taxonomy color."""
    h, w = image.shape[:2]
    name_to_color = taxonomy_info["name_to_color"]
    name_to_id = taxonomy_info["name_to_id"]
    
    category_colors = {}
    for name, color in name_to_color.items():
        category_colors[name] = np.array(color, dtype=np.float32)
    
    is_white = np.all(image > white_threshold, axis=2)
    
    cleaned_rgb = np.full_like(image, 255, dtype=np.uint8)
    class_map = np.zeros((h, w), dtype=np.int32)
    
    non_white_mask = ~is_white
    if np.any(non_white_mask):
        y_coords, x_coords = np.where(non_white_mask)
        pixels = image[y_coords, x_coords].astype(np.float32)
        
        best_names = np.full(len(pixels), "Background", dtype=object)
        best_dist = np.full(len(pixels), np.inf, dtype=np.float32)
        
        for name, color in category_colors.items():
            if name == "Background":
                continue
            dists = np.linalg.norm(pixels - color, axis=1)
            closer = dists < best_dist
            best_names[closer] = name
            best_dist[closer] = dists[closer]
        
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
# Object Extraction
# =============================================================================

def get_room_bbox(class_map: np.ndarray, background_id: int = 0) -> Dict[str, int]:
    """Get bounding box of non-background area."""
    non_bg = class_map != background_id
    if not non_bg.any():
        h, w = class_map.shape
        return {'min_x': 0, 'min_y': 0, 'max_x': w, 'max_y': h}
    
    y_coords, x_coords = np.where(non_bg)
    return {
        'min_x': int(x_coords.min()),
        'min_y': int(y_coords.min()),
        'max_x': int(x_coords.max()),
        'max_y': int(y_coords.max())
    }


def get_camera_position(resolution: int = 512, margin: int = 15) -> Tuple[float, float]:
    """Get camera position (bottom-center as per POV rendering)."""
    return (resolution / 2, resolution - margin)


def extract_objects(
    class_map: np.ndarray,
    taxonomy_info: Dict,
    exclude_classes: Optional[List[str]] = None,
    min_pixels: int = 10,
    camera_position: Optional[Tuple[float, float]] = None
) -> Dict[str, List[NormalizedObject]]:
    """Extract objects grouped by SUPERCATEGORY with camera-relative positions."""
    if exclude_classes is None:
        exclude_classes = ["Background"]
    
    id_to_name = taxonomy_info["id_to_name"]
    name_to_super = taxonomy_info["name_to_super"]
    
    h, w = class_map.shape
    
    room_bbox = get_room_bbox(class_map)
    room_width = max(room_bbox['max_x'] - room_bbox['min_x'], 1)
    room_height = max(room_bbox['max_y'] - room_bbox['min_y'], 1)
    room_diagonal = math.sqrt(room_width**2 + room_height**2)
    
    if camera_position is None:
        camera_position = get_camera_position(resolution=max(h, w))
    
    cam_x, cam_y = camera_position
    
    # Normalize camera position
    cam_x_norm = (cam_x - room_bbox['min_x']) / room_width
    cam_y_norm = (cam_y - room_bbox['min_y']) / room_height
    
    objects_by_super = defaultdict(list)
    unique_classes = np.unique(class_map)
    
    exclude_ids = set()
    for name in exclude_classes:
        if name in taxonomy_info["name_to_id"]:
            exclude_ids.add(taxonomy_info["name_to_id"][name])
    
    for class_id in unique_classes:
        if class_id in exclude_ids or class_id == 0:
            continue
        
        class_name = id_to_name.get(class_id, str(class_id))
        supercategory = name_to_super.get(class_name, "Others")
        
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
            
            bbox_w = max(bbox['max_x'] - bbox['min_x'], 1)
            bbox_h = max(bbox['max_y'] - bbox['min_y'], 1)
            bbox_area = bbox_w * bbox_h
            
            centroid_norm = (
                (centroid[0] - room_bbox['min_x']) / room_width,
                (centroid[1] - room_bbox['min_y']) / room_height
            )
            
            bbox_norm = {
                'min_x': (bbox['min_x'] - room_bbox['min_x']) / room_width,
                'min_y': (bbox['min_y'] - room_bbox['min_y']) / room_height,
                'max_x': (bbox['max_x'] - room_bbox['min_x']) / room_width,
                'max_y': (bbox['max_y'] - room_bbox['min_y']) / room_height
            }
            
            density = pixel_count / bbox_area
            
            # Camera-relative metrics (L1 distance)
            dx = centroid[0] - cam_x
            dy = centroid[1] - cam_y
            
            # Normalized offsets
            camera_x_offset = dx / room_width
            camera_y_offset = dy / room_height
            
            # L1 distance (normalized)
            camera_distance = abs(camera_x_offset) + abs(camera_y_offset)
            
            # Angle from camera
            camera_angle = math.atan2(dx, -dy)  # 0 = straight ahead, positive = right
            
            obj = NormalizedObject(
                class_id=int(class_id),
                class_name=class_name,
                supercategory=supercategory,
                centroid=centroid,
                bbox=bbox,
                pixel_count=int(pixel_count),
                mask=component,
                centroid_norm=centroid_norm,
                bbox_norm=bbox_norm,
                density=density,
                camera_distance=camera_distance,
                camera_angle=camera_angle,
                camera_x_offset=camera_x_offset,
                camera_y_offset=camera_y_offset
            )
            
            objects_by_super[supercategory].append(obj)
    
    return dict(objects_by_super)


# =============================================================================
# Supercategory-Level Metrics
# =============================================================================

def compute_supercategory_presence(
    pred_objects: Dict[str, List[NormalizedObject]],
    target_objects: Dict[str, List[NormalizedObject]],
    supercategories: List[str]
) -> Dict[str, Any]:
    """Compute presence accuracy per supercategory."""
    pred_supers = set(pred_objects.keys())
    target_supers = set(target_objects.keys())
    
    per_super = {}
    correct = 0
    total = 0
    
    for super_name in supercategories:
        if super_name in ["Structure", "Background"]:
            continue
        
        pred_present = super_name in pred_supers and len(pred_objects.get(super_name, [])) > 0
        target_present = super_name in target_supers and len(target_objects.get(super_name, [])) > 0
        
        is_correct = pred_present == target_present
        per_super[super_name] = {
            'pred_present': pred_present,
            'target_present': target_present,
            'correct': is_correct
        }
        
        if target_present or pred_present:
            total += 1
            if is_correct:
                correct += 1
    
    accuracy = correct / total if total > 0 else 1.0
    
    return {
        'presence_accuracy': float(accuracy),
        'num_correct': correct,
        'num_total': total,
        'per_supercategory': per_super
    }


def compute_supercategory_counts(
    pred_objects: Dict[str, List[NormalizedObject]],
    target_objects: Dict[str, List[NormalizedObject]],
    supercategories: List[str]
) -> Dict[str, Any]:
    """Compute count accuracy per supercategory."""
    per_super = {}
    total_correct = 0
    total_classes = 0
    total_abs_error = 0
    
    for super_name in supercategories:
        if super_name in ["Structure", "Background"]:
            continue
        
        pred_count = len(pred_objects.get(super_name, []))
        target_count = len(target_objects.get(super_name, []))
        
        if pred_count == 0 and target_count == 0:
            continue
        
        is_correct = pred_count == target_count
        abs_error = abs(pred_count - target_count)
        
        per_super[super_name] = {
            'pred_count': pred_count,
            'target_count': target_count,
            'correct': is_correct,
            'abs_error': abs_error
        }
        
        total_classes += 1
        if is_correct:
            total_correct += 1
        total_abs_error += abs_error
    
    count_accuracy = total_correct / total_classes if total_classes > 0 else 1.0
    mae = total_abs_error / total_classes if total_classes > 0 else 0.0
    
    return {
        'count_accuracy': float(count_accuracy),
        'count_mae': float(mae),
        'num_correct': total_correct,
        'num_total': total_classes,
        'per_supercategory': per_super
    }


def match_objects_by_centroid(
    pred_objects: List[NormalizedObject],
    target_objects: List[NormalizedObject],
    max_distance: float = 0.5
) -> List[Tuple[int, int, float]]:
    """Match objects by normalized centroid distance (greedy)."""
    if len(pred_objects) == 0 or len(target_objects) == 0:
        return []
    
    distances = []
    for i, pred in enumerate(pred_objects):
        for j, target in enumerate(target_objects):
            dx = pred.centroid_norm[0] - target.centroid_norm[0]
            dy = pred.centroid_norm[1] - target.centroid_norm[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist <= max_distance:
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


def normalized_bbox_iou(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    """Compute IoU between normalized bboxes."""
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


def compute_supercategory_spatial_metrics(
    pred_objects: Dict[str, List[NormalizedObject]],
    target_objects: Dict[str, List[NormalizedObject]],
    supercategories: List[str]
) -> Dict[str, Any]:
    """Compute spatial metrics (IoU, L1 distance) per supercategory."""
    per_super = {}
    all_bbox_ious = []
    all_l1_distances = []
    all_matches = []
    
    total_matched = 0
    total_pred = 0
    total_target = 0
    
    for super_name in supercategories:
        if super_name in ["Structure", "Background"]:
            continue
        
        pred_list = pred_objects.get(super_name, [])
        target_list = target_objects.get(super_name, [])
        
        n_pred = len(pred_list)
        n_target = len(target_list)
        total_pred += n_pred
        total_target += n_target
        
        if n_pred == 0 and n_target == 0:
            continue
        
        if n_pred == 0 or n_target == 0:
            per_super[super_name] = {
                'n_pred': n_pred,
                'n_target': n_target,
                'n_matched': 0,
                'mean_bbox_iou': 0.0,
                'mean_l1_distance': 0.0,
                'detection_precision': 0.0 if n_pred > 0 else 1.0,
                'detection_recall': 0.0 if n_target > 0 else 1.0,
            }
            continue
        
        matches = match_objects_by_centroid(pred_list, target_list)
        n_matched = len(matches)
        total_matched += n_matched
        
        super_bbox_ious = []
        super_l1_distances = []
        
        for pred_idx, target_idx, _ in matches:
            pred_obj = pred_list[pred_idx]
            target_obj = target_list[target_idx]
            
            all_matches.append((pred_obj, target_obj, super_name))
            
            # BBox IoU
            bbox_iou = normalized_bbox_iou(pred_obj.bbox_norm, target_obj.bbox_norm)
            super_bbox_ious.append(bbox_iou)
            all_bbox_ious.append(bbox_iou)
            
            # L1 distance error (difference in camera-relative positions)
            l1_error = (
                abs(pred_obj.camera_x_offset - target_obj.camera_x_offset) +
                abs(pred_obj.camera_y_offset - target_obj.camera_y_offset)
            )
            super_l1_distances.append(l1_error)
            all_l1_distances.append(l1_error)
        
        per_super[super_name] = {
            'n_pred': n_pred,
            'n_target': n_target,
            'n_matched': n_matched,
            'mean_bbox_iou': float(np.mean(super_bbox_ious)) if super_bbox_ious else 0.0,
            'mean_l1_distance': float(np.mean(super_l1_distances)) if super_l1_distances else 0.0,
            'detection_precision': n_matched / n_pred if n_pred > 0 else 1.0,
            'detection_recall': n_matched / n_target if n_target > 0 else 1.0,
        }
    
    detection_precision = total_matched / total_pred if total_pred > 0 else 1.0
    detection_recall = total_matched / total_target if total_target > 0 else 1.0
    detection_f1 = (
        2 * detection_precision * detection_recall / (detection_precision + detection_recall)
    ) if (detection_precision + detection_recall) > 0 else 0.0
    
    return {
        'detection_precision': float(detection_precision),
        'detection_recall': float(detection_recall),
        'detection_f1': float(detection_f1),
        'total_matched': total_matched,
        'total_pred': total_pred,
        'total_target': total_target,
        'mean_bbox_iou': float(np.mean(all_bbox_ious)) if all_bbox_ious else 0.0,
        'mean_l1_distance': float(np.mean(all_l1_distances)) if all_l1_distances else 0.0,
        'per_supercategory': per_super,
        '_matches': all_matches
    }


def compute_pixel_metrics_supercategory(
    pred_class_map: np.ndarray,
    target_class_map: np.ndarray,
    taxonomy_info: Dict
) -> Dict[str, Any]:
    """Compute pixel-level IoU metrics at supercategory level."""
    id_to_name = taxonomy_info["id_to_name"]
    name_to_super = taxonomy_info["name_to_super"]
    name_to_id = taxonomy_info["name_to_id"]
    
    # Create supercategory maps
    h, w = pred_class_map.shape
    pred_super_map = np.zeros((h, w), dtype=object)
    target_super_map = np.zeros((h, w), dtype=object)
    
    pred_super_map[:] = "Background"
    target_super_map[:] = "Background"
    
    for class_id in np.unique(pred_class_map):
        if class_id == 0:
            continue
        class_name = id_to_name.get(class_id, "Unknown")
        super_name = name_to_super.get(class_name, "Others")
        mask = pred_class_map == class_id
        pred_super_map[mask] = super_name
    
    for class_id in np.unique(target_class_map):
        if class_id == 0:
            continue
        class_name = id_to_name.get(class_id, "Unknown")
        super_name = name_to_super.get(class_name, "Others")
        mask = target_class_map == class_id
        target_super_map[mask] = super_name
    
    # Compute per-supercategory IoU
    all_supers = set(pred_super_map.flatten()) | set(target_super_map.flatten())
    per_super_iou = {}
    
    for super_name in all_supers:
        if super_name == "Background":
            continue
        
        pred_mask = pred_super_map == super_name
        target_mask = target_super_map == super_name
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        iou = float(intersection / union) if union > 0 else 0.0
        per_super_iou[super_name] = iou
    
    # Structural IoUs (Floor, Wall separate)
    structural_iou = {}
    for struct_class in ["Floor", "Wall"]:
        class_id = name_to_id.get(struct_class)
        if class_id is not None:
            pred_mask = pred_class_map == class_id
            target_mask = target_class_map == class_id
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            structural_iou[struct_class] = float(intersection / union) if union > 0 else 0.0
    
    # Openings IoU (Door + Window combined)
    door_id = name_to_id.get("Door")
    window_id = name_to_id.get("Window")
    
    pred_openings = np.zeros((h, w), dtype=bool)
    target_openings = np.zeros((h, w), dtype=bool)
    
    if door_id is not None:
        pred_openings |= (pred_class_map == door_id)
        target_openings |= (target_class_map == door_id)
    if window_id is not None:
        pred_openings |= (pred_class_map == window_id)
        target_openings |= (target_class_map == window_id)
    
    intersection = np.logical_and(pred_openings, target_openings).sum()
    union = np.logical_or(pred_openings, target_openings).sum()
    openings_iou = float(intersection / union) if union > 0 else 0.0
    
    # Mean IoU over furniture supercategories (excluding Structure, Background)
    furniture_ious = [
        iou for name, iou in per_super_iou.items()
        if name not in ["Structure", "Background", "Openings"]
    ]
    mean_furniture_iou = float(np.mean(furniture_ious)) if furniture_ious else 0.0
    
    return {
        'floor_iou': structural_iou.get("Floor", 0.0),
        'wall_iou': structural_iou.get("Wall", 0.0),
        'openings_iou': openings_iou,
        'mean_furniture_iou': mean_furniture_iou,
        'per_supercategory_iou': per_super_iou,
        'structural_iou': structural_iou
    }


def compute_unified_score(
    presence_metrics: Dict,
    count_metrics: Dict,
    spatial_metrics: Dict,
    pixel_metrics: Dict,
    is_empty: bool = False,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Compute unified score with different weights for empty vs furnished rooms."""
    if is_empty:
        # Empty rooms: focus on structural metrics
        if weights is None:
            weights = {
                'floor_iou': 0.35,
                'wall_iou': 0.35,
                'openings_iou': 0.30,
            }
        
        score = (
            weights['floor_iou'] * pixel_metrics['floor_iou'] +
            weights['wall_iou'] * pixel_metrics['wall_iou'] +
            weights['openings_iou'] * pixel_metrics['openings_iou']
        )
    else:
        # Furnished rooms: balanced metrics
        if weights is None:
            weights = {
                'floor_iou': 0.10,
                'wall_iou': 0.10,
                'openings_iou': 0.10,
                'presence_accuracy': 0.15,
                'count_accuracy': 0.15,
                'detection_f1': 0.15,
                'mean_bbox_iou': 0.10,
                'l1_accuracy': 0.15,
            }
        
        # Convert L1 distance to accuracy (lower is better)
        l1_accuracy = max(0.0, 1.0 - spatial_metrics['mean_l1_distance'])
        
        score = (
            weights['floor_iou'] * pixel_metrics['floor_iou'] +
            weights['wall_iou'] * pixel_metrics['wall_iou'] +
            weights['openings_iou'] * pixel_metrics['openings_iou'] +
            weights['presence_accuracy'] * presence_metrics['presence_accuracy'] +
            weights['count_accuracy'] * count_metrics['count_accuracy'] +
            weights['detection_f1'] * spatial_metrics['detection_f1'] +
            weights['mean_bbox_iou'] * spatial_metrics['mean_bbox_iou'] +
            weights['l1_accuracy'] * l1_accuracy
        )
    
    return float(score)


# =============================================================================
# Visualization Functions
# =============================================================================

def draw_fov_beam(
    draw: ImageDraw.Draw,
    camera_position: Tuple[float, float],
    fov_degrees: float = 80.0,
    beam_length: float = 400.0,
    offset_x: int = 0,
    offset_y: int = 0
) -> None:
    """Draw FOV beam (dashed lines) from camera position."""
    cam_x, cam_y = camera_position
    cam_x += offset_x
    cam_y += offset_y
    
    fov_rad = math.radians(fov_degrees)
    half_fov = fov_rad / 2
    
    # Camera faces up (-y direction)
    center_angle = -math.pi / 2
    
    left_angle = center_angle - half_fov
    right_angle = center_angle + half_fov
    
    # End points
    left_x = cam_x + beam_length * math.cos(left_angle)
    left_y = cam_y + beam_length * math.sin(left_angle)
    right_x = cam_x + beam_length * math.cos(right_angle)
    right_y = cam_y + beam_length * math.sin(right_angle)
    
    # Draw dashed lines
    dash_length = 10
    gap_length = 5
    
    for end_x, end_y in [(left_x, left_y), (right_x, right_y)]:
        dx = end_x - cam_x
        dy = end_y - cam_y
        length = math.sqrt(dx**2 + dy**2)
        
        if length == 0:
            continue
        
        dx /= length
        dy /= length
        
        current_dist = 0
        drawing = True
        
        while current_dist < length:
            segment_length = dash_length if drawing else gap_length
            next_dist = min(current_dist + segment_length, length)
            
            if drawing:
                x1 = cam_x + dx * current_dist
                y1 = cam_y + dy * current_dist
                x2 = cam_x + dx * next_dist
                y2 = cam_y + dy * next_dist
                draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=2)
            
            current_dist = next_dist
            drawing = not drawing


def draw_metrics_visualization(
    pred_rgb: np.ndarray,
    target_rgb: np.ndarray,
    pred_objects: Dict[str, List[NormalizedObject]],
    target_objects: Dict[str, List[NormalizedObject]],
    spatial_metrics: Dict,
    camera_position: Tuple[float, float],
    taxonomy_info: Dict,
    show_fov: bool = True,
    fov_degrees: float = 80.0
) -> Image.Image:
    """Create visualization with supercategory legend and FOV beam."""
    h, w = pred_rgb.shape[:2]
    
    super_to_color = taxonomy_info.get("super_to_color", {})
    
    # Canvas: pred | target | overlay
    canvas_width = w * 3 + 40
    canvas_height = h + 140
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Paste images
    pred_img = Image.fromarray(pred_rgb)
    target_img = Image.fromarray(target_rgb)
    
    canvas.paste(pred_img, (0, 60))
    canvas.paste(target_img, (w + 20, 60))
    
    # Create overlay
    pred_alpha = Image.fromarray(pred_rgb).convert("RGBA")
    target_alpha = Image.fromarray(target_rgb).convert("RGBA")
    blended = Image.blend(pred_alpha, target_alpha, 0.5)
    canvas.paste(blended.convert("RGB"), (2 * w + 40, 60))
    
    # Draw FOV beam on target image
    if show_fov:
        draw_fov_beam(draw, camera_position, fov_degrees, beam_length=h * 0.8,
                      offset_x=w + 20, offset_y=60)
    
    # Fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Titles
    draw.text((w // 2 - 40, 10), "Prediction", fill=(0, 0, 0), font=font)
    draw.text((w + 20 + w // 2 - 30, 10), "Target", fill=(0, 0, 0), font=font)
    draw.text((2 * w + 40 + w // 2 - 30, 10), "Overlay", fill=(0, 0, 0), font=font)
    
    if show_fov:
        draw.text((w + 20 + 5, 35), f"FOV: {fov_degrees}°", fill=(100, 100, 100), font=small_font)
    
    # Draw matched object connections
    matches = spatial_metrics.get('_matches', [])
    
    # Use supercategory colors
    for idx, (pred_obj, target_obj, super_name) in enumerate(matches):
        color = super_to_color.get(super_name, (127, 127, 127))
        
        # Draw centroids
        px, py = pred_obj.centroid
        draw.ellipse([px - 4, 60 + py - 4, px + 4, 60 + py + 4], fill=color, outline=(0, 0, 0))
        
        tx, ty = target_obj.centroid
        draw.ellipse([w + 20 + tx - 4, 60 + ty - 4, w + 20 + tx + 4, 60 + ty + 4],
                     fill=color, outline=(0, 0, 0))
        
        # Draw bboxes
        bbox = pred_obj.bbox
        draw.rectangle([bbox['min_x'], 60 + bbox['min_y'], bbox['max_x'], 60 + bbox['max_y']],
                       outline=color, width=2)
        
        bbox = target_obj.bbox
        draw.rectangle([w + 20 + bbox['min_x'], 60 + bbox['min_y'],
                        w + 20 + bbox['max_x'], 60 + bbox['max_y']],
                       outline=color, width=2)
        
        # Connection line on overlay
        draw.line([2 * w + 40 + px, 60 + py, 2 * w + 40 + tx, 60 + ty],
                  fill=color, width=2)
    
    # Draw camera position
    cam_x, cam_y = camera_position
    for offset in [0, w + 20, 2 * w + 40]:
        tri_size = 15
        draw.polygon([
            (offset + cam_x, 60 + cam_y - tri_size),
            (offset + cam_x - tri_size // 2, 60 + cam_y + tri_size // 3),
            (offset + cam_x + tri_size // 2, 60 + cam_y + tri_size // 3)
        ], fill=(0, 0, 0), outline=(255, 255, 255))
    
    # Legend at bottom - supercategory based
    legend_y = h + 70
    draw.text((10, legend_y), "Supercategories:", fill=(0, 0, 0), font=font)
    
    # Get unique supercategories from matches
    matched_supers = set()
    for _, _, super_name in matches:
        matched_supers.add(super_name)
    
    x_pos = 130
    for super_name in sorted(matched_supers):
        color = super_to_color.get(super_name, (127, 127, 127))
        draw.rectangle([x_pos, legend_y, x_pos + 15, legend_y + 15], fill=color, outline=(0, 0, 0))
        draw.text((x_pos + 20, legend_y), super_name[:15], fill=(0, 0, 0), font=small_font)
        x_pos += 130
        if x_pos > canvas_width - 150:
            legend_y += 20
            x_pos = 130
    
    # Camera and FOV legend
    draw.text((10, h + 110), "▲ = Camera    ╱╲ = FOV beam", fill=(0, 0, 0), font=small_font)
    
    return canvas


def create_metrics_summary_image(
    pred_rgb: np.ndarray,
    target_rgb: np.ndarray,
    metrics: Dict,
    sample_label: str = "",
    is_empty: bool = False
) -> Image.Image:
    """Create summary image with metrics text overlay."""
    h, w = pred_rgb.shape[:2]
    
    canvas_width = w * 2 + 20
    canvas_height = h + 250
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Paste images
    canvas.paste(Image.fromarray(pred_rgb), (0, 0))
    canvas.paste(Image.fromarray(target_rgb), (w + 20, 0))
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Labels
    draw.text((w // 2 - 40, h + 5), "Prediction", fill=(0, 0, 0), font=title_font)
    draw.text((w + 20 + w // 2 - 30, h + 5), "Target", fill=(0, 0, 0), font=title_font)
    
    if sample_label:
        draw.text((10, h + 25), sample_label, fill=(0, 0, 128), font=title_font)
    
    summary = metrics.get('summary', {})
    y_pos = h + 50
    col1_x = 10
    col2_x = canvas_width // 2 + 10
    
    # Column 1: Structure metrics
    draw.text((col1_x, y_pos), "Structure:", fill=(0, 0, 0), font=title_font)
    y_pos += 20
    
    items = [
        f"Floor IoU: {summary.get('floor_iou', 0):.3f}",
        f"Wall IoU: {summary.get('wall_iou', 0):.3f}",
        f"Openings IoU: {summary.get('openings_iou', 0):.3f}",
    ]
    
    for item in items:
        draw.text((col1_x, y_pos), item, fill=(50, 50, 50), font=font)
        y_pos += 18
    
    # Column 2: Furniture metrics (if not empty)
    y_pos = h + 50
    if not is_empty:
        draw.text((col2_x, y_pos), "Furniture:", fill=(0, 0, 0), font=title_font)
        y_pos += 20
        
        items = [
            f"Presence Acc: {summary.get('presence_accuracy', 0):.3f}",
            f"Count Acc: {summary.get('count_accuracy', 0):.3f}",
            f"Detection F1: {summary.get('detection_f1', 0):.3f}",
            f"Mean L1 Dist: {summary.get('mean_l1_distance', 0):.3f}",
        ]
        
        for item in items:
            draw.text((col2_x, y_pos), item, fill=(50, 50, 50), font=font)
            y_pos += 18
    
    # Unified score
    unified = summary.get('unified_score', 0)
    score_color = (0, 128, 0) if unified > 0.6 else (200, 100, 0) if unified > 0.4 else (200, 0, 0)
    
    score_y = h + 160
    draw.text((col1_x, score_y), f"UNIFIED SCORE: {unified:.3f}", fill=score_color, font=title_font)
    draw.rectangle([col1_x - 5, score_y - 2, col1_x + 200, score_y + 20], outline=score_color, width=2)
    
    # Per-supercategory breakdown
    per_super_spatial = metrics.get('spatial', {}).get('per_supercategory', {})
    if per_super_spatial and not is_empty:
        breakdown_y = h + 190
        draw.text((col1_x, breakdown_y), "Per-Supercategory:", fill=(0, 0, 0), font=title_font)
        breakdown_y += 18
        
        sorted_supers = sorted(per_super_spatial.items(), key=lambda x: x[1].get('n_target', 0), reverse=True)
        for super_name, super_metrics in sorted_supers[:4]:
            n_t = super_metrics.get('n_target', 0)
            n_m = super_metrics.get('n_matched', 0)
            l1 = super_metrics.get('mean_l1_distance', 0)
            text = f"{super_name[:12]}: T={n_t}, M={n_m}, L1={l1:.2f}"
            draw.text((col1_x, breakdown_y), text, fill=(80, 80, 80), font=font)
            breakdown_y += 16
    
    return canvas


# =============================================================================
# Main Evaluator Class
# =============================================================================

class FloorplanEvaluator:
    """
    Evaluator for floorplan generation with supercategory-level metrics.
    
    Metrics:
    1. Structure: Floor IoU, Wall IoU (separate)
    2. Openings: Combined Door + Window IoU
    3. Per-supercategory: presence, count, IoU, L1 distance
    4. Unified score
    """
    
    def __init__(
        self,
        taxonomy_path: Path,
        exclude_from_objects: Optional[List[str]] = None,
        min_blob_pixels: int = 10,
        resolution: int = 512,
        margin: int = 15,
        fov_degrees: float = 80.0,
        weights: Optional[Dict[str, float]] = None
    ):
        self.taxonomy_path = taxonomy_path
        self.taxonomy_info = load_taxonomy(taxonomy_path)
        self.min_blob_pixels = min_blob_pixels
        self.resolution = resolution
        self.margin = margin
        self.fov_degrees = fov_degrees
        self.weights = weights
        
        # Get supercategories from taxonomy
        self.supercategories = self.taxonomy_info.get("supercategories", [])
        
        if exclude_from_objects is None:
            exclude_from_objects = ["Background"]
        self.exclude_from_objects = exclude_from_objects
    
    def evaluate(
        self,
        pred_rgb: np.ndarray,
        target_rgb: np.ndarray,
        is_empty: bool = False,
        camera_position: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate predicted image against target at supercategory level."""
        
        # Clean images
        pred_cleaned, pred_class_map = clean_image(pred_rgb, self.taxonomy_info)
        target_cleaned, target_class_map = clean_image(target_rgb, self.taxonomy_info)
        
        # Camera position
        if camera_position is None:
            h, w = pred_class_map.shape
            camera_position = get_camera_position(resolution=max(h, w), margin=self.margin)
        
        # Extract objects grouped by supercategory
        pred_objects = extract_objects(
            pred_class_map, self.taxonomy_info,
            self.exclude_from_objects, self.min_blob_pixels, camera_position
        )
        target_objects = extract_objects(
            target_class_map, self.taxonomy_info,
            self.exclude_from_objects, self.min_blob_pixels, camera_position
        )
        
        # Compute pixel-level metrics (always computed)
        pixel_metrics = compute_pixel_metrics_supercategory(
            pred_class_map, target_class_map, self.taxonomy_info
        )
        
        # Compute furniture metrics only for non-empty rooms
        if is_empty:
            presence_metrics = {'presence_accuracy': 1.0, 'per_supercategory': {}}
            count_metrics = {'count_accuracy': 1.0, 'count_mae': 0.0, 'per_supercategory': {}}
            spatial_metrics = {
                'detection_f1': 1.0, 'detection_precision': 1.0, 'detection_recall': 1.0,
                'mean_bbox_iou': 0.0, 'mean_l1_distance': 0.0,
                'total_matched': 0, 'total_pred': 0, 'total_target': 0,
                'per_supercategory': {}, '_matches': []
            }
        else:
            presence_metrics = compute_supercategory_presence(
                pred_objects, target_objects, self.supercategories
            )
            count_metrics = compute_supercategory_counts(
                pred_objects, target_objects, self.supercategories
            )
            spatial_metrics = compute_supercategory_spatial_metrics(
                pred_objects, target_objects, self.supercategories
            )
        
        # Unified score
        unified = compute_unified_score(
            presence_metrics, count_metrics, spatial_metrics, pixel_metrics,
            is_empty=is_empty, weights=self.weights
        )
        
        # Build results
        results = {
            "presence": presence_metrics,
            "counts": count_metrics,
            "spatial": spatial_metrics,
            "pixels": pixel_metrics,
            "is_empty": is_empty,
            "summary": {
                # Structure
                "floor_iou": pixel_metrics["floor_iou"],
                "wall_iou": pixel_metrics["wall_iou"],
                "openings_iou": pixel_metrics["openings_iou"],
                
                # Furniture (0 for empty rooms)
                "presence_accuracy": presence_metrics["presence_accuracy"],
                "count_accuracy": count_metrics["count_accuracy"],
                "detection_f1": spatial_metrics["detection_f1"],
                "detection_recall": spatial_metrics["detection_recall"],
                "detection_precision": spatial_metrics["detection_precision"],
                "mean_bbox_iou": spatial_metrics["mean_bbox_iou"],
                "mean_l1_distance": spatial_metrics["mean_l1_distance"],
                
                # Unified
                "unified_score": unified
            },
            "_cleaned_pred": pred_cleaned,
            "_cleaned_target": target_cleaned,
            "_pred_objects": pred_objects,
            "_target_objects": target_objects,
            "_camera_position": camera_position,
            "_spatial_metrics": spatial_metrics
        }
        
        return results
    
    def create_visualization(
        self,
        pred_rgb: np.ndarray,
        target_rgb: np.ndarray,
        metrics: Dict,
        sample_label: str = "",
        show_fov: bool = True
    ) -> Tuple[Image.Image, Image.Image]:
        """Create visualization images for a sample."""
        pred_objects = metrics.get('_pred_objects', {})
        target_objects = metrics.get('_target_objects', {})
        spatial_metrics = metrics.get('_spatial_metrics', metrics.get('spatial', {}))
        camera_position = metrics.get('_camera_position', get_camera_position())
        is_empty = metrics.get('is_empty', False)
        
        detailed = draw_metrics_visualization(
            pred_rgb, target_rgb,
            pred_objects, target_objects,
            spatial_metrics,
            camera_position,
            self.taxonomy_info,
            show_fov=show_fov,
            fov_degrees=self.fov_degrees
        )
        
        summary = create_metrics_summary_image(
            pred_rgb, target_rgb, metrics, sample_label, is_empty=is_empty
        )
        
        return detailed, summary
    
    def get_supercategory_summary(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across samples, grouped by supercategory."""
        # Separate empty and furnished
        empty_metrics = [m for m in all_metrics if m.get('is_empty', False)]
        furnished_metrics = [m for m in all_metrics if not m.get('is_empty', False)]
        
        def aggregate_list(metrics_list: List[Dict], key_path: str) -> Dict[str, float]:
            """Extract values from nested dict and compute stats."""
            values = []
            for m in metrics_list:
                val = m
                for key in key_path.split('.'):
                    val = val.get(key, {}) if isinstance(val, dict) else None
                    if val is None:
                        break
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
            
            if not values:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
            
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        
        # Overall summary
        summary = {
            'total_samples': len(all_metrics),
            'empty_samples': len(empty_metrics),
            'furnished_samples': len(furnished_metrics),
        }
        
        # Structure metrics (all samples)
        summary['structure'] = {
            'floor_iou': aggregate_list(all_metrics, 'summary.floor_iou'),
            'wall_iou': aggregate_list(all_metrics, 'summary.wall_iou'),
            'openings_iou': aggregate_list(all_metrics, 'summary.openings_iou'),
        }
        
        # Furniture metrics (furnished only)
        if furnished_metrics:
            summary['furniture'] = {
                'presence_accuracy': aggregate_list(furnished_metrics, 'summary.presence_accuracy'),
                'count_accuracy': aggregate_list(furnished_metrics, 'summary.count_accuracy'),
                'detection_f1': aggregate_list(furnished_metrics, 'summary.detection_f1'),
                'mean_l1_distance': aggregate_list(furnished_metrics, 'summary.mean_l1_distance'),
                'mean_bbox_iou': aggregate_list(furnished_metrics, 'summary.mean_bbox_iou'),
            }
            
            # Per-supercategory aggregation
            per_super = defaultdict(lambda: defaultdict(list))
            for m in furnished_metrics:
                spatial_per_super = m.get('spatial', {}).get('per_supercategory', {})
                for super_name, super_metrics in spatial_per_super.items():
                    for metric_name, value in super_metrics.items():
                        if isinstance(value, (int, float)):
                            per_super[super_name][metric_name].append(value)
            
            summary['per_supercategory'] = {}
            for super_name, metrics_dict in per_super.items():
                summary['per_supercategory'][super_name] = {}
                for metric_name, values in metrics_dict.items():
                    if values:
                        summary['per_supercategory'][super_name][metric_name] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'count': len(values)
                        }
        
        # Unified scores
        summary['unified_score'] = {
            'all': aggregate_list(all_metrics, 'summary.unified_score'),
            'empty': aggregate_list(empty_metrics, 'summary.unified_score') if empty_metrics else None,
            'furnished': aggregate_list(furnished_metrics, 'summary.unified_score') if furnished_metrics else None,
        }
        
        return summary


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


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--is-empty", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--fov", type=float, default=80.0)
    args = parser.parse_args()
    
    evaluator = FloorplanEvaluator(args.taxonomy, fov_degrees=args.fov)
    
    pred_img = np.array(Image.open(args.pred).convert("RGB"))
    target_img = np.array(Image.open(args.target).convert("RGB"))
    
    metrics = evaluator.evaluate(pred_img, target_img, is_empty=args.is_empty)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Supercategory Level)")
    print("=" * 70)
    
    print("\n[Structure]")
    print(f"  Floor IoU: {metrics['summary']['floor_iou']:.3f}")
    print(f"  Wall IoU: {metrics['summary']['wall_iou']:.3f}")
    print(f"  Openings IoU: {metrics['summary']['openings_iou']:.3f}")
    
    if not args.is_empty:
        print("\n[Furniture]")
        print(f"  Presence Accuracy: {metrics['summary']['presence_accuracy']:.3f}")
        print(f"  Count Accuracy: {metrics['summary']['count_accuracy']:.3f}")
        print(f"  Detection F1: {metrics['summary']['detection_f1']:.3f}")
        print(f"  Mean L1 Distance: {metrics['summary']['mean_l1_distance']:.3f}")
        print(f"  Mean BBox IoU: {metrics['summary']['mean_bbox_iou']:.3f}")
        
        print("\n[Per-Supercategory]")
        for super_name, super_metrics in metrics['spatial']['per_supercategory'].items():
            if super_metrics['n_target'] > 0 or super_metrics['n_pred'] > 0:
                print(f"  {super_name:20s}: T={super_metrics['n_target']}, "
                      f"P={super_metrics['n_pred']}, M={super_metrics['n_matched']}, "
                      f"L1={super_metrics.get('mean_l1_distance', 0):.2f}")
    
    print("\n" + "=" * 70)
    print(f"UNIFIED SCORE: {metrics['summary']['unified_score']:.3f}")
    print("=" * 70)
    
    if args.visualize:
        detailed, summary = evaluator.create_visualization(
            pred_img, target_img, metrics, "Test Sample", show_fov=True
        )
        detailed.save("detailed_viz.png")
        summary.save("summary_viz.png")
        print("\nSaved visualizations: detailed_viz.png, summary_viz.png")
    
    if args.output:
        output_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        with open(args.output, "w") as f:
            json.dump(output_metrics, f, indent=2)
        print(f"\nSaved to {args.output}")
