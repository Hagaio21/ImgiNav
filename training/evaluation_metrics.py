#!/usr/bin/env python3
"""
Evaluation metrics for floorplan generation with navigation-focused metrics.

Metrics computed:
1. Class presence: precision, recall, F1 for which classes appear
2. Per-class count: exact match, count accuracy per class
3. Scale-invariant spatial: normalized bbox IoU, density similarity, centroid accuracy
4. Camera-centric: direction and distance from agent POV
5. Pixel-level: weighted accuracy, structural IoU (walls, floor)
6. Unified navigation score: weighted combination of all above

Visualization:
- Generates diagnostic images showing how metrics were calculated
- Highlights matched objects, centroids, camera position
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy import ndimage
from dataclasses import dataclass
import math
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NormalizedObject:
    """Object representation with normalized coordinates."""
    class_id: int
    class_name: str
    centroid: Tuple[float, float]
    bbox: Dict[str, int]
    pixel_count: int
    mask: np.ndarray
    centroid_norm: Tuple[float, float]
    bbox_norm: Dict[str, float]
    density: float
    camera_angle: Optional[float] = None
    camera_distance: Optional[float] = None


# =============================================================================
# Taxonomy and Image Cleaning
# =============================================================================

def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy and build color mappings."""
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)
    
    categories = taxonomy.get("categories", [])
    category_to_color = taxonomy.get("category_to_color", {})
    
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
# Object Extraction with Normalization
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
    id_to_name: Dict[int, str],
    exclude_classes: Optional[List[int]] = None,
    min_pixels: int = 10,
    camera_position: Optional[Tuple[float, float]] = None
) -> Dict[int, List[NormalizedObject]]:
    """Extract objects with normalized coordinates."""
    if exclude_classes is None:
        exclude_classes = [0]
    
    h, w = class_map.shape
    
    room_bbox = get_room_bbox(class_map)
    room_width = max(room_bbox['max_x'] - room_bbox['min_x'], 1)
    room_height = max(room_bbox['max_y'] - room_bbox['min_y'], 1)
    room_diagonal = math.sqrt(room_width**2 + room_height**2)
    
    if camera_position is None:
        camera_position = get_camera_position(resolution=max(h, w))
    
    objects_by_class = defaultdict(list)
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
            
            dx = centroid[0] - camera_position[0]
            dy = centroid[1] - camera_position[1]
            camera_angle = math.atan2(dx, -dy)
            camera_distance = math.sqrt(dx**2 + dy**2) / room_diagonal
            
            obj = NormalizedObject(
                class_id=int(class_id),
                class_name=id_to_name.get(class_id, str(class_id)),
                centroid=centroid,
                bbox=bbox,
                pixel_count=int(pixel_count),
                mask=component,
                centroid_norm=centroid_norm,
                bbox_norm=bbox_norm,
                density=density,
                camera_angle=camera_angle,
                camera_distance=camera_distance
            )
            
            objects_by_class[int(class_id)].append(obj)
    
    return dict(objects_by_class)


# =============================================================================
# Metric Computations
# =============================================================================

def compute_class_presence(
    pred_objects: Dict[int, List[NormalizedObject]],
    target_objects: Dict[int, List[NormalizedObject]],
    id_to_name: Dict[int, str]
) -> Dict[str, Any]:
    """Compute class presence metrics."""
    pred_classes = set(pred_objects.keys())
    target_classes = set(target_objects.keys())
    
    common = pred_classes & target_classes
    missing = target_classes - pred_classes
    extra = pred_classes - target_classes
    
    precision = len(common) / len(pred_classes) if pred_classes else 1.0
    recall = len(common) / len(target_classes) if target_classes else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'class_precision': float(precision),
        'class_recall': float(recall),
        'class_f1': float(f1),
        'num_common': len(common),
        'num_missing': len(missing),
        'num_extra': len(extra),
        'missing_classes': [id_to_name.get(c, str(c)) for c in missing],
        'extra_classes': [id_to_name.get(c, str(c)) for c in extra]
    }


def compute_count_metrics(
    pred_objects: Dict[int, List[NormalizedObject]],
    target_objects: Dict[int, List[NormalizedObject]],
    id_to_name: Dict[int, str]
) -> Dict[str, Any]:
    """Compute count accuracy metrics."""
    all_classes = set(pred_objects.keys()) | set(target_objects.keys())
    
    per_class = {}
    total_abs_error = 0
    exact_matches = 0
    total_target = 0
    total_pred = 0
    
    for class_id in all_classes:
        target_count = len(target_objects.get(class_id, []))
        pred_count = len(pred_objects.get(class_id, []))
        
        total_target += target_count
        total_pred += pred_count
        
        abs_error = abs(pred_count - target_count)
        total_abs_error += abs_error
        
        exact_match = 1 if pred_count == target_count else 0
        exact_matches += exact_match
        
        class_name = id_to_name.get(class_id, str(class_id))
        per_class[class_name] = {
            'target': target_count,
            'pred': pred_count,
            'diff': pred_count - target_count,
            'exact_match': exact_match
        }
    
    n_classes = len(all_classes) if all_classes else 1
    mae = total_abs_error / n_classes
    exact_match_rate = exact_matches / n_classes if n_classes > 0 else 1.0
    
    max_error = max(total_target, total_pred, 1)
    count_accuracy = max(0.0, 1.0 - (total_abs_error / (2 * max_error)))
    
    return {
        'count_mae': float(mae),
        'count_accuracy': float(count_accuracy),
        'count_exact_match_rate': float(exact_match_rate),
        'total_target_objects': total_target,
        'total_pred_objects': total_pred,
        'per_class': per_class
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


def compute_spatial_metrics(
    pred_objects: Dict[int, List[NormalizedObject]],
    target_objects: Dict[int, List[NormalizedObject]],
    id_to_name: Dict[int, str]
) -> Dict[str, Any]:
    """Compute spatial metrics on matched objects."""
    all_classes = set(pred_objects.keys()) | set(target_objects.keys())
    
    all_bbox_ious = []
    all_density_sims = []
    all_centroid_dists = []
    all_camera_sims = []
    
    total_matched = 0
    total_pred = 0
    total_target = 0
    
    per_class = {}
    all_matches = []  # Store for visualization
    
    for class_id in all_classes:
        pred_list = pred_objects.get(class_id, [])
        target_list = target_objects.get(class_id, [])
        
        n_pred = len(pred_list)
        n_target = len(target_list)
        total_pred += n_pred
        total_target += n_target
        
        class_name = id_to_name.get(class_id, str(class_id))
        
        if n_pred == 0 or n_target == 0:
            per_class[class_name] = {
                'n_pred': n_pred,
                'n_target': n_target,
                'n_matched': 0,
                'detection_precision': 0.0 if n_pred > 0 else 1.0,
                'detection_recall': 0.0 if n_target > 0 else 1.0,
            }
            continue
        
        matches = match_objects_by_centroid(pred_list, target_list)
        n_matched = len(matches)
        total_matched += n_matched
        
        class_bbox_ious = []
        class_density_sims = []
        class_centroid_dists = []
        class_camera_sims = []
        
        for pred_idx, target_idx, _ in matches:
            pred_obj = pred_list[pred_idx]
            target_obj = target_list[target_idx]
            
            all_matches.append((pred_obj, target_obj, class_name))
            
            bbox_iou = normalized_bbox_iou(pred_obj.bbox_norm, target_obj.bbox_norm)
            class_bbox_ious.append(bbox_iou)
            all_bbox_ious.append(bbox_iou)
            
            density_sim = 1.0 - abs(pred_obj.density - target_obj.density)
            class_density_sims.append(density_sim)
            all_density_sims.append(density_sim)
            
            dx = pred_obj.centroid_norm[0] - target_obj.centroid_norm[0]
            dy = pred_obj.centroid_norm[1] - target_obj.centroid_norm[1]
            centroid_dist = math.sqrt(dx**2 + dy**2)
            class_centroid_dists.append(centroid_dist)
            all_centroid_dists.append(centroid_dist)
            
            if pred_obj.camera_angle is not None and target_obj.camera_angle is not None:
                angle_diff = abs(pred_obj.camera_angle - target_obj.camera_angle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                dist_diff = abs(pred_obj.camera_distance - target_obj.camera_distance)
                
                angle_sim = 1.0 - (angle_diff / math.pi)
                dist_sim = 1.0 - min(dist_diff, 1.0)
                camera_sim = 0.5 * angle_sim + 0.5 * dist_sim
                class_camera_sims.append(camera_sim)
                all_camera_sims.append(camera_sim)
        
        per_class[class_name] = {
            'n_pred': n_pred,
            'n_target': n_target,
            'n_matched': n_matched,
            'detection_precision': n_matched / n_pred if n_pred > 0 else 1.0,
            'detection_recall': n_matched / n_target if n_target > 0 else 1.0,
            'mean_bbox_iou': float(np.mean(class_bbox_ious)) if class_bbox_ious else 0.0,
            'mean_density_sim': float(np.mean(class_density_sims)) if class_density_sims else 0.0,
            'mean_centroid_dist': float(np.mean(class_centroid_dists)) if class_centroid_dists else 0.0,
            'mean_camera_sim': float(np.mean(class_camera_sims)) if class_camera_sims else 0.0,
        }
    
    detection_precision = total_matched / total_pred if total_pred > 0 else 1.0
    detection_recall = total_matched / total_target if total_target > 0 else 1.0
    detection_f1 = (2 * detection_precision * detection_recall / 
                    (detection_precision + detection_recall)) if (detection_precision + detection_recall) > 0 else 0.0
    
    return {
        'detection_precision': float(detection_precision),
        'detection_recall': float(detection_recall),
        'detection_f1': float(detection_f1),
        'total_matched': total_matched,
        'total_pred': total_pred,
        'total_target': total_target,
        'mean_bbox_iou': float(np.mean(all_bbox_ious)) if all_bbox_ious else 0.0,
        'mean_density_sim': float(np.mean(all_density_sims)) if all_density_sims else 0.0,
        'mean_centroid_dist': float(np.mean(all_centroid_dists)) if all_centroid_dists else 0.0,
        'mean_centroid_accuracy': float(1.0 - np.mean(all_centroid_dists) / math.sqrt(2)) if all_centroid_dists else 0.0,
        'mean_camera_similarity': float(np.mean(all_camera_sims)) if all_camera_sims else 0.0,
        'per_class': per_class,
        '_matches': all_matches  # For visualization
    }


def compute_pixel_metrics(
    pred_class_map: np.ndarray,
    target_class_map: np.ndarray,
    id_to_name: Dict[int, str],
    structural_classes: List[str] = None
) -> Dict[str, Any]:
    """Compute pixel-level metrics."""
    if structural_classes is None:
        structural_classes = ["Wall", "Floor"]
    
    name_to_id = {v: k for k, v in id_to_name.items()}
    h, w = pred_class_map.shape
    
    correct = (pred_class_map == target_class_map)
    
    # Object-only accuracy
    non_background = (target_class_map != 0)
    if non_background.sum() > 0:
        object_accuracy = float(correct[non_background].sum() / non_background.sum())
    else:
        object_accuracy = 1.0
    
    # Structural IoU
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
    
    # Per-class IoU
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
    
    non_bg_ious = [iou for name, iou in per_class_iou.items() if name != "Background"]
    mean_iou = float(np.mean(non_bg_ious)) if non_bg_ious else 0.0
    
    return {
        'object_pixel_accuracy': object_accuracy,
        'mean_iou': mean_iou,
        'structural_iou': structural_iou,
        'per_class_iou': per_class_iou
    }


def compute_unified_score(
    class_metrics: Dict,
    count_metrics: Dict,
    spatial_metrics: Dict,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Compute unified navigation score."""
    if weights is None:
        weights = {
            'class_f1': 0.10,
            'count_accuracy': 0.15,
            'detection_f1': 0.15,
            'bbox_iou': 0.10,
            'density_sim': 0.05,
            'centroid_accuracy': 0.20,
            'camera_similarity': 0.25,
        }
    
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    score = 0.0
    score += weights['class_f1'] * class_metrics['class_f1']
    score += weights['count_accuracy'] * count_metrics['count_accuracy']
    score += weights['detection_f1'] * spatial_metrics['detection_f1']
    score += weights['bbox_iou'] * spatial_metrics['mean_bbox_iou']
    score += weights['density_sim'] * spatial_metrics['mean_density_sim']
    score += weights['centroid_accuracy'] * spatial_metrics['mean_centroid_accuracy']
    score += weights['camera_similarity'] * spatial_metrics['mean_camera_similarity']
    
    return float(score)


# =============================================================================
# Visualization Functions
# =============================================================================

def draw_metrics_visualization(
    pred_rgb: np.ndarray,
    target_rgb: np.ndarray,
    pred_objects: Dict[int, List[NormalizedObject]],
    target_objects: Dict[int, List[NormalizedObject]],
    spatial_metrics: Dict,
    camera_position: Tuple[float, float],
    name_to_color: Dict[str, Tuple[int, int, int]]
) -> Image.Image:
    """
    Create visualization showing how metrics were calculated.
    
    Shows:
    - Side-by-side pred/target with matched object overlays
    - Centroids and connections between matched pairs
    - Camera position and viewing direction
    - Bounding boxes with IoU annotations
    """
    h, w = pred_rgb.shape[:2]
    
    # Create canvas: pred | target | overlay
    canvas_width = w * 3 + 40
    canvas_height = h + 120  # Extra space for legend
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Paste images
    pred_img = Image.fromarray(pred_rgb)
    target_img = Image.fromarray(target_rgb)
    
    canvas.paste(pred_img, (0, 60))
    canvas.paste(target_img, (w + 20, 60))
    
    # Create overlay image (semi-transparent comparison)
    overlay = Image.new("RGB", (w, h), (255, 255, 255))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw on overlay
    pred_alpha = Image.fromarray(pred_rgb).convert("RGBA")
    target_alpha = Image.fromarray(target_rgb).convert("RGBA")
    
    # Blend for overlay
    blended = Image.blend(pred_alpha, target_alpha, 0.5)
    canvas.paste(blended.convert("RGB"), (2 * w + 40, 60))
    
    # Titles
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    draw.text((w // 2 - 40, 10), "Prediction", fill=(0, 0, 0), font=font)
    draw.text((w + 20 + w // 2 - 30, 10), "Target", fill=(0, 0, 0), font=font)
    draw.text((2 * w + 40 + w // 2 - 30, 10), "Overlay", fill=(0, 0, 0), font=font)
    
    # Draw matched object connections
    matches = spatial_metrics.get('_matches', [])
    
    colors_for_matches = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),
        (128, 0, 128), (0, 128, 128), (255, 192, 203), (139, 69, 19)
    ]
    
    for idx, (pred_obj, target_obj, class_name) in enumerate(matches):
        color = colors_for_matches[idx % len(colors_for_matches)]
        
        # Draw centroids on pred
        px, py = pred_obj.centroid
        draw.ellipse([px - 4, 60 + py - 4, px + 4, 60 + py + 4], fill=color, outline=(0, 0, 0))
        
        # Draw centroids on target
        tx, ty = target_obj.centroid
        draw.ellipse([w + 20 + tx - 4, 60 + ty - 4, w + 20 + tx + 4, 60 + ty + 4], 
                     fill=color, outline=(0, 0, 0))
        
        # Draw bboxes on pred
        bbox = pred_obj.bbox
        draw.rectangle([bbox['min_x'], 60 + bbox['min_y'], bbox['max_x'], 60 + bbox['max_y']],
                       outline=color, width=2)
        
        # Draw bboxes on target
        bbox = target_obj.bbox
        draw.rectangle([w + 20 + bbox['min_x'], 60 + bbox['min_y'], 
                        w + 20 + bbox['max_x'], 60 + bbox['max_y']],
                       outline=color, width=2)
        
        # Connection line on overlay
        draw.line([2 * w + 40 + px, 60 + py, 2 * w + 40 + tx, 60 + ty], 
                  fill=color, width=2)
    
    # Draw camera position on all panels
    cam_x, cam_y = camera_position
    for offset in [0, w + 20, 2 * w + 40]:
        # Camera triangle
        tri_size = 15
        draw.polygon([
            (offset + cam_x, 60 + cam_y - tri_size),
            (offset + cam_x - tri_size // 2, 60 + cam_y + tri_size // 3),
            (offset + cam_x + tri_size // 2, 60 + cam_y + tri_size // 3)
        ], fill=(0, 0, 0), outline=(255, 255, 255))
    
    # Legend at bottom
    legend_y = h + 70
    draw.text((10, legend_y), "Legend:", fill=(0, 0, 0), font=font)
    
    # Draw match legend
    for idx, (pred_obj, target_obj, class_name) in enumerate(matches[:6]):
        color = colors_for_matches[idx % len(colors_for_matches)]
        x_pos = 80 + idx * 120
        draw.rectangle([x_pos, legend_y, x_pos + 15, legend_y + 15], fill=color)
        draw.text((x_pos + 20, legend_y), class_name[:12], fill=(0, 0, 0), font=small_font)
    
    # Camera symbol in legend
    draw.text((10, legend_y + 25), "▲ = Camera", fill=(0, 0, 0), font=small_font)
    
    return canvas


def create_metrics_summary_image(
    pred_rgb: np.ndarray,
    target_rgb: np.ndarray,
    metrics: Dict,
    sample_label: str = ""
) -> Image.Image:
    """Create summary image with metrics text overlay."""
    h, w = pred_rgb.shape[:2]
    
    canvas_width = w * 2 + 20
    canvas_height = h + 200
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
    
    # Title
    if sample_label:
        draw.text((10, h + 25), sample_label, fill=(0, 0, 128), font=title_font)
    
    # Metrics text
    summary = metrics.get('summary', {})
    y_pos = h + 50
    
    col1_x = 10
    col2_x = canvas_width // 2 + 10
    
    # Column 1: Detection & Counts
    draw.text((col1_x, y_pos), "Detection & Counts:", fill=(0, 0, 0), font=title_font)
    y_pos += 20
    
    items = [
        f"Class F1: {summary.get('class_f1', 0):.3f}",
        f"Count Accuracy: {summary.get('count_accuracy', 0):.3f}",
        f"Detection F1: {summary.get('detection_f1', 0):.3f}",
        f"Detection Recall: {summary.get('detection_recall', 0):.3f}",
    ]
    
    for item in items:
        draw.text((col1_x, y_pos), item, fill=(50, 50, 50), font=font)
        y_pos += 18
    
    # Column 2: Spatial & Camera
    y_pos = h + 50
    draw.text((col2_x, y_pos), "Spatial & Camera:", fill=(0, 0, 0), font=title_font)
    y_pos += 20
    
    items = [
        f"BBox IoU: {summary.get('mean_bbox_iou', 0):.3f}",
        f"Centroid Acc: {summary.get('mean_centroid_accuracy', 0):.3f}",
        f"Camera Sim: {summary.get('mean_camera_similarity', 0):.3f}",
        f"Unified Score: {summary.get('unified_score', 0):.3f}",
    ]
    
    for item in items:
        draw.text((col2_x, y_pos), item, fill=(50, 50, 50), font=font)
        y_pos += 18
    
    # Highlight unified score
    unified = summary.get('unified_score', 0)
    score_color = (0, 128, 0) if unified > 0.5 else (200, 100, 0) if unified > 0.3 else (200, 0, 0)
    draw.rectangle([col2_x - 5, h + 50 + 20 + 54 - 2, col2_x + 200, h + 50 + 20 + 72], 
                   outline=score_color, width=2)
    
    return canvas


# =============================================================================
# Main Evaluator Class
# =============================================================================

class FloorplanEvaluator:
    """
    Complete evaluator for floorplan generation with navigation-focused metrics.
    
    Metrics computed:
    1. Class presence: precision, recall, F1
    2. Per-class count: exact match rate, count accuracy
    3. Scale-invariant spatial: normalized bbox IoU, density sim, centroid accuracy
    4. Camera-centric: direction and distance from POV
    5. Pixel-level: object accuracy, structural IoU
    6. Unified navigation score
    """
    
    def __init__(
        self,
        taxonomy_path: Path,
        exclude_from_objects: Optional[List[str]] = None,
        min_blob_pixels: int = 10,
        resolution: int = 512,
        margin: int = 15,
        weights: Optional[Dict[str, float]] = None
    ):
        self.taxonomy_info = load_taxonomy(taxonomy_path)
        self.min_blob_pixels = min_blob_pixels
        self.resolution = resolution
        self.margin = margin
        self.weights = weights
        
        if exclude_from_objects is None:
            exclude_from_objects = ["Background", "Floor", "Wall"]
        self.exclude_object_ids = []
        for name in exclude_from_objects:
            class_id = self.taxonomy_info["name_to_id"].get(name)
            if class_id is not None:
                self.exclude_object_ids.append(class_id)
    
    def evaluate(
        self,
        pred_rgb: np.ndarray,
        target_rgb: np.ndarray,
        compute_paths: bool = False,  # Kept for backwards compatibility
        camera_position: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate predicted image against target."""
        
        # Clean images
        pred_cleaned, pred_class_map = clean_image(pred_rgb, self.taxonomy_info)
        target_cleaned, target_class_map = clean_image(target_rgb, self.taxonomy_info)
        
        # Camera position
        if camera_position is None:
            h, w = pred_class_map.shape
            camera_position = get_camera_position(resolution=max(h, w), margin=self.margin)
        
        # Extract objects
        pred_objects = extract_objects(
            pred_class_map, self.taxonomy_info["id_to_name"],
            self.exclude_object_ids, self.min_blob_pixels, camera_position
        )
        target_objects = extract_objects(
            target_class_map, self.taxonomy_info["id_to_name"],
            self.exclude_object_ids, self.min_blob_pixels, camera_position
        )
        
        # Compute metrics
        class_metrics = compute_class_presence(
            pred_objects, target_objects, self.taxonomy_info["id_to_name"]
        )
        count_metrics = compute_count_metrics(
            pred_objects, target_objects, self.taxonomy_info["id_to_name"]
        )
        spatial_metrics = compute_spatial_metrics(
            pred_objects, target_objects, self.taxonomy_info["id_to_name"]
        )
        pixel_metrics = compute_pixel_metrics(
            pred_class_map, target_class_map, self.taxonomy_info["id_to_name"]
        )
        
        # Unified score
        unified = compute_unified_score(class_metrics, count_metrics, spatial_metrics, self.weights)
        
        # Build results
        results = {
            "class_presence": class_metrics,
            "counts": count_metrics,
            "spatial": spatial_metrics,
            "pixels": pixel_metrics,
            "summary": {
                # Class presence
                "class_f1": class_metrics["class_f1"],
                "class_recall": class_metrics["class_recall"],
                "class_precision": class_metrics["class_precision"],
                
                # Counts
                "count_accuracy": count_metrics["count_accuracy"],
                "count_exact_match_rate": count_metrics["count_exact_match_rate"],
                
                # Detection
                "detection_f1": spatial_metrics["detection_f1"],
                "detection_recall": spatial_metrics["detection_recall"],
                "detection_precision": spatial_metrics["detection_precision"],
                
                # Spatial
                "mean_bbox_iou": spatial_metrics["mean_bbox_iou"],
                "mean_density_sim": spatial_metrics["mean_density_sim"],
                "mean_centroid_dist": spatial_metrics["mean_centroid_dist"],
                "mean_centroid_accuracy": spatial_metrics["mean_centroid_accuracy"],
                
                # Camera-centric
                "mean_camera_similarity": spatial_metrics["mean_camera_similarity"],
                
                # Pixel-level
                "object_pixel_accuracy": pixel_metrics["object_pixel_accuracy"],
                "mean_iou": pixel_metrics["mean_iou"],
                "wall_iou": pixel_metrics["structural_iou"].get("Wall", 0.0),
                "floor_iou": pixel_metrics["structural_iou"].get("Floor", 0.0),
                
                # Unified
                "unified_score": unified
            },
            "_cleaned_pred": pred_cleaned,
            "_cleaned_target": target_cleaned,
            "_pred_objects": pred_objects,
            "_target_objects": target_objects,
            "_camera_position": camera_position,
            "_spatial_metrics": spatial_metrics  # Contains matches for viz
        }
        
        return results
    
    def create_visualization(
        self,
        pred_rgb: np.ndarray,
        target_rgb: np.ndarray,
        metrics: Dict,
        sample_label: str = ""
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Create visualization images for a sample.
        
        Returns:
            (detailed_viz, summary_viz)
        """
        pred_objects = metrics.get('_pred_objects', {})
        target_objects = metrics.get('_target_objects', {})
        spatial_metrics = metrics.get('_spatial_metrics', metrics.get('spatial', {}))
        camera_position = metrics.get('_camera_position', get_camera_position())
        
        detailed = draw_metrics_visualization(
            pred_rgb, target_rgb,
            pred_objects, target_objects,
            spatial_metrics,
            camera_position,
            self.taxonomy_info["name_to_color"]
        )
        
        summary = create_metrics_summary_image(
            pred_rgb, target_rgb, metrics, sample_label
        )
        
        return detailed, summary
    
    def evaluate_batch(
        self,
        pred_images: List[np.ndarray],
        target_images: List[np.ndarray],
        compute_paths: bool = False
    ) -> Dict[str, Any]:
        """Evaluate batch and compute aggregate metrics."""
        all_metrics = []
        all_per_class = defaultdict(lambda: defaultdict(list))
        
        for pred, target in zip(pred_images, target_images):
            metrics = self.evaluate(pred, target, compute_paths)
            all_metrics.append(metrics["summary"])
            
            for class_name, class_metrics in metrics["spatial"]["per_class"].items():
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
        
        # Per-class aggregation
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
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    evaluator = FloorplanEvaluator(args.taxonomy)
    
    pred_img = np.array(Image.open(args.pred).convert("RGB"))
    target_img = np.array(Image.open(args.target).convert("RGB"))
    
    metrics = evaluator.evaluate(pred_img, target_img)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n[Class Presence]")
    print(f"  Class F1: {metrics['summary']['class_f1']:.3f}")
    print(f"  Missing: {metrics['class_presence']['missing_classes']}")
    print(f"  Extra: {metrics['class_presence']['extra_classes']}")
    
    print("\n[Counts]")
    print(f"  Count Accuracy: {metrics['summary']['count_accuracy']:.3f}")
    print(f"  Exact Match Rate: {metrics['summary']['count_exact_match_rate']:.3f}")
    
    print("\n[Detection]")
    print(f"  Detection F1: {metrics['summary']['detection_f1']:.3f}")
    print(f"  Detection Recall: {metrics['summary']['detection_recall']:.3f}")
    
    print("\n[Spatial (Scale-Invariant)]")
    print(f"  BBox IoU: {metrics['summary']['mean_bbox_iou']:.3f}")
    print(f"  Density Sim: {metrics['summary']['mean_density_sim']:.3f}")
    print(f"  Centroid Accuracy: {metrics['summary']['mean_centroid_accuracy']:.3f}")
    
    print("\n[Camera-Centric]")
    print(f"  Camera Similarity: {metrics['summary']['mean_camera_similarity']:.3f}")
    
    print("\n[Pixel-Level]")
    print(f"  Object Pixel Accuracy: {metrics['summary']['object_pixel_accuracy']:.3f}")
    print(f"  Mean IoU: {metrics['summary']['mean_iou']:.3f}")
    print(f"  Wall IoU: {metrics['summary']['wall_iou']:.3f}")
    print(f"  Floor IoU: {metrics['summary']['floor_iou']:.3f}")
    
    print("\n" + "=" * 70)
    print(f"UNIFIED SCORE: {metrics['summary']['unified_score']:.3f}")
    print("=" * 70)
    
    print("\n[Per-Class Breakdown]")
    for class_name, class_metrics in metrics['spatial']['per_class'].items():
        if class_metrics['n_target'] > 0 or class_metrics['n_pred'] > 0:
            print(f"  {class_name:15s}: T={class_metrics['n_target']}, P={class_metrics['n_pred']}, "
                  f"M={class_metrics['n_matched']}, IoU={class_metrics.get('mean_bbox_iou', 0):.2f}")
    
    if args.visualize:
        detailed, summary = evaluator.create_visualization(
            pred_img, target_img, metrics, "Test Sample"
        )
        detailed.save("detailed_viz.png")
        summary.save("summary_viz.png")
        print("\nSaved visualizations: detailed_viz.png, summary_viz.png")
    
    if args.output:
        output_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        with open(args.output, "w") as f:
            json.dump(output_metrics, f, indent=2)
        print(f"\nSaved to {args.output}")