#!/usr/bin/env python3
"""Compare two layout images and compute matching metrics."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from collections import defaultdict
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy


def get_color_class(pixel_rgb, taxonomy):
    """Get class name for a color by matching to taxonomy."""
    id2color = taxonomy.data.get("id2color", {})
    pixel_array = np.array(pixel_rgb, dtype=np.float32)
    closest_dist, closest_class = float('inf'), "Unknown"
    
    # Check background
    background_rgb = (255, 255, 255)
    dist = np.linalg.norm(pixel_array - np.array(background_rgb, dtype=np.float32))
    if dist < closest_dist:
        closest_dist, closest_class = dist, "Background"
    
    # Check wall (ID 2053)
    if "2053" in id2color:
        wall_rgb = tuple(int(c) for c in id2color["2053"])
        dist = np.linalg.norm(pixel_array - np.array(wall_rgb, dtype=np.float32))
        if dist < closest_dist:
            closest_dist, closest_class = dist, "Wall"
    
    # Check floor
    for floor_id in taxonomy.get_floor_ids():
        floor_id_str = str(floor_id)
        if floor_id_str in id2color:
            floor_rgb = tuple(int(c) for c in id2color[floor_id_str])
            dist = np.linalg.norm(pixel_array - np.array(floor_rgb, dtype=np.float32))
            if dist < closest_dist:
                closest_dist, closest_class = dist, "Floor"
    
    # Check super-categories
    id2super = taxonomy.data.get("id2super", {})
    for super_id_str, super_name in id2super.items():
        if super_id_str in id2color:
            color_rgb = tuple(int(c) for c in id2color[super_id_str])
            dist = np.linalg.norm(pixel_array - np.array(color_rgb, dtype=np.float32))
            if dist < closest_dist:
                closest_dist, closest_class = dist, super_name
    
    return closest_class


def is_wall_color(pixel_rgb, taxonomy):
    """Check if pixel color matches wall."""
    id2color = taxonomy.data.get("id2color", {})
    if "2053" not in id2color:
        return False
    wall_rgb = np.array([int(c) for c in id2color["2053"]], dtype=np.float32)
    pixel_array = np.array(pixel_rgb, dtype=np.float32)
    dist = np.linalg.norm(pixel_array - wall_rgb)
    return dist < 30


def is_floor_color(pixel_rgb, taxonomy):
    """Check if pixel color matches floor."""
    id2color = taxonomy.data.get("id2color", {})
    for floor_id in taxonomy.get_floor_ids():
        floor_id_str = str(floor_id)
        if floor_id_str in id2color:
            floor_rgb = np.array([int(c) for c in id2color[floor_id_str]], dtype=np.float32)
            pixel_array = np.array(pixel_rgb, dtype=np.float32)
            dist = np.linalg.norm(pixel_array - floor_rgb)
            if dist < 30:
                return True
    return False


def is_background_color(pixel_rgb):
    """Check if pixel is white/background."""
    r, g, b = pixel_rgb
    return pixel_rgb == (255, 255, 255) or (r > 245 and g > 245 and b > 245)


def extract_all_objects(image_path, taxonomy, min_pixels=20):
    """Extract all objects including walls and floor."""
    img = Image.open(image_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)
    h, w = img_array.shape[:2]
    
    objects = []
    floor_objects = []
    wall_mask = np.zeros((h, w), dtype=bool)
    
    # Extract walls
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            if is_wall_color(pixel_rgb, taxonomy):
                wall_mask[y, x] = True
    
    if np.any(wall_mask):
        labeled_wall, num_wall = ndimage.label(wall_mask)
        for fid in range(1, num_wall + 1):
            pixels = np.argwhere(labeled_wall == fid)
            if len(pixels) >= min_pixels:
                min_y, min_x = pixels.min(axis=0)
                max_y, max_x = pixels.max(axis=0)
                centroid_y, centroid_x = pixels.mean(axis=0)
                wall_color = tuple(img_array[int(centroid_y), int(centroid_x)])
                objects.append({
                    "class": "Wall",
                    "color": wall_color,
                    "center": (int(centroid_x), int(centroid_y)),
                    "bbox": {"min_x": int(min_x), "min_y": int(min_y), "max_x": int(max_x), "max_y": int(max_y)},
                    "pixel_count": len(pixels)
                })
    
    # Extract floor
    floor_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            if is_floor_color(pixel_rgb, taxonomy):
                floor_mask[y, x] = True
    
    if np.any(floor_mask):
        labeled_floor, num_floor = ndimage.label(floor_mask)
        for fid in range(1, num_floor + 1):
            pixels = np.argwhere(labeled_floor == fid)
            if len(pixels) >= min_pixels:
                min_y, min_x = pixels.min(axis=0)
                max_y, max_x = pixels.max(axis=0)
                floor_color = tuple(img_array[int(pixels[0][0]), int(pixels[0][1])])
                floor_objects.append({
                    "class": "Floor",
                    "color": floor_color,
                    "bbox": {"min_x": int(min_x), "min_y": int(min_y), "max_x": int(max_x), "max_y": int(max_y)},
                    "pixel_count": len(pixels)
                })
    
    # Extract other objects
    seg_map = np.zeros((h, w), dtype=np.int32)
    color_to_info = {}
    next_label = 1
    
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            if is_wall_color(pixel_rgb, taxonomy) or is_floor_color(pixel_rgb, taxonomy) or is_background_color(pixel_rgb):
                continue
            color_key = pixel_rgb
            if color_key not in color_to_info:
                color_to_info[color_key] = {"color": pixel_rgb, "seg_label": next_label}
                next_label += 1
            seg_map[y, x] = color_to_info[color_key]["seg_label"]
    
    for label_id in np.unique(seg_map[seg_map > 0]):
        mask = (seg_map == label_id)
        label_info = color_to_info.get(next((k for k, v in color_to_info.items() if v["seg_label"] == label_id), None))
        if not label_info:
            continue
        
        labeled_mask, num_features = ndimage.label(mask)
        for fid in range(1, num_features + 1):
            pixels = np.argwhere(labeled_mask == fid)
            if len(pixels) < min_pixels:
                continue
            min_y, min_x = pixels.min(axis=0)
            max_y, max_x = pixels.max(axis=0)
            centroid_y, centroid_x = pixels.mean(axis=0)
            obj_color = label_info["color"]
            obj_class = get_color_class(obj_color, taxonomy)
            objects.append({
                "class": obj_class,
                "color": obj_color,
                "center": (int(centroid_x), int(centroid_y)),
                "bbox": {"min_x": int(min_x), "min_y": int(min_y), "max_x": int(max_x), "max_y": int(max_y)},
                "pixel_count": len(pixels)
            })
    
    return objects, floor_objects


def bbox_iou(bbox1, bbox2):
    """Compute Intersection over Union for two bboxes."""
    x1_min, y1_min = bbox1["min_x"], bbox1["min_y"]
    x1_max, y1_max = bbox1["max_x"], bbox1["max_y"]
    x2_min, y2_min = bbox2["min_x"], bbox2["min_y"]
    x2_max, y2_max = bbox2["max_x"], bbox2["max_y"]
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def bbox_coverage(bbox1, bbox2):
    """Compute coverage: intersection area / bbox1 area."""
    x1_min, y1_min = bbox1["min_x"], bbox1["min_y"]
    x1_max, y1_max = bbox1["max_x"], bbox1["max_y"]
    x2_min, y2_min = bbox2["min_x"], bbox2["min_y"]
    x2_max, y2_max = bbox2["max_x"], bbox2["max_y"]
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    
    return inter_area / area1 if area1 > 0 else 0.0


def centroid_distance(center1, center2):
    """Compute Euclidean distance between two centroids."""
    x1, y1 = center1
    x2, y2 = center2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def compare_layouts(img1_path, img2_path, taxonomy, min_pixels=20):
    """Compare two layouts and compute metrics."""
    objects1, floor1 = extract_all_objects(img1_path, taxonomy, min_pixels)
    objects2, floor2 = extract_all_objects(img2_path, taxonomy, min_pixels)
    
    # Add floor objects to main objects list
    all_objects1 = objects1 + [{"class": "Floor", **f} for f in floor1]
    all_objects2 = objects2 + [{"class": "Floor", **f} for f in floor2]
    
    # Count objects per class
    count1 = defaultdict(int)
    count2 = defaultdict(int)
    for obj in all_objects1:
        count1[obj["class"]] += 1
    for obj in all_objects2:
        count2[obj["class"]] += 1
    
    all_classes = set(count1.keys()) | set(count2.keys())
    
    # Class count matching
    class_counts = {}
    for cls in all_classes:
        class_counts[cls] = {
            "count1": count1[cls],
            "count2": count2[cls],
            "difference": abs(count1[cls] - count2[cls])
        }
    
    # Bbox IoU and coverage per class
    bbox_metrics = defaultdict(lambda: {"ious": [], "coverages": []})
    
    for cls in all_classes:
        objs1_cls = [obj for obj in all_objects1 if obj["class"] == cls]
        objs2_cls = [obj for obj in all_objects2 if obj["class"] == cls]
        
        if not objs1_cls or not objs2_cls:
            continue
        
        # Match objects by best IoU
        matched = set()
        for obj1 in objs1_cls:
            if "bbox" not in obj1:
                continue
            best_iou = 0.0
            best_obj2 = None
            for idx, obj2 in enumerate(objs2_cls):
                if idx in matched or "bbox" not in obj2:
                    continue
                iou = bbox_iou(obj1["bbox"], obj2["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_obj2 = obj2
            
            if best_obj2:
                matched.add(objs2_cls.index(best_obj2))
                coverage = bbox_coverage(obj1["bbox"], best_obj2["bbox"])
                bbox_metrics[cls]["ious"].append(best_iou)
                bbox_metrics[cls]["coverages"].append(coverage)
    
    # Centroid distances (match closest objects of same class)
    centroid_distances = defaultdict(list)
    
    for cls in all_classes:
        objs1_cls = [obj for obj in all_objects1 if obj["class"] == cls and "center" in obj]
        objs2_cls = [obj for obj in all_objects2 if obj["class"] == cls and "center" in obj]
        
        if not objs1_cls or not objs2_cls:
            continue
        
        # For each object in img1, find closest in img2
        for obj1 in objs1_cls:
            min_dist = float('inf')
            for obj2 in objs2_cls:
                dist = centroid_distance(obj1["center"], obj2["center"])
                min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                centroid_distances[cls].append(min_dist)
    
    # Aggregate metrics
    results = {
        "class_counts": dict(class_counts),
        "bbox_metrics": {},
        "centroid_distances": {}
    }
    
    for cls in all_classes:
        if cls in bbox_metrics:
            ious = bbox_metrics[cls]["ious"]
            coverages = bbox_metrics[cls]["coverages"]
            results["bbox_metrics"][cls] = {
                "mean_iou": np.mean(ious) if ious else 0.0,
                "mean_coverage": np.mean(coverages) if coverages else 0.0,
                "num_matched": len(ious)
            }
        
        if cls in centroid_distances:
            dists = centroid_distances[cls]
            results["centroid_distances"][cls] = {
                "mean_distance": np.mean(dists) if dists else float('inf'),
                "min_distance": np.min(dists) if dists else float('inf'),
                "max_distance": np.max(dists) if dists else 0.0,
                "num_objects": len(dists)
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare two layout images")
    parser.add_argument("--image1", type=Path, required=True, help="First image (target)")
    parser.add_argument("--image2", type=Path, required=True, help="Second image (generated)")
    parser.add_argument("--taxonomy", type=Path, required=True, help="Taxonomy JSON")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--min-pixels", type=int, default=20, help="Minimum pixels per object (default: 20)")
    args = parser.parse_args()
    
    taxonomy = Taxonomy(args.taxonomy)
    results = compare_layouts(args.image1, args.image2, taxonomy, min_pixels=args.min_pixels)
    
    # Print summary
    print("\n=== Class Count Matching ===")
    for cls, counts in sorted(results["class_counts"].items()):
        print(f"{cls}: {counts['count1']} vs {counts['count2']} (diff: {counts['difference']})")
    
    print("\n=== Bbox Metrics (IoU/Coverage) ===")
    for cls, metrics in sorted(results["bbox_metrics"].items()):
        print(f"{cls}: Mean IoU={metrics['mean_iou']:.3f}, Mean Coverage={metrics['mean_coverage']:.3f}, Matched={metrics['num_matched']}")
    
    print("\n=== Centroid Distances ===")
    for cls, dists in sorted(results["centroid_distances"].items()):
        print(f"{cls}: Mean={dists['mean_distance']:.2f}, Min={dists['min_distance']:.2f}, Max={dists['max_distance']:.2f}, Objects={dists['num_objects']}")
    
    # Save JSON
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

