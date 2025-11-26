#!/usr/bin/env python3
"""Visualize layout comparison with overlays and matching connections."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy


def get_color_class(pixel_rgb, taxonomy):
    """Get class name for a color by matching to taxonomy."""
    id2color = taxonomy.data.get("id2color", {})
    pixel_array = np.array(pixel_rgb, dtype=np.float32)
    closest_dist, closest_class = float('inf'), "Unknown"
    
    background_rgb = (255, 255, 255)
    dist = np.linalg.norm(pixel_array - np.array(background_rgb, dtype=np.float32))
    if dist < closest_dist:
        closest_dist, closest_class = dist, "Background"
    
    if "2053" in id2color:
        wall_rgb = tuple(int(c) for c in id2color["2053"])
        dist = np.linalg.norm(pixel_array - np.array(wall_rgb, dtype=np.float32))
        if dist < closest_dist:
            closest_dist, closest_class = dist, "Wall"
    
    for floor_id in taxonomy.get_floor_ids():
        floor_id_str = str(floor_id)
        if floor_id_str in id2color:
            floor_rgb = tuple(int(c) for c in id2color[floor_id_str])
            dist = np.linalg.norm(pixel_array - np.array(floor_rgb, dtype=np.float32))
            if dist < closest_dist:
                closest_dist, closest_class = dist, "Floor"
    
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


def extract_all_objects(image_path, taxonomy, min_pixels=10):
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
    
    return objects, floor_objects, img


def bbox_intersection(bbox1, bbox2):
    """Get intersection bbox."""
    x1_min, y1_min = bbox1["min_x"], bbox1["min_y"]
    x1_max, y1_max = bbox1["max_x"], bbox1["max_y"]
    x2_min, y2_min = bbox2["min_x"], bbox2["min_y"]
    x2_max, y2_max = bbox2["max_x"], bbox2["max_y"]
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return None
    
    return {"min_x": inter_x_min, "min_y": inter_y_min, "max_x": inter_x_max, "max_y": inter_y_max}


def match_objects(objects1, objects2):
    """Match objects between two images by class and proximity.
    
    Only matches objects of the same class. For each class, matches by closest distance.
    """
    matches = []
    used1 = set()
    used2 = set()
    
    # Group by class
    objs1_by_class = {}
    objs2_by_class = {}
    for idx, obj in enumerate(objects1):
        if "center" not in obj:
            continue
        cls = obj.get("class", "Unknown")
        if cls not in objs1_by_class:
            objs1_by_class[cls] = []
        objs1_by_class[cls].append((idx, obj))
    
    for idx, obj in enumerate(objects2):
        if "center" not in obj:
            continue
        cls = obj.get("class", "Unknown")
        if cls not in objs2_by_class:
            objs2_by_class[cls] = []
        objs2_by_class[cls].append((idx, obj))
    
    # Match within each class by closest distance
    for cls in set(objs1_by_class.keys()) & set(objs2_by_class.keys()):
        objs1_cls = objs1_by_class[cls]
        objs2_cls = objs2_by_class[cls]
        
        # For each object in image1, find closest unmatched object in image2 of same class
        for idx1, obj1 in objs1_cls:
            if idx1 in used1:
                continue
            
            best_dist = float('inf')
            best_obj2 = None
            best_idx2 = -1
            
            for idx2, obj2 in objs2_cls:
                if idx2 in used2:
                    continue
                
                dx = obj1["center"][0] - obj2["center"][0]
                dy = obj1["center"][1] - obj2["center"][1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < best_dist:
                    best_dist = dist
                    best_obj2 = obj2
                    best_idx2 = idx2
            
            if best_obj2:
                matches.append((obj1, best_obj2, best_dist))
                used1.add(idx1)
                used2.add(best_idx2)
    
    return matches


def create_bbox_visualization(img1_cleaned, img1_bbox, img2_cleaned, img2_bbox, taxonomy, min_pixels=20, alpha=0.6):
    """Create bbox comparison visualization."""
    # Extract objects from cleaned images for matching
    objects1, floor1, _ = extract_all_objects(img1_cleaned, taxonomy, min_pixels)
    objects2, floor2, _ = extract_all_objects(img2_cleaned, taxonomy, min_pixels)
    
    # EXCLUDE floor and walls from matching
    all_objects1 = [obj for obj in objects1 if obj.get("class") not in ["Floor", "Wall"]]
    all_objects2 = [obj for obj in objects2 if obj.get("class") not in ["Floor", "Wall"]]
    
    # Load bbox images
    bbox1 = Image.open(img1_bbox).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    bbox2 = Image.open(img2_bbox).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    
    # Create composite: overlay bbox images with transparency
    bbox1_rgba = bbox1.convert("RGBA")
    bbox2_rgba = bbox2.convert("RGBA")
    
    # Make semi-transparent
    bbox1_data = np.array(bbox1_rgba)
    bbox1_data[:, :, 3] = int(255 * alpha)
    bbox1_alpha = Image.fromarray(bbox1_data)
    
    bbox2_data = np.array(bbox2_rgba)
    bbox2_data[:, :, 3] = int(255 * alpha)
    bbox2_alpha = Image.fromarray(bbox2_data)
    
    # Start with white background
    composite = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
    composite = Image.alpha_composite(composite, bbox1_alpha)
    composite = Image.alpha_composite(composite, bbox2_alpha)
    
    # Convert to RGB for drawing
    composite = composite.convert("RGB")
    draw = ImageDraw.Draw(composite)
    
    # Match objects (excluding floor/walls)
    matches = match_objects(all_objects1, all_objects2)
    # Filter out floor/wall matches
    matches = [(o1, o2, d) for o1, o2, d in matches 
               if o1.get("class") not in ["Floor", "Wall"] and o2.get("class") not in ["Floor", "Wall"]]
    
    # Draw bbox intersections (opaque, darker)
    for obj1, obj2, dist in matches:
        if "bbox" not in obj1 or "bbox" not in obj2:
            continue
        inter = bbox_intersection(obj1["bbox"], obj2["bbox"])
        if inter:
            # Draw intersection as darker/more opaque
            inter_img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            inter_draw = ImageDraw.Draw(inter_img)
            inter_draw.rectangle(
                [inter["min_x"], inter["min_y"], inter["max_x"], inter["max_y"]],
                fill=(*obj1["color"], 200)  # More opaque
            )
            composite = Image.alpha_composite(composite.convert("RGBA"), inter_img).convert("RGB")
            draw = ImageDraw.Draw(composite)
    
    return composite, matches


def create_centroid_visualization(img1_cleaned, img1_centroid, img2_cleaned, img2_centroid, taxonomy, min_pixels=20, alpha=0.6):
    """Create centroid comparison visualization (excluding floor/walls)."""
    # Extract objects from cleaned images for matching (excluding floor/walls)
    objects1, floor1, _ = extract_all_objects(img1_cleaned, taxonomy, min_pixels)
    objects2, floor2, _ = extract_all_objects(img2_cleaned, taxonomy, min_pixels)
    
    # Filter out floor, walls, and light gray floor colors from matching
    def is_floor_gray_color(obj, taxonomy):
        """Check if object color is light gray floor."""
        color = obj.get("color", (255, 255, 255))
        id2color = taxonomy.data.get("id2color", {})
        for floor_id in taxonomy.get_floor_ids():
            floor_id_str = str(floor_id)
            if floor_id_str in id2color:
                floor_rgb = tuple(int(c) for c in id2color[floor_id_str])
                r, g, b = color
                fr, fg, fb = floor_rgb
                if abs(r - fr) < 10 and abs(g - fg) < 10 and abs(b - fb) < 10:
                    return True
        return False
    
    objects1_filtered = [obj for obj in objects1 
                        if obj.get("class") not in ["Floor", "Wall"] and not is_floor_gray_color(obj, taxonomy)]
    objects2_filtered = [obj for obj in objects2 
                        if obj.get("class") not in ["Floor", "Wall"] and not is_floor_gray_color(obj, taxonomy)]
    
    # Load centroid images
    centroid1 = Image.open(img1_centroid).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    centroid2 = Image.open(img2_centroid).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    
    # Create composite: overlay centroid images with transparency
    centroid1_rgba = centroid1.convert("RGBA")
    centroid2_rgba = centroid2.convert("RGBA")
    
    centroid1_data = np.array(centroid1_rgba)
    centroid1_data[:, :, 3] = int(255 * alpha)
    centroid1_alpha = Image.fromarray(centroid1_data)
    
    centroid2_data = np.array(centroid2_rgba)
    centroid2_data[:, :, 3] = int(255 * alpha)
    centroid2_alpha = Image.fromarray(centroid2_data)
    
    # Start with white background
    composite = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
    composite = Image.alpha_composite(composite, centroid1_alpha)
    composite = Image.alpha_composite(composite, centroid2_alpha)
    composite = composite.convert("RGB")
    draw = ImageDraw.Draw(composite)
    
    # Match objects (excluding floor/walls)
    matches = match_objects(objects1_filtered, objects2_filtered)
    
    # Get centroid colors from the actual centroid images
    centroid1_array = np.array(centroid1.convert("RGB"))
    centroid2_array = np.array(centroid2.convert("RGB"))
    
    # Draw lines connecting matched centroids
    for obj1, obj2, dist in matches:
        if "center" not in obj1 or "center" not in obj2:
            continue
        
        # Get actual centroid colors from images
        cx1, cy1 = obj1["center"]
        cx2, cy2 = obj2["center"]
        cx1, cy1 = max(0, min(255, cx1)), max(0, min(255, cy1))
        cx2, cy2 = max(0, min(255, cx2)), max(0, min(255, cy2))
        
        centroid1_color = tuple(centroid1_array[cy1, cx1])
        centroid2_color = tuple(centroid2_array[cy2, cx2])
        
        # Color by distance
        if dist < 20:
            line_color = (255, 0, 0)  # Red - close
        elif dist < 50:
            line_color = (255, 165, 0)  # Orange - medium
        else:
            line_color = (128, 128, 128)  # Gray - far
        
        draw.line([obj1["center"], obj2["center"]], fill=line_color, width=2)
        
        # Draw centroid markers with actual colors from images
        r = 5
        draw.ellipse(
            [obj1["center"][0] - r, obj1["center"][1] - r, obj1["center"][0] + r, obj1["center"][1] + r],
            fill=centroid1_color, outline=(0, 0, 0), width=2
        )
        draw.ellipse(
            [obj2["center"][0] - r, obj2["center"][1] - r, obj2["center"][0] + r, obj2["center"][1] + r],
            fill=centroid2_color, outline=(0, 0, 0), width=2
        )
    
    return composite, matches


def main():
    parser = argparse.ArgumentParser(description="Visualize layout comparison using bbox/centroid images")
    parser.add_argument("--image1-cleaned", type=Path, required=True, help="First cleaned image (for object extraction)")
    parser.add_argument("--image1-bbox", type=Path, required=True, help="First bbox image")
    parser.add_argument("--image1-centroid", type=Path, required=True, help="First centroid image")
    parser.add_argument("--image2-cleaned", type=Path, required=True, help="Second cleaned image (for object extraction)")
    parser.add_argument("--image2-bbox", type=Path, required=True, help="Second bbox image")
    parser.add_argument("--image2-centroid", type=Path, required=True, help="Second centroid image")
    parser.add_argument("--taxonomy", type=Path, required=True, help="Taxonomy JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for images")
    parser.add_argument("--min-pixels", type=int, default=20, help="Minimum pixels per object (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Transparency for overlay (0-1, default: 0.6)")
    args = parser.parse_args()
    
    taxonomy = Taxonomy(args.taxonomy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create bbox visualization
    bbox_composite, bbox_matches = create_bbox_visualization(
        args.image1_cleaned, args.image1_bbox,
        args.image2_cleaned, args.image2_bbox,
        taxonomy, min_pixels=args.min_pixels, alpha=args.alpha
    )
    bbox_output = args.output_dir / "comparison_bboxes.png"
    bbox_composite.save(bbox_output)
    print(f"Saved bbox comparison to {bbox_output}")
    print(f"  Matched {len(bbox_matches)} object pairs")
    
    # Create centroid visualization
    centroid_composite, centroid_matches = create_centroid_visualization(
        args.image1_cleaned, args.image1_centroid,
        args.image2_cleaned, args.image2_centroid,
        taxonomy, min_pixels=args.min_pixels, alpha=args.alpha
    )
    centroid_output = args.output_dir / "comparison_centroids.png"
    centroid_composite.save(centroid_output)
    print(f"Saved centroid comparison to {centroid_output}")
    print(f"  Matched {len(centroid_matches)} object pairs (excluding floor/walls)")


if __name__ == "__main__":
    main()

