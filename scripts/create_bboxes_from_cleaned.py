#!/usr/bin/env python3
"""Create bboxes/centroids from cleaned layout images."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy


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


def extract_objects(image_path, taxonomy, min_pixels=20):
    """Extract objects from cleaned image (excluding walls)."""
    img = Image.open(image_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)
    h, w = img_array.shape[:2]
    
    # Create segmentation map: group pixels by color (excluding walls, floor, background)
    seg_map = np.zeros((h, w), dtype=np.int32)
    color_to_info = {}
    next_label = 1
    
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            
            # Skip walls, floor, and background
            if is_wall_color(pixel_rgb, taxonomy) or is_floor_color(pixel_rgb, taxonomy) or is_background_color(pixel_rgb):
                continue
            
            # Group by color
            color_key = pixel_rgb
            if color_key not in color_to_info:
                color_to_info[color_key] = {"color": pixel_rgb, "seg_label": next_label}
                next_label += 1
            seg_map[y, x] = color_to_info[color_key]["seg_label"]
    
    # Extract floor separately
    floor_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            if is_floor_color(pixel_rgb, taxonomy):
                floor_mask[y, x] = True
    
    # Extract connected components for objects
    objects = []
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
            objects.append({
                "color": label_info["color"],
                "center": (int(centroid_x), int(centroid_y)),
                "bbox": {"min_x": int(min_x), "min_y": int(min_y), "max_x": int(max_x), "max_y": int(max_y)}
            })
    
    # Extract floor as single object
    floor_objects = []
    if np.any(floor_mask):
        labeled_floor, num_floor = ndimage.label(floor_mask)
        for fid in range(1, num_floor + 1):
            pixels = np.argwhere(labeled_floor == fid)
            if len(pixels) < min_pixels:
                continue
            min_y, min_x = pixels.min(axis=0)
            max_y, max_x = pixels.max(axis=0)
            # Get floor color
            floor_color = None
            for y, x in pixels[:10]:  # Sample a few pixels
                pixel_rgb = tuple(img_array[y, x])
                if is_floor_color(pixel_rgb, taxonomy):
                    floor_color = pixel_rgb
                    break
            if not floor_color:
                continue
            floor_objects.append({
                "color": floor_color,
                "bbox": {"min_x": int(min_x), "min_y": int(min_y), "max_x": int(max_x), "max_y": int(max_y)}
            })
    
    return objects, floor_objects, img


def create_bbox_image(image, objects, floor_objects, taxonomy):
    """Create image with bboxes (floor drawn first/under)."""
    img_array = np.array(image.resize((256, 256), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.uint8)
    h, w = img_array.shape[:2]
    output = np.full((h, w, 3), 255, dtype=np.uint8)  # White background
    
    # Copy walls from original
    wall_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            if is_wall_color(pixel_rgb, taxonomy):
                wall_mask[y, x] = True
    output[wall_mask] = img_array[wall_mask]
    
    # Draw floor bboxes FIRST (under everything)
    for floor_obj in floor_objects:
        bbox = floor_obj["bbox"]
        min_x, min_y = max(0, bbox["min_x"]), max(0, bbox["min_y"])
        max_x, max_y = min(w, bbox["max_x"]), min(h, bbox["max_y"])
        color = floor_obj["color"]
        bbox_mask = np.zeros((h, w), dtype=bool)
        bbox_mask[min_y:max_y+1, min_x:max_x+1] = True
        fill_mask = bbox_mask & ~wall_mask
        output[fill_mask] = color
    
    # Draw object bboxes (on top of floor)
    for obj in objects:
        bbox = obj["bbox"]
        min_x, min_y = max(0, bbox["min_x"]), max(0, bbox["min_y"])
        max_x, max_y = min(w, bbox["max_x"]), min(h, bbox["max_y"])
        color = obj["color"]
        bbox_mask = np.zeros((h, w), dtype=bool)
        bbox_mask[min_y:max_y+1, min_x:max_x+1] = True
        fill_mask = bbox_mask & ~wall_mask
        output[fill_mask] = color
    
    return Image.fromarray(output)


def create_centroid_image(image, objects, floor_objects, taxonomy, centroid_radius=8):
    """Create image with centroids (non-wall, non-floor) and floor bboxes."""
    img_array = np.array(image.resize((256, 256), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.uint8)
    h, w = img_array.shape[:2]
    output = np.full((h, w, 3), 255, dtype=np.uint8)  # White background
    
    # Copy walls from original
    wall_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            if is_wall_color(pixel_rgb, taxonomy):
                wall_mask[y, x] = True
    output[wall_mask] = img_array[wall_mask]
    
    # Draw floor bboxes FIRST (under centroids)
    for floor_obj in floor_objects:
        bbox = floor_obj["bbox"]
        min_x, min_y = max(0, bbox["min_x"]), max(0, bbox["min_y"])
        max_x, max_y = min(w, bbox["max_x"]), min(h, bbox["max_y"])
        color = floor_obj["color"]
        bbox_mask = np.zeros((h, w), dtype=bool)
        bbox_mask[min_y:max_y+1, min_x:max_x+1] = True
        fill_mask = bbox_mask & ~wall_mask
        output[fill_mask] = color
    
    # Draw centroids for objects (on top of floor)
    for obj in objects:
        color = obj["color"]
        cx, cy = obj["center"]
        cx, cy = int(cx), int(cy)
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        
        # Draw centroid as large circle
        for dy in range(-centroid_radius, centroid_radius + 1):
            for dx in range(-centroid_radius, centroid_radius + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    if not (wall_mask[py, px]):
                        dist_sq = dx*dx + dy*dy
                        if dist_sq <= centroid_radius*centroid_radius:
                            output[py, px] = color
    
    return Image.fromarray(output)


def main():
    parser = argparse.ArgumentParser(description="Create bboxes/centroids from cleaned images")
    parser.add_argument("--image", type=Path, nargs="+", required=True, help="Input cleaned image(s)")
    parser.add_argument("--taxonomy", type=Path, required=True, help="Taxonomy JSON")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--min-pixels", type=int, default=20, help="Minimum pixels per object (default: 20)")
    parser.add_argument("--centroid-radius", type=int, default=8, help="Centroid radius (default: 8)")
    args = parser.parse_args()
    
    taxonomy = Taxonomy(args.taxonomy)
    output_dir = args.output_dir or Path("outputs/samples/bboxes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in args.image:
        objects, floor_objects, image = extract_objects(img_path, taxonomy, min_pixels=args.min_pixels)
        print(f"{img_path.name}: {len(objects)} objects, {len(floor_objects)} floor objects")
        
        bbox_img = create_bbox_image(image, objects, floor_objects, taxonomy)
        centroid_img = create_centroid_image(image, objects, floor_objects, taxonomy, centroid_radius=args.centroid_radius)
        
        stem = img_path.stem.replace("_cleaned", "")  # Remove _cleaned suffix if present
        bbox_img.save(output_dir / f"{stem}_bboxes.png")
        centroid_img.save(output_dir / f"{stem}_centroids.png")
        print(f"  Saved to {output_dir / f'{stem}_bboxes.png'} and {output_dir / f'{stem}_centroids.png'}")


if __name__ == "__main__":
    main()

