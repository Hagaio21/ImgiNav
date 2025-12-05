#!/usr/bin/env python3
"""Clean layout image by matching pixels to taxonomy category colors and detect objects."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy


def match_pixel_to_class(pixel_rgb, taxonomy):
    """Match a pixel RGB to its closest category name from taxonomy."""
    category_to_color = taxonomy.data.get("category_to_color", {})
    categories = taxonomy.data.get("categories", [])
    
    # Create category to ID mapping (1-indexed, 0 is Unknown)
    category2id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    category2id["Unknown"] = 0
    
    pixel_array = np.array(pixel_rgb, dtype=np.float32)
    closest_dist = float('inf')
    closest_category = "Unknown"
    
    # Check background (white)
    background_rgb = np.array([255, 255, 255], dtype=np.float32)
    dist = np.linalg.norm(pixel_array - background_rgb)
    if dist < closest_dist:
        closest_dist = dist
        closest_category = "Unknown"
    
    # Check all category colors
    for category_name, color_list in category_to_color.items():
        if not isinstance(color_list, list) or len(color_list) != 3:
            continue
        color_rgb = np.array(color_list, dtype=np.float32)
        dist = np.linalg.norm(pixel_array - color_rgb)
        if dist < closest_dist:
            closest_dist = dist
            closest_category = category_name
    
    # Convert category name to ID
    category_id = category2id.get(closest_category, 0)
    return category_id, closest_category


def find_object_centroids(img_array, taxonomy, obj_detection_scale=4, exclude_classes=None):
    """
    Find centroids and bounding boxes of objects in the image.
    
    Args:
        img_array: 2D numpy array of image (h, w, 3)
        taxonomy: Taxonomy object to get category names
        obj_detection_scale: Scale factor for object detection (process every Nth pixel)
        exclude_classes: List of class IDs to exclude (e.g., [0] for background, plus Wall, Floor)
    
    Returns:
        List of dicts with keys: class_id, centroid (x, y), bbox (min_x, min_y, max_x, max_y), pixel_count, mask
    """
    img_h, img_w = img_array.shape[:2]
    
    # Create downsampled grid for object detection
    downsampled_h = img_h // obj_detection_scale
    downsampled_w = img_w // obj_detection_scale
    full_res_grid = np.zeros((downsampled_h, downsampled_w), dtype=np.int32)
    
    # Sample pixels for object detection
    for y in range(0, img_h, obj_detection_scale):
        for x in range(0, img_w, obj_detection_scale):
            pixel_rgb = tuple(img_array[y, x])
            class_id, _ = match_pixel_to_class(pixel_rgb, taxonomy)
            full_res_grid[y // obj_detection_scale, x // obj_detection_scale] = class_id
    
    # Get category names and IDs
    categories = taxonomy.data.get("categories", [])
    category2id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    category2id["Unknown"] = 0
    
    # Exclude background, floor, and wall
    wall_id = category2id.get("Wall", None)
    floor_id = category2id.get("Floor", None)
    
    if exclude_classes is None:
        exclude_classes = [0]  # Background
        if wall_id is not None:
            exclude_classes.append(wall_id)
        if floor_id is not None:
            exclude_classes.append(floor_id)
    
    objects = []
    unique_classes = np.unique(full_res_grid)
    
    for class_id in unique_classes:
        if class_id in exclude_classes:
            continue
        
        # Create mask for this class
        mask = (full_res_grid == class_id)
        
        if not np.any(mask):
            continue
        
        # Find connected components using simple flood fill
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        
        def flood_fill(start_y, start_x):
            """Flood fill to find connected component."""
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                return []
            component = []
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                
                # Check 4 neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            
            return component
        
        # Find all components
        for y in range(0, h, 2):  # Sample every 2nd row for speed
            for x in range(0, w, 2):  # Sample every 2nd column for speed
                if mask[y, x] and not visited[y, x]:
                    component = flood_fill(y, x)
                    if len(component) >= 5:  # Lower threshold since we're sampling
                        components.append(component)
        
        # Calculate centroids and bboxes for each component (scale back to image coordinates)
        for component in components:
            pixels = np.array(component)
            centroid_y, centroid_x = pixels.mean(axis=0)
            
            # Calculate bounding box
            min_y, min_x = pixels.min(axis=0)
            max_y, max_x = pixels.max(axis=0)
            
            # Scale back to image coordinates
            img_centroid_x = int(centroid_x * obj_detection_scale)
            img_centroid_y = int(centroid_y * obj_detection_scale)
            img_min_x = int(min_x * obj_detection_scale)
            img_min_y = int(min_y * obj_detection_scale)
            img_max_x = int((max_x + 1) * obj_detection_scale)
            img_max_y = int((max_y + 1) * obj_detection_scale)
            
            # Create mask for this specific component (in downsampled grid space)
            comp_mask = np.zeros((h, w), dtype=bool)
            for py, px in component:
                comp_mask[py, px] = True
            
            objects.append({
                'class_id': int(class_id),
                'centroid': (img_centroid_x, img_centroid_y),  # Image coordinates
                'bbox': {
                    'min_x': img_min_x,
                    'min_y': img_min_y,
                    'max_x': img_max_x,
                    'max_y': img_max_y
                },
                'pixel_count': len(component),
                'mask': comp_mask,  # Mask in downsampled space
                'mask_scale': obj_detection_scale
            })
    
    return objects


def clean_layout(layout_path, taxonomy):
    """
    Clean layout image by matching each pixel to its closest category color.
    Optimized with vectorized operations.
    
    Args:
        layout_path: Path to layout image
        taxonomy: Taxonomy object
    
    Returns:
        cleaned_layout: PIL Image with cleaned colors
    """
    # Load layout image
    img = Image.open(layout_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    h, w = img_array.shape[:2]
    
    category_to_color = taxonomy.data.get("category_to_color", {})
    categories = taxonomy.data.get("categories", [])
    category2id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    category2id["Unknown"] = 0
    
    # Pre-compute category colors as numpy arrays for fast matching
    category_colors = {}
    for cat_name, color_list in category_to_color.items():
        if isinstance(color_list, list) and len(color_list) == 3:
            category_colors[cat_name] = np.array(color_list, dtype=np.float32)
    
    # Vectorized: identify white pixels (background) - ignore these
    white_threshold = 250  # Pixels with all RGB > 250 are considered white
    is_white = np.all(img_array > white_threshold, axis=2)
    
    # Initialize cleaned pixels as white (background)
    cleaned_pixels = np.full_like(img_array, 255, dtype=np.uint8)
    
    # Only process non-white pixels
    non_white_mask = ~is_white
    
    if np.any(non_white_mask):
        # Get coordinates of non-white pixels
        y_coords, x_coords = np.where(non_white_mask)
        non_white_pixels = img_array[y_coords, x_coords].astype(np.float32)  # Shape: (N, 3)
        
        # Find closest category for each non-white pixel
        best_category = np.full(non_white_pixels.shape[0], "Unknown", dtype=object)
        best_dist = np.full(non_white_pixels.shape[0], np.inf, dtype=np.float32)
        
        # Check each category color
        for cat_name, cat_color in category_colors.items():
            # Compute distance for all pixels at once
            dists = np.linalg.norm(non_white_pixels - cat_color, axis=1)
            # Update where this category is closer
            closer_mask = dists < best_dist
            best_category[closer_mask] = cat_name
            best_dist[closer_mask] = dists[closer_mask]
        
        # Assign colors based on best category
        for cat_name in category_colors.keys():
            mask = (best_category == cat_name)
            if np.any(mask):
                color = category_colors[cat_name].astype(np.uint8)
                # Use the y_coords and x_coords where this category matches
                matched_y = y_coords[mask]
                matched_x = x_coords[mask]
                cleaned_pixels[matched_y, matched_x] = color
    
    # Create cleaned layout image
    cleaned_layout = Image.fromarray(cleaned_pixels.astype(np.uint8))
    
    return cleaned_layout


def main():
    parser = argparse.ArgumentParser(
        description="Clean layout image by matching pixels to taxonomy colors"
    )
    parser.add_argument(
        "layout_path",
        type=Path,
        help="Path to layout image (PNG)"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("../data_preparation_v2/taxonomy.json"),
        help="Path to taxonomy JSON file (default: ../data_preparation_v2/taxonomy.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: {layout_name}_cleaned.png)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.layout_path.exists():
        print(f"Error: Layout file not found: {args.layout_path}")
        sys.exit(1)
    
    if not args.taxonomy.exists():
        print(f"Error: Taxonomy file not found: {args.taxonomy}")
        sys.exit(1)
    
    # Load taxonomy
    print(f"Loading taxonomy from {args.taxonomy}...")
    taxonomy = Taxonomy(args.taxonomy)
    
    # Clean layout
    print(f"Cleaning layout: {args.layout_path}...")
    cleaned_layout = clean_layout(args.layout_path, taxonomy)
    
    # Detect objects (only from non-white pixels)
    print("Detecting objects...")
    img = Image.open(args.layout_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    
    # Ignore white pixels for object detection
    white_threshold = 250
    is_white = np.all(img_array > white_threshold, axis=2)
    # Set white pixels to white in a copy for object detection
    img_array_for_detection = img_array.copy()
    img_array_for_detection[is_white] = [255, 255, 255]
    
    objects = find_object_centroids(img_array_for_detection, taxonomy, obj_detection_scale=4)
    print(f"  Found {len(objects)} objects")
    
    # Determine output paths
    if args.output is None:
        base_name = args.layout_path.stem
        output_dir = args.layout_path.parent
    else:
        base_name = args.output.stem
        output_dir = args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_path = output_dir / f"{base_name}_cleaned.png"
    objects_path = output_dir / f"{base_name}_objects.json"
    
    # Save cleaned layout
    print(f"Saving cleaned layout to {cleaned_path}...")
    cleaned_layout.save(cleaned_path)
    
    # Save objects data
    print(f"Saving objects data to {objects_path}...")
    objects_data = {
        'objects': []
    }
    for obj in objects:
        objects_data['objects'].append({
            'class_id': obj['class_id'],
            'centroid': list(obj['centroid']),
            'bbox': obj['bbox'],
            'pixel_count': obj['pixel_count'],
            'mask_scale': obj['mask_scale']
        })
    
    with open(objects_path, 'w') as f:
        json.dump(objects_data, f, indent=2)
    
    print(f"\nDone!")
    print(f"  Cleaned layout: {cleaned_path}")
    print(f"  Objects data: {objects_path}")
    print(f"  Objects found: {len(objects)}")


if __name__ == "__main__":
    main()

