#!/usr/bin/env python3
"""Clean layout by matching pixels to super-category colors."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy


def is_white_or_background(pixel_rgb):
    """Check if pixel is white/background - should never be matched to floor."""
    r, g, b = pixel_rgb
    # Pure white
    if pixel_rgb == (255, 255, 255):
        return True
    # Very close to white (within 10 RGB units)
    if r > 245 and g > 245 and b > 245:
        return True
    return False


def find_closest_color(pixel_rgb, taxonomy):
    """Find closest taxonomy color (background, wall, floor, or super-category)."""
    id2color = taxonomy.data.get("id2color", {})
    pixel_array = np.array(pixel_rgb, dtype=np.float32)
    closest_dist, closest_color = float('inf'), None
    matched_unknown_or_structure = False
    
    # Check background FIRST (white) - highest priority
    background_rgb = (255, 255, 255)
    dist = np.linalg.norm(pixel_array - np.array(background_rgb, dtype=np.float32))
    if dist < closest_dist:
        closest_dist, closest_color = dist, background_rgb
    
    # Check wall (ID 2053)
    if "2053" in id2color:
        wall_rgb = tuple(int(c) for c in id2color["2053"])
        dist = np.linalg.norm(pixel_array - np.array(wall_rgb, dtype=np.float32))
        if dist < closest_dist:
            closest_dist, closest_color = dist, wall_rgb
    
    # Check floor
    floor_rgb_list = []
    for floor_id in taxonomy.get_floor_ids():
        floor_id_str = str(floor_id)
        if floor_id_str in id2color:
            floor_rgb = tuple(int(c) for c in id2color[floor_id_str])
            floor_rgb_list.append(floor_rgb)
            dist = np.linalg.norm(pixel_array - np.array(floor_rgb, dtype=np.float32))
            if dist < closest_dist:
                closest_dist, closest_color = dist, floor_rgb
    
    # Check unknown (ID 0) and structure (ID 1008) - these should go to floor
    unknown_structure_ids = ["0", "1008"]
    for uid in unknown_structure_ids:
        if uid in id2color:
            unknown_rgb = tuple(int(c) for c in id2color[uid])
            dist = np.linalg.norm(pixel_array - np.array(unknown_rgb, dtype=np.float32))
            if dist < closest_dist:
                closest_dist = dist
                matched_unknown_or_structure = True
                # Use first available floor color, or keep unknown if no floor found
                if floor_rgb_list:
                    closest_color = floor_rgb_list[0]
                else:
                    closest_color = unknown_rgb
    
    # Check super-categories
    id2super = taxonomy.data.get("id2super", {})
    for super_id_str, super_name in id2super.items():
        if super_id_str in id2color:
            color_rgb = tuple(int(c) for c in id2color[super_id_str])
            dist = np.linalg.norm(pixel_array - np.array(color_rgb, dtype=np.float32))
            if dist < closest_dist:
                closest_dist, closest_color = dist, color_rgb
                matched_unknown_or_structure = False
    
    # If matched unknown/structure but floor wasn't the closest, force to floor
    if matched_unknown_or_structure and floor_rgb_list:
        return floor_rgb_list[0]
    
    return closest_color if closest_color else pixel_rgb  # Return original if no match


def clean_image(image_path, taxonomy):
    """Clean image by matching pixels to taxonomy colors."""
    img = Image.open(image_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)
    h, w = img_array.shape[:2]
    cleaned_array = np.zeros_like(img_array)
    
    for y in range(h):
        for x in range(w):
            pixel_rgb = tuple(img_array[y, x])
            closest_color = find_closest_color(pixel_rgb, taxonomy)
            cleaned_array[y, x] = closest_color
    
    return Image.fromarray(cleaned_array)


def main():
    parser = argparse.ArgumentParser(description="Clean layout by matching to super-category colors")
    parser.add_argument("--image", type=Path, nargs="+", required=True, help="Input image(s)")
    parser.add_argument("--taxonomy", type=Path, required=True, help="Taxonomy JSON")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()
    
    taxonomy = Taxonomy(args.taxonomy)
    output_dir = args.output_dir or Path("outputs/samples/bboxes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in args.image:
        cleaned_img = clean_image(img_path, taxonomy)
        stem = img_path.stem
        output_path = output_dir / f"{stem}_cleaned.png"
        cleaned_img.save(output_path)
        print(f"{img_path.name} -> {output_path}")


if __name__ == "__main__":
    main()

