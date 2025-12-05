#!/usr/bin/env python3
"""Create occupancy grid - floor/door/window are traversable."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.taxonomy import Taxonomy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned_layout_path", type=Path)
    parser.add_argument("--taxonomy", type=Path, default=Path("../data_preparation_v2/taxonomy.json"))
    parser.add_argument("--grid-resolution", type=int, default=128)
    args = parser.parse_args()

    taxonomy = Taxonomy(args.taxonomy)
    categories = taxonomy.data.get("categories", [])
    category2id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    
    # Traversable classes
    traversable = {category2id.get("Floor"), category2id.get("Door"), category2id.get("Window")} - {None}
    
    # Color lookup
    cat_colors = taxonomy.data.get("category_to_color", {})
    color_to_id = {tuple(c): category2id.get(name, 0) for name, c in cat_colors.items() if len(c) == 3}

    img = Image.open(args.cleaned_layout_path).convert("RGB")
    img_arr = np.array(img)
    img_h, img_w = img_arr.shape[:2]
    
    res = args.grid_resolution
    scale_x, scale_y = img_w / res, img_h / res
    grid = np.full((res, res), -1, dtype=np.int32)

    for gy in range(res):
        for gx in range(res):
            px, py = int((gx + 0.5) * scale_x), int((gy + 0.5) * scale_y)
            class_id = color_to_id.get(tuple(img_arr[py, px]), 0)
            if class_id in traversable:
                grid[gy, gx] = class_id

    # Load objects and mark as obstacles
    base = args.cleaned_layout_path.stem.replace("_cleaned", "")
    obj_path = args.cleaned_layout_path.parent / f"{base}_objects.json"
    with open(obj_path) as f:
        obj_data = json.load(f)

    for obj in obj_data['objects']:
        b = obj['bbox']
        gx1, gy1 = max(0, int(b['min_x']/scale_x)), max(0, int(b['min_y']/scale_y))
        gx2, gy2 = min(res-1, int(b['max_x']/scale_x)), min(res-1, int(b['max_y']/scale_y))
        grid[gy1:gy2+1, gx1:gx2+1] = -1
        obj['centroid_grid'] = [int(obj['centroid'][0]/scale_x), int(obj['centroid'][1]/scale_y)]
        obj['bbox_grid'] = {'min_x': gx1, 'min_y': gy1, 'max_x': gx2, 'max_y': gy2}
        obj['mask_coords'] = [[y, x] for y in range(gy1, gy2+1) for x in range(gx1, gx2+1)]

    cam_pos = (res // 2, res - 2)
    obj_data['camera_pos'], obj_data['scale'], obj_data['grid_resolution'] = cam_pos, (scale_x, scale_y), res

    np.save(args.cleaned_layout_path.parent / f"{base}_occupancy_grid.npy", grid)
    with open(obj_path, 'w') as f:
        json.dump(obj_data, f, indent=2)

    print(f"Grid: {res}x{res}, Camera: {cam_pos}, Traversable: {np.sum(grid != -1)}")


if __name__ == "__main__":
    main()