#!/usr/bin/env python3
"""
Update taxonomy colors from old taxonomy.json to new taxonomy.json
Maps colors from config/taxonomy.json (id2color) to data_preparation_v2/taxonomy.json (category_to_color)
"""

import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path):
    """Save JSON file with proper formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_taxonomy_colors(old_taxonomy_path: Path, new_taxonomy_path: Path):
    """Update colors in new taxonomy from old taxonomy."""
    print(f"Loading old taxonomy from: {old_taxonomy_path}")
    old_tax = load_json(old_taxonomy_path)
    
    print(f"Loading new taxonomy from: {new_taxonomy_path}")
    new_tax = load_json(new_taxonomy_path)
    
    # Get mappings from old taxonomy
    id2category = old_tax.get("id2category", {})
    id2color = old_tax.get("id2color", {})
    id2super = old_tax.get("id2super", {})
    
    # Get supercategory colors
    super2id = old_tax.get("super2id", {})
    supercategory_to_color = {}
    for super_name, super_id in super2id.items():
        if str(super_id) in id2color:
            supercategory_to_color[super_name] = id2color[str(super_id)]
    
    # Map category colors
    category_to_color = {}
    color_to_category = {}
    color_to_super = {}
    
    # Map each category ID to its color
    for cat_id_str, cat_name in id2category.items():
        if cat_id_str in id2color:
            color = id2color[cat_id_str]
            category_to_color[cat_name] = color
            color_key = f"{color[0]},{color[1]},{color[2]}"
            color_to_category[color_key] = cat_name
    
    # Map supercategory colors
    for super_name, super_id in super2id.items():
        if str(super_id) in id2color:
            color = id2color[str(super_id)]
            supercategory_to_color[super_name] = color
            color_key = f"{color[0]},{color[1]},{color[2]}"
            color_to_super[color_key] = super_name
    
    # Handle special cases: floor, wall, ceiling (lowercase in old, uppercase in new)
    label2id = old_tax.get("label2id", {})
    for label, label_id in label2id.items():
        if str(label_id) in id2color:
            color = id2color[str(label_id)]
            # Map to new taxonomy names
            if label == "floor":
                category_to_color["Floor"] = color
            elif label == "wall":
                category_to_color["Wall"] = color
            elif label == "ceiling":
                category_to_color["Ceiling"] = color
    
    # Handle Door and Window - they might not be in the old taxonomy's category2id
    # Use fixed colors from recolor_taxonomy.py if not found in old taxonomy
    if "Door" not in category_to_color:
        category_to_color["Door"] = [255, 100, 100]  # red/salmon
        print("Added Door color: RGB(255, 100, 100)")
    if "Window" not in category_to_color:
        category_to_color["Window"] = [100, 200, 255]  # light blue
        print("Added Window color: RGB(100, 200, 255)")
    
    # Update color_to_category for Door and Window
    door_color_key = f"{category_to_color['Door'][0]},{category_to_color['Door'][1]},{category_to_color['Door'][2]}"
    window_color_key = f"{category_to_color['Window'][0]},{category_to_color['Window'][1]},{category_to_color['Window'][2]}"
    color_to_category[door_color_key] = "Door"
    color_to_category[window_color_key] = "Window"
    
    # Handle Openings supercategory if it doesn't exist
    if "Openings" not in supercategory_to_color:
        # Use a color between door and window (pinkish)
        supercategory_to_color["Openings"] = [177, 150, 177]  # light purple/pink
        print("Added Openings supercategory color: RGB(177, 150, 177)")
    
    # Update new taxonomy
    print("\nUpdating colors in new taxonomy...")
    new_tax["category_to_color"] = category_to_color
    new_tax["supercategory_to_color"] = supercategory_to_color
    new_tax["color_to_category"] = color_to_category
    new_tax["color_to_super"] = color_to_super
    
    # Count updates
    print(f"\nUpdated {len(category_to_color)} category colors")
    print(f"Updated {len(supercategory_to_color)} supercategory colors")
    
    # Show some examples
    print("\nSample category colors:")
    for i, (cat, color) in enumerate(list(category_to_color.items())[:5]):
        print(f"  {cat}: RGB{tuple(color)}")
    
    print("\nSupercategory colors:")
    for super_name, color in supercategory_to_color.items():
        print(f"  {super_name}: RGB{tuple(color)}")
    
    # Save updated taxonomy
    print(f"\nSaving updated taxonomy to: {new_taxonomy_path}")
    save_json(new_tax, new_taxonomy_path)
    print("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update taxonomy colors from old to new taxonomy")
    parser.add_argument(
        "--old-taxonomy",
        type=str,
        default="config/taxonomy.json",
        help="Path to old taxonomy.json"
    )
    parser.add_argument(
        "--new-taxonomy",
        type=str,
        default="data_preparation_v2/taxonomy.json",
        help="Path to new taxonomy.json"
    )
    
    args = parser.parse_args()
    
    old_path = Path(args.old_taxonomy)
    new_path = Path(args.new_taxonomy)
    
    if not old_path.exists():
        print(f"ERROR: Old taxonomy not found: {old_path}")
        exit(1)
    
    if not new_path.exists():
        print(f"ERROR: New taxonomy not found: {new_path}")
        exit(1)
    
    update_taxonomy_colors(old_path, new_path)

