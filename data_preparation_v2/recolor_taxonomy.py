#!/usr/bin/env python3
"""
Recolor Taxonomy

Updates colors in an existing taxonomy.json file without re-scanning scenes.
Uses evenly spaced supercategory colors and derives category colors from supercategories.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set

import colorsys


def assign_colors(
    categories: Set[str],
    supercategories: Set[str],
    category_to_super: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Assign deterministic colors to categories and supercategories.
    
    Supercategories: evenly spaced hues for maximum separation
    Categories: derived from supercategory colors (darker/lighter variations)
    
    Fixed colors for structural elements:
    - Floor: dark gray
    - Wall: light gray
    - Ceiling: white
    - Door: red/salmon (warm)
    - Window: light blue (cool)
    
    Returns:
        Dictionary with:
        - category_to_color: {category_name: [R, G, B]}
        - supercategory_to_color: {supercategory_name: [R, G, B]}
        - color_to_category: {"R,G,B": category_name}
        - color_to_super: {"R,G,B": supercategory_name}
    """
    category_to_color = {}
    supercategory_to_color = {}
    color_to_category = {}
    color_to_super = {}
    
    # Fixed colors for structural elements and openings
    FIXED_COLORS = {
        # Structure (grays)
        "Floor": [180, 180, 180],     # light gray
        "Wall": [60, 60, 60],         # dark gray
        "Ceiling": [220, 220, 220],   # near white
        # Openings (distinct colors)
        "Door": [255, 100, 100],      # red/salmon
        "Window": [100, 200, 255],    # light blue
    }
    
    # Group categories by supercategory for color derivation
    categories_by_super = {}
    for category in categories:
        if category not in FIXED_COLORS:
            sup = category_to_super.get(category, "UnknownSuper")
            if sup not in categories_by_super:
                categories_by_super[sup] = []
            categories_by_super[sup].append(category)
    
    # Assign colors to supercategories with maximum separation
    sorted_supers = sorted(supercategories)
    non_fixed_supers = [s for s in sorted_supers if s.lower() not in ("unknown", "others", "other", "unknownsuper") 
                        and s not in ("Structure", "Openings")]
    
    for sup in sorted_supers:
        if sup.lower() in ("unknown", "others", "other", "unknownsuper"):
            color = [127, 127, 127]  # gray
        elif sup == "Structure":
            color = [100, 100, 100]  # dark gray
        elif sup == "Openings":
            color = [200, 150, 150]  # muted red
        else:
            # Evenly space hues across 360 degrees for maximum separation
            idx = non_fixed_supers.index(sup)
            hue = (idx * 360.0 / len(non_fixed_supers)) / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            color = [int(r * 255), int(g * 255), int(b * 255)]
        
        supercategory_to_color[sup] = color
        color_key = f"{color[0]},{color[1]},{color[2]}"
        color_to_super[color_key] = sup
    
    # Assign colors to categories (derived from supercategory)
    for category in sorted(categories):
        # Fixed colors for structural/opening categories
        if category in FIXED_COLORS:
            color = FIXED_COLORS[category]
        else:
            # Derive from supercategory color
            sup = category_to_super.get(category, "UnknownSuper")
            sup_color = supercategory_to_color.get(sup, [127, 127, 127])
            
            # Get categories in this supercategory for consistent ordering
            cats_in_sup = sorted(categories_by_super.get(sup, []))
            cat_idx = cats_in_sup.index(category)
            num_cats = len(cats_in_sup)
            
            # Convert supercategory RGB to HSV
            r, g, b = [c / 255.0 for c in sup_color]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            
            # Vary the color: alternate between darker and lighter
            # Use index to create consistent variations
            if num_cats == 1:
                # Single category: slightly darker
                new_v = max(0.6, v - 0.1)
                new_s = min(0.95, s + 0.1)
            else:
                # Multiple categories: distribute across value range
                # Alternate between darker (lower value) and lighter (higher value)
                if cat_idx % 2 == 0:
                    # Darker variation
                    value_offset = -0.15 - (cat_idx // 2) * 0.1
                    new_v = max(0.5, v + value_offset)
                    new_s = min(0.95, s + 0.05)
                else:
                    # Lighter variation
                    value_offset = 0.1 + (cat_idx // 2) * 0.05
                    new_v = min(0.95, v + value_offset)
                    new_s = max(0.6, s - 0.05)
            
            # Slight hue shift for additional distinction (within ±10 degrees)
            hue_shift = ((cat_idx % 5) - 2) * 10.0 / 360.0
            new_h = (h + hue_shift) % 1.0
            
            # Convert back to RGB
            r, g, b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
            color = [int(r * 255), int(g * 255), int(b * 255)]
        
        category_to_color[category] = color
        color_key = f"{color[0]},{color[1]},{color[2]}"
        color_to_category[color_key] = category
    
    return {
        "category_to_color": category_to_color,
        "supercategory_to_color": supercategory_to_color,
        "color_to_category": color_to_category,
        "color_to_super": color_to_super
    }


def recolor_taxonomy(taxonomy_path: Path, output_path: Path = None):
    """
    Recolor an existing taxonomy.json file.
    
    Args:
        taxonomy_path: Path to existing taxonomy.json
        output_path: Optional output path (default: overwrite input file)
    """
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
    
    print(f"[Recolor] Loading taxonomy from {taxonomy_path}...")
    
    # Load existing taxonomy
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    
    # Extract data
    categories = set(taxonomy.get("categories", []))
    supercategories = set(taxonomy.get("supercategories", []))
    category_to_super = taxonomy.get("category_to_super", {})
    
    print(f"  Found {len(categories)} categories, {len(supercategories)} supercategories")
    
    # Reassign colors
    print("[Recolor] Reassigning colors...")
    color_mappings = assign_colors(categories, supercategories, category_to_super)
    
    # Update taxonomy with new colors
    taxonomy.update(color_mappings)
    
    # Update version
    taxonomy["version"] = taxonomy.get("version", "2.0").split(".")[0] + ".2"
    
    # Save updated taxonomy
    output_path = output_path or taxonomy_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    
    print(f"[Recolor] Saved recolored taxonomy to {output_path}")
    print(f"  Updated {len(color_mappings['category_to_color'])} category colors")
    print(f"  Updated {len(color_mappings['supercategory_to_color'])} supercategory colors")


def main():
    parser = argparse.ArgumentParser(
        description="Recolor an existing taxonomy.json file"
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Path to existing taxonomy.json file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: overwrite input file)"
    )
    args = parser.parse_args()
    
    recolor_taxonomy(
        Path(args.taxonomy),
        Path(args.output) if args.output else None
    )


if __name__ == "__main__":
    main()

