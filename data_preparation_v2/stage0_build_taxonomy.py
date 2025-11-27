#!/usr/bin/env python3
"""
Stage 0: Taxonomy Generation

Generates deterministic category/supercategory ↔ color mappings from
3D-FUTURE model_info.json and 3D-FRONT scene JSON files.

Output: dataset/taxonomy/taxonomy.json with category_to_color and color_to_category mappings.

Improvements:
- Single pass over scenes (collects room types and titles together)
- Includes doors and windows as distinct categories
- Fixed colors for architectural elements (floor, wall, door, window)
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import colorsys
from tqdm import tqdm


def hash_to_color(category_name: str, saturation: float = 0.8, value: float = 0.9) -> Tuple[int, int, int]:
    """
    Generate a deterministic RGB color from a category name using hash.
    
    Args:
        category_name: Category or supercategory name
        saturation: HSV saturation (0-1)
        value: HSV value/brightness (0-1)
    
    Returns:
        RGB tuple (0-255)
    """
    hash_obj = hashlib.md5(category_name.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Use hash to get hue (0-360 degrees)
    hue = (hash_int % 360) / 360.0
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    return (int(r * 255), int(g * 255), int(b * 255))


def collect_from_scenes(
    model_info_path: Path,
    scenes_dir: Path,
    limit: Optional[int] = None
) -> Tuple[Dict[str, str], Set[str], Set[str], Dict[str, str], Dict[str, str], Set[str]]:
    """
    Single pass over scenes to collect:
    - Categories and supercategories from model_info
    - Title to category/supercategory mappings from scenes
    - Room types from scenes
    
    Returns:
        Tuple of:
        - category_to_super: mapping
        - categories: set
        - supercategories: set
        - title_to_category: mapping
        - title_to_super: mapping
        - room_types: set
    """
    # Load model_info.json
    with open(model_info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)
    
    # Build model_id -> info lookup
    model_info_map = {m["model_id"]: m for m in model_info}
    
    category_to_super = {}
    categories = set()
    supercategories = set()
    
    # Extract from model_info.json
    for model in model_info:
        category = model.get("category") or "UnknownCategory"
        supercategory = model.get("super-category") or "UnknownSuper"
        
        categories.add(category)
        supercategories.add(supercategory)
        category_to_super[category] = supercategory
    
    # Add structural categories (floors, walls, doors, windows)
    structural_categories = {
        "Floor": "Structure",
        "Wall": "Structure",
        "Ceiling": "Structure",
        "Door": "Openings",
        "Window": "Openings",
    }
    
    for cat, sup in structural_categories.items():
        categories.add(cat)
        supercategories.add(sup)
        category_to_super[cat] = sup
    
    # Single pass over scenes
    scene_files = list(scenes_dir.glob("*.json"))
    if limit:
        scene_files = scene_files[:limit]
    
    title_to_category = {}
    title_to_super = {}
    room_types = set()
    
    for scene_file in tqdm(scene_files, desc="Scanning scenes"):
        with open(scene_file, "r", encoding="utf-8") as f:
            scene = json.load(f)
        
        # Build furniture lookup for this scene
        furniture_map = {f['uid']: f for f in scene.get('furniture', [])}
        
        # Process rooms
        for room in scene.get("scene", {}).get("room", []):
            # Collect room type
            room_type = room.get("type")
            if isinstance(room_type, str) and room_type.strip():
                room_types.add(room_type.strip())
            else:
                room_types.add("OtherRoom")
            
            # Process children to collect title -> category mappings
            for child in room.get("children", []):
                ref_id = child.get("ref")
                if ref_id in furniture_map:
                    item = furniture_map[ref_id]
                    jid = item.get("jid")
                    title = item.get("title")
                    
                    if title and jid and jid in model_info_map:
                        model = model_info_map[jid]
                        category = model.get("category") or "UnknownCategory"
                        supercategory = model.get("super-category") or "UnknownSuper"
                        title_to_category[title] = category
                        title_to_super[title] = supercategory
    
    return (
        category_to_super,
        categories,
        supercategories,
        title_to_category,
        title_to_super,
        room_types
    )


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
        "Floor": [60, 60, 60],        # dark gray
        "Wall": [180, 180, 180],      # light gray
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


def build_taxonomy(
    model_info_path: Path,
    scenes_dir: Path,
    out_path: Path,
    limit: Optional[int] = None
) -> None:
    """
    Build taxonomy with deterministic color assignments.
    
    Args:
        model_info_path: Path to 3D-FUTURE model_info.json
        scenes_dir: Path to directory containing 3D-FRONT scene JSON files
        out_path: Output path for taxonomy.json
        limit: Optional limit on number of scenes to scan
    """
    if not model_info_path.exists():
        raise FileNotFoundError(f"model_info.json not found: {model_info_path}")
    
    print("[Stage 0] Building taxonomy...")
    
    # Single pass collection
    (
        category_to_super,
        categories,
        supercategories,
        title_to_category,
        title_to_super,
        room_types
    ) = collect_from_scenes(model_info_path, scenes_dir, limit)
    
    print(f"  Found {len(categories)} categories, {len(supercategories)} supercategories, {len(room_types)} room types")
    
    # Assign colors
    color_mappings = assign_colors(categories, supercategories, category_to_super)
    
    # Build final taxonomy structure
    taxonomy = {
        "version": "2.1",
        "categories": sorted(list(categories)),
        "supercategories": sorted(list(supercategories)),
        "room_types": sorted(list(room_types)),
        "category_to_super": category_to_super,
        "title_to_category": title_to_category,
        "title_to_super": title_to_super,
        **color_mappings
    }
    
    # Save taxonomy
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    
    print(f"[Stage 0] Saved taxonomy to {out_path}")
    print(f"  {len(categories)} categories, {len(supercategories)} supercategories")
    print(f"  {len(room_types)} room types, {len(title_to_category)} titles")
    
    # Print fixed colors for verification
    print("\n  Fixed colors:")
    for cat in ["Floor", "Wall", "Door", "Window"]:
        if cat in color_mappings["category_to_color"]:
            c = color_mappings["category_to_color"][cat]
            print(f"    {cat}: RGB({c[0]}, {c[1]}, {c[2]})")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Build taxonomy with deterministic colors"
    )
    parser.add_argument(
        "--model-info",
        required=True,
        help="Path to 3D-FUTURE model_info.json"
    )
    parser.add_argument(
        "--scenes-dir",
        required=True,
        help="Path to 3D-FRONT scenes directory"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output taxonomy.json path (e.g., dataset/taxonomy/taxonomy.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenes to scan (default: all)"
    )
    args = parser.parse_args()
    
    build_taxonomy(
        Path(args.model_info),
        Path(args.scenes_dir),
        Path(args.out),
        args.limit
    )


if __name__ == "__main__":
    main()