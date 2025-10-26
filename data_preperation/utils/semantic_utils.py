#!/usr/bin/env python3
"""
taxonomy_builder.py

Build a unified taxonomy for 3D-FRONT scenes and 3D-FUTURE models.

- Scans scene JSON files for furniture, titles, and rooms
- Resolves labels/categories/super-categories using model_info.json (Stage 1 logic)
- Adds structural classes (floor/wall/ceiling)
- Assigns IDs to supers, categories, rooms, labels, and titles in fixed ranges
- Generates category→super mapping and a color palette
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import colorsys


# ------------------------------
# Class
# ------------------------------
class Taxonomy:

    def __init__(self, taxonomy_path: str | Path):
        with open(Path(taxonomy_path), "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.ranges = self.data.get("ranges", {})
        self.structural_categories = {'wall', 'floor', 'ceiling'}



    # ----------------------------
    # Core name ↔ id
    # ----------------------------
    def id_to_name(self, val: int) -> str:
        val_str = str(val)
        if self.ranges["super"][0] <= val <= self.ranges["super"][1]:
            return self.data["id2super"].get(val_str, "UnknownSuper")
        if self.ranges["category"][0] <= val <= self.ranges["category"][1]:
            return self.data["id2category"].get(val_str, "UnknownCategory")
        if self.ranges["room"][0] <= val <= self.ranges["room"][1]:
            return self.data["id2room"].get(val_str, "UnknownRoom")
        if self.ranges["label"][0] <= val <= self.ranges["label"][1]:
            return self.data["id2label"].get(val_str, "UnknownLabel")
        if self.ranges["title"][0] <= val <= self.ranges["title"][1]:
            return self.data["id2title"].get(val_str, "UnknownTitle")
        return "Unknown"

    def name_to_id(self, val: str) -> int:
        return (
            self.data.get("room2id", {}).get(val)
            or self.data.get("title2id", {}).get(val)
            or self.data.get("label2id", {}).get(val)
            or self.data.get("category2id", {}).get(val)
            or self.data.get("super2id", {}).get(val, 0)
            or 0
        )


    def translate(self, val, output: str = "id"):
        """
        Generic translator.
        Args:
            val: int or str
            output: 'id' or 'name'
        """
        if isinstance(val, int):
            return self.id_to_name(val) if output == "name" else val
        elif isinstance(val, str):
            return self.name_to_id(val) if output == "id" else val
        return None

    def get_floor_ids(self) -> list[int]:
        """
        Return taxonomy IDs that correspond to 'floor' surfaces.
        Includes both category and label definitions.
        """
        floor_ids = []
        for name, idx in self.data.get("category2id", {}).items():
            if name.lower() == "floor":
                floor_ids.append(int(idx))
        for name, idx in self.data.get("label2id", {}).items():
            if name.lower() == "floor":
                floor_ids.append(int(idx))
        return floor_ids

    # ----------------------------
    # Super-category
    # ----------------------------
    def get_sup(self, val, output: str = "name"):
        """
        Get super-category for a given category/title/etc.
        Args:
            val: str name or int id
            output: 'name' or 'id'
        """
        if isinstance(val, int):
            name = self.id_to_name(val)
        else:
            name = val

        super_name = (
            self.data.get("title2super", {}).get(name)
            or self.data.get("category2super", {}).get(name)
            or (name if name in self.data["super2id"] else "UnknownSuper")
        )

        if output == "name":
            return super_name
        return self.data["super2id"].get(super_name, 0)
    
    def get_category(self, title: str) -> str:
        return self.data.get("title2category", {}).get(title, "UnknownCategory")

    def get_room_id(self, room_type: str) -> int:
        return self.data.get("room2id", {}).get(room_type, 0)

    def get_color(self, val, mode: str = "none"):
        """
        Get color for any taxonomy ID or name.
        For structural elements (floor/wall/ceiling), always returns category color (2000 range).
        For other elements, returns super-category color.
        Returns RGB tuple.
        """
        default_color = (127, 127, 127)  # gray fallback

        if val is None:
            return default_color

        # Convert to int ID if it's a string
        if isinstance(val, str):
            if val.isdigit():
                val = int(val)
            else:
                # Convert name to ID
                val = self.name_to_id(val)
                if val == 0:  # name_to_id returns 0 for unknown
                    return default_color

        # Get the name to check if it's structural
        name = self.id_to_name(val)
        
        # Special handling for structural elements - ALWAYS use category color (2000 range)
        if name.lower() in self.structural_categories:
            # Find the corresponding category ID (2000 range) for this structural element
            category_id = self.data.get("category2id", {}).get(name.lower())
            if category_id is not None:
                color = self.data.get("id2color", {}).get(str(category_id), default_color)
                return tuple(color) if isinstance(color, list) else default_color
            else:
                print(f"DEBUG: No category found for structural element '{name}'")
                return default_color

        # For non-structural items, resolve to super-category and get color
        super_id = self.resolve_super(val)
        if super_id is None:
            print(f"DEBUG get_color: could not resolve super for {val} ({name})")
            return default_color

        # Look up color by super-category ID in id2color
        color = self.data.get("id2color", {}).get(str(super_id), default_color)
        return tuple(color) if isinstance(color, list) else default_color


# Additional helper method for debugging specific failures
    def debug_title_resolution(self, title_id):
        """
        Debug why a specific title ID can't resolve to super-category.
        """
        print(f"\n--- Debug title resolution for ID: {title_id} ---")
        
        title_name = self.id_to_name(title_id)
        print(f"Title ID {title_id} → Name: '{title_name}'")
        
        # Check title2super mapping
        title2super_direct = self.data.get("title2super", {}).get(str(title_id))
        print(f"Direct title2super lookup: {title2super_direct}")
        
        # Check title2category mapping
        title2category = self.data.get("title2category", {}).get(str(title_id))
        print(f"title2category lookup: {title2category}")
        
        if title2category:
            # Check if category has super mapping
            category2super = self.data.get("category2super", {}).get(title2category)
            print(f"category2super for '{title2category}': {category2super}")
            
            if category2super:
                super_id = self.data.get("super2id", {}).get(category2super)
                print(f"super2id for '{category2super}': {super_id}")
        
        # Check available mappings for debugging
        print(f"\nAvailable title2super keys (first 10): {list(self.data.get('title2super', {}).keys())[:10]}")
        print(f"Available title2category keys (first 10): {list(self.data.get('title2category', {}).keys())[:10]}")
        
        print("--- End debug ---\n")


    # Fixed resolve_super method with better case handling and fallbacks
    def resolve_super(self, val: int) -> int | None:
        """
        Resolve any ID (title, label, category, super) to its super-category ID.
        Returns the super-category ID or None if not found.
        """
        if val is None or val == 0:
            return None
        
        val_str = str(val)
        
        # Already a super-category ID
        if self.ranges["super"][0] <= val <= self.ranges["super"][1]:
            return val

        # Title ID → super-category ID
        if self.ranges["title"][0] <= val <= self.ranges["title"][1]:
            # Try direct title → super mapping (using title ID as key)
            super_name = self.data.get("title2super", {}).get(val_str)
            if super_name:
                # Try exact match first
                super_id = self.data.get("super2id", {}).get(super_name)
                if super_id:
                    return super_id
                
                # Try case variations
                for key in self.data.get("super2id", {}):
                    if key.lower() == super_name.lower():
                        return self.data["super2id"][key]
                
                # If "others", try common variations
                if super_name.lower() in ["others", "other"]:
                    for variant in ["Others", "Other", "others", "other", "UnknownSuper"]:
                        super_id = self.data.get("super2id", {}).get(variant)
                        if super_id:
                            return super_id
            
            # Fallback: title → category → super (using title ID as key)
            category_name = self.data.get("title2category", {}).get(val_str)
            if category_name:
                super_name = self.data.get("category2super", {}).get(category_name)
                if super_name:
                    super_id = self.data.get("super2id", {}).get(super_name)
                    if super_id:
                        return super_id
                    
                    # Try case variations for category→super lookup too
                    for key in self.data.get("super2id", {}):
                        if key.lower() == super_name.lower():
                            return self.data["super2id"][key]
            
            # Additional fallback: try using title name as key (legacy support)
            title_name = self.id_to_name(val)
            super_name = self.data.get("title2super", {}).get(title_name)
            if super_name:
                super_id = self.data.get("super2id", {}).get(super_name)
                if super_id:
                    return super_id
                
                # Try case variations
                for key in self.data.get("super2id", {}):
                    if key.lower() == super_name.lower():
                        return self.data["super2id"][key]
            
            category_name = self.data.get("title2category", {}).get(title_name)
            if category_name:
                super_name = self.data.get("category2super", {}).get(category_name)
                if super_name:
                    super_id = self.data.get("super2id", {}).get(super_name)
                    if super_id:
                        return super_id

        # Label ID → category → super-category
        if self.ranges["label"][0] <= val <= self.ranges["label"][1]:
            # Get label name, then find its category
            label_name = self.id_to_name(val)
            if label_name.lower() in self.structural_categories:
                # For structural labels, find the corresponding category
                category_name = label_name.lower()  # floor label -> floor category
                super_name = self.data.get("category2super", {}).get(category_name)
                if super_name:
                    super_id = self.data.get("super2id", {}).get(super_name)
                    if super_id:
                        return super_id
            else:
                # For non-structural labels, use label2category mapping if it exists
                category_id = self.data.get("label2category", {}).get(val_str)
                if category_id:
                    category_name = self.id_to_name(int(category_id))
                    super_name = self.data.get("category2super", {}).get(category_name)
                    if super_name:
                        super_id = self.data.get("super2id", {}).get(super_name)
                        if super_id:
                            return super_id

        # Category ID → super-category
        if self.ranges["category"][0] <= val <= self.ranges["category"][1]:
            category_name = self.id_to_name(val)
            super_name = self.data.get("category2super", {}).get(category_name)
            if super_name:
                super_id = self.data.get("super2id", {}).get(super_name)
                if super_id:
                    return super_id

        # Room IDs don't have super-categories
        if self.ranges["room"][0] <= val <= self.ranges["room"][1]:
            return None

        return None


    # Additional helper method for debugging
    def debug_color_resolution(self, val):
        """
        Debug method to trace color resolution process.
        """
        print(f"\n--- Debug color resolution for: {val} ---")
        
        # Convert to ID if needed
        original_val = val
        if isinstance(val, str) and not val.isdigit():
            val = self.name_to_id(val)
            print(f"Converted '{original_val}' to ID: {val}")
        elif isinstance(val, str):
            val = int(val)
        
        if val == 0:
            print("Result: Unknown item, using default color")
            return
        
        name = self.id_to_name(val)
        print(f"ID {val} → Name: '{name}'")
        
        # Check range
        for range_name, (start, end) in self.ranges.items():
            if start <= val <= end:
                print(f"Range: {range_name} ({start}-{end})")
                break
        
        # Check if structural
        if name.lower() in self.structural_categories:
            print(f"Structural element detected: {name}")
            category_id = self.data.get("category2id", {}).get(name.lower())
            print(f"Corresponding category ID (2000 range): {category_id}")
            if category_id:
                color = self.data.get("id2color", {}).get(str(category_id))
                print(f"Category color: {color}")
            return
        
        # Resolve super for non-structural
        super_id = self.resolve_super(val)
        print(f"Resolved super ID: {super_id}")
        
        if super_id:
            super_name = self.id_to_name(super_id)
            print(f"Super name: {super_name}")
            color = self.data.get("id2color", {}).get(str(super_id))
            print(f"Super color: {color}")
        
        final_color = self.get_color(original_val)
        print(f"Final color: {final_color}")
        print("--- End debug ---\n")
# ------------------------------
# Utilities
# ------------------------------

def _make_flat_mapping(items, base=0, unknown_name="Unknown"):
    """Return mapping {name: id} with explicit 0 reserved for Unknown."""
    items = sorted(list(items))
    mapping = {unknown_name: 0}
    mapping.update({name: base + i + 1 for i, name in enumerate(items)})
    return mapping, list(mapping.values())


def _invert_mapping(mapping: dict) -> dict:
    """Build reverse dict {id: name}."""
    return {v: k for k, v in mapping.items()}

# ------------------------------
# Taxonomy Builder
# ------------------------------

def build_taxonomy_full(model_info_path: Path, scenes_dir: Path):
    """
    Build taxonomy by scanning scene JSONs with Stage 1 resolution logic.
    - Supers come only from model_info.json (+Structure).
    - Collects labels, categories, and titles from furniture + room children.
    - Returns one dict with id mappings and category→super mapping.
    """

    # --- load model_info.json ---
    with open(model_info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    # jid -> (category, super)
    jid_to_cat_super_map = {}
    category_set, super_set = set(), set()
    category2super = {}

    for model in model_info:
        # The key in model_info.json is 'model_id', which corresponds to 'jid' in scene files
        jid = model.get("model_id")
        cat = model.get("category") or "UnknownCategory"
        sup = model.get("super-category") or "UnknownSuper"
        if jid:
            jid_to_cat_super_map[jid] = (cat, sup)
        category_set.add(cat)
        super_set.add(sup)
        if cat != "UnknownCategory":
            category2super[cat] = sup

    # --- inject structural classes ---
    STRUCTURAL = {
        "floor": ("floor", "Structure"),
        "wall": ("wall", "Structure"),
        "ceiling": ("ceiling", "Structure"),
    }
    label_set = set() # only structural labels
    for lbl, (cat, sup) in STRUCTURAL.items():
        label_set.add(lbl)
        category_set.add(cat)
        super_set.add(sup)
        category2super[cat] = sup

    # --- scan scenes to get titles and build title->cat/super mappings ---
    title_set = set()
    title2super = {}
    title2category = {}

    scene_files = list(scenes_dir.glob("*.json"))
    for scene_file in tqdm(scene_files, desc="[Taxonomy] Scanning scenes"):
        with open(scene_file, "r", encoding="utf-8") as f:
            scene = json.load(f)

        def process_furniture(furniture_list):
            for furn in furniture_list:
                jid = furn.get("jid")
                title = furn.get("title")

                if title:
                    title_set.add(title)

                if jid and jid in jid_to_cat_super_map:
                    cat, sup = jid_to_cat_super_map[jid]
                    if title:
                        title2super[title] = sup
                        title2category[title] = cat

        process_furniture(scene.get("furniture", []))
        for room in scene.get("scene", {}).get("room", []):
            process_furniture(room.get("children", []))


    # --- build final mappings ---
    super2id, super_ids = _make_flat_mapping(super_set, base=1000)
    category2id, category_ids = _make_flat_mapping(category_set, base=2000)
    label2id, label_ids = _make_flat_mapping(label_set, base=4000)
    title2id, title_ids = _make_flat_mapping(title_set, base=5000)


    return {
        "super2id": super2id,
        "category2id": category2id,
        "label2id": label2id,
        "title2id": title2id,
        "category2super": category2super,
        "title2super": title2super,
        "title2category": title2category
    }

def build_room_taxonomy(scenes_dir: Path):
    """Collect unique room types and count empty rooms."""
    room_set = set()
    empty_count = 0
    scene_files = list(scenes_dir.glob("*.json"))
    for scene_file in tqdm(scene_files, desc="[Rooms] Scanning scenes"):
        with open(scene_file, "r", encoding="utf-8") as f:
            scene = json.load(f)
        for room in scene.get("scene", {}).get("room", []):
            if room.get("empty", 0) == 1:
                empty_count += 1
            rtype = room.get("type")
            if isinstance(rtype, str) and rtype.strip():
                room_set.add(rtype.strip())
            else:
                room_set.add("OtherRoom")

    room2id, room_ids = _make_flat_mapping(room_set, base=3000, unknown_name="UnknownRoom")
    return room2id, room_ids, empty_count


# ------------------------------
# Color Palette
# ------------------------------
def assign_colors(super2id: dict, category2id: dict, category2super: dict):
    """
    Assign colors:
      - Structural (floor, wall, ceiling) -> fixed distinct colors
      - Unknown -> gray
      - Supers -> anchor hues
      - Categories -> variations of super hue
    """

    id2color = {}

    # 1. Fixed structural colors
    STRUCTURAL_COLORS = {
        "floor": [50, 50, 50],      # dark gray
        "wall": [200, 200, 200],    # light gray
        "ceiling": [255, 255, 255], # white
    }

    for cat, supercat in category2super.items():
        if cat in STRUCTURAL_COLORS:
            cid = category2id[cat]
            id2color[str(cid)] = STRUCTURAL_COLORS[cat]

    # 2. Anchors for non-structural supers
    super_anchors = [
        (228, 26, 28),    # red
        (55, 126, 184),   # blue
        (77, 175, 74),    # green
        (152, 78, 163),   # purple
        (255, 127, 0),    # orange
        (255, 255, 51),   # yellow
        (166, 86, 40),    # brown
        (0, 191, 196),    # cyan
    ]

    supers = sorted(super2id.keys())
    anchor_index = 0

    for supercat in supers:
        sid = super2id[supercat]
        cats = [c for c, s in category2super.items() if s == supercat]

        # Unknown super
        if supercat.lower() in ("unknown", "others", "other"):
            id2color[str(sid)] = [127, 127, 127]
            for cat in cats:
                cid = category2id[cat]
                id2color[str(cid)] = [127, 127, 127]
            continue

        # Structural handled above
        if supercat == "Structure":
            id2color[str(sid)] = [0, 0, 0]  # black for structure super
            continue

        # Assign anchor color to this super
        base_rgb = super_anchors[anchor_index % len(super_anchors)]
        id2color[str(sid)] = list(base_rgb)
        anchor_index += 1

        # Categories under this super
        n = len(cats)
        if n == 0:
            continue
        if n == 1:
            cid = category2id[cats[0]]
            id2color[str(cid)] = list(base_rgb)
            continue

        import colorsys, numpy as np
        r, g, b = [x / 255 for x in base_rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        vals = np.linspace(0.5, 0.95, n)
        sats = np.linspace(0.6, 1.0, n)

        for i, cat in enumerate(sorted(cats)):
            if cat in STRUCTURAL_COLORS:
                continue  # already assigned
            cid = category2id[cat]
            new_s = sats[i % len(sats)]
            new_v = vals[i % len(vals)]
            rr, gg, bb = colorsys.hsv_to_rgb(h, new_s, new_v)
            col = [int(rr * 255), int(gg * 255), int(bb * 255)]
            id2color[str(cid)] = col

    return id2color

# ------------------------------
# Wrapper
# ------------------------------

def build_taxonomy(model_info_path: str, scenes_dir: str, out_path: str) -> None:
    model_info_path = Path(model_info_path)
    scenes_dir = Path(scenes_dir)

    if not model_info_path.exists():
        raise FileNotFoundError(f"model_info.json not found: {model_info_path}")

    taxonomy_dict = build_taxonomy_full(model_info_path, scenes_dir)
    room2id, _, empty_count = build_room_taxonomy(scenes_dir)

    print(f"  Found {len(room2id)} unique room types, {empty_count} empty rooms flagged")

    id2color = assign_colors(
        taxonomy_dict["super2id"],
        taxonomy_dict["category2id"],
        taxonomy_dict["category2super"],
    )

    taxonomy = {
        **taxonomy_dict,
        "room2id": room2id,
        "id2color": id2color,
        "id2label": _invert_mapping(taxonomy_dict["label2id"]),
        "id2category": _invert_mapping(taxonomy_dict["category2id"]),
        "id2super": _invert_mapping(taxonomy_dict["super2id"]),
        "id2room": _invert_mapping(room2id),
        "id2title": _invert_mapping(taxonomy_dict["title2id"]),
        "ranges": {
            "super": [1000, 1999],
            "category": [2000, 2999],
            "room": [3000, 3999],
            "label": [4000, 4999],
            "title": [5000, 5999],
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"[INFO] Saved taxonomy to {out_path}")
    print(f"  {len(taxonomy_dict['label2id'])} labels, "
          f"{len(taxonomy_dict['category2id'])} categories, "
          f"{len(taxonomy_dict['super2id'])} super-categories, "
          f"{len(room2id)} room types, "
          f"{len(taxonomy_dict['title2id'])} titles")


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build taxonomy from 3D-FUTURE and 3D-FRONT.")
    parser.add_argument("--model-info", required=True, help="Path to 3D-FUTURE model_info.json")
    parser.add_argument("--scenes-dir", required=True, help="Path to 3D-FRONT scenes directory")
    parser.add_argument("--out", required=True, help="Output taxonomy.json path")
    args = parser.parse_args()

    print("[INFO] Building taxonomy")
    build_taxonomy(args.model_info, args.scenes_dir, args.out)


if __name__ == "__main__":
    main()
