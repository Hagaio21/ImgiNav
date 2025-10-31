#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
from pathlib import Path
from common.utils import write_json, safe_mkdir
from tqdm import tqdm
import numpy as np
import colorsys



class Taxonomy:

    def __init__(self, taxonomy_path: str | Path):
        with open(Path(taxonomy_path), "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.ranges = self.data.get("ranges", {})
        self.structural_categories = {'wall', 'floor', 'ceiling'}
        
        # Precompute frequently used mappings for performance
        self._init_quick_lookups()
    
    def _init_quick_lookups(self):
        """Precompute direct mappings for super-categories + walls and rooms."""
        # Super-category + wall color mappings (main use case)
        self.color_to_super_id = {}
        self.super_id_to_color = {}
        id2color = self.data.get("id2color", {})
        
        for sid_str, rgb in id2color.items():
            sid = int(sid_str)
            # Only super categories (1000-1999) and wall (2053)
            if (1000 <= sid <= 1999) or sid == 2053:
                rgb_t = tuple(rgb) if isinstance(rgb, list) else tuple(rgb)
                self.color_to_super_id[rgb_t] = sid
                self.super_id_to_color[sid] = rgb_t


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
        """Get all IDs for floor (category or label)."""
        ids = []
        for mapping in [self.data.get("category2id", {}), self.data.get("label2id", {})]:
            ids.extend([int(idx) for name, idx in mapping.items() if name.lower() == "floor"])
        return ids

    def _is_structural(self, name: str) -> bool:
        """Check if a name represents a structural category."""
        return name.lower() in self.structural_categories

    def _normalize_to_id(self, val):
        """Convert val (int/str) to int ID, or None if invalid."""
        if isinstance(val, int):
            return val if val != 0 else None
        if isinstance(val, str):
            if val.isdigit():
                return int(val) if int(val) != 0 else None
            return self.name_to_id(val) if self.name_to_id(val) != 0 else None
        return None

    def _lookup_super_id(self, super_name: str) -> int | None:
        """Internal: Convert super_name to super_id with case-insensitive fallback."""
        if not super_name:
            return None
        
        super2id = self.data.get("super2id", {})
        
        # Exact match first
        super_id = super2id.get(super_name)
        if super_id:
            return super_id
        
        # Case-insensitive search
        super_name_lower = super_name.lower()
        for key in super2id:
            if key.lower() == super_name_lower:
                return super2id[key]
        
        # "others"/"other" variations
        if super_name_lower in ["others", "other"]:
            for variant in ["Others", "Other", "others", "other", "UnknownSuper"]:
                super_id = super2id.get(variant)
                if super_id:
                    return super_id
        
        return None

    def _resolve_category_to_super_id(self, category_name: str) -> int | None:
        """Internal: Resolve category_name → super_name → super_id."""
        if not category_name:
            return None
        super_name = self.data.get("category2super", {}).get(category_name)
        if not super_name:
            return None
        return self._lookup_super_id(super_name)
    
    def _resolve_category(self, val: int) -> int | None:
        """Resolve any ID (title, label, category) to its category ID."""
        if val is None or val == 0:
            return None
        
        val_str = str(val)
        
        # Already a category ID
        if self.ranges["category"][0] <= val <= self.ranges["category"][1]:
            return val
        
        # Title ID → category ID
        if self.ranges["title"][0] <= val <= self.ranges["title"][1]:
            # PRIMARY: Use title name as key (taxonomy stores title2category with names as keys)
            title_name = self.id_to_name(val)
            if title_name and title_name != "UnknownTitle":
                category_name = self.data.get("title2category", {}).get(title_name)
                if category_name:
                    category_id = self.data.get("category2id", {}).get(category_name)
                    if category_id is not None:
                        return int(category_id)
            
            # FALLBACK: Try using title ID as key
            category_name = self.data.get("title2category", {}).get(val_str)
            if category_name:
                category_id = self.data.get("category2id", {}).get(category_name)
                if category_id is not None:
                    return int(category_id)
        
        # Label ID → category ID
        if self.ranges["label"][0] <= val <= self.ranges["label"][1]:
            label_name = self.id_to_name(val)
            if self._is_structural(label_name):
                # For structural labels, the category ID should match the label name
                category_id = self.data.get("category2id", {}).get(label_name.lower())
                if category_id is not None:
                    return int(category_id)
            else:
                # For non-structural labels, try label2category mapping
                category_id = self.data.get("label2category", {}).get(val_str)
                if category_id:
                    return int(category_id) if isinstance(category_id, (int, str)) else None
        
        return None

    def get_sup(self, val, output: str = "name"):

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

    def get_color(self, val, mode: str = "none"):
        """Get color for a given value, with fallback to default color."""
        default_color = (127, 127, 127)

        if val is None:
            return default_color

        # Normalize to ID
        val = self._normalize_to_id(val)
        if val is None:
            return default_color

        # Get the name to check if it's structural
        name = self.id_to_name(val)
        
        # Special handling for structural elements - ALWAYS use category color (2000 range)
        if self._is_structural(name):
            category_id = self.data.get("category2id", {}).get(name.lower())
            if category_id is not None:
                color = self.data.get("id2color", {}).get(str(category_id), default_color)
                return tuple(color) if isinstance(color, list) else default_color
            return default_color

        # Handle mode: "category" vs "super"/"none" (default)
        if mode == "category":
            # Try to resolve to category ID first
            category_id = self._resolve_category(val)
            if category_id is not None:
                color = self.data.get("id2color", {}).get(str(category_id), None)
                if color is not None:
                    return tuple(color) if isinstance(color, list) else default_color
            # Fallback to super if category color not found
            super_id = self.resolve_super(val)
            if super_id and super_id in self.super_id_to_color:
                return self.super_id_to_color[super_id]
            if super_id:
                color = self.data.get("id2color", {}).get(str(super_id), default_color)
                return tuple(color) if isinstance(color, list) else default_color
            return default_color
        else:
            # Default/super mode: use super-category color
            super_id = self.resolve_super(val)
            
            if super_id and super_id in self.super_id_to_color:
                # Use precomputed mapping for fast lookup
                return self.super_id_to_color[super_id]
            
            if super_id is None:
                # FALLBACK: Check if it's a category with direct color assignment
                if self.ranges["category"][0] <= val <= self.ranges["category"][1]:
                    color = self.data.get("id2color", {}).get(str(val), None)
                    if color is not None:
                        return tuple(color) if isinstance(color, list) else default_color
                    # Try resolving category name → super manually
                    category_name = self.id_to_name(val)
                    super_name = self.data.get("category2super", {}).get(category_name)
                    if super_name:
                        super_id_fallback = self._lookup_super_id(super_name)
                        if super_id_fallback and super_id_fallback in self.super_id_to_color:
                            return self.super_id_to_color[super_id_fallback]
                return default_color

            # Fallback to id2color lookup if not in precomputed map
            color = self.data.get("id2color", {}).get(str(super_id), default_color)
            return tuple(color) if isinstance(color, list) else default_color

    def get_color_to_label_dict(self) -> dict:
        """Get color -> label mapping for super-categories + wall (uses precomputed mappings)."""
        color_to_label = {}
        
        # Use precomputed super_id_to_color for faster iteration
        for sid, rgb in self.super_id_to_color.items():
            if 1000 <= sid <= 1999:
                # Super categories
                super_label = self.id_to_name(sid)
                color_to_label[rgb] = {
                    "label_id": sid,
                    "label": super_label
                }
            elif sid == 2053:
                # Wall - get category name first, then map to super
                category_name = self.id_to_name(sid)
                super_label = self.get_sup(category_name, output="name")
                color_to_label[rgb] = {
                    "label_id": sid,
                    "label": super_label
                }
        
        return color_to_label

    def get_color_to_class_dict(self, class_ids: list[int]) -> dict:
        """
        Get RGB tuple -> class index mapping for given IDs.
        
        Args:
            class_ids: List of IDs (can be any mix of label/title/category/super IDs)
        
        Returns:
            dict mapping (R, G, B) tuple -> class_index (0 to len(class_ids)-1)
        """
        rgb_to_class = {}
        for class_idx, val in enumerate(class_ids):
            # Resolve to super ID or use directly if already super/wall
            sid = self.resolve_super(val) if val not in self.super_id_to_color else val
            if sid and sid in self.super_id_to_color:
                rgb_to_class[self.super_id_to_color[sid]] = class_idx
        return rgb_to_class

    def get_room_mappings(self) -> tuple[dict, dict]:
        """
        Get room ID mappings as tuple (id2room, room2id).
        Convenience method for graph building.
        """
        return self.data.get("id2room", {}), self.data.get("room2id", {})
    
    def id2room(self, room_id: int | str) -> str:
        """Get room name from room ID. Convenience method for Tax.id2room(room_id)."""
        room_id_str = str(room_id)
        return self.data.get("id2room", {}).get(room_id_str, f"UnknownRoom_{room_id}")
    
    def get_color_from_super_id(self, super_id: int) -> tuple[int, int, int] | None:
        """Fast lookup: Get RGB color for super-category ID or wall (2053)."""
        return self.super_id_to_color.get(super_id)
    
    def get_super_id_from_color(self, rgb: tuple[int, int, int]) -> int | None:
        """Fast lookup: Get super-category ID or wall ID from RGB color."""
        return self.color_to_super_id.get(rgb)

    def _try_super_lookup(self, name: str, val_str: str) -> int | None:
        """Helper method to try super-category lookup by name and ID."""
        # Try by name first
        if name and name != "UnknownTitle":
            super_name = self.data.get("title2super", {}).get(name)
            if super_name:
                super_id = self._lookup_super_id(super_name)
                if super_id:
                    return super_id
        
        # Try by ID as fallback
        super_name = self.data.get("title2super", {}).get(val_str)
        if super_name:
            super_id = self._lookup_super_id(super_name)
            if super_id:
                return super_id
        
        return None

    def _try_category_to_super_lookup(self, name: str, val_str: str) -> int | None:
        """Helper method to try category → super-category lookup."""
        # Try by name first
        if name and name != "UnknownTitle":
            category_name = self.data.get("title2category", {}).get(name)
            if category_name:
                super_id = self._resolve_category_to_super_id(category_name)
                if super_id:
                    return super_id
        
        # Try by ID as fallback
        category_name = self.data.get("title2category", {}).get(val_str)
        if category_name:
            super_id = self._resolve_category_to_super_id(category_name)
            if super_id:
                return super_id
        
        return None

    def resolve_super(self, val: int) -> int | None:
        """Resolve any ID (title, label, category, super) to its super-category ID."""
        if val is None or val == 0:
            return None
        
        val_str = str(val)
        
        # Already a super-category ID
        if self.ranges["super"][0] <= val <= self.ranges["super"][1]:
            return val

        # Title ID → super-category ID
        if self.ranges["title"][0] <= val <= self.ranges["title"][1]:
            title_name = self.id_to_name(val)
            # Try direct super lookup
            super_id = self._try_super_lookup(title_name, val_str)
            if super_id:
                return super_id
            # Try category → super lookup
            super_id = self._try_category_to_super_lookup(title_name, val_str)
            if super_id:
                return super_id

        # Label ID → category → super-category
        if self.ranges["label"][0] <= val <= self.ranges["label"][1]:
            label_name = self.id_to_name(val)
            if self._is_structural(label_name):
                # For structural labels, find the corresponding category
                category_name = label_name.lower()
                super_id = self._resolve_category_to_super_id(category_name)
                if super_id:
                    return super_id
            else:
                # For non-structural labels, use label2category mapping if it exists
                category_id = self.data.get("label2category", {}).get(val_str)
                if category_id:
                    category_name = self.id_to_name(int(category_id))
                    super_id = self._resolve_category_to_super_id(category_name)
                    if super_id:
                        return super_id

        # Category ID → super-category
        if self.ranges["category"][0] <= val <= self.ranges["category"][1]:
            category_name = self.id_to_name(val)
            super_id = self._resolve_category_to_super_id(category_name)
            if super_id:
                return super_id

        # Room IDs don't have super-categories
        if self.ranges["room"][0] <= val <= self.ranges["room"][1]:
            return None

        return None

def load_valid_colors(taxonomy_path: str | Path, include_background: bool = True):

    tax = Taxonomy(taxonomy_path)
    
    valid_ids = [
        1001, 1002, 1003, 1004, 1005,
        1006, 1007, 1008, 1009,  # super categories
        2051, 2052, 2053         # ceiling, floor, wall
    ]

    if include_background:
        tax.data["id2color"]["9000"] = [255, 255, 255]
        valid_ids.append(9000)

    filtered = {str(i): tax.data["id2color"][str(i)] for i in valid_ids if str(i) in tax.data["id2color"]}
    return filtered, valid_ids

def _make_flat_mapping(items, base=0, unknown_name="Unknown"):
    """Return mapping {name: id} with explicit 0 reserved for Unknown."""
    items = sorted(list(items))
    mapping = {unknown_name: 0}
    mapping.update({name: base + i + 1 for i, name in enumerate(items)})
    return mapping, list(mapping.values())

def _invert_mapping(mapping: dict) -> dict:
    """Build reverse dict {id: name}."""
    return {v: k for k, v in mapping.items()}

def build_taxonomy_full(model_info_path: Path, scenes_dir: Path):

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


def assign_colors(super2id: dict, category2id: dict, category2super: dict):

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


def assign_colors_golden_ratio(label2id: dict) -> dict:

    import colorsys
    
    ids = sorted(int(v) for v in label2id.values())
    palette = {}
    phi = 0.61803398875  # golden ratio
    
    for i, lid in enumerate(ids):
        h = (lid * phi) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        palette[str(lid)] = [int(r*255), int(g*255), int(b*255)]
    
    return palette


def generate_palette_for_labels(json_path: Path) -> bool:

    data = json.loads(json_path.read_text(encoding="utf-8"))
    
    if "id2color" in data:
        print("id2color already exists, skipping.")
        return False
    
    if "label2id" not in data:
        raise ValueError(f"'label2id' not found in {json_path}")
    
    palette = assign_colors_golden_ratio(data["label2id"])
    data["id2color"] = palette
    write_json(data, json_path)
    print(f"✔ Added id2color to {json_path}")
    return True


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
    safe_mkdir(out_path.parent)
    write_json(taxonomy, out_path)

    print(f"[INFO] Saved taxonomy to {out_path}")
    print(f"  {len(taxonomy_dict['label2id'])} labels, "
          f"{len(taxonomy_dict['category2id'])} categories, "
          f"{len(taxonomy_dict['super2id'])} super-categories, "
          f"{len(room2id)} room types, "
          f"{len(taxonomy_dict['title2id'])} titles")

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
