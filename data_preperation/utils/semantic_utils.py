"""
semantic_utils.py
-----------------
Handles taxonomy and semantic mappings for Stage-1:
  - normalize category names
  - build taxonomy from 3D-FUTURE models
  - assign distinct RGB colors per category
  - load taxonomy
  - resolve category string -> (category_id, color)
  - translation helpers: id <-> label <-> color
"""

import json
import colorsys
from pathlib import Path
from typing import Dict, Tuple, List


# ------------------------------------------------------
# Normalization
# ------------------------------------------------------
def _norm(name: str) -> str:
    """Normalize category name (lowercase, strip spaces, unify separators)."""
    if name is None:
        return "unknown"
    return name.strip().lower().replace(" ", "_").replace("-", "_")


# ------------------------------------------------------
# Distinct Color Generator
# ------------------------------------------------------
def _distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct RGB colors using evenly spaced HSV."""
    colors = []
    for i in range(n):
        h = i / n
        s, v = 0.65, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


# ------------------------------------------------------
# Taxonomy
# ------------------------------------------------------
def build_taxonomy(models_dir: str, out_path: str) -> None:
    """
    Build taxonomy from all 3D-FUTURE models.
    Assigns each unique category:
      - unique category_id
      - distinct RGB color
    """
    categories = {}
    model_dirs = [d for d in Path(models_dir).iterdir() if d.is_dir()]

    # collect unique categories
    category_list = []
    for model_dir in model_dirs:
        meta_file = model_dir / "model.json"
        if not meta_file.exists():
            continue
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        category_raw = meta.get("category", "unknown")
        category = _norm(category_raw)

        if category not in category_list:
            category_list.append(category)

    # assign ids + distinct colors
    colors = _distinct_colors(len(category_list))
    for i, category in enumerate(sorted(category_list)):
        categories[category] = {
            "id": i + 1,
            "color": colors[i]
        }

    # save taxonomy
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)


def load_taxonomy(path: str) -> Dict[str, Dict]:
    """Load taxonomy.json into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_category(category_name: str, taxonomy: Dict[str, Dict]) -> Tuple[int, Tuple[int, int, int]]:
    """Resolve category string to (category_id, color)."""
    category = _norm(category_name)
    if category in taxonomy:
        entry = taxonomy[category]
        return entry["id"], tuple(entry["color"])
    return 0, (127, 127, 127)


# ------------------------------------------------------
# Translation helpers
# ------------------------------------------------------
def id2label(category_id: int, taxonomy: Dict[str, Dict]) -> str:
    """Translate category_id -> label."""
    for label, entry in taxonomy.items():
        if entry["id"] == category_id:
            return label
    return "unknown"


def label2id(label: str, taxonomy: Dict[str, Dict]) -> int:
    """Translate label -> category_id."""
    label = _norm(label)
    return taxonomy.get(label, {}).get("id", 0)


def id2color(category_id: int, taxonomy: Dict[str, Dict]) -> Tuple[int, int, int]:
    """Translate category_id -> color."""
    for entry in taxonomy.values():
        if entry["id"] == category_id:
            return tuple(entry["color"])
    return (127, 127, 127)


def color2id(color: Tuple[int, int, int], taxonomy: Dict[str, Dict]) -> int:
    """Translate color -> category_id."""
    for entry in taxonomy.values():
        if tuple(entry["color"]) == tuple(color):
            return entry["id"]
    return 0


def label2color(label: str, taxonomy: Dict[str, Dict]) -> Tuple[int, int, int]:
    """Translate label -> color."""
    label = _norm(label)
    if label in taxonomy:
        return tuple(taxonomy[label]["color"])
    return (127, 127, 127)


def color2label(color: Tuple[int, int, int], taxonomy: Dict[str, Dict]) -> str:
    """Translate color -> label."""
    for label, entry in taxonomy.items():
        if tuple(entry["color"]) == tuple(color):
            return label
    return "unknown"
