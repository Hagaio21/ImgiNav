import argparse
import json
import colorsys
from pathlib import Path
from tqdm import tqdm


# ------------------------------
# Color utilities
# ------------------------------

def _distinct_colors(n: int):
    """Generate n visually distinct RGB colors using HSV."""
    colors = []
    for i in range(n):
        h = i / n
        s, v = 0.65, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors


# ------------------------------
# Taxonomy building
# ------------------------------

def _make_flat_mapping(items):
    """Return mapping {name: id} and ordered list of ids."""
    items = sorted(list(items))
    mapping = {name: i + 1 for i, name in enumerate(items)}
    return mapping, list(mapping.values())



def build_taxonomy(model_info_path: str, scenes_dir: str, out_path: str) -> None:
    """
    Build taxonomy from 3D-FUTURE (labels, categories, super) + 3D-FRONT (rooms).
    """
    model_info_path = Path(model_info_path)
    scenes_dir = Path(scenes_dir)

    if not model_info_path.exists():
        raise FileNotFoundError(f"model_info.json not found: {model_info_path}")

    with open(model_info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    # collect sets
    label_set, category_set, super_set = set(), set(), set()
    for model in model_info:
        label = model.get("label") or model.get("name") or "unknown"
        category = model.get("category") or "unknown"
        supercat = model.get("super-category") or "Other"

        label_set.add(label)
        category_set.add(category)
        super_set.add(supercat)

    # collect room types with progress bar
    room_set = set()
    scene_files = list(scenes_dir.glob("*.json"))
    for scene_file in tqdm(scene_files, desc="[Building taxonomy] Scanning scenes"):
        with open(scene_file, "r", encoding="utf-8") as f:
            scene = json.load(f)
        for room in scene.get("rooms", []):
            room_set.add(room.get("roomType", "OtherRoom"))

    # build mappings
    label2id, label_ids = _make_flat_mapping(label_set)
    category2id, category_ids = _make_flat_mapping(category_set)
    super2id, super_ids = _make_flat_mapping(super_set)
    merged2id = dict(super2id)  # copy
    room2id, room_ids = _make_flat_mapping(room_set)

    # assign colors for ALL ids
    all_ids = label_ids + category_ids + super_ids + room_ids
    colors = _distinct_colors(len(all_ids))
    id2color = {str(i): c for i, c in zip(all_ids, colors)}

    taxonomy = {
        "label2id": label2id,
        "room2id": room2id,
        "category2id": category2id,
        "super2id": super2id,
        "merged2id": merged2id,
        "id2color": id2color,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"[INFO] Saved taxonomy to {out_path}")
    print(f"  {len(label2id)} labels, {len(category2id)} categories, "
          f"{len(super2id)} super-categories, {len(room2id)} room types, "
          f"{len(id2color)} colors")


# ------------------------------
# Loaders
# ------------------------------

def load_taxonomy(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------
# Translation functions
# ------------------------------

def label2id(label: str, taxonomy: dict) -> int:
    return taxonomy["label2id"].get(label, 0)

def id2label(idx: int, taxonomy: dict) -> str:
    for k, v in taxonomy["label2id"].items():
        if v == idx:
            return k
    return "unknown"


def category2id(cat: str, taxonomy: dict) -> int:
    return taxonomy["category2id"].get(cat, 0)

def id2category(idx: int, taxonomy: dict) -> str:
    for k, v in taxonomy["category2id"].items():
        if v == idx:
            return k
    return "unknown"


def super2id(supercat: str, taxonomy: dict) -> int:
    return taxonomy["super2id"].get(supercat, 0)

def id2super(idx: int, taxonomy: dict) -> str:
    for k, v in taxonomy["super2id"].items():
        if v == idx:
            return k
    return "unknown"


def merged2id(m: str, taxonomy: dict) -> int:
    return taxonomy["merged2id"].get(m, 0)

def id2merged(idx: int, taxonomy: dict) -> str:
    for k, v in taxonomy["merged2id"].items():
        if v == idx:
            return k
    return "unknown"


def room2id(room: str, taxonomy: dict) -> int:
    return taxonomy["room2id"].get(room, 0)

def id2room(idx: int, taxonomy: dict) -> str:
    for k, v in taxonomy["room2id"].items():
        if v == idx:
            return k
    return "unknown"


def id2color(idx: int, taxonomy: dict) -> tuple:
    return tuple(taxonomy["id2color"].get(str(idx), (127, 127, 127)))

def color2id(color: tuple, taxonomy: dict) -> int:
    for k, v in taxonomy["id2color"].items():
        if tuple(v) == tuple(color):
            return int(k)
    return 0


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build taxonomy from 3D-FUTURE and 3D-FRONT.")
    parser.add_argument("--model-info", required=True, help="Path to 3D-FUTURE model_info.json")
    parser.add_argument("--scenes-dir", required=True, help="Path to 3D-FRONT scenes directory (with scene JSONs)")
    parser.add_argument("--out", required=True, help="Output taxonomy.json path")

    args = parser.parse_args()
    print("[INFO] Building taxonomoy")
    build_taxonomy(args.model_info, args.scenes_dir, args.out)



if __name__ == "__main__":

    main()
