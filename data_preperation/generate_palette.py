#!/usr/bin/env python3
import json
from pathlib import Path
import colorsys

def assign_colors(label2id: dict) -> dict:
    """Assign deterministic colors for each label id."""
    ids = sorted(int(v) for v in label2id.values())
    palette = {}
    phi = 0.61803398875  # golden ratio
    for i, lid in enumerate(ids):
        h = (lid * phi) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        palette[str(lid)] = [int(r*255), int(g*255), int(b*255)]
    return palette

def main(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "id2color" in data:
        print("id2color already exists, skipping.")
        return
    palette = assign_colors(data["label2id"])
    data["id2color"] = palette
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✔ Added id2color to {json_path}")

if __name__ == "__main__":
    root = Path("/work3/s233249/ImgiNav/datasets/scenes/semantic_maps.json")  # adjust path
    main(root)
