import json
from pathlib import Path

def load_taxonomy(taxonomy_path: str):
    """Load the full taxonomy JSON (includes id2color, id2room, etc.)."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def articleize(label: str) -> str:
    """Add 'the', 'a', or 'an' before a label depending on plurality."""
    clean = label.strip().replace("_", " ")
    lower = clean.lower()

    # heuristic plural detection
    if lower.endswith(("s", "x", "z", "ch", "sh")) and not lower.endswith(("ss", "us")):
        article = "a"
    else:
        # singular
        vowels = "aeiou"
        article = "an" if lower[0] in vowels else "the"
    return f"{article} {clean}"


def graph2text(graph_path: str, taxonomy: dict, max_edges: int = 10_000):
    """
    Converts either a 3D-FRONT room graph or scene graph JSON to text.
    Uses taxonomy to decode room_id when available.
    Removes underscores and adds articles ('the', 'a', 'an').
    """
    path = Path(graph_path)
    
    if not path.exists():
        return ""
    
    g = json.loads(path.read_text(encoding="utf-8"))

    nodes = g.get("nodes", [])
    edges = g.get("edges", [])
    if not edges:
        return ""

    is_scene_graph = "room_a" in edges[0] or "room_b" in edges[0]

    # build node label map
    id_to_label = {}
    for n in nodes:
        if is_scene_graph:
            rid = n.get("room_id")
            raw_label = taxonomy.get(rid, n.get("room_type", str(rid)))
        else:
            raw_label = n.get("label", n.get("id"))
        id_to_label[n["id"]] = articleize(raw_label)

    sentences = []
    seen = set()

    for e in edges[:max_edges]:
        a = e.get("room_a") if is_scene_graph else e.get("obj_a")
        b = e.get("room_b") if is_scene_graph else e.get("obj_b")
        if not a or not b:
            continue

        label_a = id_to_label.get(a)
        label_b = id_to_label.get(b)
        if not label_a or not label_b:
            continue

        key = tuple(sorted([label_a, label_b]))
        if key in seen:
            continue
        seen.add(key)

        dist = e.get("distance_relation")
        direc = e.get("direction_relation")

        if dist and direc:
            sentence = f"{label_a} is {dist} and {direc} {label_b}."
        elif dist:
            sentence = f"{label_a} is {dist} {label_b}."
        elif direc:
            sentence = f"{label_a} is {direc} {label_b}."
        else:
            sentence = f"{label_a} relates to {label_b}."

        sentences.append(sentence)

    text = " ".join(sentences)
    return text.replace("_", " ")
