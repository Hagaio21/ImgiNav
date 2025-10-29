#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Union


def articleize(label: str) -> str:
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


def graph2text(graph_path: Union[str, Path], taxonomy: Union[dict, object], max_edges: int = 10_000) -> str:
    path = Path(graph_path)
    
    if not path.exists():
        return ""
    
    g = json.loads(path.read_text(encoding="utf-8"))

    nodes = g.get("nodes", [])
    edges = g.get("edges", [])
    if not edges:
        return ""

    is_scene_graph = "room_a" in edges[0] or "room_b" in edges[0]

    # Extract taxonomy dict if Taxonomy instance is provided
    if hasattr(taxonomy, 'data'):
        # Taxonomy class instance
        id2room = taxonomy.data.get("id2room", {})
    else:
        # Already a dict
        id2room = taxonomy if isinstance(taxonomy, dict) else {}

    # build node label map
    id_to_label = {}
    for n in nodes:
        if is_scene_graph:
            rid = n.get("room_id")
            if hasattr(taxonomy, 'id_to_name'):
                # Taxonomy instance - use id_to_name
                raw_label = taxonomy.id_to_name(rid) if rid else n.get("room_type", str(rid))
            else:
                # Dict lookup
                raw_label = id2room.get(str(rid), id2room.get(rid, n.get("room_type", str(rid))))
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

