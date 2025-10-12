#!/usr/bin/env python3
"""
stage8_create_graph_embeddings.py

Reads graphs.csv manifest, converts graphs to text, generates embeddings,
and creates graphs_with_embeddings.csv manifest.
"""

import argparse
import csv
import json
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_taxonomy(taxonomy_path: str):
    """Load taxonomy mapping from taxonomy.json."""
    t = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in t.get("id2room", {}).items()}


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


def process_graphs(manifest_path: str, taxonomy_path: str, output_manifest: str, 
                   model_name: str = "all-MiniLM-L6-v2", save_format: str = "pt"):
    """
    Process all graphs in manifest: convert to text, generate embeddings, save.
    
    Args:
        manifest_path: Path to graphs.csv
        taxonomy_path: Path to taxonomy.json
        output_manifest: Path for graphs_with_embeddings.csv
        model_name: SentenceTransformer model to use
        save_format: 'pt' for PyTorch or 'npy' for NumPy
    """
    # Load taxonomy
    print(f"Loading taxonomy from {taxonomy_path}")
    taxonomy = load_taxonomy(taxonomy_path)
    
    # Load embedding model
    print(f"Loading SentenceTransformer model: {model_name}")
    embedder = SentenceTransformer(model_name)
    
    # Read manifest
    print(f"Reading manifest: {manifest_path}")
    manifest_path = Path(manifest_path)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} graphs to process")
    
    # Process each graph
    output_rows = []
    skipped = 0
    
    for row in tqdm(rows, desc="Processing graphs"):
        graph_path = row['graph_path']
        
        # Convert graph to text
        try:
            text = graph2text(graph_path, taxonomy)
            
            if not text:
                print(f"Warning: Empty text for {graph_path}")
                skipped += 1
                # Add row without embedding
                output_row = row.copy()
                output_row['embedding_path'] = ''
                output_rows.append(output_row)
                continue
            
            # Generate embedding
            embedding = embedder.encode(text, normalize_embeddings=True)
            
            # Determine save path
            graph_path_obj = Path(graph_path)
            if save_format == "pt":
                embedding_path = graph_path_obj.with_suffix('.pt')
                torch.save(torch.from_numpy(embedding), embedding_path)
            else:  # npy
                embedding_path = graph_path_obj.with_suffix('.npy')
                np.save(embedding_path, embedding)
            
            # Add to output manifest
            output_row = row.copy()
            output_row['embedding_path'] = str(embedding_path)
            output_rows.append(output_row)
            
        except Exception as e:
            print(f"Error processing {graph_path}: {e}")
            skipped += 1
            output_row = row.copy()
            output_row['embedding_path'] = ''
            output_rows.append(output_row)
    
    # Write output manifest
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(rows[0].keys()) + ['embedding_path']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"\n✓ Processed {len(rows) - skipped}/{len(rows)} graphs successfully")
    print(f"✓ Skipped {skipped} graphs")
    print(f"✓ Output manifest: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for graph files and create updated manifest"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to graphs.csv manifest"
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Path to taxonomy.json"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output graphs_with_embeddings.csv"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--format",
        choices=["pt", "npy"],
        default="pt",
        help="Embedding save format: pt (PyTorch) or npy (NumPy)"
    )
    
    args = parser.parse_args()
    
    process_graphs(
        manifest_path=args.manifest,
        taxonomy_path=args.taxonomy,
        output_manifest=args.output,
        model_name=args.model,
        save_format=args.format
    )


if __name__ == "__main__":
    main()