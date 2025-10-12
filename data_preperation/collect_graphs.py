#!/usr/bin/env python3
"""
collect_graph_manifest.py

Scans filesystem for existing graph JSON files and creates manifest.
"""

import argparse
import csv
from pathlib import Path


def collect_graphs(root_dir: Path):
    """Scan filesystem for graph JSON files that exist."""
    graphs = []
    
    print("Scanning for graph files...")
    
    for graph_path in root_dir.rglob("*_graph.json"):
        filename = graph_path.stem
        
        if filename.endswith("_scene_graph"):
            scene_id = filename.replace("_scene_graph", "")
            graph_type = "scene"
            room_id = "scene"
        else:
            parts = filename.replace("_graph", "").split("_")
            if len(parts) >= 2:
                room_id = parts[-1]
                scene_id = "_".join(parts[:-1])
                graph_type = "room"
            else:
                continue
        
        if graph_type == "scene":
            layout_filename = f"{scene_id}_scene_layout.png"
        else:
            layout_filename = f"{scene_id}_{room_id}_room_seg_layout.png"
        
        layout_path = graph_path.parent / layout_filename
        
        graphs.append({
            'scene_id': scene_id,
            'type': graph_type,
            'room_id': room_id,
            'layout_path': str(layout_path) if layout_path.exists() else '',
            'graph_path': str(graph_path),
            'is_empty': 'false'
        })
    
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    output_path = Path(args.output)
    
    if not root_dir.exists():
        print(f"[error] Directory not found: {root_dir}")
        return
    
    graphs = collect_graphs(root_dir)
    
    if not graphs:
        print("[error] No graphs found")
        return
    
    scene_count = sum(1 for g in graphs if g['type'] == 'scene')
    room_count = sum(1 for g in graphs if g['type'] == 'room')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['scene_id', 'type', 'room_id', 'layout_path', 'graph_path', 'is_empty']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(graphs)
    
    print(f"\nTotal graphs: {len(graphs)}")
    print(f"Scene graphs: {scene_count}")
    print(f"Room graphs:  {room_count}")
    print(f"\nManifest: {output_path}")


if __name__ == "__main__":
    main()