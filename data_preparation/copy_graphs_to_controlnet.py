#!/usr/bin/env python3
"""
Copy graphs from manifest to ControlNet dataset structure.

This script:
1. Reads graphs manifest CSV
2. Copies JSON graph files to dataset/controlnet/graphs/jsons/
3. Converts graphs to text and saves to dataset/controlnet/graphs/text/
4. Creates embeddings directories
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest
from common.taxonomy import Taxonomy
from data_preparation.utils.text_utils import graph2text


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """Copy a file, creating parent directories if needed."""
    if dst.exists() and not overwrite:
        return True
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """Create and return dictionary of dataset directories."""
    dirs = {
        "graphs_json": base_dir / "controlnet" / "graphs" / "jsons",
        "graphs_text": base_dir / "controlnet" / "graphs" / "text",
        "graphs_json_embeddings": base_dir / "controlnet" / "graphs" / "jsons" / "embeddings",
        "graphs_text_embeddings": base_dir / "controlnet" / "graphs" / "text" / "embeddings",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def copy_graphs_to_dataset(graphs_manifest: Path, output_dirs: Dict[str, Path], 
                           taxonomy: Taxonomy, overwrite: bool = False) -> Dict[str, Tuple[str, str]]:
    """
    Copy graph JSON files and convert to text.
    Returns mapping: original_graph_path -> (json_path, text_path)
    """
    print("Processing graphs...")
    rows = read_manifest(graphs_manifest)
    
    graph_mapping = {}
    skipped = 0
    
    for row in tqdm(rows, desc="Copying and converting graphs"):
        graph_path_str = row.get("graph_path", "")
        if not graph_path_str:
            skipped += 1
            continue
        
        graph_path = Path(graph_path_str)
        if not graph_path.exists():
            print(f"Warning: Graph file not found: {graph_path}")
            skipped += 1
            continue
        
        # Check if empty
        is_empty = row.get("is_empty", "false")
        if str(is_empty).lower() in ("true", "1"):
            skipped += 1
            continue
        
        # Determine output filename
        # Use scene_id and room_id to create unique filename
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        graph_type = row.get("type", "room")
        
        if graph_type == "scene":
            filename = f"{scene_id}_scene_graph.json"
        else:
            filename = f"{scene_id}_{room_id}_room_graph.json"
        
        # Copy JSON
        json_dst = output_dirs["graphs_json"] / filename
        if copy_file(graph_path, json_dst, overwrite):
            # Convert to text using the copied JSON (for consistency)
            try:
                text = graph2text(json_dst, taxonomy)
                if text:
                    text_dst = output_dirs["graphs_text"] / filename.replace(".json", ".txt")
                    text_dst.write_text(text, encoding="utf-8")
                    graph_mapping[graph_path_str] = (str(json_dst), str(text_dst))
                else:
                    print(f"Warning: Empty text for {graph_path}")
                    skipped += 1
            except Exception as e:
                print(f"Error converting graph {graph_path} to text: {e}")
                skipped += 1
    
    print(f"✓ Processed {len(graph_mapping)} graphs, skipped {skipped}")
    return graph_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Copy graphs from manifest to ControlNet dataset structure"
    )
    
    parser.add_argument(
        "--graphs-manifest",
        required=True,
        type=Path,
        help="Path to graphs manifest CSV"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Base directory for dataset (will create controlnet subdirectories)"
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        type=Path,
        help="Path to taxonomy.json"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.graphs_manifest.exists():
        parser.error(f"Graphs manifest not found: {args.graphs_manifest}")
    
    if not args.taxonomy.exists():
        parser.error(f"Taxonomy file not found: {args.taxonomy}")
    
    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = Taxonomy(args.taxonomy)
    
    # Setup directories
    print("Setting up directories...")
    output_dirs = setup_directories(args.dataset_dir)
    
    # Copy and convert graphs
    graph_mapping = copy_graphs_to_dataset(
        args.graphs_manifest,
        output_dirs,
        taxonomy,
        overwrite=args.overwrite
    )
    
    print(f"\n✓ Graphs processing complete!")
    print(f"  Dataset directory: {args.dataset_dir / 'controlnet'}")
    print(f"  Processed {len(graph_mapping)} graphs")


if __name__ == "__main__":
    main()

