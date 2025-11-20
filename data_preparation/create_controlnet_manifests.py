#!/usr/bin/env python3
"""
Create ControlNet manifests (tex and seg) from copied dataset files.

This script:
1. Reads the original manifests (layouts, graphs, povs)
2. Finds the copied files in the controlnet dataset structure
3. Creates two manifests (tex and seg) with data points:
   - For rooms: {layout_path, graph_path, pov_path}
   - For scenes: {layout_path, graph_path, "0"}
4. Handles duplicate POVs by taking the first one found per room
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tqdm import tqdm

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest, create_manifest


def find_copied_file(dataset_dir: Path, subdir: str, filename: str) -> Optional[Path]:
    """
    Find a copied file in the controlnet dataset structure.
    
    Args:
        dataset_dir: Base dataset directory
        subdir: Subdirectory in controlnet (e.g., "graphs/jsons", "layouts", "povs/tex")
        filename: Expected filename
    
    Returns:
        Path to copied file, or None if not found
    """
    controlnet_dir = dataset_dir / "controlnet" / subdir
    file_path = controlnet_dir / filename
    
    if file_path.exists() and file_path.is_file():
        return file_path
    
    return None


def build_file_mappings(dataset_dir: Path, layouts_manifest: Path, 
                        graphs_manifest: Path, povs_manifest: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Build mappings from (scene_id, room_id, type) to copied file paths.
    
    Returns:
        layout_mapping: (scene_id, room_id, type) -> layout_path
        graph_mapping: (scene_id, room_id, type) -> (json_path, text_path)
        pov_mapping: (scene_id, room_id, type) -> pov_path
    """
    print("Building file mappings from copied dataset...")
    
    layouts_rows = read_manifest(layouts_manifest)
    graphs_rows = read_manifest(graphs_manifest)
    povs_rows = read_manifest(povs_manifest)
    
    layout_mapping = {}
    graph_mapping = {}
    pov_mapping = {}
    
    # Map layouts
    print("  Mapping layouts...")
    for row in tqdm(layouts_rows, desc="  Layouts"):
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        layout_type = row.get("type", "")
        layout_path_str = row.get("layout_path", "")
        
        if not layout_path_str:
            continue
        
        # Check if empty
        is_empty = row.get("is_empty", "0")
        if str(is_empty) in ("1", "true", "True"):
            continue
        
        # Find copied layout file
        if layout_type == "scene":
            filename = f"{scene_id}_scene_layout.png"
        else:
            filename = f"{scene_id}_{room_id}_room_layout.png"
        
        copied_path = find_copied_file(dataset_dir, "layouts", filename)
        if copied_path:
            key = (scene_id, room_id, layout_type)
            layout_mapping[key] = str(copied_path)
    
    # Map graphs
    print("  Mapping graphs...")
    for row in tqdm(graphs_rows, desc="  Graphs"):
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        graph_type = row.get("type", "")
        graph_path_str = row.get("graph_path", "")
        
        if not graph_path_str:
            continue
        
        # Check if empty
        is_empty = row.get("is_empty", "false")
        if str(is_empty).lower() in ("true", "1"):
            continue
        
        # Normalize room_id for scenes
        if graph_type == "scene":
            room_id = "scene"
        
        # Find copied graph files
        if graph_type == "scene":
            json_filename = f"{scene_id}_scene_graph.json"
            text_filename = f"{scene_id}_scene_graph.txt"
        else:
            json_filename = f"{scene_id}_{room_id}_room_graph.json"
            text_filename = f"{scene_id}_{room_id}_room_graph.txt"
        
        json_path = find_copied_file(dataset_dir, "graphs/jsons", json_filename)
        text_path = find_copied_file(dataset_dir, "graphs/text", text_filename)
        
        if json_path and text_path:
            key = (scene_id, room_id, graph_type)
            graph_mapping[key] = (str(json_path), str(text_path))
    
    # Map POVs
    print("  Mapping POVs...")
    for row in tqdm(povs_rows, desc="  POVs"):
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        pov_type = row.get("type", "")
        pov_path_str = row.get("pov_path", "")
        
        if not pov_path_str:
            continue
        
        # Check if empty
        is_empty = row.get("is_empty", "0")
        if str(is_empty) in ("1", "true", "True"):
            continue
        
        # Find copied POV file (uses original filename)
        pov_filename = Path(pov_path_str).name
        if pov_type == "tex":
            copied_path = find_copied_file(dataset_dir, "povs/tex", pov_filename)
        elif pov_type == "seg":
            copied_path = find_copied_file(dataset_dir, "povs/seg", pov_filename)
        else:
            continue
        
        if copied_path:
            key = (scene_id, room_id, pov_type)
            pov_mapping[key] = str(copied_path)
    
    print(f"  ✓ Mapped {len(layout_mapping)} layouts, {len(graph_mapping)} graphs, {len(pov_mapping)} POVs")
    return layout_mapping, graph_mapping, pov_mapping


def create_manifest_for_pov_type(pov_type: str, layouts_manifest: Path, 
                                  graphs_manifest: Path, povs_manifest: Path,
                                  layout_mapping: Dict, graph_mapping: Dict, 
                                  pov_mapping: Dict, output_path: Path):
    """
    Create a manifest for a specific POV type (tex or seg).
    
    Args:
        pov_type: "tex" or "seg"
        layouts_manifest: Path to layouts manifest
        graphs_manifest: Path to graphs manifest
        povs_manifest: Path to POVs manifest
        layout_mapping: Mapping from (scene_id, room_id, type) to layout_path
        graph_mapping: Mapping from (scene_id, room_id, type) to (json_path, text_path)
        pov_mapping: Mapping from (scene_id, room_id, type) to pov_path
        output_path: Path to output manifest CSV
    """
    print(f"\nCreating {pov_type} manifest...")
    
    layouts_rows = read_manifest(layouts_manifest)
    graphs_rows = read_manifest(graphs_manifest)
    povs_rows = read_manifest(povs_manifest)
    
    data_points = []
    seen_rooms = set()  # Track rooms we've already added (to handle duplicates)
    
    # Step 1: Create data points for rooms with POVs of this type
    print(f"  Processing {pov_type} POVs for rooms...")
    for pov_row in tqdm(povs_rows, desc=f"  {pov_type} POVs"):
        scene_id = pov_row.get("scene_id", "")
        room_id = pov_row.get("room_id", "")
        pov_type_row = pov_row.get("type", "")
        
        # Only process POVs of the target type
        if pov_type_row != pov_type:
            continue
        
        # Check if empty
        is_empty = pov_row.get("is_empty", "0")
        if str(is_empty) in ("1", "true", "True"):
            continue
        
        # Skip if we've already added this room (handle duplicates - take first)
        room_key = (scene_id, room_id)
        if room_key in seen_rooms:
            continue
        
        # Get POV path
        pov_key = (scene_id, room_id, pov_type)
        if pov_key not in pov_mapping:
            continue
        
        pov_path = pov_mapping[pov_key]
        
        # Get matching room's graph and layout
        graph_key = (scene_id, room_id, "room")
        layout_key = (scene_id, room_id, "room")
        
        if graph_key not in graph_mapping or layout_key not in layout_mapping:
            continue
        
        graph_json, graph_text = graph_mapping[graph_key]
        layout_path = layout_mapping[layout_key]
        
        # Create data point: {layout, graph, pov}
        data_point = {
            "scene_id": scene_id,
            "room_id": room_id,
            "type": "room",
            "layout_path": layout_path,
            "graph_json_path": graph_json,
            "graph_text_path": graph_text,
            "pov_path": pov_path,
        }
        data_points.append(data_point)
        seen_rooms.add(room_key)
    
    # Step 2: Create data points for scenes (POV = "0")
    print("  Processing scenes...")
    for layout_row in tqdm(layouts_rows, desc="  Scenes"):
        scene_id = layout_row.get("scene_id", "")
        layout_type = layout_row.get("type", "")
        
        # Only process scenes
        if layout_type != "scene":
            continue
        
        # Check if empty
        is_empty = layout_row.get("is_empty", "0")
        if str(is_empty) in ("1", "true", "True"):
            continue
        
        room_id = "scene"
        layout_key = (scene_id, room_id, "scene")
        
        if layout_key not in layout_mapping:
            continue
        
        layout_path = layout_mapping[layout_key]
        
        # Get scene graph
        graph_key = (scene_id, room_id, "scene")
        if graph_key not in graph_mapping:
            continue
        
        graph_json, graph_text = graph_mapping[graph_key]
        
        # Create data point: {layout, graph, "0"}
        data_point = {
            "scene_id": scene_id,
            "room_id": "scene",
            "type": "scene",
            "layout_path": layout_path,
            "graph_json_path": graph_json,
            "graph_text_path": graph_text,
            "pov_path": "0",
        }
        data_points.append(data_point)
    
    # Step 3: Filter out data points with missing files
    print("  Filtering data points with missing files...")
    initial_count = len(data_points)
    filtered_points = []
    for dp in data_points:
        # Check all required files exist
        layout_path = Path(dp["layout_path"])
        graph_json_path = Path(dp["graph_json_path"])
        graph_text_path = Path(dp["graph_text_path"])
        
        if not (layout_path.exists() and graph_json_path.exists() and graph_text_path.exists()):
            continue
        
        # For rooms, also check POV exists (scenes have pov_path="0")
        if dp["type"] == "room":
            pov_path = Path(dp["pov_path"])
            if not pov_path.exists():
                continue
        
        filtered_points.append(dp)
    
    removed_count = initial_count - len(filtered_points)
    if removed_count > 0:
        print(f"    Removed {removed_count} data points with missing files")
    
    # Step 4: Write manifest
    print(f"  Writing manifest to {output_path}...")
    fieldnames = [
        "scene_id", "room_id", "type",
        "layout_path", "graph_json_path", "graph_text_path", "pov_path"
    ]
    create_manifest(filtered_points, output_path, fieldnames)
    
    room_count = len([dp for dp in filtered_points if dp["type"] == "room"])
    scene_count = len([dp for dp in filtered_points if dp["type"] == "scene"])
    
    print(f"  ✓ Created {len(filtered_points)} data points ({room_count} rooms, {scene_count} scenes)")


def main():
    parser = argparse.ArgumentParser(
        description="Create ControlNet manifests (tex and seg) from copied dataset files"
    )
    
    parser.add_argument(
        "--layouts-manifest",
        required=True,
        type=Path,
        help="Path to layouts manifest CSV"
    )
    parser.add_argument(
        "--graphs-manifest",
        required=True,
        type=Path,
        help="Path to graphs manifest CSV"
    )
    parser.add_argument(
        "--povs-manifest",
        required=True,
        type=Path,
        help="Path to POVs manifest CSV"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Base directory for dataset (should contain controlnet subdirectory)"
    )
    parser.add_argument(
        "--output-tex",
        required=True,
        type=Path,
        help="Path to output tex manifest CSV"
    )
    parser.add_argument(
        "--output-seg",
        required=True,
        type=Path,
        help="Path to output seg manifest CSV"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for manifest_path, name in [
        (args.layouts_manifest, "layouts"),
        (args.graphs_manifest, "graphs"),
        (args.povs_manifest, "POVs"),
    ]:
        if not manifest_path.exists():
            parser.error(f"{name} manifest not found: {manifest_path}")
    
    controlnet_dir = args.dataset_dir / "controlnet"
    if not controlnet_dir.exists():
        parser.error(f"ControlNet dataset directory not found: {controlnet_dir}")
    
    # Build file mappings
    layout_mapping, graph_mapping, pov_mapping = build_file_mappings(
        args.dataset_dir,
        args.layouts_manifest,
        args.graphs_manifest,
        args.povs_manifest
    )
    
    # Create tex manifest
    create_manifest_for_pov_type(
        "tex",
        args.layouts_manifest,
        args.graphs_manifest,
        args.povs_manifest,
        layout_mapping,
        graph_mapping,
        pov_mapping,
        args.output_tex
    )
    
    # Create seg manifest
    create_manifest_for_pov_type(
        "seg",
        args.layouts_manifest,
        args.graphs_manifest,
        args.povs_manifest,
        layout_mapping,
        graph_mapping,
        pov_mapping,
        args.output_seg
    )
    
    print("\n" + "="*60)
    print("✓ ControlNet manifests creation complete!")
    print("="*60)
    print(f"  Tex manifest: {args.output_tex}")
    print(f"  Seg manifest: {args.output_seg}")
    print("="*60)


if __name__ == "__main__":
    main()

