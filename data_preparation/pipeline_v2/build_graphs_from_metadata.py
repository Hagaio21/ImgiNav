#!/usr/bin/env python3
"""
Build graphs from metadata JSON files.
This script uses metadata (furniture positions, room centers) instead of layout images.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.graph_builder import (
    build_scene_graph_from_metadata,
    build_room_graph_from_metadata
)


def main():
    parser = argparse.ArgumentParser(description="Build graphs from metadata JSON files")
    parser.add_argument("--scene_id", required=True, help="Scene ID (e.g., 002c110c-9bbc-4ab4-affa-4225fb127bad)")
    parser.add_argument("--output_dir", required=True, help="Output dataset root directory (must contain geometry/ folder)")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    
    args = parser.parse_args()
    
    # Load taxonomy
    taxonomy = Taxonomy(Path(args.taxonomy))
    
    # Load metadata
    geometry_dir = Path(args.output_dir) / "geometry"
    metadata_path = geometry_dir / f"{args.scene_id}_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Run geometry export first.")
    
    with open(metadata_path, 'r') as f:
        scene_metadata = json.load(f)
    
    print(f"Building graphs for scene: {args.scene_id}")
    print(f"Loaded metadata from: {metadata_path}")
    
    output_dir = Path(args.output_dir)
    
    # Build scene-level graph
    print("Building scene-level graph from metadata...")
    try:
        build_scene_graph_from_metadata(
            args.scene_id, scene_metadata, taxonomy, output_dir
        )
        print("  Scene graph built successfully")
    except Exception as e:
        print(f"  ERROR: Failed to build scene graph: {e}")
        import traceback
        traceback.print_exc()
    
    # Build room-level graphs
    if 'rooms' in scene_metadata:
        print(f"Building room-level graphs from metadata...")
        for room_name, room_info in scene_metadata['rooms'].items():
            print(f"  Building graph for room: {room_name}")
            try:
                build_room_graph_from_metadata(
                    args.scene_id, room_name, room_info, taxonomy, output_dir
                )
                print(f"  Room graph built for {room_name}")
            except Exception as e:
                print(f"  ERROR: Failed to build graph for room {room_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"Completed graph building for scene: {args.scene_id}")


if __name__ == "__main__":
    main()

