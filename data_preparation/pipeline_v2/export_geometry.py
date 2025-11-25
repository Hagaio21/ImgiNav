#!/usr/bin/env python3
"""
Standalone script to export scene geometry to OBJ files.
Run this first to create geometry files before rendering.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.geometry_exporter import export_scene_geometry


def main():
    parser = argparse.ArgumentParser(description="Export 3D-FRONT scene geometry to OBJ files")
    parser.add_argument("--scene_json", required=True, help="Path to 3D-FRONT JSON file")
    parser.add_argument("--future_root", required=True, help="Path to 3D-FUTURE model directory")
    parser.add_argument("--output_dir", required=True, help="Output dataset root directory")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    
    args = parser.parse_args()
    
    # Load taxonomy
    taxonomy = Taxonomy(Path(args.taxonomy))
    
    # Export geometry
    scene_metadata = export_scene_geometry(
        Path(args.scene_json),
        Path(args.future_root),
        taxonomy,
        Path(args.output_dir)
    )
    
    print(f"\n✓ Successfully exported geometry for scene: {scene_metadata['scene_id']}")
    print(f"  - Regular GLB: {scene_metadata['scene_id']}.glb")
    print(f"  - Segmented GLB: {scene_metadata['scene_id']}_seg.glb")
    print(f"  - Metadata JSON: {scene_metadata['scene_id']}_metadata.json")
    print(f"  - Rooms: {scene_metadata['statistics']['room_count']}")


if __name__ == "__main__":
    main()

