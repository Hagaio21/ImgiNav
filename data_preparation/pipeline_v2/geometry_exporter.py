#!/usr/bin/env python3
"""
Export scene geometry to GLB files (regular and segmented).
This is fast and should be done first, before rendering.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.scene_loader import load_front_scene, extract_rooms_from_scene


def create_segmented_scene(trimesh_scene: trimesh.Scene, taxonomy: Taxonomy) -> trimesh.Scene:
    """
    Create a new scene with vertex colors set based on taxonomy category colors.
    
    Args:
        trimesh_scene: Original scene
        taxonomy: Taxonomy object
        
    Returns:
        New trimesh.Scene with taxonomy colors as vertex colors
    """
    seg_scene = trimesh.Scene()
    
    for node_name in trimesh_scene.graph.nodes_geometry:
        try:
            transform, geometry_name = trimesh_scene.graph.get(node_name)
            if geometry_name not in trimesh_scene.geometry:
                if node_name not in trimesh_scene.geometry:
                    continue
                geometry = trimesh_scene.geometry[node_name]
            else:
                geometry = trimesh_scene.geometry[geometry_name]
            
            if isinstance(geometry, trimesh.Trimesh):
                mesh_copy = geometry.copy()
                metadata = getattr(geometry, 'metadata', {})
                
                # Get category color
                category_id = metadata.get('category_id', 0)
                color_rgb = taxonomy.get_color(category_id, mode="category")
                if color_rgb is None:
                    color_rgb = (127, 127, 127)
                
                # Apply color to all vertices
                num_vertices = len(mesh_copy.vertices)
                vertex_colors = np.tile(np.array(color_rgb, dtype=np.uint8), (num_vertices, 1))
                mesh_copy.visual.vertex_colors = vertex_colors
                
                # Preserve metadata
                if hasattr(geometry, 'metadata'):
                    mesh_copy.metadata = geometry.metadata.copy()
                
                seg_scene.add_geometry(mesh_copy, node_name=node_name, transform=transform)
        except (KeyError, ValueError, IndexError):
            continue
    
    return seg_scene


def export_scene_geometry(scene_json: Path, future_root: Path, taxonomy: Taxonomy, 
                         output_dir: Path) -> Dict:
    """
    Export scene geometry to GLB files and metadata JSON.
    
    Args:
        scene_json: Path to 3D-FRONT JSON file
        future_root: Path to 3D-FUTURE model directory
        taxonomy: Taxonomy object
        output_dir: Output directory (geometry subdirectory will be created)
        
    Returns:
        Dictionary with scene metadata
    """
    scene_id = scene_json.stem
    geometry_dir = output_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading scene: {scene_id}")
    trimesh_scene = load_front_scene(scene_json, future_root, taxonomy)
    
    print("Extracting rooms from scene...")
    room_scenes = extract_rooms_from_scene(trimesh_scene)
    print(f"  Found {len(room_scenes)} rooms: {list(room_scenes.keys())}")
    
    # Save regular GLB
    print("Saving scene geometry...")
    glb_path = geometry_dir / f"{scene_id}.glb"
    trimesh_scene.export(glb_path, file_type="glb")
    print(f"  Saved scene geometry: {glb_path}")
    
    # Create and save segmented GLB
    print("  Creating segmented GLB...")
    seg_scene = create_segmented_scene(trimesh_scene, taxonomy)
    seg_glb_path = geometry_dir / f"{scene_id}_seg.glb"
    seg_scene.export(seg_glb_path, file_type="glb")
    print(f"  Saved segmented geometry: {seg_glb_path}")
    
    # Save processed scene metadata JSON
    scene_metadata = {
        "scene_id": scene_id,
        "rooms": {room_name: {
            "room_name": room_name,
            "mesh_count": len(list(room_scene.graph.nodes_geometry))
        } for room_name, room_scene in room_scenes.items()},
        "total_meshes": len(list(trimesh_scene.graph.nodes_geometry)),
        "room_count": len(room_scenes)
    }
    metadata_path = geometry_dir / f"{scene_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(scene_metadata, f, indent=2)
    print(f"  Saved scene metadata: {metadata_path}")
    
    return scene_metadata

