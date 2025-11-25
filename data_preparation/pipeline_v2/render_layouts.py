#!/usr/bin/env python3
"""
Render layouts only (scene and room layouts, RGB and segmentation).
No POV rendering, no geometry export.
"""

import os
import time

# Try to import xvfbwrapper (like old pipeline)
try:
    from xvfbwrapper import Xvfb
    XVFBWRAPPER_AVAILABLE = True
except ImportError:
    Xvfb = None
    XVFBWRAPPER_AVAILABLE = False

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

# Global variable to hold Xvfb instance
VFB = None

import sys

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.scene_loader import extract_rooms_from_scene
from data_preparation.pipeline_v2.renderer import render_layout_rgb, render_layout_seg

# Set Open3D verbosity to errors only
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def load_scene_from_obj(obj_path: Path) -> trimesh.Scene:
    """
    Load scene from OBJ file.
    OBJ files preserve textures (via MTL files) and vertex colors.
    Use process=False and maintain_order=True to preserve textures and materials.
    """
    print(f"Loading scene from OBJ: {obj_path}")
    # Load with process=False to preserve textures and materials
     # OBJ files need a resolver to find MTL and texture files in materials/ folder
    geometry_dir = obj_path.parent
    materials_dir = geometry_dir / "materials"
    
    # Create resolver that checks materials/ folder first, then geometry/ folder
    class MultiPathResolver:
        def __init__(self, base_dir, materials_dir):
            self.base_dir = Path(base_dir)
            self.materials_dir = Path(materials_dir)
        
        def get(self, name):
            # Remove "materials/" prefix if present (from updated MTL file)
            clean_name = name.replace("materials/", "")
            
            # Check materials/ folder first
            materials_path = self.materials_dir / clean_name
            if materials_path.exists():
                return str(materials_path)
            
            # Fallback to base directory
            base_path = self.base_dir / clean_name
            if base_path.exists():
                return str(base_path)
            
            return None
    
    resolver = MultiPathResolver(geometry_dir, materials_dir)
    
    # Check if OBJ file exists and is not empty
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    
    if obj_path.stat().st_size == 0:
        raise ValueError(f"OBJ file is empty: {obj_path}")
    
    try:
        scene = trimesh.load(str(obj_path), file_type='obj', process=False, maintain_order=True, 
                           force='scene', resolver=resolver)
    except Exception as e:
        raise ValueError(f"Failed to load OBJ file {obj_path}: {e}")
    
    if scene is None:
        raise ValueError(f"OBJ file loaded as None: {obj_path}")
    
    if not isinstance(scene, trimesh.Scene):
        # If it's a single mesh, wrap it in a scene
        if isinstance(scene, trimesh.Trimesh):
            new_scene = trimesh.Scene()
            new_scene.add_geometry(scene)
            scene = new_scene
        else:
            raise ValueError(f"OBJ file did not load as a scene or mesh: {type(scene)}")
    
    # Validate scene has geometry
    if len(scene.geometry) == 0:
        raise ValueError(f"OBJ file has no geometry: {obj_path}")
    
    return scene


def extract_rooms_from_obj_scene(trimesh_scene: trimesh.Scene, scene_metadata: Dict) -> Dict[str, trimesh.Scene]:
    """
    Extract room scenes from OBJ using room bounds from metadata file.
    Matches meshes to rooms by checking if mesh center is within room bounds.
    """
    if scene_metadata is None or 'rooms' not in scene_metadata:
        print("Warning: No room metadata available, returning full scene as single 'room'")
        return {'scene': trimesh_scene}
    
    room_scenes = {}
    
    # Extract rooms using bounds from metadata
    for room_name, room_info in scene_metadata['rooms'].items():
        room_scene = trimesh.Scene()
        room_bounds = room_info.get('bounds', {})
        
        if 'min' not in room_bounds or 'max' not in room_bounds:
            print(f"Warning: Room {room_name} has no bounds, skipping")
            continue
        
        room_min = np.array(room_bounds['min'])
        room_max = np.array(room_bounds['max'])
        
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
                    # Transform vertices to world space
                    vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                    vertices_world = (transform @ vertices_hom.T).T[:, :3]
                    
                    # Check if mesh center is within room bounds
                    mesh_center = vertices_world.mean(axis=0)
                    
                    # Also check if any vertex is within bounds (for walls/floors that span rooms)
                    vertices_in_bounds = np.all((vertices_world >= room_min) & (vertices_world <= room_max), axis=1)
                    has_vertices_in_bounds = np.any(vertices_in_bounds)
                    
                    # Include mesh if center is in bounds OR if it has vertices in bounds
                    # (this catches walls/floors that span multiple rooms)
                    if np.all(mesh_center >= room_min) and np.all(mesh_center <= room_max) or has_vertices_in_bounds:
                        # For architectural elements (walls/floors), be more selective
                        metadata = getattr(geometry, 'metadata', {})
                        category_name = metadata.get('category_name', '').lower()
                        is_architectural = any(arch in category_name for arch in ['wall', 'floor', 'structure'])
                        
                        if is_architectural:
                            # Only include if significant portion is in this room
                            if np.sum(vertices_in_bounds) / len(vertices_world) > 0.3:  # 30% threshold
                                mesh_copy = geometry.copy()
                                if hasattr(geometry, 'metadata'):
                                    mesh_copy.metadata = geometry.metadata.copy()
                                room_scene.add_geometry(mesh_copy, node_name=node_name, transform=transform)
                        else:
                            # Furniture: include if center is in room
                            if np.all(mesh_center >= room_min) and np.all(mesh_center <= room_max):
                                mesh_copy = geometry.copy()
                                if hasattr(geometry, 'metadata'):
                                    mesh_copy.metadata = geometry.metadata.copy()
                                room_scene.add_geometry(mesh_copy, node_name=node_name, transform=transform)
            except (KeyError, ValueError, IndexError) as e:
                continue
        
        if len(room_scene.graph.nodes_geometry) > 0:
            room_scenes[room_name] = room_scene
            print(f"  Extracted room '{room_name}': {len(room_scene.graph.nodes_geometry)} meshes")
    
    if len(room_scenes) == 0:
        print("Warning: Could not extract rooms from OBJ, using full scene")
        room_scenes['scene'] = trimesh_scene
    
    return room_scenes


def main():
    parser = argparse.ArgumentParser(description="Render scene layouts from OBJ files using Open3D")
    parser.add_argument("--scene_id", required=True, help="Scene ID (e.g., 002c110c-9bbc-4ab4-affa-4225fb127bad)")
    parser.add_argument("--output_dir", required=True, help="Output dataset root directory (must contain geometry/ folder)")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hpc", action="store_true", default=False,
                        help="Run inside Xvfb for headless HPC rendering (like old pipeline)")
    
    args = parser.parse_args()
    
    # Start Xvfb if needed (like old pipeline)
    global VFB
    if args.hpc:
        if XVFBWRAPPER_AVAILABLE:
            try:
                VFB = Xvfb(width=256, height=256, colordepth=24)
                VFB.start()
                os.environ['DISPLAY'] = f':{VFB.new_display}'
                print(f"Started Xvfb virtual display: {os.environ['DISPLAY']}")
            except Exception as e:
                print(f"Warning: Failed to start Xvfb: {e}")
                print("Continuing without Xvfb (may fail if no display available)")
                VFB = None
        else:
            print("Warning: --hpc flag set but xvfbwrapper not installed")
            print("Install with: pip install xvfbwrapper")
            if 'DISPLAY' not in os.environ:
                raise RuntimeError("Cannot render without display. Install xvfbwrapper or use xvfb-run.")
    
    # Load taxonomy
    taxonomy = Taxonomy(Path(args.taxonomy))
    
    # Get scene ID and paths
    scene_id = args.scene_id
    output_dir = Path(args.output_dir)
    geometry_dir = output_dir / "geometry"
    
    # Load OBJ file (textured version)
    obj_path = geometry_dir / f"{scene_id}.obj"
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}. Run geometry export first.")
    
    # Load metadata (required for room extraction)
    metadata_path = geometry_dir / f"{scene_id}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Run geometry export first.")
    
    import json
    with open(metadata_path, 'r') as f:
        scene_metadata = json.load(f)
    print(f"Loaded metadata from: {metadata_path}")
    
    # Load scene from OBJ
    print(f"Loading scene from OBJ: {scene_id}")
    trimesh_scene = load_scene_from_obj(obj_path)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    layouts_rgb_dir = output_dir / "layouts" / "rgb"
    layouts_seg_dir = output_dir / "layouts" / "seg"
    
    for d in [layouts_rgb_dir, layouts_seg_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Extract rooms from OBJ scene using metadata
    print("Extracting rooms from OBJ scene using metadata...")
    room_scenes = extract_rooms_from_obj_scene(trimesh_scene, scene_metadata)
    print(f"  Found {len(room_scenes)} rooms: {list(room_scenes.keys())}")
    
    # Scene-level Layout Pass: RGB
    print("Rendering scene-level layout RGB...")
    try:
        layout_rgb = render_layout_rgb(trimesh_scene, hide_ceilings=True, width=256, height=256, 
                                      clip_top_meters=1.0, scene_metadata=scene_metadata)
        layout_rgb_path = layouts_rgb_dir / f"{scene_id}_scene.png"
        Image.fromarray(layout_rgb).save(layout_rgb_path)
        print(f"  Saved scene layout RGB: {layout_rgb_path}")
        if not layout_rgb_path.exists():
            raise FileNotFoundError(f"Scene layout RGB file was not created: {layout_rgb_path}")
    except Exception as e:
        print(f"ERROR: Failed to render scene layout RGB: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Scene-level Layout Pass: Segmentation
    print("Rendering scene-level layout segmentation...")
    try:
        layout_seg = render_layout_seg(trimesh_scene, taxonomy, hide_ceilings=True, width=256, height=256,
                                      clip_top_meters=1.0, scene_metadata=scene_metadata)
        layout_seg_path = layouts_seg_dir / f"{scene_id}_scene.png"
        Image.fromarray(layout_seg).save(layout_seg_path)
        print(f"  Saved scene layout segmentation: {layout_seg_path}")
        if not layout_seg_path.exists():
            raise FileNotFoundError(f"Scene layout segmentation file was not created: {layout_seg_path}")
    except Exception as e:
        print(f"ERROR: Failed to render scene layout segmentation: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Room-level Layout Pass
    print("Rendering room-level layouts...")
    for room_name, room_scene in room_scenes.items():
        # Sanitize room name for filename
        safe_room_name = room_name.replace(" ", "_").replace("/", "_").lower()
        
        print(f"  Processing room: {room_name}")
        
        # Room RGB layout
        try:
            # Get room metadata if available
            room_metadata = None
            if scene_metadata and 'rooms' in scene_metadata and room_name in scene_metadata['rooms']:
                room_metadata = {'scene_bounds': scene_metadata['rooms'][room_name].get('bounds', {})}
            room_layout_rgb = render_layout_rgb(room_scene, hide_ceilings=True, width=256, height=256,
                                               clip_top_meters=1.0, scene_metadata=room_metadata)
            room_layout_rgb_path = layouts_rgb_dir / f"{scene_id}_{safe_room_name}_room.png"
            Image.fromarray(room_layout_rgb).save(room_layout_rgb_path)
            print(f"    Saved room RGB layout: {room_layout_rgb_path}")
            if not room_layout_rgb_path.exists():
                raise FileNotFoundError(f"Room RGB layout file was not created: {room_layout_rgb_path}")
        except Exception as e:
            print(f"    ERROR: Failed to render room RGB layout for {room_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Room segmentation layout
        try:
            # Get room metadata if available
            room_metadata = None
            if scene_metadata and 'rooms' in scene_metadata and room_name in scene_metadata['rooms']:
                room_metadata = {'scene_bounds': scene_metadata['rooms'][room_name].get('bounds', {})}
            room_layout_seg = render_layout_seg(room_scene, taxonomy, hide_ceilings=True, width=256, height=256,
                                               clip_top_meters=1.0, scene_metadata=room_metadata)
            room_layout_seg_path = layouts_seg_dir / f"{scene_id}_{safe_room_name}_room.png"
            Image.fromarray(room_layout_seg).save(room_layout_seg_path)
            print(f"    Saved room segmentation layout: {room_layout_seg_path}")
            if not room_layout_seg_path.exists():
                raise FileNotFoundError(f"Room segmentation layout file was not created: {room_layout_seg_path}")
        except Exception as e:
            print(f"    ERROR: Failed to render room segmentation layout for {room_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Completed layout rendering for scene: {scene_id}")
    print(f"  Scene layouts: RGB and segmentation")
    print(f"  Room layouts: {len(room_scenes)} rooms (RGB and segmentation each)")
    
    # Stop Xvfb if we started it
    if VFB is not None:
        try:
            VFB.stop()
            print("Stopped Xvfb virtual display")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure Xvfb is stopped even on error
        if 'VFB' in globals() and globals()['VFB'] is not None:
            try:
                globals()['VFB'].stop()
            except Exception:
                pass

