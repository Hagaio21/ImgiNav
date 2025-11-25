#!/usr/bin/env python3
"""
Render POV (point-of-view) images from OBJ files using Open3D.
Separate script for POV rendering to speed up the pipeline.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

# Global variable to hold Xvfb instance
VFB = None

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.renderer import render_pov, render_pov_seg, sample_camera_positions_from_corners
from data_preparation.pipeline_v2.scene_loader import extract_rooms_from_scene

# Set Open3D verbosity to errors only
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def load_scene_from_obj(obj_path: Path) -> trimesh.Scene:
    """
    Load scene from OBJ file.
    """
    print(f"Loading scene from OBJ: {obj_path}")
    
    # Simply load the OBJ file - trimesh will handle MTL files automatically
    # Use the directory containing the OBJ as the base for resolving relative paths
    scene = trimesh.load(str(obj_path), file_type='obj', process=False, maintain_order=True)
    
    if scene is None:
        scene = trimesh.load(str(obj_path), file_type='obj', process=False, maintain_order=True, force='scene')
    
    if scene is None:
        raise ValueError(f"OBJ file loaded as None: {obj_path}")
    
    if not isinstance(scene, trimesh.Scene):
        if isinstance(scene, trimesh.Trimesh):
            new_scene = trimesh.Scene()
            new_scene.add_geometry(scene)
            scene = new_scene
        else:
            raise ValueError(f"OBJ file did not load as a scene or mesh: {type(scene)}")
    
    return scene


def extract_rooms_from_obj_scene(trimesh_scene: trimesh.Scene, scene_metadata: dict) -> dict:
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
    parser = argparse.ArgumentParser(description="Render scene POVs from OBJ files using Open3D")
    parser.add_argument("--scene_id", required=True, help="Scene ID (e.g., 002c110c-9bbc-4ab4-affa-4225fb127bad)")
    parser.add_argument("--output_dir", required=True, help="Output dataset root directory (must contain geometry/ folder)")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    parser.add_argument("--num_povs", type=int, default=6, help="Number of POVs to render per room")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hpc", action="store_true", default=False,
                        help="Run inside Xvfb for headless HPC rendering (like old pipeline)")
    
    args = parser.parse_args()
    
    # Start Xvfb if needed (like old pipeline)
    if args.hpc:
        try:
            import xvfbwrapper
            global VFB
            VFB = xvfbwrapper.Xvfb(width=1920, height=1080)
            VFB.start()
            print("Started Xvfb for headless rendering")
        except ImportError:
            print("WARNING: xvfbwrapper not available, continuing without Xvfb")
        except Exception as e:
            print(f"WARNING: Failed to start Xvfb: {e}, continuing without it")
    
    try:
        # Load taxonomy
        taxonomy = Taxonomy(Path(args.taxonomy))
        
        # Load scene geometry
        geometry_dir = Path(args.output_dir) / "geometry"
        obj_path = geometry_dir / f"{args.scene_id}.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"OBJ file not found: {obj_path}. Run geometry export first.")
        
        # Load metadata (required for room extraction)
        metadata_path = geometry_dir / f"{args.scene_id}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Run geometry export first.")
        
        import json
        with open(metadata_path, 'r') as f:
            scene_metadata = json.load(f)
        print(f"Loaded metadata from: {metadata_path}")
        
        # Load scene from OBJ
        print(f"Loading scene from OBJ: {args.scene_id}")
        trimesh_scene = load_scene_from_obj(obj_path)
        
        # Create output directories
        output_dir = Path(args.output_dir)
        povs_rgb_dir = output_dir / "povs" / "rgb"
        povs_seg_dir = output_dir / "povs" / "seg"
        
        for d in [povs_rgb_dir, povs_seg_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Extract rooms from OBJ scene using metadata
        print("Extracting rooms from OBJ scene using metadata...")
        room_scenes = extract_rooms_from_obj_scene(trimesh_scene, scene_metadata)
        print(f"  Found {len(room_scenes)} rooms: {list(room_scenes.keys())}")
        
        # Get up direction from metadata
        up_direction_info = scene_metadata.get('up_direction', {})
        up_axis = up_direction_info.get('up_axis', 1)
        up_vector = np.array(up_direction_info.get('up_vector', [0, 1, 0]))
        
        # Scene-level POVs
        print("Rendering scene-level POVs...")
        scene_bounds = scene_metadata.get('bounds', {})
        if 'min' in scene_bounds and 'max' in scene_bounds:
            scene_min = np.array(scene_bounds['min'])
            scene_max = np.array(scene_bounds['max'])
            scene_center = (scene_min + scene_max) / 2
            
            # Sample camera positions from corners
            np.random.seed(args.seed)
            camera_positions = sample_camera_positions_from_corners(
                trimesh_scene, scene_min, scene_max, scene_center, 
                num_attempts=20, num_povs=args.num_povs, up_axis=up_axis
            )
            
            for pov_idx, camera_pos in enumerate(camera_positions):
                try:
                    pov_rgb = render_pov(trimesh_scene, camera_pos, hide_ceilings=True, 
                                        width=256, height=256, camera_target=scene_center)
                    pov_rgb_path = povs_rgb_dir / f"{args.scene_id}_scene_v{pov_idx+1:02d}.png"
                    Image.fromarray(pov_rgb).save(pov_rgb_path)
                    print(f"  Saved scene POV RGB {pov_idx+1}: {pov_rgb_path}")
                    
                    pov_seg = render_pov_seg(trimesh_scene, camera_pos, taxonomy, hide_ceilings=True,
                                            width=256, height=256, camera_target=scene_center)
                    pov_seg_path = povs_seg_dir / f"{args.scene_id}_scene_v{pov_idx+1:02d}.png"
                    Image.fromarray(pov_seg).save(pov_seg_path)
                    print(f"  Saved scene POV segmentation {pov_idx+1}: {pov_seg_path}")
                except Exception as e:
                    print(f"  ERROR: Failed to render scene POV {pov_idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Room-level POVs
        for room_name, room_scene in room_scenes.items():
            if room_name == 'scene':
                continue  # Already handled above
            
            safe_room_name = room_name.replace(" ", "_").replace("/", "_").lower()
            print(f"Rendering POVs for room: {room_name}")
            
            room_info = scene_metadata.get('rooms', {}).get(room_name, {})
            room_bounds = room_info.get('bounds', {})
            
            if 'min' not in room_bounds or 'max' not in room_bounds:
                print(f"  Warning: Room {room_name} has no bounds, skipping POVs")
                continue
            
            room_min = np.array(room_bounds['min'])
            room_max = np.array(room_bounds['max'])
            room_center = np.array(room_info.get('center', (room_min + room_max) / 2))
            
            # Sample camera positions from corners
            np.random.seed(args.seed + hash(room_name) % 1000)
            camera_positions = sample_camera_positions_from_corners(
                room_scene, room_min, room_max, room_center,
                num_attempts=20, num_povs=args.num_povs, up_axis=up_axis
            )
            
            for pov_idx, camera_pos in enumerate(camera_positions):
                try:
                    pov_rgb = render_pov(room_scene, camera_pos, hide_ceilings=True,
                                        width=256, height=256, camera_target=room_center)
                    pov_rgb_path = povs_rgb_dir / f"{args.scene_id}_{safe_room_name}_room_v{pov_idx+1:02d}.png"
                    Image.fromarray(pov_rgb).save(pov_rgb_path)
                    print(f"  Saved room POV RGB {pov_idx+1}: {pov_rgb_path}")
                    
                    pov_seg = render_pov_seg(room_scene, camera_pos, taxonomy, hide_ceilings=True,
                                            width=256, height=256, camera_target=room_center)
                    pov_seg_path = povs_seg_dir / f"{args.scene_id}_{safe_room_name}_room_v{pov_idx+1:02d}.png"
                    Image.fromarray(pov_seg).save(pov_seg_path)
                    print(f"  Saved room POV segmentation {pov_idx+1}: {pov_seg_path}")
                except Exception as e:
                    print(f"  ERROR: Failed to render room POV {pov_idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"Completed POV rendering for scene: {args.scene_id}")
        
    finally:
        # Stop Xvfb if started
        if VFB is not None:
            try:
                VFB.stop()
                print("Stopped Xvfb")
            except Exception:
                pass


if __name__ == "__main__":
    main()

