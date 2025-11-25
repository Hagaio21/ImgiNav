#!/usr/bin/env python3
"""
Scene loader for 3D-FRONT to trimesh.Scene conversion.
Parses 3D-FRONT JSON files and loads 3D-FUTURE meshes into a unified scene.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from common.taxonomy import Taxonomy


def parse_transform(item: Dict) -> np.ndarray:
    """
    Extract pos, rot, scale from JSON item and convert to 4x4 transformation matrix.
    
    Args:
        item: Dictionary with 'pos', 'rot' (quaternion w,x,y,z), and 'scale' keys
        
    Returns:
        4x4 transformation matrix
    """
    pos = np.array(item.get("pos", [0, 0, 0]), dtype=np.float64)
    rot = np.array(item.get("rot", [0, 0, 0, 1]), dtype=np.float64)  # w, x, y, z
    scl = np.array(item.get("scale", [1, 1, 1]), dtype=np.float64)
    
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    Rm = Rotation.from_quat(rot).as_matrix()
    Sm = np.diag(scl)
    T[:3, :3] = Rm @ Sm
    return T


def load_future_model(jid: str, future_root: Path) -> trimesh.Trimesh:
    """
    Load 3D-FUTURE model mesh from raw_model.obj or raw_model.glb.
    
    Args:
        jid: Model ID (directory name in 3D-FUTURE)
        future_root: Root directory of 3D-FUTURE dataset
        
    Returns:
        trimesh.Trimesh object
        
    Raises:
        FileNotFoundError: If neither .obj nor .glb file exists
    """
    obj_path = future_root / jid / "raw_model.obj"
    glb_path = future_root / jid / "raw_model.glb"
    
    if obj_path.exists():
        resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
        return trimesh.load(str(obj_path), force="mesh", process=False,
                           maintain_order=True, resolver=resolver)
    elif glb_path.exists():
        return trimesh.load(str(glb_path), force="mesh", process=False,
                           maintain_order=True)
    else:
        raise FileNotFoundError(f"Model not found: {obj_path} or {glb_path}")


def create_arch_mesh(arch: Dict) -> trimesh.Trimesh:
    """
    Create trimesh from architectural mesh definition (walls, floors, ceilings).
    
    Args:
        arch: Dictionary with 'xyz' (vertices) and 'faces' keys
        
    Returns:
        trimesh.Trimesh object
    """
    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
    faces = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # Default gray color for architectural elements
    mesh.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)
    return mesh


def load_front_scene(json_path: Path, future_root: Path, taxonomy: Taxonomy) -> trimesh.Scene:
    """
    Load 3D-FRONT scene from JSON and assemble into trimesh.Scene.
    
    Includes both furniture (from 3D-FUTURE) and architectural meshes (walls/floors/ceilings).
    Attaches metadata: category_id, is_ceiling flag, room_name.
    
    Args:
        json_path: Path to 3D-FRONT JSON file
        future_root: Root directory of 3D-FUTURE dataset
        taxonomy: Taxonomy object for category mapping
        
    Returns:
        Assembled trimesh.Scene with all meshes and metadata
    """
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)
    
    # Build furniture lookup dict {uid: {jid, title, category, ...}}
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    
    # Build architectural mesh lookup dict {uid: mesh_data}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}
    
    # Get ceiling category/label IDs for detection
    ceiling_ids = []
    if "ceiling" in taxonomy.data.get("category2id", {}):
        ceiling_ids.append(taxonomy.data["category2id"]["ceiling"])
    if "ceiling" in taxonomy.data.get("label2id", {}):
        ceiling_ids.append(taxonomy.data["label2id"]["ceiling"])
    
    # Create scene
    scene = trimesh.Scene()
    
    # Iterate through rooms
    for room in scene_data.get("scene", {}).get("room", []):
        # Get room type/name
        room_type = room.get("type", "UnknownRoom")
        if not room_type or room_type.strip() == "":
            room_type = "UnknownRoom"
        room_name = room_type.strip()
        
        # Skip empty rooms
        if room.get("empty", 0) == 1:
            continue
        
        for child_index, child in enumerate(room.get("children", [])):
            ref_id = child.get("ref")
            if not ref_id:
                continue
            
            try:
                mesh = None
                category_id = 0
                is_ceiling = False
                label = "unknown"
                
                # Check if it's furniture
                if ref_id in furniture_map:
                    item_info = furniture_map[ref_id]
                    jid = item_info.get('jid')
                    if not jid:
                        continue
                    
                    # Load furniture mesh
                    mesh = load_future_model(jid, future_root)
                    
                    # Get category from title
                    title = item_info.get('title', 'unknown')
                    category_id = taxonomy.translate(title, output="id")
                    if category_id == 0:
                        # Try to get category name first
                        category_name = taxonomy.data.get("title2category", {}).get(title)
                        if category_name:
                            category_id = taxonomy.data.get("category2id", {}).get(category_name, 0)
                    
                    label = title
                
                # Check if it's architectural mesh (wall, floor, ceiling)
                elif ref_id in arch_map:
                    arch = arch_map[ref_id]
                    arch_type = arch.get("type", "")
                    
                    # Skip ceiling for layout rendering (will be filtered later)
                    if "Ceiling" in arch_type:
                        is_ceiling = True
                        # Still create the mesh but mark it
                        mesh = create_arch_mesh(arch)
                        label = "ceiling"
                        category_id = taxonomy.data.get("category2id", {}).get("ceiling", 0)
                    elif "Floor" in arch_type:
                        mesh = create_arch_mesh(arch)
                        label = "floor"
                        category_id = taxonomy.data.get("category2id", {}).get("floor", 0)
                    elif "Wall" in arch_type:
                        mesh = create_arch_mesh(arch)
                        label = "wall"
                        category_id = taxonomy.data.get("category2id", {}).get("wall", 0)
                    else:
                        # Other architectural elements
                        mesh = create_arch_mesh(arch)
                        label = "structure"
                        category_id = taxonomy.data.get("super2id", {}).get("Structure", 0)
                
                if mesh is None or mesh.is_empty:
                    continue
                
                # Apply transform
                transform = parse_transform(child)
                
                # Attach metadata
                if not hasattr(mesh, 'metadata'):
                    mesh.metadata = {}
                mesh.metadata['category_id'] = category_id
                mesh.metadata['is_ceiling'] = is_ceiling
                mesh.metadata['label'] = label
                mesh.metadata['ref_id'] = ref_id
                mesh.metadata['room_name'] = room_name  # Track which room this belongs to
                
                # Add to scene
                node_name = f"{ref_id}_{child_index}"
                scene.add_geometry(mesh, node_name=node_name, transform=transform)
                
            except Exception as e:
                print(f"Warning: Failed to load {ref_id}: {e}")
                continue
    
    return scene


def extract_rooms_from_scene(trimesh_scene: trimesh.Scene) -> Dict[str, trimesh.Scene]:
    """
    Extract separate trimesh scenes for each room.
    
    Args:
        trimesh_scene: Full scene with all rooms
        
    Returns:
        Dictionary mapping room_name -> trimesh.Scene for that room
    """
    # Group meshes by room
    room_meshes = defaultdict(list)
    for node_name in trimesh_scene.graph.nodes_geometry:
        try:
            transform, geometry_name = trimesh_scene.graph.get(node_name)
            if geometry_name not in trimesh_scene.geometry:
                if node_name not in trimesh_scene.geometry:
                    continue
                geometry = trimesh_scene.geometry[node_name]
            else:
                geometry = trimesh_scene.geometry[geometry_name]
            
            metadata = getattr(geometry, 'metadata', {})
            room_name = metadata.get('room_name', 'UnknownRoom')
            
            room_meshes[room_name].append((node_name, geometry, transform))
        except (KeyError, ValueError, IndexError):
            continue
    
    # Create separate scene for each room
    rooms = {}
    for room_name, meshes in room_meshes.items():
        room_scene = trimesh.Scene()
        for node_name, geometry, transform in meshes:
            # Create a copy of the mesh to avoid modifying the original
            mesh_copy = geometry.copy()
            # Preserve metadata
            if hasattr(geometry, 'metadata'):
                mesh_copy.metadata = geometry.metadata.copy()
            room_scene.add_geometry(mesh_copy, node_name=node_name, transform=transform)
        rooms[room_name] = room_scene
    
    return rooms

