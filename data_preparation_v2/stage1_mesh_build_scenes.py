#!/usr/bin/env python3

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# --- imports ---
from common.taxonomy import Taxonomy
from common.utils import (
    load_config_with_profile, create_progress_tracker,
    safe_mkdir, write_json
)
from data_preparation_v2.utils.file_discovery import gather_paths_from_sources, infer_ids_from_path

# Configure logging
def setup_logging(log_dir: Path = None):
    """Setup logging to both console and file."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Get the module logger
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    module_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    module_logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "stage1_errors.log"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        module_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    module_logger.propagate = False
    
    return module_logger

# Initialize a basic logger
logger = logging.getLogger(__name__)

# --- Global Taxonomy Object ---
TAXONOMY: Taxonomy = None
ARGS = None

# ---------------------------------------------------------------------
# Utility Functions (reused from stage1_build_scenes.py)
# ---------------------------------------------------------------------

def load_mesh(model_dir: Path, jid: str):
    # Try multiple possible paths for the model directory
    possible_dirs = [
        model_dir,  # Direct: model_dir/jid/raw_model.obj
        model_dir / "3D-FUTURE-model",  # Nested: model_dir/3D-FUTURE-model/jid/raw_model.obj
    ]
    
    for base_dir in possible_dirs:
        obj_path = base_dir / jid / "raw_model.obj"
        glb_path = base_dir / jid / "raw_model.glb"
        
        if obj_path.exists():
            # Use FilePathResolver to properly resolve texture paths relative to the OBJ file
            resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
            mesh = trimesh.load(str(obj_path), force="mesh", process=False,
                                maintain_order=True, resolver=resolver)
            
            # Verify UVs and textures are loaded
            if hasattr(mesh, 'visual') and mesh.visual is not None:
                if hasattr(mesh.visual, 'uv'):
                    if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                        logger.debug(f"Loaded mesh {jid} with {len(mesh.visual.uv)} UV coordinates")
                    else:
                        logger.warning(f"Mesh {jid} loaded but has no UV coordinates")
                
                # Check for texture/material
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
                        logger.debug(f"Loaded mesh {jid} with texture image")
                    else:
                        logger.debug(f"Mesh {jid} has material but no texture image")
                else:
                    logger.debug(f"Mesh {jid} has no material")
            
            return mesh
        elif glb_path.exists():
            mesh = trimesh.load(str(glb_path), force="mesh", process=False,
                                maintain_order=True)
            return mesh
    
    # If none found, raise error with all attempted paths
    attempted_paths = []
    for base_dir in possible_dirs:
        attempted_paths.append(str(base_dir / jid / "raw_model.obj"))
        attempted_paths.append(str(base_dir / jid / "raw_model.glb"))
    raise FileNotFoundError(f"Model not found for jid {jid}. Tried: {', '.join(attempted_paths)}")

def create_arch_mesh(arch: Dict):
    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
    faces = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)
    return mesh

def build_transform(child: Dict):
    pos = np.array(child.get("pos", [0, 0, 0]), dtype=np.float64)
    rot = np.array(child.get("rot", [0, 0, 0, 1]), dtype=np.float64)
    scl = np.array(child.get("scale", [1, 1, 1]), dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    Rm = Rotation.from_quat(rot).as_matrix()
    Sm = np.diag(scl)
    T[:3, :3] = Rm @ Sm
    return T

def apply_segmented_colors(mesh: trimesh.Trimesh, label: str, taxonomy: Taxonomy) -> trimesh.Trimesh:
    """Apply taxonomy category colors to mesh vertices."""
    # Get color from taxonomy (using category mode)
    color = taxonomy.get_color(label, mode="category")
    if color is None:
        color = (128, 128, 128)  # Default gray
    
    # Ensure color is RGB tuple
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        rgb = tuple(int(c) for c in color[:3])
    else:
        rgb = (128, 128, 128)
    
    # Apply color to all vertices
    num_vertices = len(mesh.vertices)
    vertex_colors = np.full((num_vertices, 4), 255, dtype=np.uint8)
    vertex_colors[:, :3] = rgb
    
    mesh_seg = mesh.copy()
    mesh_seg.visual.vertex_colors = vertex_colors
    
    # For segmented version, remove material/texture to use vertex colors only
    # Create a simple material to avoid None material issues
    try:
        from trimesh.visual.material import SimpleMaterial
        mesh_seg.visual.material = SimpleMaterial(
            diffuse=[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, 1.0]
        )
    except:
        # If material creation fails, try to clear it safely
        if hasattr(mesh_seg.visual, 'material'):
            try:
                mesh_seg.visual.material = None
            except:
                pass
    
    return mesh_seg

def fix_mesh_materials(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fix material issues for GLB export while preserving UVs and textures."""
    # Only fix if material is None - don't overwrite existing materials with textures
    if hasattr(mesh, 'visual') and mesh.visual is not None:
        if hasattr(mesh.visual, 'material'):
            if mesh.visual.material is None:
                # Check if we have UVs - if so, we might need a material for texture
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    try:
                        from trimesh.visual.material import SimpleMaterial
                        # Create a default material that can hold textures
                        mesh.visual.material = SimpleMaterial()
                    except:
                        pass
            else:
                # Material exists - ensure it has a name attribute for GLB export
                # But don't modify materials that have textures
                try:
                    has_texture = (hasattr(mesh.visual.material, 'image') and 
                                 mesh.visual.material.image is not None)
                    if not has_texture:
                        # Only set name if no texture (textured materials should keep their original name)
                        if not hasattr(mesh.visual.material, 'name') or mesh.visual.material.name is None:
                            mesh.visual.material.name = "Material"
                except:
                    pass
    return mesh

# ---------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------

def compute_mesh_bbox(mesh: trimesh.Trimesh) -> Dict:
    """Compute bounding box of a mesh."""
    if mesh.is_empty or len(mesh.vertices) == 0:
        return {"min": [0, 0, 0], "max": [0, 0, 0]}
    vertices = mesh.vertices
    return {
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist()
    }

def compute_mesh_centroid(mesh: trimesh.Trimesh) -> List[float]:
    """Compute centroid of a mesh."""
    if mesh.is_empty or len(mesh.vertices) == 0:
        return [0, 0, 0]
    return mesh.vertices.mean(axis=0).tolist()

def compute_room_floor_plane(room_meshes: List[Dict], floor_label_ids: List[int], taxonomy: Taxonomy) -> Tuple[np.ndarray, np.ndarray]:
    """Compute floor plane from room meshes using floor-labeled geometry."""
    floor_points = []
    
    for obj_info in room_meshes:
        label = obj_info.get('label', '')
        label_id = taxonomy.translate(label, output="id")
        
        if label_id in floor_label_ids or 'floor' in label.lower():
            mesh_world = obj_info['mesh'].copy()
            mesh_world.apply_transform(obj_info['transform'])
            vertices = mesh_world.vertices
            floor_points.append(vertices)
    
    if len(floor_points) == 0:
        # Fallback: use lowest Z points
        all_vertices = []
        for obj_info in room_meshes:
            mesh_world = obj_info['mesh'].copy()
            mesh_world.apply_transform(obj_info['transform'])
            all_vertices.append(mesh_world.vertices)
        if len(all_vertices) == 0:
            return np.array([0, 0, 0]), np.array([0, 0, 1])
        all_vertices = np.vstack(all_vertices)
        z_min = all_vertices[:, 2].min()
        z_threshold = z_min + 0.1  # 10cm above minimum
        floor_points = [all_vertices[all_vertices[:, 2] <= z_threshold]]
    
    if len(floor_points) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 1])
    
    floor_vertices = np.vstack(floor_points)
    if len(floor_vertices) < 3:
        return np.array([0, 0, 0]), np.array([0, 0, 1])
    
    from data_preparation_v2.utils.geometry_utils import pca_plane_fit
    origin, normal = pca_plane_fit(floor_vertices)
    # Ensure normal points upward
    if normal[2] < 0:
        normal = -normal
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    
    return origin, normal

def process_scene_meshes(scene_data, model_dir, model_info_map, taxonomy: Taxonomy, scene_id: str):
    """Process scene and create both segmented and textured meshes, plus metadata."""
    failed_models = {}
    scene_objects = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}
    
    # Store arch_map for door/window detection later
    global _arch_map_for_detection
    _arch_map_for_detection = arch_map
    
    # Get floor label IDs from taxonomy
    floor_label_ids = taxonomy.get_floor_ids()
    
    # Track statistics
    furniture_loaded = 0
    furniture_failed = 0
    arch_loaded = 0
    
    # Group objects by room for metadata
    room_data = {}

    for room_idx, room in enumerate(scene_data.get("scene", {}).get("room", [])):
        room_type = room.get("type", "UnknownRoom")
        room_id = room_idx
        room_objects = []

        for child_index, child in enumerate(room.get("children", [])):
            ref_id = child.get("ref")
            if not ref_id:
                continue

            try:
                mesh, label, model_item = None, "unknown", None

                if ref_id in furniture_map:
                    item_info = furniture_map[ref_id]
                    jid = item_info.get('jid')
                    if not jid:
                        raise ValueError("Missing 'jid'")

                    mesh = load_mesh(model_dir, jid)
                    label = (model_info_map.get(jid, {}).get('category')
                             or item_info.get('title') or "unknown")
                    model_item = item_info
                    furniture_loaded += 1

                elif ref_id in arch_map:
                    arch = arch_map[ref_id]
                    if "Ceiling" in arch.get("type", ""):
                        continue  # Skip ceilings
                    mesh = create_arch_mesh(arch)
                    label = 'floor' if 'Floor' in arch.get("type", "") else 'wall'
                    arch_loaded += 1
                else:
                    failed_models[ref_id] = "Reference not found"
                    continue

                if mesh is None or mesh.is_empty:
                    raise ValueError("Empty mesh")

                transform = build_transform(child)
                
                obj_info = {
                    "mesh": mesh,
                    "transform": transform,
                    "label": label,
                    "room_type": room_type,
                    "node_name": f"{ref_id}_{child_index}",
                    "model_item": model_item,
                    "ref_id": ref_id
                }
                
                scene_objects.append(obj_info)
                room_objects.append(obj_info)

            except Exception as e:
                if ref_id in furniture_map:
                    furniture_failed += 1
                logger.warning(f"Failed to process model {ref_id} in scene: {e}")
                failed_models[ref_id] = str(e)
        
        # Store room objects for metadata
        if room_objects:
            room_data[room_id] = {
                "room_id": room_id,
                "room_type": room_type,
                "objects": room_objects
            }

    # Log statistics
    logger.info(f"Scene {scene_id}: Loaded {furniture_loaded} furniture, {furniture_failed} furniture failed, {arch_loaded} architectural elements")
    
    if not scene_objects:
        logger.warning(f"Scene {scene_id}: No objects loaded (all models failed or scene is empty)")
        return None, None, failed_models, None

    # Build segmented and textured scenes
    segmented_scene = trimesh.Scene()
    textured_scene = trimesh.Scene()
    
    all_vertices = []
    all_objects_metadata = []
    
    for obj in scene_objects:
        # Create segmented version with taxonomy colors
        mesh_seg = apply_segmented_colors(obj['mesh'], obj['label'], taxonomy)
        segmented_scene.add_geometry(mesh_seg, node_name=obj['node_name'],
                                     transform=obj['transform'])
        
        # Keep original textured version - preserve all visual data including UVs and textures
        # For textured meshes, we need to ensure the visual data is fully preserved
        # Use copy() which should preserve visual data, but verify and restore if needed
        mesh_tex = obj['mesh'].copy()
        
        # Ensure visual data is fully preserved - trimesh.copy() should handle this,
        # but we verify and restore critical attributes
        if hasattr(obj['mesh'], 'visual') and obj['mesh'].visual is not None:
            orig_visual = obj['mesh'].visual
            
            # Ensure mesh_tex has a visual object
            if not hasattr(mesh_tex, 'visual') or mesh_tex.visual is None:
                # Create a new visual object of the same type
                mesh_tex.visual = type(orig_visual)()
            
            # Copy UVs explicitly (critical for texture mapping)
            if hasattr(orig_visual, 'uv') and orig_visual.uv is not None:
                if not hasattr(mesh_tex.visual, 'uv') or mesh_tex.visual.uv is None:
                    mesh_tex.visual.uv = orig_visual.uv.copy()
                elif not np.array_equal(mesh_tex.visual.uv, orig_visual.uv):
                    mesh_tex.visual.uv = orig_visual.uv.copy()
            
            # Copy material (contains texture image data)
            if hasattr(orig_visual, 'material') and orig_visual.material is not None:
                # Preserve the original material - it contains texture/image data
                # Don't overwrite if it already exists and has texture data
                if not hasattr(mesh_tex.visual, 'material') or mesh_tex.visual.material is None:
                    mesh_tex.visual.material = orig_visual.material
                elif hasattr(orig_visual.material, 'image') and orig_visual.material.image is not None:
                    # Original has texture - preserve it
                    mesh_tex.visual.material = orig_visual.material
            
            # Copy any other visual attributes that might be needed
            for attr in ['vertex_colors', 'face_colors']:
                if hasattr(orig_visual, attr):
                    orig_val = getattr(orig_visual, attr)
                    if orig_val is not None:
                        if not hasattr(mesh_tex.visual, attr) or getattr(mesh_tex.visual, attr) is None:
                            setattr(mesh_tex.visual, attr, orig_val.copy() if hasattr(orig_val, 'copy') else orig_val)
        
        # Fix materials only if needed (don't overwrite valid materials with textures)
        mesh_tex = fix_mesh_materials(mesh_tex)
        
        # Add to scene - trimesh will apply transform without modifying UVs (UVs are in texture space)
        textured_scene.add_geometry(mesh_tex, node_name=obj['node_name'],
                                  transform=obj['transform'])
        
        # Collect vertices for bbox computation
        mesh_world = obj['mesh'].copy()
        mesh_world.apply_transform(obj['transform'])
        all_vertices.append(mesh_world.vertices)
        
        # Collect comprehensive object metadata
        label_id = taxonomy.translate(obj['label'], output="id") or 0
        category_id = taxonomy.translate(obj['label'], output="id") or 0
        super_id = taxonomy.get_sup(obj['label'], output="id") or 0
        category_name = taxonomy.translate(label_id, output="name") if label_id else obj['label']
        super_name = taxonomy.get_sup(obj['label'], output="name") or "UnknownSuper"
        room_id_tax = taxonomy.translate(obj['room_type'], output="id") or 0
        
        obj_bbox = compute_mesh_bbox(mesh_world)
        obj_centroid = compute_mesh_centroid(mesh_world)
        
        # Compute mesh properties
        mesh_area = float(mesh_world.area) if hasattr(mesh_world, 'area') else 0.0
        mesh_volume = float(mesh_world.volume) if hasattr(mesh_world, 'volume') and mesh_world.is_volume else 0.0
        num_vertices = len(mesh_world.vertices)
        num_faces = len(mesh_world.faces) if hasattr(mesh_world, 'faces') else 0
        
        # Extract transform components
        transform = obj['transform']
        position = transform[:3, 3].tolist()
        rotation_matrix = transform[:3, :3].tolist()
        scale = np.linalg.norm(transform[:3, 0]), np.linalg.norm(transform[:3, 1]), np.linalg.norm(transform[:3, 2])
        scale = [float(s) for s in scale]
        
        # Get model info if available
        model_info = {}
        if obj.get('model_item'):
            model_item = obj['model_item']
            model_info = {
                "jid": model_item.get('jid'),
                "title": model_item.get('title'),
                "uid": model_item.get('uid'),
                "valid": model_item.get('valid', True)
            }
            # Add model_info.json data if available
            if model_item.get('jid') and model_item.get('jid') in model_info_map:
                mi = model_info_map[model_item.get('jid')]
                model_info.update({
                    "category": mi.get('category'),
                    "super-category": mi.get('super-category'),
                    "style": mi.get('style'),
                    "material": mi.get('material'),
                    "theme": mi.get('theme')
                })
        
        # Detect doors and windows
        label_lower = obj['label'].lower()
        category_lower = category_name.lower()
        is_door = 'door' in label_lower or 'door' in category_lower
        is_window = 'window' in label_lower or 'window' in category_lower
        # Also check architectural mesh types (e.g., "BayWindow")
        if not is_door and not is_window:
            # Check if it's an architectural element that's a door/window
            arch_type = ""
            ref_id = obj.get('ref_id')
            if ref_id and ref_id in arch_map:
                arch_type = arch_map[ref_id].get('type', '').lower()
            is_door = 'door' in arch_type
            is_window = 'window' in arch_type or 'baywindow' in arch_type
        
        # Get door/window height for clipping walls above them
        door_window_height = None
        if is_door or is_window:
            bbox_size = np.array(obj_bbox["max"]) - np.array(obj_bbox["min"])
            door_window_height = float(bbox_size[2])  # Height (z-dimension)
        
        all_objects_metadata.append({
            "object_id": obj['ref_id'],
            "label": obj['label'],
            "label_id": int(label_id),
            "category": category_name,
            "category_id": int(category_id),
            "super": super_name,
            "super_id": int(super_id),
            "room_type": obj['room_type'],
            "room_id_taxonomy": int(room_id_tax),
            "bbox": obj_bbox,
            "bbox_size": (np.array(obj_bbox["max"]) - np.array(obj_bbox["min"])).tolist(),
            "location": obj_centroid,
            "position": position,
            "transform": transform.tolist(),
            "rotation_matrix": rotation_matrix,
            "scale": scale,
            "mesh_properties": {
                "area": mesh_area,
                "volume": mesh_volume,
                "num_vertices": num_vertices,
                "num_faces": num_faces,
                "is_watertight": bool(mesh_world.is_watertight) if hasattr(mesh_world, 'is_watertight') else False
            },
            "model_info": model_info,
            "is_door": is_door,
            "is_window": is_window,
            "door_window_height": door_window_height,  # Height for clipping walls above
            "room_id": None  # Will be set below
        })
    
    if not all_vertices:
        return None, None, failed_models, None
    
    # Compute scene bounding box and center
    all_vertices = np.vstack(all_vertices)
    scene_bbox = {
        "min": all_vertices.min(axis=0).tolist(),
        "max": all_vertices.max(axis=0).tolist()
    }
    scene_bbox_size = (np.array(scene_bbox["max"]) - np.array(scene_bbox["min"])).tolist()
    scene_center = all_vertices.mean(axis=0).tolist()
    
    # Build comprehensive metadata
    metadata_rooms = []
    scene_up = [0, 0, 1]
    
    for room_id, room_info in room_data.items():
        room_objects_meta = []
        room_meshes = []
        
        for obj in room_info["objects"]:
            # Find corresponding metadata
            obj_meta = next((o for o in all_objects_metadata if o["object_id"] == obj['ref_id']), None)
            if obj_meta:
                obj_meta["room_id"] = room_id
                room_objects_meta.append(obj_meta)
                room_meshes.append(obj)
        
        if not room_objects_meta:
            continue
        
        # Compute room bounding box
        room_vertices = []
        for obj in room_info["objects"]:
            mesh_world = obj['mesh'].copy()
            mesh_world.apply_transform(obj['transform'])
            room_vertices.append(mesh_world.vertices)
        
        if not room_vertices:
            continue
        
        room_vertices = np.vstack(room_vertices)
        room_bbox = {
            "min": room_vertices.min(axis=0).tolist(),
            "max": room_vertices.max(axis=0).tolist()
        }
        room_bbox_size = (np.array(room_bbox["max"]) - np.array(room_bbox["min"])).tolist()
        room_centroid = room_vertices.mean(axis=0).tolist()
        
        # Compute room floor plane and up direction
        floor_origin, up_normal = compute_room_floor_plane(room_meshes, floor_label_ids, taxonomy)
        if room_id == 0:  # Use first room's up direction for scene
            scene_up = up_normal.tolist()
        
        # Compute floor area (from floor objects)
        floor_area = 0.0
        floor_objects = []
        for obj_meta in room_objects_meta:
            if 'floor' in obj_meta.get('label', '').lower() or obj_meta.get('label_id') in floor_label_ids:
                floor_area += obj_meta.get('mesh_properties', {}).get('area', 0.0)
                floor_objects.append(obj_meta['object_id'])
        
        # Count objects by category
        object_counts = {}
        for obj_meta in room_objects_meta:
            cat = obj_meta.get('category', 'Unknown')
            object_counts[cat] = object_counts.get(cat, 0) + 1
        
        # Compute room dimensions
        room_dimensions = {
            "width": room_bbox_size[0],
            "depth": room_bbox_size[1],
            "height": room_bbox_size[2]
        }
        
        metadata_rooms.append({
            "room_id": room_id,
            "room_type": room_info["room_type"],
            "room_id_taxonomy": int(taxonomy.translate(room_info["room_type"], output="id") or 0),
            "room_bbox": room_bbox,
            "room_bbox_size": room_bbox_size,
            "room_dimensions": room_dimensions,
            "room_center": room_centroid,
            "room_location": room_centroid,  # Alias for compatibility
            "floor_origin": floor_origin.tolist(),
            "floor_area": float(floor_area),
            "floor_objects": floor_objects,
            "up_direction": up_normal.tolist(),
            "num_objects": len(room_objects_meta),
            "object_counts_by_category": object_counts,
            "objects": room_objects_meta
        })
    
    # Compute scene-level statistics
    total_floor_area = sum(r.get('floor_area', 0.0) for r in metadata_rooms)
    total_volume = sum(r.get('room_dimensions', {}).get('width', 0) * 
                      r.get('room_dimensions', {}).get('depth', 0) * 
                      r.get('room_dimensions', {}).get('height', 0) for r in metadata_rooms)
    
    # Room type distribution
    room_type_counts = {}
    for room in metadata_rooms:
        rt = room.get('room_type', 'Unknown')
        room_type_counts[rt] = room_type_counts.get(rt, 0) + 1
    
    # Object category distribution across scene
    scene_object_counts = {}
    for obj in all_objects_metadata:
        cat = obj.get('category', 'Unknown')
        scene_object_counts[cat] = scene_object_counts.get(cat, 0) + 1
    
    # Collect doors and windows for easy access
    doors = [obj for obj in all_objects_metadata if obj.get('is_door', False)]
    windows = [obj for obj in all_objects_metadata if obj.get('is_window', False)]
    
    # Compute room adjacency (simple check based on bbox proximity)
    room_adjacency = []
    for i, room_a in enumerate(metadata_rooms):
        for j, room_b in enumerate(metadata_rooms):
            if j <= i:
                continue
            bbox_a = room_a['room_bbox']
            bbox_b = room_b['room_bbox']
            # Check if bboxes are close (within 0.5m)
            a_min = np.array(bbox_a['min'])
            a_max = np.array(bbox_a['max'])
            b_min = np.array(bbox_b['min'])
            b_max = np.array(bbox_b['max'])
            
            # Compute minimum distance between bboxes
            dist = np.max([0, 
                          np.max(a_min - b_max),
                          np.max(b_min - a_max)])
            
            if dist < 0.5:  # Adjacent if within 0.5m
                room_adjacency.append({
                    "room_a": room_a['room_id'],
                    "room_b": room_b['room_id'],
                    "distance": float(dist)
                })
    
    # Build comprehensive scene metadata
    scene_metadata = {
        "scene_id": scene_id,
        "scene_center": scene_center,
        "up_direction": scene_up,
        "scene_bbox": scene_bbox,
        "scene_bbox_size": scene_bbox_size,
        "scene_dimensions": {
            "width": scene_bbox_size[0],
            "depth": scene_bbox_size[1],
            "height": scene_bbox_size[2]
        },
        "scene_size": scene_bbox_size,  # Alias for compatibility
        "num_rooms": len(metadata_rooms),
        "num_objects": len(all_objects_metadata),
        "total_floor_area": float(total_floor_area),
        "total_volume": float(total_volume),
        "room_type_distribution": room_type_counts,
        "object_category_distribution": scene_object_counts,
        "room_adjacency": room_adjacency,
        "doors": doors,  # List of door objects with locations
        "windows": windows,  # List of window objects with locations
        "num_doors": len(doors),
        "num_windows": len(windows),
        "rooms": metadata_rooms,
        "all_objects": all_objects_metadata  # Include all objects for reference
    }
    
    return segmented_scene, textured_scene, failed_models, scene_metadata

def process_one_scene(
    scene_path: Path, model_dir: Path, model_info_file: Path, out_root: Path,
    args: argparse.Namespace, taxonomy: Taxonomy
) -> Tuple[bool, Optional[str]]:
    scene_id = infer_ids_from_path(scene_path)
    if isinstance(scene_id, tuple):
        scene_id = scene_id[0]
    scene_id = str(scene_id)

    # Flat structure: geometry/seg, geometry/tex, and metadatas/ folders
    geometry_dir = out_root / "geometry"
    seg_dir = geometry_dir / "seg"
    tex_dir = geometry_dir / "tex"
    metadatas_dir = out_root / "metadatas"
    safe_mkdir(seg_dir)
    safe_mkdir(tex_dir)
    safe_mkdir(metadatas_dir)

    # Load scene JSON
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    # Load model_info.json
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}

    # Process scene: returns (segmented_scene, textured_scene, failed_models, scene_metadata)
    segmented_scene, textured_scene, failed, scene_metadata = process_scene_meshes(
        scene, model_dir, model_info_map, taxonomy, scene_id)

    if segmented_scene is None or textured_scene is None:
        error_msg = f"No meshes created for scene {scene_id}"
        logger.warning(error_msg)
        return False, error_msg

    if scene_metadata is None:
        error_msg = f"No metadata created for scene {scene_id}"
        logger.warning(error_msg)
        return False, error_msg

    # Export meshes
    try:
        # Determine export format
        export_format = args.format.lower()
        
        # Export segmented mesh to geometry/seg/
        if export_format == "glb":
            segmented_scene.export(seg_dir / f"{scene_id}.glb", file_type="glb")
        elif export_format == "obj":
            segmented_scene.export(seg_dir / f"{scene_id}.obj", file_type="obj")
        elif export_format == "ply":
            # For PLY, we need to merge all meshes first
            meshes_list = list(segmented_scene.geometry.values())
            if meshes_list:
                merged = trimesh.util.concatenate(meshes_list)
                merged.export(seg_dir / f"{scene_id}.ply", file_type="ply")
        else:
            raise ValueError(f"Unsupported format: {export_format}")
        
        # Export textured mesh to geometry/tex/
        # For GLB export, ensure textures are embedded
        if export_format == "glb":
            try:
                # Export with texture embedding enabled
                textured_scene.export(
                    tex_dir / f"{scene_id}.glb", 
                    file_type="glb",
                    include_normals=True
                )
            except Exception as e:
                logger.warning(f"GLB export failed for {scene_id}, trying without options: {e}")
                # Fallback: try without extra options
                textured_scene.export(tex_dir / f"{scene_id}.glb", file_type="glb")
        elif export_format == "obj":
            textured_scene.export(tex_dir / f"{scene_id}.obj", file_type="obj")
        elif export_format == "ply":
            meshes_list = list(textured_scene.geometry.values())
            if meshes_list:
                merged = trimesh.util.concatenate(meshes_list)
                merged.export(tex_dir / f"{scene_id}.ply", file_type="ply")
        
        # Save comprehensive scene metadata to metadatas/
        scene_metadata["segmented_mesh"] = f"geometry/seg/{scene_id}.{export_format}"
        scene_metadata["textured_mesh"] = f"geometry/tex/{scene_id}.{export_format}"
        scene_metadata["export_format"] = export_format
        
        # Validate metadata size (should be > 100 lines when serialized)
        metadata_json = json.dumps(scene_metadata, indent=2)
        metadata_lines = len(metadata_json.split('\n'))
        
        if metadata_lines < 100:
            logger.warning(f"Metadata for scene {scene_id} is only {metadata_lines} lines, expected >= 100")
        
        write_json(scene_metadata, metadatas_dir / f"{scene_id}_metadata.json")
        
        return True, None
    except Exception as e:
        error_msg = f"Failed to export meshes for scene {scene_id}: {e}"
        logger.exception(error_msg)
        return False, error_msg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global TAXONOMY
    global ARGS
    global logger
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--scene_list")
    ap.add_argument("--scene_file")
    ap.add_argument("--in_dir", help="Input directory to scan for JSON scene files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_info", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--format", choices=["obj", "glb", "ply"], default="glb",
                    help="Mesh export format")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenes to process (default: all)"
    )
    args = ap.parse_args()
    ARGS = args
    TAXONOMY = Taxonomy(Path(args.taxonomy))
    
    # Setup logging with file output
    out_root = Path(args.out_dir)
    log_dir = out_root / "logs"
    logger = setup_logging(log_dir)
    logger.info(f"Stage 1: Building scene meshes. Log file: {log_dir / 'stage1_errors.log'}")

    scene_paths = gather_paths_from_sources(args.scene_file, args.scenes, args.scene_list)
    
    # If no paths found and in_dir is provided, scan directory
    if not scene_paths and args.in_dir:
        in_dir = Path(args.in_dir)
        if in_dir.exists() and in_dir.is_dir():
            scene_paths = list(in_dir.glob("*.json"))
            scene_paths = [p.resolve() for p in scene_paths]
    
    if not scene_paths:
        print("No scenes found")
        print(f"  scene_file: {args.scene_file}")
        print(f"  scenes: {args.scenes}")
        print(f"  scene_list: {args.scene_list}")
        print(f"  in_dir: {args.in_dir}")
        return

    if args.limit is not None:
        scene_paths = scene_paths[:args.limit]

    out_root = Path(args.out_dir)
    safe_mkdir(out_root)

    progress = create_progress_tracker(len(scene_paths), "scenes")
    success_count = 0
    failed_scenes = []
    
    for i, scene_path in enumerate(scene_paths, 1):
        try:
            success, error_msg = process_one_scene(scene_path, Path(args.model_dir),
                                                   Path(args.model_info), out_root, args, TAXONOMY)
            if success:
                success_count += 1
                progress(i, scene_path.name, True)
            else:
                failed_scenes.append({
                    "scene_id": scene_path.stem,
                    "scene_path": str(scene_path),
                    "error": error_msg or "Unknown error"
                })
                progress(i, f"failed {scene_path.name}: {error_msg or 'Unknown error'}", False)
        except Exception as e:
            error_msg = f"Exception processing scene {scene_path.name}: {e}"
            logger.exception(error_msg)
            failed_scenes.append({
                "scene_id": scene_path.stem,
                "scene_path": str(scene_path),
                "error": str(e)
            })
            progress(i, f"failed {scene_path.name}: {e}", False)
    
    # Write failed scenes manifest if any failures occurred
    if failed_scenes:
        failed_manifest_path = out_root / "failed_scenes.csv"
        try:
            import csv
            with open(failed_manifest_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["scene_id", "scene_path", "error"])
                writer.writeheader()
                writer.writerows(failed_scenes)
            logger.info(f"Wrote failed scenes manifest to {failed_manifest_path} ({len(failed_scenes)} failures)")
        except Exception as e:
            logger.error(f"Failed to write failed scenes manifest: {e}")

    print(f"\nSuccessfully processed {success_count}/{len(scene_paths)} scenes")


if __name__ == "__main__":
    main()

