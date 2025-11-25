#!/usr/bin/env python3
"""
Export scene geometry to GLB files (regular and segmented).
This is fast and should be done first, before rendering.
"""

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh
import trimesh.visual.material

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.scene_loader import load_front_scene, extract_rooms_from_scene


def _organize_material_files(obj_path: Path, materials_dir: Path):
    """
    Move ALL material files (MTL and textures) to materials/ folder.
    Only the OBJ file should remain in geometry/ folder.
    
    Args:
        obj_path: Path to OBJ file (in geometry/ folder)
        materials_dir: Directory to move materials to (geometry/materials/)
    """
    obj_dir = obj_path.parent
    obj_name = obj_path.stem
    
    # Find MTL file (usually same name as OBJ)
    mtl_path = obj_dir / f"{obj_name}.mtl"
    if not mtl_path.exists():
        # Try alternative naming
        mtl_path = obj_dir / f"{obj_name}_materials.mtl"
        if not mtl_path.exists():
            print(f"  WARNING: MTL file not found for {obj_path.name}")
            return
    
    # Read MTL file to find texture references
    texture_files = []
    with open(mtl_path, 'r') as f:
        mtl_content = f.read()
        # Find all texture file references (map_Kd, map_Ka, map_Ks, etc.)
        texture_pattern = r'map_\w+\s+([^\s\n]+)'
        for match in re.finditer(texture_pattern, mtl_content):
            texture_path = match.group(1)
            # Handle both absolute and relative paths
            if Path(texture_path).is_absolute():
                texture_files.append(Path(texture_path))
            else:
                # Check if it's already in materials/ or in obj_dir
                if texture_path.startswith("materials/"):
                    texture_files.append(obj_dir / texture_path)
                else:
                    texture_files.append(obj_dir / texture_path)
    
    # Move MTL file to materials/
    new_mtl_path = materials_dir / mtl_path.name
    shutil.move(str(mtl_path), str(new_mtl_path))
    print(f"  Moved MTL file to: {new_mtl_path}")
    
    # Move texture files to materials/ (only if they're in obj_dir or its subdirectories)
    moved_textures = set()
    for texture_path in texture_files:
        if texture_path.exists():
            # Only move if texture is in or under obj_dir (don't move external textures)
            try:
                texture_path.relative_to(obj_dir)
                # Get just the filename
                texture_name = texture_path.name
                new_texture_path = materials_dir / texture_name
                if not new_texture_path.exists():  # Don't overwrite if already moved
                    shutil.move(str(texture_path), str(new_texture_path))
                    moved_textures.add(texture_name)
                    print(f"  Moved texture to: {new_texture_path}")
            except ValueError:
                # Texture is outside obj_dir, skip it
                print(f"  WARNING: Texture {texture_path} is outside geometry folder, skipping move")
    
    # Also find any image files in geometry/ that might have been created by trimesh
    # (trimesh sometimes creates texture files with different names)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tga', '.tiff']
    for ext in image_extensions:
        for img_file in obj_dir.glob(f'*{ext}'):
            # Skip if it's already in materials/ or if we already moved it
            if img_file.parent == materials_dir:
                continue
            if img_file.name in moved_textures:
                continue
            # Move any image files found in geometry/ to materials/
            new_img_path = materials_dir / img_file.name
            if not new_img_path.exists():
                shutil.move(str(img_file), str(new_img_path))
                print(f"  Moved image file to: {new_img_path}")
    
    # MTL content doesn't need updating - textures are referenced by filename only
    # (they're in the same materials/ folder as the MTL file)
    # Write MTL file (no changes needed to content)
    with open(new_mtl_path, 'w') as f:
        f.write(mtl_content)
    
    # Update OBJ file to reference MTL in materials/ folder
    with open(obj_path, 'r') as f:
        obj_content = f.read()
    
    # Update mtllib reference to point to materials/ folder
    obj_content = re.sub(
        r'mtllib\s+[^\s\n]+',
        f'mtllib materials/{mtl_path.name}',
        obj_content
    )
    
    with open(obj_path, 'w') as f:
        f.write(obj_content)
    print(f"  Updated OBJ file to reference MTL in materials/ folder")


def create_scene_without_ceilings(trimesh_scene: trimesh.Scene) -> trimesh.Scene:
    """
    Create a new scene excluding ceiling meshes.
    
    Args:
        trimesh_scene: Original scene
        
    Returns:
        New trimesh.Scene without ceilings
    """
    scene_no_ceiling = trimesh.Scene()
    
    for node_name in trimesh_scene.graph.nodes_geometry:
        try:
            transform, geometry_name = trimesh_scene.graph.get(node_name)
            if geometry_name not in trimesh_scene.geometry:
                if node_name not in trimesh_scene.geometry:
                    continue
                geometry = trimesh_scene.geometry[node_name]
            else:
                geometry = trimesh_scene.geometry[geometry_name]
            
            # Skip ceilings
            metadata = getattr(geometry, 'metadata', {})
            if metadata.get('is_ceiling', False):
                continue
            
            if isinstance(geometry, trimesh.Trimesh):
                # Copy mesh with all its properties (including textures)
                mesh_copy = geometry.copy()
                
                # Preserve metadata
                if hasattr(geometry, 'metadata'):
                    mesh_copy.metadata = geometry.metadata.copy()
                
                scene_no_ceiling.add_geometry(mesh_copy, node_name=node_name, transform=transform)
        except (KeyError, ValueError, IndexError):
            continue
    
    return scene_no_ceiling


def create_segmented_scene(trimesh_scene: trimesh.Scene, taxonomy: Taxonomy) -> trimesh.Scene:
    """
    Create a new scene with vertex colors set based on taxonomy category colors.
    Excludes ceilings.
    
    Args:
        trimesh_scene: Original scene
        taxonomy: Taxonomy object
        
    Returns:
        New trimesh.Scene with taxonomy colors as vertex colors (no ceilings)
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
            
            # Skip ceilings
            metadata = getattr(geometry, 'metadata', {})
            if metadata.get('is_ceiling', False):
                continue
            
            if isinstance(geometry, trimesh.Trimesh):
                # Get category color
                category_id = metadata.get('category_id', 0)
                color_rgb = taxonomy.get_color(category_id, mode="category")
                if color_rgb is None:
                    color_rgb = (127, 127, 127)
                
                # Create a new mesh with only geometry (no materials/textures)
                # This ensures vertex colors are used in GLB export
                mesh_seg = trimesh.Trimesh(
                    vertices=geometry.vertices.copy(),
                    faces=geometry.faces.copy(),
                    process=False
                )
                
                # Apply color to all vertices (RGBA format: R, G, B, A)
                num_vertices = len(mesh_seg.vertices)
                # Ensure RGBA format (4 channels)
                if len(color_rgb) == 3:
                    color_rgba = (*color_rgb, 255)  # Add alpha channel
                else:
                    color_rgba = color_rgb[:4] if len(color_rgb) >= 4 else (*color_rgb[:3], 255)
                
                vertex_colors = np.tile(np.array(color_rgba, dtype=np.uint8), (num_vertices, 1))
                
                # Set vertex colors directly (no material to override)
                mesh_seg.visual.vertex_colors = vertex_colors
                
                # Preserve metadata
                if hasattr(geometry, 'metadata'):
                    mesh_seg.metadata = geometry.metadata.copy()
                
                seg_scene.add_geometry(mesh_seg, node_name=node_name, transform=transform)
        except (KeyError, ValueError, IndexError):
            continue
    
    return seg_scene


def export_scene_geometry(scene_json: Path, future_root: Path, taxonomy: Taxonomy, 
                         output_dir: Path) -> Dict:
    """
    Export scene geometry to OBJ files (textured and segmented) and metadata JSON.
    OBJ format preserves textures (via MTL files) and vertex colors.
    
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
    
    # Create scene without ceilings
    print("Removing ceilings from scene...")
    scene_no_ceiling = create_scene_without_ceilings(trimesh_scene)
    
    # Validate scene has geometry
    if len(scene_no_ceiling.geometry) == 0:
        raise ValueError(f"Scene {scene_id} has no geometry after removing ceilings!")
    
    # Count vertices to ensure we have actual geometry
    total_vertices = 0
    for node_name in scene_no_ceiling.graph.nodes_geometry:
        try:
            transform, geometry_name = scene_no_ceiling.graph.get(node_name)
            if geometry_name not in scene_no_ceiling.geometry:
                if node_name not in scene_no_ceiling.geometry:
                    continue
                geometry = scene_no_ceiling.geometry[node_name]
            else:
                geometry = scene_no_ceiling.geometry[geometry_name]
            
            if isinstance(geometry, trimesh.Trimesh):
                total_vertices += len(geometry.vertices)
        except (KeyError, ValueError, IndexError):
            continue
    
    if total_vertices == 0:
        raise ValueError(f"Scene {scene_id} has no vertices after removing ceilings!")
    
    print(f"  Scene has {len(scene_no_ceiling.geometry)} meshes with {total_vertices} total vertices")
    
    # Save regular GLB with textures (no ceilings)
    print("Saving scene geometry...")
    glb_path = geometry_dir / f"{scene_id}.glb"
    
    try:
        scene_no_ceiling.export(glb_path, file_type="glb")
        print(f"  Saved scene geometry: {glb_path}")
    except Exception as e:
        print(f"  WARNING: GLB export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Create and save segmented GLB (no ceilings) - GLB preserves vertex colors
    print("  Creating segmented GLB...")
    seg_scene = create_segmented_scene(trimesh_scene, taxonomy)
    seg_glb_path = geometry_dir / f"{scene_id}_seg.glb"
    try:
        seg_scene.export(seg_glb_path, file_type="glb")
        print(f"  Saved segmented geometry: {seg_glb_path}")
    except Exception as e:
        print(f"  WARNING: Segmented GLB export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Generate comprehensive metadata
    print("  Generating scene metadata...")
    scene_metadata = generate_scene_metadata(
        scene_id, trimesh_scene, room_scenes, scene_no_ceiling, taxonomy
    )
    
    metadata_path = geometry_dir / f"{scene_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(scene_metadata, f, indent=2)
    print(f"  Saved scene metadata: {metadata_path}")
    
    return scene_metadata


def detect_up_direction(trimesh_scene: trimesh.Scene) -> Dict:
    """
    Detect the up direction by finding floor and ceiling meshes.
    The axis with the largest difference between floor and ceiling is the up axis.
    
    Args:
        trimesh_scene: Full scene (must include ceilings for this calculation)
        
    Returns:
        Dictionary with 'up_axis' (0=X, 1=Y, 2=Z), 'up_vector' ([1,0,0], [0,1,0], or [0,0,1]),
        'floor_center', 'ceiling_center', and 'vertical_separation'
    """
    floor_positions = []
    ceiling_positions = []
    
    for node_name in trimesh_scene.graph.nodes_geometry:
        try:
            transform, geometry_name = trimesh_scene.graph.get(node_name)
            if geometry_name not in trimesh_scene.geometry:
                if node_name not in trimesh_scene.geometry:
                    continue
                geometry = trimesh_scene.geometry[node_name]
            else:
                geometry = trimesh_scene.geometry[geometry_name]
            
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            
            metadata = getattr(geometry, 'metadata', {})
            label = metadata.get('label', '').lower()
            is_ceiling = metadata.get('is_ceiling', False)
            
            # Get world-space vertices
            vertices_world = trimesh.transform_points(geometry.vertices, transform)
            mesh_center = vertices_world.mean(axis=0)
            
            if label == 'floor' or (label == '' and not is_ceiling and 'floor' in str(geometry_name).lower()):
                floor_positions.append(mesh_center)
            elif label == 'ceiling' or is_ceiling:
                ceiling_positions.append(mesh_center)
        except (KeyError, ValueError, IndexError):
            continue
    
    if len(floor_positions) == 0 or len(ceiling_positions) == 0:
        # Fallback: assume Y-up (3D-FRONT convention)
        print(f"WARNING: Could not detect up direction (floor={len(floor_positions)}, ceiling={len(ceiling_positions)}), assuming Y-up")
        return {
            'up_axis': 1,  # Y-axis
            'up_vector': [0, 1, 0],
            'floor_center': None,
            'ceiling_center': None,
            'vertical_separation': None
        }
    
    # Calculate average positions
    floor_center = np.array(floor_positions).mean(axis=0)
    ceiling_center = np.array(ceiling_positions).mean(axis=0)
    
    # Calculate separation along each axis
    separation = ceiling_center - floor_center
    abs_separation = np.abs(separation)
    
    # The axis with the largest separation is the up axis
    up_axis = int(np.argmax(abs_separation))
    
    # Determine up direction (positive or negative)
    if separation[up_axis] > 0:
        up_vector = np.zeros(3)
        up_vector[up_axis] = 1.0
    else:
        up_vector = np.zeros(3)
        up_vector[up_axis] = -1.0
    
    print(f"Detected up direction: axis={up_axis} ({['X', 'Y', 'Z'][up_axis]}), "
          f"vector={up_vector}, separation={abs_separation[up_axis]:.2f}")
    print(f"  Floor center: {floor_center}")
    print(f"  Ceiling center: {ceiling_center}")
    
    return {
        'up_axis': int(up_axis),
        'up_vector': up_vector.tolist(),
        'floor_center': floor_center.tolist(),
        'ceiling_center': ceiling_center.tolist(),
        'vertical_separation': float(abs_separation[up_axis])
    }


def generate_scene_metadata(scene_id: str,
                            trimesh_scene: trimesh.Scene, 
                            room_scenes: Dict[str, trimesh.Scene],
                            scene_no_ceiling: trimesh.Scene,
                            taxonomy: Taxonomy) -> Dict:
    """
    Generate comprehensive metadata for a scene.
    
    Args:
        scene_id: Scene identifier
        trimesh_scene: Full scene (with ceilings)
        room_scenes: Dictionary of room_name -> room_scene
        scene_no_ceiling: Scene without ceilings
        taxonomy: Taxonomy object
        
    Returns:
        Dictionary with comprehensive scene metadata
    """
    # Helper to get mesh info from a scene
    def get_mesh_info(scene: trimesh.Scene, exclude_ceilings: bool = False, include_furniture_positions: bool = False):
        category_counts = Counter()
        label_counts = Counter()
        furniture_count = 0
        architectural_count = 0
        total_vertices = 0
        total_faces = 0
        categories_present = set()
        labels_present = set()
        furniture_positions = []  # List of furniture with positions
        
        for node_name in scene.graph.nodes_geometry:
            try:
                transform, geometry_name = scene.graph.get(node_name)
                if geometry_name not in scene.geometry:
                    if node_name not in scene.geometry:
                        continue
                    geometry = scene.geometry[node_name]
                else:
                    geometry = scene.geometry[geometry_name]
                
                if not isinstance(geometry, trimesh.Trimesh):
                    continue
                
                metadata = getattr(geometry, 'metadata', {})
                
                # Skip ceilings if requested
                if exclude_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                category_id = metadata.get('category_id', 0)
                label = metadata.get('label', 'unknown')
                
                if category_id > 0:
                    category_name = taxonomy.id_to_name(category_id)
                    category_counts[category_name] += 1
                    categories_present.add(category_name)
                
                if label and label != 'unknown':
                    label_counts[label] += 1
                    labels_present.add(label)
                
                # Apply transform to get world-space bounds
                vertices_world = trimesh.transform_points(geometry.vertices, transform)
                total_vertices += len(vertices_world)
                total_faces += len(geometry.faces)
                
                # Calculate mesh center (position) in world space
                mesh_center = vertices_world.mean(axis=0)
                mesh_bounds_min = vertices_world.min(axis=0)
                mesh_bounds_max = vertices_world.max(axis=0)
                mesh_size = mesh_bounds_max - mesh_bounds_min
                
                # Count furniture vs architectural
                is_architectural = False
                if metadata.get('is_ceiling', False):
                    architectural_count += 1
                    is_architectural = True
                elif label in ['wall', 'floor', 'ceiling', 'structure']:
                    architectural_count += 1
                    is_architectural = True
                else:
                    furniture_count += 1
                
                # Store furniture position if requested
                if include_furniture_positions and not is_architectural:
                    furniture_info = {
                        'category_id': category_id,
                        'category_name': taxonomy.id_to_name(category_id) if category_id > 0 else 'unknown',
                        'label': label,
                        'position': mesh_center.tolist(),  # [x, y, z] center position
                        'bounds': {
                            'min': mesh_bounds_min.tolist(),
                            'max': mesh_bounds_max.tolist(),
                            'size': mesh_size.tolist()
                        },
                        'ref_id': metadata.get('ref_id', 'unknown')
                    }
                    furniture_positions.append(furniture_info)
                
            except (KeyError, ValueError, IndexError):
                continue
        
        result = {
            'category_counts': dict(category_counts),
            'label_counts': dict(label_counts),
            'furniture_count': furniture_count,
            'architectural_count': architectural_count,
            'total_vertices': total_vertices,
            'total_faces': total_faces,
            'categories_present': sorted(list(categories_present)),
            'labels_present': sorted(list(labels_present))
        }
        
        if include_furniture_positions:
            result['furniture_positions'] = furniture_positions
        
        return result
    
    # Detect up direction using floor and ceiling meshes
    up_direction_info = detect_up_direction(trimesh_scene)
    
    # Get scene bounds (without ceilings)
    try:
        scene_bounds = scene_no_ceiling.bounds
        scene_min = scene_bounds[0].tolist()
        scene_max = scene_bounds[1].tolist()
        scene_center = scene_no_ceiling.centroid.tolist()
        scene_size = (scene_bounds[1] - scene_bounds[0]).tolist()
    except:
        scene_min = scene_max = scene_center = scene_size = None
    
    # Get overall scene statistics
    scene_info = get_mesh_info(trimesh_scene, exclude_ceilings=True, include_furniture_positions=False)
    
    # Get per-room statistics with furniture positions
    rooms_metadata = {}
    for room_name, room_scene in room_scenes.items():
        try:
            # Get room bounds
            room_bounds = room_scene.bounds
            room_min = room_bounds[0].tolist()
            room_max = room_bounds[1].tolist()
            room_center = room_scene.centroid.tolist()
            room_size = (room_bounds[1] - room_bounds[0]).tolist()
        except:
            room_min = room_max = room_center = room_size = None
        
        # Get room info with furniture positions
        room_info = get_mesh_info(room_scene, exclude_ceilings=True, include_furniture_positions=True)
        
        rooms_metadata[room_name] = {
            "room_name": room_name,
            "mesh_count": len(list(room_scene.graph.nodes_geometry)),
            "center": room_center,  # Room center for graph building
            "bounds": {
                "min": room_min,
                "max": room_max,
                "center": room_center,  # Also in bounds for compatibility
                "size": room_size
            },
            "statistics": {
                "furniture_count": room_info['furniture_count'],
                "architectural_count": room_info['architectural_count'],
                "total_vertices": room_info['total_vertices'],
                "total_faces": room_info['total_faces']
            },
            "category_distribution": room_info['category_counts'],
            "label_distribution": room_info['label_counts'],
            "categories_present": room_info['categories_present'],
            "labels_present": room_info['labels_present'],
            "furniture": room_info.get('furniture_positions', [])  # List of furniture with positions
        }
    
    # Build comprehensive metadata
    metadata = {
        "scene_id": scene_id,
        "scene_bounds": {
            "min": scene_min,
            "max": scene_max,
            "center": scene_center,
            "size": scene_size
        },
        "up_direction": up_direction_info,  # Detected up direction from floor/ceiling
        "statistics": {
            "total_meshes": len(list(trimesh_scene.graph.nodes_geometry)),
            "total_meshes_no_ceiling": len(list(scene_no_ceiling.graph.nodes_geometry)),
            "room_count": len(room_scenes),
            "furniture_count": scene_info['furniture_count'],
            "architectural_count": scene_info['architectural_count'],
            "total_vertices": scene_info['total_vertices'],
            "total_faces": scene_info['total_faces']
        },
        "category_distribution": scene_info['category_counts'],
        "label_distribution": scene_info['label_counts'],
        "categories_present": scene_info['categories_present'],
        "labels_present": scene_info['labels_present'],
        "rooms": rooms_metadata
    }
    
    return metadata

