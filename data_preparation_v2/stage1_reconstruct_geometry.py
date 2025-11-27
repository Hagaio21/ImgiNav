#!/usr/bin/env python3
"""
Stage 1: Scene Geometry Reconstruction - FIXED VERSION v2

Fixes:
1. Floor = dark gray, Wall = light gray (for visibility in layouts)
2. Furniture textures/colors are PRESERVED (not overwritten)
3. Proper vertex color handling for GLB export
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy JSON file."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_category_color(category: str, taxonomy: Dict) -> Tuple[int, int, int]:
    """Get color for a category from taxonomy."""
    category_to_color = taxonomy.get("category_to_color", {})
    
    # Try exact match first
    if category in category_to_color:
        return tuple(category_to_color[category])
    
    # Try lowercase match (taxonomy has "floor", "wall" in lowercase)
    if category.lower() in category_to_color:
        return tuple(category_to_color[category.lower()])
    
    # Try capitalized match
    if category.capitalize() in category_to_color:
        return tuple(category_to_color[category.capitalize()])
    
    # Fallback to default gray
    return (127, 127, 127)


def load_mesh(model_dir: Path, jid: str):
    """
    Load a furniture model mesh from 3D-FUTURE with textures.
    Returns a list of meshes (to preserve individual materials).
    
    Handles:
    - UV-mapped textures (texture images in .mtl)
    - Material colors (diffuse color in .mtl)
    - Vertex colors
    """
    obj_path = model_dir / jid / "raw_model.obj"
    glb_path = model_dir / jid / "raw_model.glb"
    
    if obj_path.exists():
        # Load OBJ with materials/textures
        # Use file path resolver to find texture images
        resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
        try:
            loaded = trimesh.load(
                str(obj_path),
                process=False,
                maintain_order=True,
                resolver=resolver,
                skip_materials=False,  # Load materials
            )
        except Exception as e:
            # Fallback: load without materials
            logger.debug(f"Failed to load {jid} with materials: {e}, trying without")
            loaded = trimesh.load(
                str(obj_path),
                process=False,
                force="mesh",
            )
    elif glb_path.exists():
        loaded = trimesh.load(
            str(glb_path),
            process=False,
            maintain_order=True
        )
    else:
        raise FileNotFoundError(f"Model not found: {obj_path} or {glb_path}")
    
    # Return list of meshes to preserve materials
    meshes = []
    
    if isinstance(loaded, trimesh.Scene):
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                # Ensure mesh has valid visual
                mesh = ensure_mesh_visual(geom)
                meshes.append(mesh)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = ensure_mesh_visual(loaded)
        meshes.append(mesh)
    else:
        raise ValueError(f"Unexpected type: {type(loaded)}")
    
    if not meshes:
        raise ValueError("No valid meshes loaded")
    
    return meshes


def ensure_mesh_visual(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Ensure mesh has valid visual that can be exported to GLB.
    
    Converts TextureVisuals with embedded images to ColorVisuals if needed.
    """
    if not hasattr(mesh, 'visual') or mesh.visual is None:
        # No visual - add default gray color
        n_vertices = len(mesh.vertices)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.full((n_vertices, 4), [180, 180, 180, 255], dtype=np.uint8)
        )
        return mesh
    
    visual = mesh.visual
    
    # If it's already ColorVisuals with valid vertex colors, keep it
    if isinstance(visual, trimesh.visual.ColorVisuals):
        if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
            return mesh
    
    # Try to convert to vertex colors
    try:
        if hasattr(visual, 'to_color'):
            color_visual = visual.to_color()
            if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                mesh.visual = color_visual
                return mesh
    except Exception:
        pass
    
    # Try to get material color
    try:
        if hasattr(visual, 'material'):
            material = visual.material
            if hasattr(material, 'diffuse'):
                diffuse = material.diffuse
                if diffuse is not None:
                    # Convert to vertex colors
                    color = np.array(diffuse[:3] * 255, dtype=np.uint8)
                    n_vertices = len(mesh.vertices)
                    vertex_colors = np.tile([*color, 255], (n_vertices, 1)).astype(np.uint8)
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=mesh,
                        vertex_colors=vertex_colors
                    )
                    return mesh
    except Exception:
        pass
    
    # Fallback: gray color
    n_vertices = len(mesh.vertices)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.full((n_vertices, 4), [180, 180, 180, 255], dtype=np.uint8)
    )
    return mesh


def create_arch_mesh(arch: Dict, arch_type: str, taxonomy: Dict = None) -> trimesh.Trimesh:
    """
    Create an architectural mesh (wall/floor/door/window) from 3D-FRONT JSON data.
    
    Args:
        arch: Architectural element dictionary from 3D-FRONT JSON
        arch_type: "floor", "wall", "door", or "window"
        taxonomy: Taxonomy dictionary for colors (optional)
    """
    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
    faces = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Get color from taxonomy or use defaults
    # Taxonomy has lowercase keys: "floor", "wall", "Door", "Window"
    if taxonomy:
        category_to_color = taxonomy.get("category_to_color", {})
        if arch_type == "floor":
            color = category_to_color.get("floor", [200, 200, 200])
        elif arch_type == "wall":
            color = category_to_color.get("wall", [50, 50, 50])
        elif arch_type == "door":
            color = category_to_color.get("Door", [255, 100, 100])
        elif arch_type == "window":
            color = category_to_color.get("Window", [100, 200, 255])
        else:
            color = [127, 127, 127]
        color = list(color) + [255]  # Add alpha
    else:
        # Fallback defaults matching taxonomy
        if arch_type == "floor":
            color = [200, 200, 200, 255]  # Light gray for floor
        elif arch_type == "wall":
            color = [50, 50, 50, 255]  # Dark gray for walls
        elif arch_type == "door":
            color = [255, 100, 100, 255]  # Red/salmon for doors
        elif arch_type == "window":
            color = [100, 200, 255, 255]  # Light blue for windows
        else:
            color = [127, 127, 127, 255]
    
    # Apply vertex colors using ColorVisuals
    n_vertices = len(mesh.vertices)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(color, (n_vertices, 1)).astype(np.uint8)
    )
    
    return mesh


def build_transform(child: Dict) -> np.ndarray:
    """Build 4x4 transformation matrix from child transform data."""
    pos = np.array(child.get("pos", [0, 0, 0]), dtype=np.float64)
    rot = np.array(child.get("rot", [0, 0, 0, 1]), dtype=np.float64)
    scl = np.array(child.get("scale", [1, 1, 1]), dtype=np.float64)
    
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    Rm = Rotation.from_quat(rot).as_matrix()
    Sm = np.diag(scl)
    T[:3, :3] = Rm @ Sm
    return T


def apply_category_color(mesh: trimesh.Trimesh, category: str, taxonomy: Dict) -> trimesh.Trimesh:
    """
    Apply category color to a mesh for segmentation.
    Creates a copy with solid vertex colors.
    """
    color = get_category_color(category, taxonomy)
    mesh_copy = mesh.copy()
    
    n_vertices = len(mesh_copy.vertices)
    colors = np.array([list(color) + [255]] * n_vertices, dtype=np.uint8)
    
    mesh_copy.visual = trimesh.visual.ColorVisuals(
        mesh=mesh_copy,
        vertex_colors=colors
    )
    
    # Debug log for floor/wall
    if category in ('Floor', 'Wall', 'floor', 'wall'):
        logger.debug(f"apply_category_color: {category} -> color={color}")
    
    return mesh_copy


def process_scene_geometry(
    scene_data: Dict,
    model_dir: Path,
    model_info_map: Dict,
    taxonomy: Dict,
    texture_dir: Optional[Path] = None
) -> Tuple[trimesh.Scene, trimesh.Scene, Dict]:
    """
    Process a scene and build textured and segmented geometry.
    """
    failed_models = {}
    scene_objects = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}
    
    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "UnknownRoom")
        
        for child_index, child in enumerate(room.get("children", [])):
            ref_id = child.get("ref")
            if not ref_id:
                continue
            
            try:
                mesh = None
                category = "UnknownCategory"
                is_arch = False  # Is this an architectural element (floor/wall)?
                
                if ref_id in furniture_map:
                    # FURNITURE - load from 3D-FUTURE, keep original textures
                    item_info = furniture_map[ref_id]
                    jid = item_info.get('jid')
                    if not jid:
                        raise ValueError("Missing 'jid'")
                    
                    meshes = load_mesh(model_dir, jid)  # Returns list of meshes
                    model_info = model_info_map.get(jid, {})
                    
                    # Check if this is a door or window based on title
                    title = (item_info.get('title') or '').lower()
                    if title.startswith('door/') or title.startswith('door\\'):
                        category = 'Door'
                        is_arch = False
                    elif title.startswith('window/') or title.startswith('window\\'):
                        category = 'Window'
                        is_arch = False
                    else:
                        category = model_info.get('category') or item_info.get('title') or "UnknownCategory"
                        is_arch = False
                    
                    transform = build_transform(child)
                    
                    # Add each mesh separately to preserve materials
                    for mesh_idx, mesh in enumerate(meshes):
                        if mesh is None or mesh.is_empty:
                            continue
                        scene_objects.append({
                            "mesh": mesh,
                            "transform": transform,
                            "category": category,
                            "is_arch": is_arch,
                            "node_name": f"{ref_id}_{child_index}_{mesh_idx}"
                        })
                    
                    # Skip the arch handling below
                    continue
                
                elif ref_id in arch_map:
                    # ARCHITECTURAL ELEMENT - floor/wall/door/window
                    arch = arch_map[ref_id]
                    arch_type_str = arch.get("type", "")
                    
                    if "Ceiling" in arch_type_str:
                        continue
                    
                    if 'Floor' in arch_type_str:
                        category = 'Floor'
                        mesh = create_arch_mesh(arch, "floor", taxonomy)
                        is_arch = True
                    elif 'Door' in arch_type_str or 'door' in arch_type_str:
                        category = 'Door'
                        mesh = create_arch_mesh(arch, "door", taxonomy)
                        is_arch = False  # Doors should be in graph
                    elif 'Window' in arch_type_str or 'window' in arch_type_str:
                        category = 'Window'
                        mesh = create_arch_mesh(arch, "window", taxonomy)
                        is_arch = False  # Windows should be in graph
                    else:
                        category = 'Wall'
                        mesh = create_arch_mesh(arch, "wall", taxonomy)
                        is_arch = True
                    
                    if mesh is None or mesh.is_empty:
                        raise ValueError("Empty mesh")
                    
                    transform = build_transform(child)
                    
                    scene_objects.append({
                        "mesh": mesh,
                        "transform": transform,
                        "category": category,
                        "is_arch": is_arch,
                        "node_name": f"{ref_id}_{child_index}"
                    })
                    
                else:
                    failed_models[ref_id] = "Reference not found"
                    continue
            
            except Exception as e:
                logger.debug(f"Failed to process {ref_id}: {e}")
                failed_models[ref_id] = str(e)
    
    if not scene_objects:
        return None, None, failed_models
    
    # Build textured scene
    # - Furniture: keep original mesh with textures
    # - Floor/Wall: already have colors set in create_arch_mesh()
    textured_scene = trimesh.Scene()
    for obj in scene_objects:
        # Use mesh as-is (furniture has textures, arch has vertex colors)
        textured_scene.add_geometry(
            obj['mesh'].copy(),
            node_name=obj['node_name'],
            transform=obj['transform']
        )
    
    # Build segmented scene
    # - All objects get solid category colors
    segmented_scene = trimesh.Scene()
    for obj in scene_objects:
        seg_mesh = apply_category_color(obj['mesh'], obj['category'], taxonomy)
        segmented_scene.add_geometry(
            seg_mesh,
            node_name=obj['node_name'],
            transform=obj['transform']
        )
    
    return textured_scene, segmented_scene, failed_models


def process_one_scene(
    scene_path: Path,
    model_dir: Path,
    model_info_file: Path,
    taxonomy_path: Path,
    output_dir: Path,
    texture_dir: Optional[Path] = None
) -> Tuple[bool, Optional[str]]:
    """Process a single scene and export GLB files."""
    scene_id = scene_path.stem
    
    with open(scene_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)
    
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}
    
    taxonomy = load_taxonomy(taxonomy_path)
    
    try:
        textured_scene, segmented_scene, failed = process_scene_geometry(
            scene_data,
            model_dir,
            model_info_map,
            taxonomy,
            texture_dir
        )
        
        if textured_scene is None or segmented_scene is None:
            return False, "No geometry generated"
        
        tex_output = output_dir / "tex" / f"{scene_id}_tex.glb"
        seg_output = output_dir / "seg" / f"{scene_id}_seg.glb"
        
        tex_output.parent.mkdir(parents=True, exist_ok=True)
        seg_output.parent.mkdir(parents=True, exist_ok=True)
        
        textured_scene.export(str(tex_output), file_type="glb")
        segmented_scene.export(str(seg_output), file_type="glb")
        
        if failed:
            logger.debug(f"Scene {scene_id}: {len(failed)} models failed")
        
        return True, None
    
    except Exception as e:
        logger.exception(f"Failed {scene_id}: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Reconstruct scene geometry (fixed v2)")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-info", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--texture-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    scenes_dir = Path(args.scenes_dir)
    model_dir = Path(args.model_dir)
    model_info_file = Path(args.model_info)
    taxonomy_path = Path(args.taxonomy)
    output_dir = Path(args.output_dir)
    texture_dir = Path(args.texture_dir) if args.texture_dir else None
    
    scene_files = list(scenes_dir.glob("*.json"))
    if not scene_files:
        logger.error(f"No scene files found in {scenes_dir}")
        return
    
    if args.limit:
        scene_files = scene_files[:args.limit]
    
    logger.info(f"Processing {len(scene_files)} scenes...")
    
    success_count = 0
    
    for i, scene_path in enumerate(scene_files, 1):
        success, error = process_one_scene(
            scene_path, model_dir, model_info_file, taxonomy_path, output_dir, texture_dir
        )
        
        if success:
            success_count += 1
            logger.info(f"[{i}/{len(scene_files)}] ✓ {scene_path.name}")
        else:
            logger.warning(f"[{i}/{len(scene_files)}] ✗ {scene_path.name}: {error}")
    
    logger.info(f"\nDone: {success_count}/{len(scene_files)}")


if __name__ == "__main__":
    main()