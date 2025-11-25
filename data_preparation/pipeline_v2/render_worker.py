#!/usr/bin/env python3
"""
Render worker for 3D-FRONT scenes.
Renders top-down layouts and perspective POVs using pyrender.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyrender
import trimesh
from PIL import Image

import sys
from pathlib import Path

# Add project root to path for imports
# __file__ is at: .../ImgiNav/data_preparation/pipeline_v2/render_worker.py
# Project root is: .../ImgiNav/
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Go up from pipeline_v2 -> data_preparation -> ImgiNav
sys.path.insert(0, str(project_root))

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.scene_loader import load_front_scene

# Set EGL platform for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def trimesh_to_pyrender_scene(trimesh_scene: trimesh.Scene, 
                               hide_ceilings: bool = False) -> pyrender.Scene:
    """
    Convert trimesh.Scene to pyrender.Scene.
    
    Args:
        trimesh_scene: Input trimesh scene
        hide_ceilings: If True, exclude nodes with is_ceiling=True
        
    Returns:
        pyrender.Scene object
    """
    pyrender_scene = pyrender.Scene()
    
    for node_name in trimesh_scene.graph.nodes_geometry:
        geometry = trimesh_scene.geometry[node_name]
        
        # Check if ceiling should be hidden
        if hide_ceilings:
            metadata = getattr(geometry, 'metadata', {})
            if metadata.get('is_ceiling', False):
                continue
        
        # Convert trimesh to pyrender mesh
        if isinstance(geometry, trimesh.Trimesh):
            # Get vertex colors if available
            vertex_colors = None
            if hasattr(geometry.visual, 'vertex_colors'):
                vc = geometry.visual.vertex_colors
                if vc is not None and len(vc) > 0:
                    # Convert to uint8 if needed
                    if vc.dtype != np.uint8:
                        vc = np.clip(vc, 0, 255).astype(np.uint8)
                    vertex_colors = vc[:, :3]  # RGB only
            
            # Create pyrender mesh
            pyrender_mesh = pyrender.Mesh.from_trimesh(geometry, 
                                                       smooth=False,
                                                       vertex_colors=vertex_colors)
            
            # Get transform from scene graph
            transform = trimesh_scene.graph.get(node_name)[0]
            
            # Add to pyrender scene
            pyrender_scene.add(pyrender_mesh, pose=transform, name=node_name)
    
    return pyrender_scene


def render_layout_rgb(pyrender_scene: pyrender.Scene, 
                      width: int = 256, height: int = 256) -> np.ndarray:
    """
    Render top-down RGB layout.
    
    Args:
        pyrender_scene: pyrender scene (ceilings should be filtered)
        width: Image width
        height: Image height
        
    Returns:
        RGB image array (H, W, 3) uint8
    """
    # Calculate scene bounds
    bounds = []
    for node in pyrender_scene.mesh_nodes:
        mesh = node.mesh
        if mesh.positions.shape[0] > 0:
            # Transform vertices to world space
            vertices = mesh.positions
            transform = node.matrix
            if transform is not None:
                vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                vertices_world = (transform @ vertices_hom.T).T[:, :3]
            else:
                vertices_world = vertices
            bounds.append(vertices_world)
    
    if not bounds:
        # Default bounds if no geometry
        min_bounds = np.array([-5, -5, 0])
        max_bounds = np.array([5, 5, 3])
    else:
        all_vertices = np.vstack(bounds)
        min_bounds = all_vertices.min(axis=0)
        max_bounds = all_vertices.max(axis=0)
    
    # Center and size
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    max_size = max(size[0], size[1])
    
    # Camera setup: orthographic top-down
    camera = pyrender.OrthographicCamera(xmag=max_size/2, ymag=max_size/2)
    
    # Camera pose: looking down from above
    camera_pose = np.eye(4)
    camera_pose[2, 3] = center[2] + max_size  # Height above scene
    camera_pose[1, 1] = -1  # Flip Y for correct orientation
    camera_pose[2, 2] = -1
    
    # Add camera and light
    camera_node = pyrender_scene.add(camera, pose=camera_pose)
    
    # Add directional light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = np.eye(4)
    light_pose[2, 3] = center[2] + max_size
    pyrender_scene.add(light, pose=light_pose)
    
    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(pyrender_scene)
    renderer.delete()
    
    # Remove camera and light
    pyrender_scene.remove_node(camera_node)
    
    return color


def render_layout_seg(pyrender_scene: pyrender.Scene, 
                     taxonomy: Taxonomy,
                     width: int = 256, height: int = 256) -> np.ndarray:
    """
    Render top-down segmentation layout with taxonomy colors.
    
    Args:
        pyrender_scene: pyrender scene (ceilings should be filtered)
        taxonomy: Taxonomy object for color mapping
        width: Image width
        height: Image height
        
    Returns:
        Segmentation image array (H, W, 3) uint8
    """
    # Swap materials for segmentation
    original_materials = {}
    for node in pyrender_scene.mesh_nodes:
        mesh = node.mesh
        # Get category_id from metadata (stored in node name or need to track)
        # We'll need to store category_id mapping when converting
        
        # For now, we'll need to access the original trimesh scene metadata
        # This is a limitation - we may need to pass metadata separately
        pass
    
    # For segmentation, we'll create a new scene with colored materials
    # This is simplified - in practice, we need to map each mesh to its category_id
    # and apply the taxonomy color
    
    # Re-render with flat materials
    # Calculate bounds (same as RGB)
    bounds = []
    for node in pyrender_scene.mesh_nodes:
        mesh = node.mesh
        if mesh.positions.shape[0] > 0:
            vertices = mesh.positions
            transform = node.matrix
            if transform is not None:
                vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                vertices_world = (transform @ vertices_hom.T).T[:, :3]
            else:
                vertices_world = vertices
            bounds.append(vertices_world)
    
    if not bounds:
        min_bounds = np.array([-5, -5, 0])
        max_bounds = np.array([5, 5, 3])
    else:
        all_vertices = np.vstack(bounds)
        min_bounds = all_vertices.min(axis=0)
        max_bounds = all_vertices.max(axis=0)
    
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    max_size = max(size[0], size[1])
    
    # Create new scene with colored materials
    seg_scene = pyrender.Scene()
    
    # We need to rebuild with colored materials
    # This requires access to the original trimesh scene metadata
    # For now, return a placeholder - this will be improved
    
    camera = pyrender.OrthographicCamera(xmag=max_size/2, ymag=max_size/2)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = center[2] + max_size
    camera_pose[1, 1] = -1
    camera_pose[2, 2] = -1
    
    camera_node = seg_scene.add(camera, pose=camera_pose)
    
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(seg_scene)
    renderer.delete()
    
    return color


def render_layout_seg_improved(trimesh_scene: trimesh.Scene,
                               taxonomy: Taxonomy,
                               width: int = 256, height: int = 256) -> np.ndarray:
    """
    Render segmentation layout by rebuilding scene with colored materials.
    
    Args:
        trimesh_scene: Original trimesh scene
        taxonomy: Taxonomy object
        width: Image width
        height: Image height
        
    Returns:
        Segmentation image array
    """
    # Create new pyrender scene with colored meshes
    seg_scene = pyrender.Scene()
    
    # Calculate bounds
    all_vertices = []
    for node_name in trimesh_scene.graph.nodes_geometry:
        geometry = trimesh_scene.geometry[node_name]
        if isinstance(geometry, trimesh.Trimesh):
            transform = trimesh_scene.graph.get(node_name)[0]
            vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
            vertices_world = (transform @ vertices_hom.T).T[:, :3]
            all_vertices.append(vertices_world)
    
    if not all_vertices:
        min_bounds = np.array([-5, -5, 0])
        max_bounds = np.array([5, 5, 3])
    else:
        all_vertices = np.vstack(all_vertices)
        min_bounds = all_vertices.min(axis=0)
        max_bounds = all_vertices.max(axis=0)
    
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    max_size = max(size[0], size[1])
    
    # Add meshes with colored materials
    for node_name in trimesh_scene.graph.nodes_geometry:
        geometry = trimesh_scene.geometry[node_name]
        metadata = getattr(geometry, 'metadata', {})
        
        # Skip ceilings
        if metadata.get('is_ceiling', False):
            continue
        
        if isinstance(geometry, trimesh.Trimesh):
            # Get category color
            category_id = metadata.get('category_id', 0)
            color_rgb = taxonomy.get_color(category_id, mode="category")
            if color_rgb is None:
                color_rgb = (127, 127, 127)  # Default gray
            
            # Create flat material with taxonomy color
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[c/255.0 for c in color_rgb] + [1.0],
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
            
            # Create mesh with material
            pyrender_mesh = pyrender.Mesh.from_trimesh(geometry, material=material)
            transform = trimesh_scene.graph.get(node_name)[0]
            seg_scene.add(pyrender_mesh, pose=transform, name=node_name)
    
    # Camera setup
    camera = pyrender.OrthographicCamera(xmag=max_size/2, ymag=max_size/2)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = center[2] + max_size
    camera_pose[1, 1] = -1
    camera_pose[2, 2] = -1
    
    camera_node = seg_scene.add(camera, pose=camera_pose)
    
    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(seg_scene)
    renderer.delete()
    
    return color


def find_floor_meshes(trimesh_scene: trimesh.Scene, taxonomy: Taxonomy) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Find floor meshes and return their bounding boxes.
    
    Args:
        trimesh_scene: trimesh scene
        taxonomy: Taxonomy object
        
    Returns:
        List of (min_bounds, max_bounds) tuples for floor meshes
    """
    floor_bboxes = []
    floor_ids = [taxonomy.data.get("category2id", {}).get("floor", 0),
                 taxonomy.data.get("label2id", {}).get("floor", 0)]
    
    for node_name in trimesh_scene.graph.nodes_geometry:
        geometry = trimesh_scene.geometry[node_name]
        metadata = getattr(geometry, 'metadata', {})
        
        category_id = metadata.get('category_id', 0)
        label = metadata.get('label', '').lower()
        
        if category_id in floor_ids or label == 'floor':
            if isinstance(geometry, trimesh.Trimesh):
                transform = trimesh_scene.graph.get(node_name)[0]
                vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                vertices_world = (transform @ vertices_hom.T).T[:, :3]
                
                min_bounds = vertices_world.min(axis=0)
                max_bounds = vertices_world.max(axis=0)
                floor_bboxes.append((min_bounds, max_bounds))
    
    return floor_bboxes


def sample_camera_position(floor_bboxes: List[Tuple[np.ndarray, np.ndarray]],
                           furniture_bboxes: List[Tuple[np.ndarray, np.ndarray]],
                           eye_height: float = 1.6,
                           max_attempts: int = 20) -> Optional[np.ndarray]:
    """
    Sample a valid camera position on the floor, not inside furniture.
    
    Args:
        floor_bboxes: List of floor bounding boxes
        furniture_bboxes: List of furniture bounding boxes
        eye_height: Camera eye height in meters
        max_attempts: Maximum sampling attempts
        
    Returns:
        Camera position (x, y, z) or None if failed
    """
    if not floor_bboxes:
        return None
    
    # Combine all floor bounds
    all_min = np.array([bbox[0] for bbox in floor_bboxes]).min(axis=0)
    all_max = np.array([bbox[1] for bbox in floor_bboxes]).max(axis=0)
    
    for attempt in range(max_attempts):
        # Sample random X, Z within floor bounds
        x = np.random.uniform(all_min[0], all_max[0])
        z = np.random.uniform(all_min[2], all_max[2])
        y = eye_height
        
        camera_pos = np.array([x, y, z])
        
        # Check if inside any furniture bbox
        inside_furniture = False
        for f_min, f_max in furniture_bboxes:
            if (f_min[0] <= x <= f_max[0] and
                f_min[1] <= y <= f_max[1] and
                f_min[2] <= z <= f_max[2]):
                inside_furniture = True
                break
        
        if not inside_furniture:
            return camera_pos
    
    return None


def render_pov_seg(trimesh_scene: trimesh.Scene,
                   camera_pos: np.ndarray,
                   taxonomy: Taxonomy,
                   camera_target: Optional[np.ndarray] = None,
                   width: int = 256,
                   height: int = 256,
                   fov: float = 70.0) -> np.ndarray:
    """
    Render POV segmentation with colored materials.
    
    Args:
        trimesh_scene: Original trimesh scene
        camera_pos: Camera position
        taxonomy: Taxonomy object
        camera_target: Look-at target
        width: Image width
        height: Image height
        fov: Field of view in degrees
        
    Returns:
        Segmentation image array
    """
    # Create scene with colored materials
    seg_scene = pyrender.Scene()
    
    # Calculate scene center for camera target
    all_vertices = []
    for node_name in trimesh_scene.graph.nodes_geometry:
        geometry = trimesh_scene.geometry[node_name]
        if isinstance(geometry, trimesh.Trimesh):
            transform = trimesh_scene.graph.get(node_name)[0]
            vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
            vertices_world = (transform @ vertices_hom.T).T[:, :3]
            all_vertices.append(vertices_world)
    
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        if camera_target is None:
            camera_target = all_vertices.mean(axis=0)
    else:
        if camera_target is None:
            camera_target = camera_pos + np.array([0, 0, -1])
    
    # Add meshes with colored materials
    for node_name in trimesh_scene.graph.nodes_geometry:
        geometry = trimesh_scene.geometry[node_name]
        if isinstance(geometry, trimesh.Trimesh):
            metadata = getattr(geometry, 'metadata', {})
            
            # Get category color
            category_id = metadata.get('category_id', 0)
            color_rgb = taxonomy.get_color(category_id, mode="category")
            if color_rgb is None:
                color_rgb = (127, 127, 127)
            
            # Create flat material
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[c/255.0 for c in color_rgb] + [1.0],
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
            
            pyrender_mesh = pyrender.Mesh.from_trimesh(geometry, material=material)
            transform = trimesh_scene.graph.get(node_name)[0]
            seg_scene.add(pyrender_mesh, pose=transform, name=node_name)
    
    # Camera setup
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov), aspectRatio=width/height)
    
    forward = camera_target - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = camera_pos
    
    camera_node = seg_scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = camera_pose.copy()
    seg_scene.add(light, pose=light_pose)
    
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(seg_scene)
    renderer.delete()
    
    return color


def render_pov(pyrender_scene: pyrender.Scene,
               camera_pos: np.ndarray,
               camera_target: Optional[np.ndarray] = None,
               width: int = 256,
               height: int = 256,
               fov: float = 70.0) -> np.ndarray:
    """
    Render perspective POV from camera position.
    
    Args:
        pyrender_scene: pyrender scene
        camera_pos: Camera position (x, y, z)
        camera_target: Look-at target (default: scene center)
        width: Image width
        height: Image height
        fov: Field of view in degrees
        
    Returns:
        RGB image array
    """
    if camera_target is None:
        # Calculate scene center
        bounds = []
        for node in pyrender_scene.mesh_nodes:
            mesh = node.mesh
            if mesh.positions.shape[0] > 0:
                vertices = mesh.positions
                transform = node.matrix
                if transform is not None:
                    vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                    vertices_world = (transform @ vertices_hom.T).T[:, :3]
                else:
                    vertices_world = vertices
                bounds.append(vertices_world)
        
        if bounds:
            all_vertices = np.vstack(bounds)
            camera_target = all_vertices.mean(axis=0)
        else:
            camera_target = camera_pos + np.array([0, 0, -1])
    
    # Camera setup
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov), aspectRatio=width/height)
    
    # Look-at matrix
    forward = camera_target - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = camera_pos
    
    # Add camera and light
    camera_node = pyrender_scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = camera_pose.copy()
    pyrender_scene.add(light, pose=light_pose)
    
    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(pyrender_scene)
    renderer.delete()
    
    # Remove camera and light
    pyrender_scene.remove_node(camera_node)
    
    return color


def main():
    parser = argparse.ArgumentParser(description="Render 3D-FRONT scene layouts and POVs")
    parser.add_argument("--scene_json", required=True, help="Path to 3D-FRONT JSON file")
    parser.add_argument("--future_root", required=True, help="Path to 3D-FUTURE model directory")
    parser.add_argument("--output_dir", required=True, help="Output dataset root directory")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    parser.add_argument("--num_povs", type=int, default=6, help="Number of POVs to render")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load taxonomy
    taxonomy = Taxonomy(Path(args.taxonomy))
    
    # Get scene ID from JSON filename
    scene_id = Path(args.scene_json).stem
    
    # Load scene
    print(f"Loading scene: {scene_id}")
    trimesh_scene = load_front_scene(
        Path(args.scene_json),
        Path(args.future_root),
        taxonomy
    )
    
    # Create output directories
    output_dir = Path(args.output_dir)
    layouts_rgb_dir = output_dir / "layouts" / "rgb"
    layouts_seg_dir = output_dir / "layouts" / "seg"
    povs_rgb_dir = output_dir / "povs" / "rgb"
    povs_seg_dir = output_dir / "povs" / "seg"
    graphs_dir = output_dir / "graphs"
    
    for d in [layouts_rgb_dir, layouts_seg_dir, povs_rgb_dir, povs_seg_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Layout Pass: RGB
    print("Rendering layout RGB...")
    pyrender_scene_no_ceiling = trimesh_to_pyrender_scene(trimesh_scene, hide_ceilings=True)
    layout_rgb = render_layout_rgb(pyrender_scene_no_ceiling, width=256, height=256)
    Image.fromarray(layout_rgb).save(layouts_rgb_dir / f"{scene_id}.png")
    
    # Layout Pass: Segmentation
    print("Rendering layout segmentation...")
    layout_seg = render_layout_seg_improved(trimesh_scene, taxonomy, width=256, height=256)
    Image.fromarray(layout_seg).save(layouts_seg_dir / f"{scene_id}.png")
    
    # POV Pass: Sample camera positions
    print("Sampling camera positions...")
    floor_bboxes = find_floor_meshes(trimesh_scene, taxonomy)
    
    # Get furniture bboxes for collision checking
    furniture_bboxes = []
    for node_name in trimesh_scene.graph.nodes_geometry:
        geometry = trimesh_scene.geometry[node_name]
        metadata = getattr(geometry, 'metadata', {})
        if not metadata.get('is_ceiling', False) and metadata.get('label', '') not in ['floor', 'wall']:
            if isinstance(geometry, trimesh.Trimesh):
                transform = trimesh_scene.graph.get(node_name)[0]
                vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                vertices_world = (transform @ vertices_hom.T).T[:, :3]
                min_bounds = vertices_world.min(axis=0)
                max_bounds = vertices_world.max(axis=0)
                furniture_bboxes.append((min_bounds, max_bounds))
    
    # Restore ceiling for POV rendering
    pyrender_scene_full = trimesh_to_pyrender_scene(trimesh_scene, hide_ceilings=False)
    
    # Render POVs
    pov_count = 0
    for i in range(args.num_povs * 2):  # Try more than needed
        if pov_count >= args.num_povs:
            break
        
        camera_pos = sample_camera_position(floor_bboxes, furniture_bboxes, 
                                          eye_height=1.6, max_attempts=20)
        if camera_pos is None:
            print(f"Warning: Failed to sample camera position for POV {i+1}")
            continue
        
        # Render RGB
        pov_rgb = render_pov(pyrender_scene_full, camera_pos, width=256, height=256, fov=70.0)
        Image.fromarray(pov_rgb).save(povs_rgb_dir / f"{scene_id}_v{pov_count+1:02d}.png")
        
        # Render segmentation with colored materials
        pov_seg = render_pov_seg(trimesh_scene, camera_pos, taxonomy, width=256, height=256, fov=70.0)
        Image.fromarray(pov_seg).save(povs_seg_dir / f"{scene_id}_v{pov_count+1:02d}.png")
        
        pov_count += 1
        print(f"Rendered POV {pov_count}/{args.num_povs}")
    
    # Build graphs from segmentation layout
    print("Building graphs...")
    try:
        from data_preparation.pipeline_v2.graph_builder import build_room_graph_from_layout as build_graph
        layout_seg_path = layouts_seg_dir / f"{scene_id}.png"
        if layout_seg_path.exists():
            # Use "scene" as room name for scene-level graphs
            build_graph(
                scene_id, "scene", layout_seg_path, taxonomy, graphs_dir
            )
            print("Graphs built successfully")
    except Exception as e:
        print(f"Warning: Failed to build graphs: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Completed rendering for scene: {scene_id}")
    print(f"  Layouts: RGB and segmentation")
    print(f"  POVs: {pov_count} RGB and segmentation images")


if __name__ == "__main__":
    main()

