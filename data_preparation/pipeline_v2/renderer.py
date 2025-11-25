#!/usr/bin/env python3
"""
Rendering functions for layouts and POVs using Open3D.
Fixed camera setup and coordinate system issues.
"""

import os
import math
import time
import inspect
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from common.taxonomy import Taxonomy

# Set Open3D verbosity to errors only
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def trimesh_to_o3d_mesh(trimesh_mesh: trimesh.Trimesh, transform: np.ndarray = None, 
                       use_texture: bool = True, apply_transform: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Convert trimesh.Trimesh to Open3D TriangleMesh with transform applied.
    Handles textures and vertex colors properly.
    
    Args:
        trimesh_mesh: Input trimesh mesh
        transform: 4x4 transformation matrix (optional)
        use_texture: Whether to attempt texture loading
        apply_transform: Whether to apply transform to vertices (set False if vertices already transformed)
        
    Returns:
        Open3D TriangleMesh
    """
    # Apply transform to vertices if needed
    if transform is not None and apply_transform:
        if not np.allclose(transform, np.eye(4)):
            vertices_hom = np.column_stack([trimesh_mesh.vertices, np.ones(len(trimesh_mesh.vertices))])
            vertices_transformed = (transform @ vertices_hom.T).T[:, :3]
        else:
            vertices_transformed = trimesh_mesh.vertices
    else:
        vertices_transformed = trimesh_mesh.vertices
    
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_transformed.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    
    # Try to load texture
    texture_loaded = False
    if use_texture and hasattr(trimesh_mesh.visual, 'material'):
        material = trimesh_mesh.visual.material
        if hasattr(material, 'image') and material.image is not None:
            try:
                # Convert PIL image to numpy array
                if hasattr(material.image, 'size'):
                    img_array = np.asarray(material.image)
                    if len(img_array.shape) == 3:
                        # Convert RGB to BGR for Open3D
                        img_bgr = img_array[:, :, ::-1].copy()
                        o3d_mesh.textures = [o3d.geometry.Image(img_bgr.astype(np.uint8))]
                        
                        # Apply UV coordinates if available
                        if hasattr(trimesh_mesh.visual, 'uv') and trimesh_mesh.visual.uv is not None:
                            uv_coords = trimesh_mesh.visual.uv
                            if len(uv_coords) == len(trimesh_mesh.vertices):
                                o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coords.astype(np.float64))
                                texture_loaded = True
            except Exception as e:
                print(f"  Warning: Failed to load texture: {e}")
    
    # If no texture, try vertex colors
    if not texture_loaded:
        if hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
            vertex_colors = trimesh_mesh.visual.vertex_colors
            if len(vertex_colors) == len(trimesh_mesh.vertices):
                # Convert to float [0, 1]
                if vertex_colors.dtype == np.uint8:
                    colors_float = vertex_colors.astype(np.float64) / 255.0
                else:
                    colors_float = vertex_colors.astype(np.float64)
                # Ensure 3 channels
                if colors_float.shape[1] >= 3:
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_float[:, :3])
        elif hasattr(trimesh_mesh.visual, 'material'):
            material = trimesh_mesh.visual.material
            if hasattr(material, 'main_color') and material.main_color is not None:
                color = np.array(material.main_color[:3]) / 255.0 if len(material.main_color) >= 3 else np.array([0.5, 0.5, 0.5])
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(color, (len(trimesh_mesh.vertices), 1))
                )
    
    # Make mesh double-sided to prevent backface culling issues
    triangles_np = np.asarray(o3d_mesh.triangles)
    triangles_double = np.vstack([triangles_np, triangles_np[:, ::-1]])
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles_double)
    
    return o3d_mesh


def get_scene_bounds(trimesh_scene: trimesh.Scene, hide_ceilings: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box of scene, optionally excluding ceilings.
    
    Args:
        trimesh_scene: Input trimesh scene
        hide_ceilings: Whether to exclude ceiling meshes
        
    Returns:
        Tuple of (min_bounds, max_bounds) as 3D numpy arrays
    """
    all_vertices = []
    
    # Handle both scene graph (GLB) and single mesh (PLY) cases
    if len(trimesh_scene.graph.nodes_geometry) > 0:
        # Scene has graph structure (from GLB)
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
                if hide_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                    vertices_world = (transform @ vertices_hom.T).T[:, :3]
                    all_vertices.append(vertices_world)
            except (KeyError, ValueError, IndexError):
                continue
    else:
        # Scene has no graph (from PLY merge) - use geometry directly
        for geometry in trimesh_scene.geometry.values():
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            
            metadata = getattr(geometry, 'metadata', {})
            if hide_ceilings and metadata.get('is_ceiling', False):
                continue
            
            all_vertices.append(geometry.vertices)
    
    if not all_vertices:
        return np.array([0, 0, 0]), np.array([1, 1, 1])
    
    all_vertices = np.vstack(all_vertices)
    min_bounds = all_vertices.min(axis=0)
    max_bounds = all_vertices.max(axis=0)
    
    return min_bounds, max_bounds


def clip_mesh_above_height(mesh: trimesh.Trimesh, max_height: float, 
                           transform: Optional[np.ndarray] = None,
                           up_axis: int = 1,
                           floor_threshold: Optional[float] = None) -> Optional[trimesh.Trimesh]:
    """
    Clip mesh above a certain height along the up axis.
    Simple approach: keep faces where ALL vertices are below threshold.
    
    Args:
        mesh: Input mesh
        max_height: Maximum height (along up_axis) to keep
        transform: Optional transform to apply before clipping
        up_axis: Which axis is up (0=X, 1=Y, 2=Z)
        floor_threshold: If provided, meshes below this height are considered floor and not clipped
        
    Returns:
        Clipped mesh or None if completely above threshold
    """
    # Transform vertices to world space
    if transform is not None and not np.allclose(transform, np.eye(4)):
        vertices_hom = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
        vertices_world = (transform @ vertices_hom.T).T[:, :3]
    else:
        vertices_world = mesh.vertices
    
    # Check if this is a floor mesh (skip clipping)
    if floor_threshold is not None:
        min_height = vertices_world[:, up_axis].min()
        if min_height < floor_threshold:
            # This is likely a floor mesh, don't clip it
            return mesh
    
    # Check if any vertices are below threshold along the up axis
    below_mask = vertices_world[:, up_axis] <= max_height
    
    if not np.any(below_mask):
        # All vertices above threshold, return None
        return None
    
    if np.all(below_mask):
        # All vertices below threshold, return original mesh
        return mesh
    
    # Some vertices are above, need to clip
    # Simple approach: keep faces where ALL vertices are below threshold
    # This is more conservative but avoids boolean operations
    face_mask = np.all(below_mask[mesh.faces], axis=1)
    
    if not np.any(face_mask):
        # No faces completely below threshold, return None
        return None
    
    # Create new mesh with filtered faces
    filtered_faces = mesh.faces[face_mask]
    
    # Remap vertex indices
    used_vertices = np.unique(filtered_faces.flatten())
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    remapped_faces = np.array([[vertex_map[v] for v in face] for face in filtered_faces])
    
    clipped_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[used_vertices],
        faces=remapped_faces,
        process=False
    )
    
    # Preserve metadata and visual properties
    if hasattr(mesh, 'metadata'):
        clipped_mesh.metadata = mesh.metadata.copy()
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        if mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(mesh.vertices):
            clipped_mesh.visual.vertex_colors = mesh.visual.vertex_colors[used_vertices]
    
    return clipped_mesh


def render_layout_rgb(trimesh_scene: trimesh.Scene,
                      hide_ceilings: bool = True,
                      width: int = 256, height: int = 256,
                      clip_top_meters: float = 0.5,
                      scene_metadata: Optional[Dict] = None) -> np.ndarray:
    """
    Render top-down layout RGB image using orthographic projection.
    
    Args:
        trimesh_scene: Input trimesh scene
        hide_ceilings: Whether to hide ceiling meshes
        width: Image width
        height: Image height
        clip_top_meters: How many meters to clip from the top
        scene_metadata: Optional scene metadata (for up_axis detection)
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    # Get up axis from metadata if available
    up_axis = 1  # Default Y-up
    if scene_metadata and 'up_axis' in scene_metadata:
        up_axis = scene_metadata['up_axis']
    
    # Get scene bounds
    min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings=hide_ceilings)
    center = (min_bounds + max_bounds) / 2
    
    # Calculate clipping height
    clip_height = max_bounds[up_axis] - clip_top_meters
    
    # Calculate room height for floor threshold
    room_height = max_bounds[up_axis] - min_bounds[up_axis]
    floor_threshold = min_bounds[up_axis] + (room_height * 0.1)  # Bottom 10% is floor
    
    # Calculate target view size (with 3% margin)
    base_max_size = max(max_bounds - min_bounds)
    max_size = base_max_size * 1.03
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes to visualizer
    mesh_count = 0
    
    # Handle both scene graph (GLB) and single mesh (PLY) cases
    if len(trimesh_scene.graph.nodes_geometry) > 0:
        # Scene has graph structure (from GLB)
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
                if hide_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    # Clip mesh if needed
                    clipped_mesh = clip_mesh_above_height(
                        geometry, clip_height, transform, up_axis, floor_threshold
                    )
                    if clipped_mesh is None:
                        continue
                    
                    # Convert to Open3D mesh (don't apply transform, already in world space from GLB)
                    o3d_mesh = trimesh_to_o3d_mesh(clipped_mesh, transform=None, apply_transform=False)
                    vis.add_geometry(o3d_mesh)
                    mesh_count += 1
            except (KeyError, ValueError, IndexError) as e:
                continue
    else:
        # Scene has no graph (from PLY merge) - use geometry directly
        for geometry in trimesh_scene.geometry.values():
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            
            metadata = getattr(geometry, 'metadata', {})
            if hide_ceilings and metadata.get('is_ceiling', False):
                continue
            
            # Clip mesh if needed (no transform needed, already merged)
            clipped_mesh = clip_mesh_above_height(
                geometry, clip_height, None, up_axis, floor_threshold
            )
            if clipped_mesh is None:
                continue
            
            # Convert to Open3D mesh
            o3d_mesh = trimesh_to_o3d_mesh(clipped_mesh, transform=None, apply_transform=False)
            vis.add_geometry(o3d_mesh)
            mesh_count += 1
    
    if mesh_count == 0:
        # Return white image if no meshes
        vis.destroy_window()
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Set up camera for orthographic top-down view
    # Calculate up vector from metadata
    up_vector = np.zeros(3)
    up_vector[up_axis] = 1.0
    
    # Calculate camera distance to fit max_size in view
    # For orthographic: distance = (target_size / 2) * (focal_length / (image_size / 2))
    # We want a 1-degree FOV for near-orthographic effect
    fov_degrees = 1.0
    fov_rad = math.radians(fov_degrees)
    image_size = min(width, height)
    focal_length = (image_size / 2.0) / math.tan(fov_rad / 2.0)
    
    # Calculate camera distance to fit max_size in view
    camera_distance = (max_size / 2.0) * focal_length / (image_size / 2.0)
    
    # Position camera along the up direction, looking down
    camera_height_pos = max_bounds[up_axis] + camera_distance
    eye = center.copy().astype(np.float64)
    eye[up_axis] = camera_height_pos
    center_point = center.astype(np.float64)
    
    # Find a horizontal axis to use as the camera's "up" in the image
    scene_size_arr = np.array([max_bounds[0] - min_bounds[0], 
                               max_bounds[1] - min_bounds[1], 
                               max_bounds[2] - min_bounds[2]])
    scene_size_arr[up_axis] = 0
    horizontal_axis = int(np.argmax(scene_size_arr))
    camera_up = np.zeros(3, dtype=np.float64)
    camera_up[horizontal_axis] = 1.0
    
    # Set up camera parameters
    cx, cy = width / 2.0, height / 2.0
    fx = fy = focal_length
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        # Match old pipeline's look_at function exactly
        f = center_ - eye_
        f = f / np.linalg.norm(f)
        upn = up_ / np.linalg.norm(up_)
        s = np.cross(f, upn)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        R = np.array([[s[0], u[0], -f[0]],
                     [s[1], u[1], -f[1]],
                     [s[2], u[2], -f[2]]])
        t = eye_
        return np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]])
    
    pin.extrinsic = look_at(eye, center_point, camera_up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin)
    
    # Set background color (dark gray for RGB)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.2, 0.2, 0.25])
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    
    # Capture image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    img = Image.open(tmp_path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_np = np.asarray(img)
    
    os.unlink(tmp_path)
    
    return img_np


def render_layout_seg(trimesh_scene: trimesh.Scene,
                      taxonomy: Taxonomy,
                      hide_ceilings: bool = True,
                      width: int = 256, height: int = 256,
                      clip_top_meters: float = 0.5,
                      scene_metadata: Optional[Dict] = None) -> np.ndarray:
    """
    Render top-down layout segmentation image using orthographic projection.
    
    Args:
        trimesh_scene: Input trimesh scene
        taxonomy: Taxonomy object for coloring
        hide_ceilings: Whether to hide ceiling meshes
        width: Image width
        height: Image height
        clip_top_meters: How many meters to clip from the top
        scene_metadata: Optional scene metadata (for up_axis detection)
        
    Returns:
        Segmentation image as numpy array (H, W, 3)
    """
    # Get up axis from metadata if available
    up_axis = 1  # Default Y-up
    if scene_metadata and 'up_axis' in scene_metadata:
        up_axis = scene_metadata['up_axis']
    
    # Get scene bounds
    min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings=hide_ceilings)
    center = (min_bounds + max_bounds) / 2
    
    # Calculate clipping height
    clip_height = max_bounds[up_axis] - clip_top_meters
    
    # Calculate room height for floor threshold
    room_height = max_bounds[up_axis] - min_bounds[up_axis]
    floor_threshold = min_bounds[up_axis] + (room_height * 0.1)  # Bottom 10% is floor
    
    # Calculate target view size (with 3% margin)
    base_max_size = max(max_bounds - min_bounds)
    max_size = base_max_size * 1.03
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes with taxonomy colors
    mesh_count = 0
    
    # Handle both scene graph (GLB) and single mesh (PLY) cases
    if len(trimesh_scene.graph.nodes_geometry) > 0:
        # Scene has graph structure (from GLB)
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
                if hide_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    # Clip mesh if needed
                    clipped_mesh = clip_mesh_above_height(
                        geometry, clip_height, transform, up_axis, floor_threshold
                    )
                    if clipped_mesh is None:
                        continue
                    
                    category_id = metadata.get('category_id', 0)
                    color_rgb = taxonomy.get_color(category_id, mode="category")
                    if color_rgb is None:
                        color_rgb = (127, 127, 127)
                    
                # Convert to Open3D mesh (don't apply transform, already in world space from GLB)
                o3d_mesh = trimesh_to_o3d_mesh(clipped_mesh, transform=None, apply_transform=False)
                num_vertices = len(o3d_mesh.vertices)
                seg_color = np.array([c/255.0 for c in color_rgb], dtype=np.float64)
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(seg_color, (num_vertices, 1))
                )
                vis.add_geometry(o3d_mesh)
                mesh_count += 1
            except (KeyError, ValueError, IndexError):
                continue
    else:
        # Scene has no graph (from PLY merge) - use geometry directly
        for geometry in trimesh_scene.geometry.values():
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            
            metadata = getattr(geometry, 'metadata', {})
            if hide_ceilings and metadata.get('is_ceiling', False):
                continue
            
            # Clip mesh if needed (no transform needed, already merged)
            clipped_mesh = clip_mesh_above_height(
                geometry, clip_height, None, up_axis, floor_threshold
            )
            if clipped_mesh is None:
                continue
            
            category_id = metadata.get('category_id', 0)
            color_rgb = taxonomy.get_color(category_id, mode="category")
            if color_rgb is None:
                color_rgb = (127, 127, 127)
            
            # Convert to Open3D mesh
            o3d_mesh = trimesh_to_o3d_mesh(clipped_mesh, transform=None, apply_transform=False)
            num_vertices = len(o3d_mesh.vertices)
            seg_color = np.array([c/255.0 for c in color_rgb], dtype=np.float64)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(seg_color, (num_vertices, 1))
            )
            vis.add_geometry(o3d_mesh)
            mesh_count += 1
    
    if mesh_count == 0:
        # Return white image if no meshes
        vis.destroy_window()
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Set up camera for orthographic top-down view (same as RGB)
    up_vector = np.zeros(3)
    up_vector[up_axis] = 1.0
    
    fov_degrees = 1.0
    fov_rad = math.radians(fov_degrees)
    image_size = min(width, height)
    focal_length = (image_size / 2.0) / math.tan(fov_rad / 2.0)
    
    camera_distance = (max_size / 2.0) * focal_length / (image_size / 2.0)
    
    camera_height_pos = max_bounds[up_axis] + camera_distance
    eye = center.copy().astype(np.float64)
    eye[up_axis] = camera_height_pos
    center_point = center.astype(np.float64)
    
    scene_size_arr = np.array([max_bounds[0] - min_bounds[0], 
                               max_bounds[1] - min_bounds[1], 
                               max_bounds[2] - min_bounds[2]])
    scene_size_arr[up_axis] = 0
    horizontal_axis = int(np.argmax(scene_size_arr))
    camera_up = np.zeros(3, dtype=np.float64)
    camera_up[horizontal_axis] = 1.0
    
    cx, cy = width / 2.0, height / 2.0
    fx = fy = focal_length
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / np.linalg.norm(f)
        upn = up_ / np.linalg.norm(up_)
        s = np.cross(f, upn)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        R = np.array([[s[0], u[0], -f[0]],
                     [s[1], u[1], -f[1]],
                     [s[2], u[2], -f[2]]])
        t = eye_
        return np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]])
    
    pin.extrinsic = look_at(eye, center_point, camera_up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin)
    
    # Set background color (white for segmentation)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    
    # Capture image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    img = Image.open(tmp_path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_np = np.asarray(img)
    
    os.unlink(tmp_path)
    
    return img_np


def render_pov(trimesh_scene: trimesh.Scene,
                camera_pos: np.ndarray,
                hide_ceilings: bool = False,
                camera_target: Optional[np.ndarray] = None,
                width: int = 256,
                height: int = 256,
                fov: float = 70.0) -> np.ndarray:
    """
    Render POV RGB image from a camera position.
    
    Args:
        trimesh_scene: Input trimesh scene
        camera_pos: Camera position (x, y, z)
        hide_ceilings: Whether to hide ceiling meshes
        camera_target: Target point to look at (defaults to scene center)
        width: Image width
        height: Image height
        fov: Field of view in degrees
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    if camera_target is None:
        min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings=False)
        camera_target = (min_bounds + max_bounds) / 2
    else:
        camera_target = np.asarray(camera_target)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes with textures
    # Handle both scene graph (GLB) and single mesh (PLY) cases
    if len(trimesh_scene.graph.nodes_geometry) > 0:
        # Scene has graph structure (from GLB)
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
                if hide_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    o3d_mesh = trimesh_to_o3d_mesh(geometry, transform)
                    vis.add_geometry(o3d_mesh)
            except (KeyError, ValueError, IndexError):
                continue
    else:
        # Scene has no graph (from PLY merge) - use geometry directly
        for geometry in trimesh_scene.geometry.values():
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            
            metadata = getattr(geometry, 'metadata', {})
            if hide_ceilings and metadata.get('is_ceiling', False):
                continue
            
            o3d_mesh = trimesh_to_o3d_mesh(geometry, transform=None)
            vis.add_geometry(o3d_mesh)
    
    # Set up camera
    eye = np.asarray(camera_pos, dtype=np.float64)
    center = np.asarray(camera_target, dtype=np.float64)
    up = np.array([0, 1, 0], dtype=np.float64)  # Y-up
    
    fov_rad = math.radians(fov)
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / np.linalg.norm(f)
        upn = up_ / np.linalg.norm(up_)
        s = np.cross(f, upn)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        R = np.array([[s[0], u[0], -f[0]],
                     [s[1], u[1], -f[1]],
                     [s[2], u[2], -f[2]]])
        t = eye_
        return np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]])
    
    pin.extrinsic = look_at(eye, center, up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin)
    
    # Set background color
    opt = vis.get_render_option()
    opt.background_color = np.array([0.2, 0.2, 0.25])
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    
    # Capture image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    img = Image.open(tmp_path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_np = np.asarray(img)
    
    os.unlink(tmp_path)
    
    return img_np


def render_pov_seg(trimesh_scene: trimesh.Scene,
                  camera_pos: np.ndarray,
                  taxonomy: Taxonomy,
                  hide_ceilings: bool = False,
                  camera_target: Optional[np.ndarray] = None,
                  width: int = 256,
                  height: int = 256,
                  fov: float = 70.0) -> np.ndarray:
    """
    Render POV segmentation with taxonomy colors.
    """
    if camera_target is None:
        min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings=False)
        camera_target = (min_bounds + max_bounds) / 2
    else:
        camera_target = np.asarray(camera_target)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes with taxonomy colors
    # Handle both scene graph (GLB) and single mesh (PLY) cases
    if len(trimesh_scene.graph.nodes_geometry) > 0:
        # Scene has graph structure (from GLB)
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
                if hide_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    category_id = metadata.get('category_id', 0)
                    color_rgb = taxonomy.get_color(category_id, mode="category")
                    if color_rgb is None:
                        color_rgb = (127, 127, 127)
                    
                    o3d_mesh = trimesh_to_o3d_mesh(geometry, transform)
                    num_vertices = len(o3d_mesh.vertices)
                    seg_color = np.array([c/255.0 for c in color_rgb], dtype=np.float64)
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                        np.tile(seg_color, (num_vertices, 1))
                    )
                    vis.add_geometry(o3d_mesh)
            except (KeyError, ValueError, IndexError):
                continue
    else:
        # Scene has no graph (from PLY merge) - use geometry directly
        for geometry in trimesh_scene.geometry.values():
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            
            metadata = getattr(geometry, 'metadata', {})
            if hide_ceilings and metadata.get('is_ceiling', False):
                continue
            
            category_id = metadata.get('category_id', 0)
            color_rgb = taxonomy.get_color(category_id, mode="category")
            if color_rgb is None:
                color_rgb = (127, 127, 127)
            
            o3d_mesh = trimesh_to_o3d_mesh(geometry, transform=None)
            num_vertices = len(o3d_mesh.vertices)
            seg_color = np.array([c/255.0 for c in color_rgb], dtype=np.float64)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(seg_color, (num_vertices, 1))
            )
            vis.add_geometry(o3d_mesh)

    # Set up camera
    eye = np.asarray(camera_pos, dtype=np.float64)
    center = np.asarray(camera_target, dtype=np.float64)
    up = np.array([0, 1, 0], dtype=np.float64)  # Y-up
    
    fov_rad = math.radians(fov)
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / np.linalg.norm(f)
        upn = up_ / np.linalg.norm(up_)
        s = np.cross(f, upn)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        R = np.array([[s[0], u[0], -f[0]],
                     [s[1], u[1], -f[1]],
                     [s[2], u[2], -f[2]]])
        t = eye_
        return np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]])
    
    pin.extrinsic = look_at(eye, center, up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin)
    
    # Set background color
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    
    # Capture image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    img = Image.open(tmp_path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_np = np.asarray(img)
    
    os.unlink(tmp_path)
    
    return img_np


def sample_camera_positions_from_corners(trimesh_scene: trimesh.Scene,
                                         room_min: np.ndarray,
                                         room_max: np.ndarray,
                                         room_center: np.ndarray,
                                         num_attempts: int = 20,
                                         num_povs: int = 6,
                                         up_axis: int = 1,
                                         eye_height: float = 1.6,
                                         margin: float = 0.5) -> list:
    """
    Sample camera positions from room corners looking toward the center.
    Places cameras at corners of the room bounding box, looking inward.
    
    Args:
        trimesh_scene: trimesh scene for validation
        room_min: Minimum bounds of the room (3D array)
        room_max: Maximum bounds of the room (3D array)
        room_center: Center of the room (3D array) - camera target
        num_attempts: Maximum number of attempts to find valid positions
        num_povs: Number of camera positions to generate
        up_axis: Which axis is up (0=X, 1=Y, 2=Z)
        eye_height: Camera eye height offset from floor (in meters)
        margin: Margin from walls (in meters)
        
    Returns:
        List of camera positions (x, y, z) as numpy arrays
    """
    room_min = np.asarray(room_min)
    room_max = np.asarray(room_max)
    room_center = np.asarray(room_center)
    
    # Calculate room dimensions
    room_size = room_max - room_min
    
    # Determine horizontal axes (not the up axis)
    horizontal_axes = [i for i in range(3) if i != up_axis]
    h1, h2 = horizontal_axes[0], horizontal_axes[1]
    
    # Calculate floor height (minimum along up axis)
    floor_height = room_min[up_axis]
    camera_height = floor_height + eye_height
    
    # Generate corner positions in horizontal plane
    corners = []
    if num_povs >= 4:
        # Four main corners
        corners.append([room_min[h1] + margin, camera_height, room_min[h2] + margin])
        corners.append([room_max[h1] - margin, camera_height, room_min[h2] + margin])
        corners.append([room_min[h1] + margin, camera_height, room_max[h2] - margin])
        corners.append([room_max[h1] - margin, camera_height, room_max[h2] - margin])
    
    # Add midpoints if we need more POVs
    if num_povs > 4:
        mid_h1 = (room_min[h1] + room_max[h1]) / 2
        mid_h2 = (room_min[h2] + room_max[h2]) / 2
        
        corners.append([room_min[h1] + margin, camera_height, mid_h2])  # Left wall center
        corners.append([room_max[h1] - margin, camera_height, mid_h2])  # Right wall center
        corners.append([mid_h1, camera_height, room_min[h2] + margin])  # Bottom wall center
        corners.append([mid_h1, camera_height, room_max[h2] - margin])  # Top wall center
    
    # Convert to 3D positions with correct up axis
    valid_positions = []
    for corner_2d in corners[:num_povs]:
        pos = np.zeros(3)
        pos[h1] = corner_2d[0]
        pos[up_axis] = corner_2d[1]  # Camera height
        pos[h2] = corner_2d[2]
        
        # Validate position is not inside furniture (simple bbox check)
        # For now, just add all corners - can add ray-casting later if needed
        valid_positions.append(pos)
    
    return valid_positions[:num_povs]
