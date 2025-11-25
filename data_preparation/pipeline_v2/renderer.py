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
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from common.taxonomy import Taxonomy

# Set Open3D verbosity to errors only
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def trimesh_to_o3d_mesh(trimesh_mesh: trimesh.Trimesh, transform: np.ndarray, 
                       use_texture: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Convert trimesh.Trimesh to Open3D TriangleMesh with transform applied.
    Handles textures and vertex colors properly.
    
    Args:
        trimesh_mesh: Input trimesh mesh
        transform: 4x4 transformation matrix
        use_texture: If True, try to use textures/materials
        
    Returns:
        Open3D TriangleMesh
    """
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces.astype(np.int32))
    
    # Apply transform to vertices
    if transform is not None and not np.allclose(transform, np.eye(4)):
        vertices_hom = np.column_stack([trimesh_mesh.vertices, np.ones(len(trimesh_mesh.vertices))])
        vertices_world = (transform @ vertices_hom.T).T[:, :3]
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_world.astype(np.float64))
    
    # Handle textures/materials if available
    has_texture = False
    if use_texture and hasattr(trimesh_mesh.visual, 'material'):
        material = trimesh_mesh.visual.material
        if hasattr(material, 'image') and material.image is not None:
            # Try to use texture
            try:
                # Convert PIL image to numpy
                img_array = np.asarray(material.image.convert('RGB'))
                # Get UV coordinates
                if hasattr(trimesh_mesh.visual, 'uv') and trimesh_mesh.visual.uv is not None:
                    uv = trimesh_mesh.visual.uv
                    if len(uv) == len(trimesh_mesh.vertices):
                        # Create texture
                        o3d_texture = o3d.geometry.Image(img_array)
                        o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(uv[trimesh_mesh.faces].reshape(-1, 2))
                        o3d_mesh.triangle_material_ids = o3d.utility.Vector1iVector(np.zeros(len(trimesh_mesh.faces), dtype=np.int32))
                        o3d_mesh.textures = [o3d_texture]
                        has_texture = True
            except Exception:
                pass
    
    # If no texture, use vertex colors
    if not has_texture and hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
        vcolors = trimesh_mesh.visual.vertex_colors
        if len(vcolors.shape) == 2 and vcolors.shape[1] >= 3:
            # Convert to 0-1 range if needed
            if vcolors.max() > 1.0:
                vcolors = vcolors.astype(np.float32) / 255.0
            else:
                vcolors = vcolors.astype(np.float32)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vcolors[:, :3].astype(np.float64))
    
    # Compute normals (important for lighting)
    o3d_mesh.compute_vertex_normals()
    
    return o3d_mesh


def get_scene_bounds(trimesh_scene: trimesh.Scene, hide_ceilings: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Get scene bounding box."""
    all_vertices = []
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
    
    if not all_vertices:
        min_bounds = np.array([-5, -5, 0])
        max_bounds = np.array([5, 5, 3])
    else:
        all_vertices = np.vstack(all_vertices)
        min_bounds = all_vertices.min(axis=0)
        max_bounds = all_vertices.max(axis=0)
    
    return min_bounds, max_bounds


def render_layout_rgb(trimesh_scene: trimesh.Scene,
                      hide_ceilings: bool = True,
                      width: int = 256, height: int = 256) -> np.ndarray:
    """
    Render top-down RGB layout using Open3D.
    Fixed coordinate system: Y-up (Open3D standard).
    
    Args:
        trimesh_scene: trimesh scene
        hide_ceilings: If True, exclude ceiling meshes
        width: Image width
        height: Image height
        
    Returns:
        RGB image array (H, W, 3) uint8
    """
    min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings)
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    max_size = max(size[0], size[2])  # X and Z dimensions for top-down
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes
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
    
    # Set render options with lighting
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)
    opt.light_on = True  # Enable lighting for textures
    
    # Camera setup: top-down orthographic view
    # 3D-FRONT uses Z-up, Open3D uses Y-up
    # For top-down: camera at high Y, looking down at -Y
    camera_height = max_size * 1.5
    eye = np.array([center[0], center[1] + camera_height, center[2]], dtype=np.float64)
    center_point = center.astype(np.float64)
    up = np.array([0, 0, 1], dtype=np.float64)  # Z-up for 3D-FRONT coordinate system
    
    # Orthographic-like camera: large focal length for top-down
    # Scale to fit scene in view
    zoom = max_size * 1.2
    fx = fy = (width / 2.0) / zoom
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        s = np.cross(f, upn)
        s = s / (np.linalg.norm(s) + 1e-12)
        u = np.cross(s, f)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    pin.extrinsic = look_at(eye, center_point, up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)
    
    # Capture image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    # Read and resize if needed
    img = Image.open(tmp_path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_np = np.asarray(img)
    
    # Clean up
    os.unlink(tmp_path)
    
    # Flip vertically for POV (Open3D captures upside down)
    if 'pov' in str(inspect.stack()[1].function).lower():
        img_np = np.flipud(img_np)
    
    return img_np


def render_layout_seg(trimesh_scene: trimesh.Scene,
                      taxonomy: Taxonomy,
                      hide_ceilings: bool = True,
                      width: int = 256, height: int = 256) -> np.ndarray:
    """
    Render top-down segmentation layout with taxonomy colors.
    """
    min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings)
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    max_size = max(size[0], size[2])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes with taxonomy colors
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
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)
    opt.light_on = True
    
    # Camera setup (same as RGB)
    camera_height = max_size * 1.5
    eye = np.array([center[0], center[1] + camera_height, center[2]], dtype=np.float64)
    center_point = center.astype(np.float64)
    up = np.array([0, 0, 1], dtype=np.float64)  # Z-up for 3D-FRONT
    
    zoom = max_size * 1.2
    fx = fy = (width / 2.0) / zoom
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        s = np.cross(f, upn)
        s = s / (np.linalg.norm(s) + 1e-12)
        u = np.cross(s, f)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    pin.extrinsic = look_at(eye, center_point, up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)
    
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
    Render perspective POV from camera position.
    Fixed coordinate system: Y-up.
    """
    if camera_target is None:
        min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings=False)
        camera_target = (min_bounds + max_bounds) / 2
    else:
        camera_target = np.asarray(camera_target)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes
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
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)
    opt.light_on = True  # Enable lighting for textures
    
    # Camera setup: perspective
    # 3D-FRONT uses Z-up, so camera_pos is (x, y, z) where y is height
    fx = (0.5 * width) / math.tan(math.radians(fov) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        s = np.cross(f, upn)
        s = s / (np.linalg.norm(s) + 1e-12)
        u = np.cross(s, f)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    # 3D-FRONT coordinate system: X-right, Y-up (height), Z-forward
    # Open3D uses Y-up, so we need to convert
    # camera_pos is (x, y, z) in 3D-FRONT coords -> (x, y, z) in Open3D
    pin.extrinsic = look_at(
        camera_pos.astype(np.float64),
        camera_target.astype(np.float64),
        np.array([0, 1, 0], dtype=np.float64)  # Y-up
    )
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)
    
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
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)
    
    fx = (0.5 * width) / math.tan(math.radians(fov) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        s = np.cross(f, upn)
        s = s / (np.linalg.norm(s) + 1e-12)
        u = np.cross(s, f)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    # 3D-FRONT coordinate system: X-right, Y-up (height), Z-forward
    pin.extrinsic = look_at(
        camera_pos.astype(np.float64),
        camera_target.astype(np.float64),
        np.array([0, 1, 0], dtype=np.float64)  # Y-up
    )
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)
    
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

