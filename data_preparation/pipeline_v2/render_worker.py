#!/usr/bin/env python3
"""
Render worker for 3D-FRONT scenes.
Renders top-down layouts and perspective POVs using Open3D (like old pipeline).
"""

import os
import time
import math

# Try to import xvfbwrapper (like old pipeline)
try:
    from xvfbwrapper import Xvfb
    XVFBWRAPPER_AVAILABLE = True
except ImportError:
    Xvfb = None
    XVFBWRAPPER_AVAILABLE = False

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

# Global variable to hold Xvfb instance
VFB = None

import sys

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Go up from pipeline_v2 -> data_preparation -> ImgiNav
sys.path.insert(0, str(project_root))

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.scene_loader import load_front_scene, extract_rooms_from_scene
from data_preparation.pipeline_v2.renderer import (
    render_layout_rgb,
    render_layout_seg,
    render_pov,
    render_pov_seg,
    sample_camera_positions_from_corners,
    get_scene_bounds,
    find_floor_meshes
)

# Set Open3D verbosity to errors only
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def trimesh_to_o3d_mesh(trimesh_mesh: trimesh.Trimesh, transform: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Convert trimesh.Trimesh to Open3D TriangleMesh with transform applied.
    
    Args:
        trimesh_mesh: Input trimesh mesh
        transform: 4x4 transformation matrix
        
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
    
    # Copy vertex colors if available
    if hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
        vcolors = trimesh_mesh.visual.vertex_colors
        if len(vcolors.shape) == 2 and vcolors.shape[1] >= 3:
            # Convert to 0-1 range if needed
            if vcolors.max() > 1.0:
                vcolors = vcolors.astype(np.float32) / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vcolors[:, :3].astype(np.float64))
    
    # Compute normals
    o3d_mesh.compute_vertex_normals()
    
    return o3d_mesh


def render_layout_rgb(trimesh_scene: trimesh.Scene,
                      hide_ceilings: bool = True,
                      width: int = 256, height: int = 256) -> np.ndarray:
    """
    Render top-down RGB layout using Open3D.
    
    Args:
        trimesh_scene: trimesh scene
        hide_ceilings: If True, exclude ceiling meshes
        width: Image width
        height: Image height
        
    Returns:
        RGB image array (H, W, 3) uint8
    """
    # Calculate scene bounds
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
    
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    max_size = max(size[0], size[1])
    
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
                # Create mesh without transform (we'll apply it to vertices)
                o3d_mesh = trimesh_to_o3d_mesh(geometry, transform)
                vis.add_geometry(o3d_mesh)
        except (KeyError, ValueError, IndexError):
            continue
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)  # Black background
    
    # Camera setup: orthographic top-down
    # Calculate camera parameters for orthographic view
    camera_height = center[2] + max_size * 1.5
    
    # Create look-at matrix: camera at (center_x, camera_height, center_z) looking at center
    eye = np.array([center[0], camera_height, center[2]], dtype=np.float64)
    center_point = center.astype(np.float64)
    up = np.array([0, 0, -1], dtype=np.float64)  # Negative Z is up for top-down
    
    # Set up camera using pinhole camera parameters
    fx = fy = width / (2 * max_size)  # Orthographic-like scaling
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    pin.extrinsic = look_at(eye, center_point, up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.12)  # Give OpenGL time
    
    # Capture image to temporary file (Open3D requires filename)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    # Read image back as numpy array
    img = Image.open(tmp_path)
    img_np = np.asarray(img)
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    return img_np


def render_layout_seg(trimesh_scene: trimesh.Scene,
                      taxonomy: Taxonomy,
                      hide_ceilings: bool = True,
                      width: int = 256, height: int = 256) -> np.ndarray:
    """
    Render top-down segmentation layout with taxonomy colors using Open3D.
    
    Args:
        trimesh_scene: trimesh scene
        taxonomy: Taxonomy object
        hide_ceilings: If True, exclude ceiling meshes
        width: Image width
        height: Image height
        
    Returns:
        Segmentation image array (H, W, 3) uint8
    """
    # Calculate scene bounds (same as RGB)
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
    
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    max_size = max(size[0], size[1])
    
    # Create Open3D visualizer
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
                # Get category color
                category_id = metadata.get('category_id', 0)
                color_rgb = taxonomy.get_color(category_id, mode="category")
                if color_rgb is None:
                    color_rgb = (127, 127, 127)  # Default gray
                
                # Create mesh
                o3d_mesh = trimesh_to_o3d_mesh(geometry, transform)
                
                # Set uniform vertex colors based on taxonomy
                num_vertices = len(o3d_mesh.vertices)
                seg_color = np.array([c/255.0 for c in color_rgb], dtype=np.float64)
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(seg_color, (num_vertices, 1))
                )
                
                vis.add_geometry(o3d_mesh)
        except (KeyError, ValueError, IndexError):
            continue
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)  # Black background
    
    # Camera setup (same as RGB)
    camera_height = center[2] + max_size * 1.5
    eye = np.array([center[0], camera_height, center[2]], dtype=np.float64)
    center_point = center.astype(np.float64)
    up = np.array([0, 0, -1], dtype=np.float64)
    
    fx = fy = width / (2 * max_size)
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    pin.extrinsic = look_at(eye, center_point, up)
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.12)
    
    # Capture image to temporary file (Open3D requires filename)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    # Read image back as numpy array
    img = Image.open(tmp_path)
    img_np = np.asarray(img)
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    return img_np


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
        try:
            transform, geometry_name = trimesh_scene.graph.get(node_name)
            if geometry_name not in trimesh_scene.geometry:
                if node_name not in trimesh_scene.geometry:
                    continue
                geometry = trimesh_scene.geometry[node_name]
            else:
                geometry = trimesh_scene.geometry[geometry_name]
            
            metadata = getattr(geometry, 'metadata', {})
            category_id = metadata.get('category_id', 0)
            label = metadata.get('label', '').lower()
            
            if category_id in floor_ids or label == 'floor':
                if isinstance(geometry, trimesh.Trimesh):
                    vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                    vertices_world = (transform @ vertices_hom.T).T[:, :3]
                    
                    min_bounds = vertices_world.min(axis=0)
                    max_bounds = vertices_world.max(axis=0)
                    floor_bboxes.append((min_bounds, max_bounds))
        except (KeyError, ValueError, IndexError):
            continue
    
    return floor_bboxes


def sample_camera_positions_from_corners(trimesh_scene: trimesh.Scene,
                                         floor_bboxes: List[Tuple[np.ndarray, np.ndarray]],
                                         furniture_bboxes: List[Tuple[np.ndarray, np.ndarray]],
                                         taxonomy: Taxonomy,
                                         num_povs: int = 6,
                                         eye_height: float = 1.6,
                                         margin: float = 0.5) -> List[np.ndarray]:
    """
    Sample camera positions from room corners looking toward the center.
    Places cameras at corners of the floor bounding box, looking inward.
    
    Args:
        trimesh_scene: trimesh scene for ray-casting
        floor_bboxes: List of floor bounding boxes
        furniture_bboxes: List of furniture bounding boxes
        taxonomy: Taxonomy object for identifying floor meshes
        num_povs: Number of camera positions to generate
        eye_height: Camera eye height in meters
        margin: Margin from walls (in meters)
        
    Returns:
        List of camera positions (x, y, z)
    """
    if not floor_bboxes:
        return []
    
    # Combine all floor bounds to get room bounds
    all_min = np.array([bbox[0] for bbox in floor_bboxes]).min(axis=0)
    all_max = np.array([bbox[1] for bbox in floor_bboxes]).max(axis=0)
    
    # Calculate room center (for looking direction)
    room_center = (all_min + all_max) / 2
    room_center[1] = eye_height  # Set height to eye level
    
    # Calculate room dimensions
    room_size_x = all_max[0] - all_min[0]
    room_size_z = all_max[2] - all_min[2]
    
    # Generate corner positions
    corners = []
    if num_povs >= 4:
        # Four main corners
        corners.append([all_min[0] + margin, eye_height, all_min[2] + margin])  # Bottom-left
        corners.append([all_max[0] - margin, eye_height, all_min[2] + margin])  # Bottom-right
        corners.append([all_min[0] + margin, eye_height, all_max[2] - margin])  # Top-left
        corners.append([all_max[0] - margin, eye_height, all_max[2] - margin])  # Top-right
    
    # Add midpoints if we need more POVs
    if num_povs > 4:
        # Midpoints of each wall
        mid_x = (all_min[0] + all_max[0]) / 2
        mid_z = (all_min[2] + all_max[2]) / 2
        
        corners.append([all_min[0] + margin, eye_height, mid_z])  # Left wall center
        corners.append([all_max[0] - margin, eye_height, mid_z])  # Right wall center
        corners.append([mid_x, eye_height, all_min[2] + margin])  # Bottom wall center
        corners.append([mid_x, eye_height, all_max[2] - margin])  # Top wall center
    
    # Filter out positions that are inside furniture
    valid_corners = []
    for corner in corners[:num_povs]:
        x, y, z = corner
        inside_furniture = False
        for f_min, f_max in furniture_bboxes:
            if (f_min[0] <= x <= f_max[0] and
                f_min[1] <= y <= f_max[1] and
                f_min[2] <= z <= f_max[2]):
                inside_furniture = True
                break
        
        if not inside_furniture:
            valid_corners.append(np.array(corner))
    
    return valid_corners


def sample_camera_position(trimesh_scene: trimesh.Scene,
                           floor_bboxes: List[Tuple[np.ndarray, np.ndarray]],
                           furniture_bboxes: List[Tuple[np.ndarray, np.ndarray]],
                           taxonomy: Taxonomy,
                           eye_height: float = 1.6,
                           max_attempts: int = 50) -> Optional[np.ndarray]:
    """
    Sample a valid camera position on the floor, not inside furniture.
    DEPRECATED: Use sample_camera_positions_from_corners instead.
    """
    # Fallback to corner-based sampling
    corners = sample_camera_positions_from_corners(
        trimesh_scene, floor_bboxes, furniture_bboxes, taxonomy,
        num_povs=1, eye_height=eye_height
    )
    if corners:
        return corners[0]
    return None


def render_pov(trimesh_scene: trimesh.Scene,
                camera_pos: np.ndarray,
                hide_ceilings: bool = False,
                camera_target: Optional[np.ndarray] = None,
                width: int = 256,
                height: int = 256,
                fov: float = 70.0) -> np.ndarray:
    """
    Render perspective POV from camera position using Open3D.
    
    Args:
        trimesh_scene: trimesh scene
        camera_pos: Camera position (x, y, z)
        hide_ceilings: If True, exclude ceiling meshes
        camera_target: Look-at target (default: scene center)
        width: Image width
        height: Image height
        fov: Field of view in degrees
        
    Returns:
        RGB image array
    """
    # Calculate scene center if not provided
    if camera_target is None:
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
                
                if isinstance(geometry, trimesh.Trimesh):
                    vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                    vertices_world = (transform @ vertices_hom.T).T[:, :3]
                    all_vertices.append(vertices_world)
            except (KeyError, ValueError, IndexError):
                continue
        
        if all_vertices:
            all_vertices = np.vstack(all_vertices)
            camera_target = all_vertices.mean(axis=0)
        else:
            camera_target = camera_pos + np.array([0, 0, -1])
    
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
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)
    
    # Camera setup: perspective
    fx = (0.5 * width) / math.tan(math.radians(fov) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    pin.extrinsic = look_at(
        camera_pos.astype(np.float64),
        camera_target.astype(np.float64),
        np.array([0, 1, 0], dtype=np.float64)
    )
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.12)
    
    # Capture image to temporary file (Open3D requires filename)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    # Read image back as numpy array
    img = Image.open(tmp_path)
    img_np = np.asarray(img)
    
    # Clean up temp file
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
    Render POV segmentation with taxonomy colors using Open3D.
    
    Args:
        trimesh_scene: trimesh scene
        camera_pos: Camera position
        taxonomy: Taxonomy object
        hide_ceilings: If True, exclude ceiling meshes
        camera_target: Look-at target
        width: Image width
        height: Image height
        fov: Field of view in degrees
        
    Returns:
        Segmentation image array
    """
    # Calculate scene center if not provided
    if camera_target is None:
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
                
                if isinstance(geometry, trimesh.Trimesh):
                    vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                    vertices_world = (transform @ vertices_hom.T).T[:, :3]
                    all_vertices.append(vertices_world)
            except (KeyError, ValueError, IndexError):
                continue
        
        if all_vertices:
            all_vertices = np.vstack(all_vertices)
            camera_target = all_vertices.mean(axis=0)
        else:
            camera_target = camera_pos + np.array([0, 0, -1])
    
    # Create Open3D visualizer
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
                # Get category color
                category_id = metadata.get('category_id', 0)
                color_rgb = taxonomy.get_color(category_id, mode="category")
                if color_rgb is None:
                    color_rgb = (127, 127, 127)
                
                # Create mesh
                o3d_mesh = trimesh_to_o3d_mesh(geometry, transform)
                
                # Set uniform vertex colors based on taxonomy
                num_vertices = len(o3d_mesh.vertices)
                seg_color = np.array([c/255.0 for c in color_rgb], dtype=np.float64)
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(seg_color, (num_vertices, 1))
                )
                
                vis.add_geometry(o3d_mesh)
        except (KeyError, ValueError, IndexError):
            continue
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)
    
    # Camera setup (same as RGB POV)
    fx = (0.5 * width) / math.tan(math.radians(fov) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    pin.extrinsic = look_at(
        camera_pos.astype(np.float64),
        camera_target.astype(np.float64),
        np.array([0, 1, 0], dtype=np.float64)
    )
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.12)
    
    # Capture image to temporary file (Open3D requires filename)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    vis.capture_screen_image(tmp_path, do_render=True)
    vis.destroy_window()
    
    # Read image back as numpy array
    img = Image.open(tmp_path)
    img_np = np.asarray(img)
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    return img_np


def main():
    parser = argparse.ArgumentParser(description="Render 3D-FRONT scene layouts and POVs using Open3D")
    parser.add_argument("--scene_json", required=True, help="Path to 3D-FRONT JSON file")
    parser.add_argument("--future_root", required=True, help="Path to 3D-FUTURE model directory")
    parser.add_argument("--output_dir", required=True, help="Output dataset root directory")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    parser.add_argument("--num_povs", type=int, default=6, help="Number of POVs to render")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hpc", action="store_true", default=False,
                        help="Run inside Xvfb for headless HPC rendering (like old pipeline)")
    
    args = parser.parse_args()
    
    # Start Xvfb if needed (like old pipeline)
    global VFB
    if args.hpc:
        if XVFBWRAPPER_AVAILABLE:
            try:
                VFB = Xvfb(width=args.num_povs * 256, height=256, colordepth=24)
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
    geometry_dir = output_dir / "geometry"
    
    for d in [layouts_rgb_dir, layouts_seg_dir, povs_rgb_dir, povs_seg_dir, graphs_dir, geometry_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Extract rooms from scene
    print("Extracting rooms from scene...")
    room_scenes = extract_rooms_from_scene(trimesh_scene)
    print(f"  Found {len(room_scenes)} rooms: {list(room_scenes.keys())}")
    
    # Scene-level Layout Pass: RGB
    print("Rendering scene-level layout RGB...")
    try:
        layout_rgb = render_layout_rgb(trimesh_scene, hide_ceilings=True, width=256, height=256)
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
        layout_seg = render_layout_seg(trimesh_scene, taxonomy, hide_ceilings=True, width=256, height=256)
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
            room_layout_rgb = render_layout_rgb(room_scene, hide_ceilings=True, width=256, height=256)
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
            room_layout_seg = render_layout_seg(room_scene, taxonomy, hide_ceilings=True, width=256, height=256)
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
    
    # POV Pass: Sample camera positions
    print("Sampling camera positions...")
    floor_bboxes = find_floor_meshes(trimesh_scene, taxonomy)
    
    # Get furniture bboxes for collision checking
    furniture_bboxes = []
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
            if not metadata.get('is_ceiling', False) and metadata.get('label', '') not in ['floor', 'wall']:
                if isinstance(geometry, trimesh.Trimesh):
                    vertices_hom = np.column_stack([geometry.vertices, np.ones(len(geometry.vertices))])
                    vertices_world = (transform @ vertices_hom.T).T[:, :3]
                    min_bounds = vertices_world.min(axis=0)
                    max_bounds = vertices_world.max(axis=0)
                    furniture_bboxes.append((min_bounds, max_bounds))
        except (KeyError, ValueError, IndexError):
            continue
    
    # Render POVs from corners
    print("Sampling camera positions from room corners...")
    camera_positions = sample_camera_positions_from_corners(
        trimesh_scene, floor_bboxes, furniture_bboxes, taxonomy,
        num_povs=args.num_povs, eye_height=1.6
    )
    
    if len(camera_positions) < args.num_povs:
        print(f"Warning: Only found {len(camera_positions)} valid corner positions (requested {args.num_povs})")
    
    # Calculate room center for look-at target
    if floor_bboxes:
        all_min = np.array([bbox[0] for bbox in floor_bboxes]).min(axis=0)
        all_max = np.array([bbox[1] for bbox in floor_bboxes]).max(axis=0)
        room_center = (all_min + all_max) / 2
        room_center[1] = 1.6  # Eye height
    else:
        room_center = None
    
    pov_count = 0
    for i, camera_pos in enumerate(camera_positions[:args.num_povs]):
        
        # Render RGB (looking toward room center)
        try:
            pov_rgb = render_pov(trimesh_scene, camera_pos, hide_ceilings=False, 
                               camera_target=room_center, width=256, height=256, fov=70.0)
            pov_rgb_path = povs_rgb_dir / f"{scene_id}_v{pov_count+1:02d}.png"
            Image.fromarray(pov_rgb).save(pov_rgb_path)
            if not pov_rgb_path.exists():
                raise FileNotFoundError(f"POV RGB file was not created: {pov_rgb_path}")
        except Exception as e:
            print(f"ERROR: Failed to render POV RGB: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Render segmentation (looking toward room center)
        try:
            pov_seg = render_pov_seg(trimesh_scene, camera_pos, taxonomy, hide_ceilings=False,
                                    camera_target=room_center, width=256, height=256, fov=70.0)
            pov_seg_path = povs_seg_dir / f"{scene_id}_v{pov_count+1:02d}.png"
            Image.fromarray(pov_seg).save(pov_seg_path)
            if not pov_seg_path.exists():
                raise FileNotFoundError(f"POV segmentation file was not created: {pov_seg_path}")
        except Exception as e:
            print(f"ERROR: Failed to render POV segmentation: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        pov_count += 1
        print(f"  Rendered POV {pov_count}/{args.num_povs} from corner")
    
    # Save scene geometry as GLB
    print("Saving scene geometry...")
    try:
        glb_path = geometry_dir / f"{scene_id}.glb"
        trimesh_scene.export(glb_path, file_type="glb")
        print(f"  Saved scene geometry: {glb_path}")
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file was not created: {glb_path}")
    except Exception as e:
        print(f"ERROR: Failed to save scene geometry: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - geometry export is optional
    
    # Build graphs from segmentation layouts
    print("Building graphs...")
    from data_preparation.pipeline_v2.graph_builder import build_room_graph_from_layout as build_graph
    
    # Scene-level graph
    try:
        scene_layout_seg_path = layouts_seg_dir / f"{scene_id}_scene.png"
        if scene_layout_seg_path.exists():
            build_graph(
                scene_id, "scene", scene_layout_seg_path, taxonomy, graphs_dir
            )
            print("  Scene graph built successfully")
    except Exception as e:
        print(f"  Warning: Failed to build scene graph: {e}")
        import traceback
        traceback.print_exc()
    
    # Room-level graphs
    for room_name, room_scene in room_scenes.items():
        safe_room_name = room_name.replace(" ", "_").replace("/", "_").lower()
        try:
            room_layout_seg_path = layouts_seg_dir / f"{scene_id}_{safe_room_name}_room.png"
            if room_layout_seg_path.exists():
                build_graph(
                    scene_id, room_name, room_layout_seg_path, taxonomy, graphs_dir
                )
                print(f"  Room graph built for {room_name}")
        except Exception as e:
            print(f"  Warning: Failed to build graph for room {room_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Completed rendering for scene: {scene_id}")
    print(f"  Scene layouts: RGB and segmentation")
    print(f"  Room layouts: {len(room_scenes)} rooms (RGB and segmentation each)")
    print(f"  POVs: {pov_count} RGB and segmentation images")
    print(f"  Geometry: GLB file")
    print(f"  Graphs: 1 scene graph + {len(room_scenes)} room graphs")
    
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
