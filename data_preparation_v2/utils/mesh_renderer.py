#!/usr/bin/env python3

import numpy as np
import trimesh
import open3d as o3d
from typing import Tuple, Optional, List
from pathlib import Path

def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert trimesh to Open3D mesh."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    
    # Handle vertex colors
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vcolors = mesh.visual.vertex_colors
        if len(vcolors.shape) == 2 and vcolors.shape[1] >= 3:
            # Convert to RGB (0-1 range)
            colors = vcolors[:, :3].astype(np.float64) / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Handle vertex normals
    if hasattr(mesh.visual, 'vertex_normals') and mesh.visual.vertex_normals is not None:
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(
            mesh.visual.vertex_normals.astype(np.float64)
        )
    else:
        o3d_mesh.compute_vertex_normals()
    
    return o3d_mesh

def filter_ceiling_geometry(mesh: trimesh.Trimesh, label: str) -> bool:
    """Check if mesh should be excluded (ceiling)."""
    return "ceiling" in label.lower()

def clip_walls_from_top(mesh: trimesh.Trimesh, clip_amount: float = 0.2) -> trimesh.Trimesh:
    """Clip walls from top by removing vertices/faces above a certain height threshold."""
    if mesh.is_empty:
        return mesh
    
    vertices = mesh.vertices.copy()
    z_max = vertices[:, 2].max()
    z_threshold = z_max - clip_amount
    
    # Keep only vertices below threshold
    mask = vertices[:, 2] <= z_threshold
    
    if mask.sum() == 0:
        return mesh  # Nothing to clip
    
    # Create new mesh with clipped vertices
    # This is a simplified approach - for better results, we'd need to properly clip faces
    # For now, we'll just remove vertices above threshold and reindex faces
    vertex_map = {}
    new_vertices = []
    new_vertex_idx = 0
    
    for i, keep in enumerate(mask):
        if keep:
            vertex_map[i] = new_vertex_idx
            new_vertices.append(vertices[i])
            new_vertex_idx += 1
    
    if len(new_vertices) == 0:
        return mesh
    
    new_vertices = np.array(new_vertices)
    
    # Reindex faces - only keep faces where all vertices are kept
    new_faces = []
    for face in mesh.faces:
        if all(v in vertex_map for v in face):
            new_face = [vertex_map[v] for v in face]
            new_faces.append(new_face)
    
    if len(new_faces) == 0:
        return mesh
    
    new_faces = np.array(new_faces)
    
    # Create new mesh
    clipped_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    # Copy visual properties if they exist
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vcolors = mesh.visual.vertex_colors[mask]
        clipped_mesh.visual.vertex_colors = vcolors
    
    return clipped_mesh

def render_orthographic_topdown(
    meshes: List[trimesh.Trimesh],
    bbox: dict,
    resolution: int = 512,
    margin: float = 0.1,
    exclude_ceiling: bool = True,
    clip_walls: bool = True,
    clip_amount: float = 0.2,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Render top-down orthographic view of meshes.
    
    Args:
        meshes: List of trimesh meshes to render
        bbox: Bounding box dict with 'min' and 'max' keys
        resolution: Output image resolution
        margin: Margin as fraction of bbox size
        exclude_ceiling: Whether to exclude ceiling geometry
        clip_walls: Whether to clip walls from top
        clip_amount: Amount to clip from top (in meters)
        labels: Optional list of labels for each mesh (for ceiling filtering)
    
    Returns:
        Rendered image as numpy array (H, W, 3) in RGB format
    """
    if not meshes:
        return np.full((resolution, resolution, 3), 240, dtype=np.uint8)
    
    # Compute view bounds
    bbox_min = np.array(bbox['min'])
    bbox_max = np.array(bbox['max'])
    bbox_size = bbox_max - bbox_min
    
    # Add margin
    margin_size = bbox_size * margin
    view_min = bbox_min - margin_size
    view_max = bbox_max + margin_size
    view_size = view_max - view_min
    
    # Compute camera parameters for orthographic projection
    center = (view_min + view_max) / 2.0
    eye = center + np.array([0, 0, view_size.max() * 2])  # High above
    lookat = center
    up = np.array([0, 1, 0])  # Y-up
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=resolution, height=resolution)
    
    # Add meshes
    for i, mesh in enumerate(meshes):
        label = labels[i] if labels and i < len(labels) else ""
        
        # Filter ceiling if requested
        if exclude_ceiling and filter_ceiling_geometry(mesh, label):
            continue
        
        # Clip walls if requested
        if clip_walls:
            mesh = clip_walls_from_top(mesh, clip_amount)
        
        if mesh.is_empty:
            continue
        
        # Convert to Open3D and add
        o3d_mesh = trimesh_to_o3d(mesh)
        vis.add_geometry(o3d_mesh)
    
    # Set up camera
    ctr = vis.get_view_control()
    
    # Set orthographic projection
    ctr.change_field_of_view(step=0)  # Reset FOV
    
    # Set camera to look down
    param = ctr.convert_to_pinhole_camera_parameters()
    
    # Compute orthographic projection matrix
    # For orthographic, we need to set up the view manually
    # Open3D doesn't have direct orthographic support, so we use a very narrow FOV
    # and adjust the camera distance
    
    # Set camera parameters
    param.extrinsic = np.eye(4)
    param.extrinsic[:3, 3] = -eye
    
    # Create look-at matrix
    forward = lookat - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    up_corrected = np.cross(right, forward)
    
    param.extrinsic[:3, 0] = right
    param.extrinsic[:3, 1] = -up_corrected
    param.extrinsic[:3, 2] = -forward
    
    # Set intrinsic for orthographic-like view
    # Use very large focal length to approximate orthographic
    focal_length = resolution * 10  # Very large to approximate orthographic
    param.intrinsic.set_intrinsics(
        resolution, resolution,
        focal_length, focal_length,
        resolution / 2, resolution / 2
    )
    
    ctr.convert_from_pinhole_camera_parameters(param)
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    
    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    image_np = np.asarray(image)
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert BGR to RGB
    image_np = image_np[:, :, ::-1]
    
    vis.destroy_window()
    
    return image_np

def render_perspective(
    meshes: List[trimesh.Trimesh],
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    fov_deg: float = 70.0,
    resolution: Tuple[int, int] = (1280, 800),
    eye_height: float = 1.6,
    exclude_ceiling: bool = True,
    clip_walls: bool = True,
    clip_amount: float = 0.2,
    labels: Optional[List[str]] = None,
    bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Render perspective view of meshes from camera position.
    
    Args:
        meshes: List of trimesh meshes to render
        camera_pos: Camera position (3D point)
        camera_target: Point camera looks at
        fov_deg: Field of view in degrees
        resolution: Output image resolution (width, height)
        eye_height: Height offset for camera (added to Z)
        exclude_ceiling: Whether to exclude ceiling geometry
        clip_walls: Whether to clip walls from top
        clip_amount: Amount to clip from top (in meters)
        labels: Optional list of labels for each mesh
        bg_color: Background color (RGB)
    
    Returns:
        Rendered image as numpy array (H, W, 3) in RGB format
    """
    if not meshes:
        return np.full((resolution[1], resolution[0], 3), bg_color, dtype=np.uint8)
    
    width, height = resolution
    
    # Adjust camera position with eye height
    eye = camera_pos.copy()
    eye[2] += eye_height
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes
    for i, mesh in enumerate(meshes):
        label = labels[i] if labels and i < len(labels) else ""
        
        # Filter ceiling if requested
        if exclude_ceiling and filter_ceiling_geometry(mesh, label):
            continue
        
        # Clip walls if requested
        if clip_walls:
            mesh = clip_walls_from_top(mesh, clip_amount)
        
        if mesh.is_empty:
            continue
        
        # Convert to Open3D and add
        o3d_mesh = trimesh_to_o3d(mesh)
        vis.add_geometry(o3d_mesh)
    
    # Set up camera
    ctr = vis.get_view_control()
    
    # Compute camera parameters
    fx = (0.5 * width) / np.tan(np.radians(fov_deg) / 2.0)
    fy = (0.5 * height) / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # Create look-at matrix
    forward = camera_target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    up = np.array([0, 0, 1])  # Z-up
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    up_corrected = np.cross(right, forward)
    
    # Build extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, 0] = right
    extrinsic[:3, 1] = -up_corrected
    extrinsic[:3, 2] = -forward
    extrinsic[:3, 3] = -eye
    
    param.extrinsic = extrinsic
    
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    
    # Set background color
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color, dtype=np.float32) / 255.0
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    
    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    image_np = np.asarray(image)
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert BGR to RGB
    image_np = image_np[:, :, ::-1]
    
    vis.destroy_window()
    
    return image_np

