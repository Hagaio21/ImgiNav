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
        transform: 4x4 transformation matrix (optional, can be None if vertices already in world space)
        use_texture: If True, try to use textures/materials
        apply_transform: If False, don't apply transform (for GLB files where transforms are already baked)
        
    Returns:
        Open3D TriangleMesh
    """
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces.astype(np.int32))
    
    # Apply transform to vertices (if needed)
    if apply_transform and transform is not None and not np.allclose(transform, np.eye(4)):
        vertices_hom = np.column_stack([trimesh_mesh.vertices, np.ones(len(trimesh_mesh.vertices))])
        vertices_world = (transform @ vertices_hom.T).T[:, :3]
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_world.astype(np.float64))
    
    # Handle textures/materials if available
    has_texture = False
    if use_texture:
        # Try multiple ways to get texture from trimesh
        texture_image = None
        uv_coords = None
        
        # Debug: check what visual attributes are available
        visual_attrs = [attr for attr in dir(trimesh_mesh.visual) if not attr.startswith('_')]
        
        # Method 1: Check visual.material.image
        if hasattr(trimesh_mesh.visual, 'material') and trimesh_mesh.visual.material is not None:
            material = trimesh_mesh.visual.material
            material_attrs = [attr for attr in dir(material) if not attr.startswith('_')]
            
            # Check for image attribute
            if hasattr(material, 'image') and material.image is not None:
                texture_image = material.image
                print(f"DEBUG: Found texture in material.image")
            
            # Check for baseColorTexture (glTF/GLB format)
            if texture_image is None:
                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                    texture_image = material.baseColorTexture
                    print(f"DEBUG: Found texture in material.baseColorTexture")
                elif hasattr(material, 'image') and material.image is not None:
                    # Sometimes it's stored differently
                    texture_image = material.image
                    print(f"DEBUG: Found texture in material.image (second check)")
        
        # Method 2: Check visual.uv (needed for texture mapping)
        if hasattr(trimesh_mesh.visual, 'uv') and trimesh_mesh.visual.uv is not None:
            uv_coords = trimesh_mesh.visual.uv
            print(f"DEBUG: Found UV coordinates: shape={uv_coords.shape}")
        elif hasattr(trimesh_mesh.visual, 'vertex_attributes'):
            # Sometimes UVs are in vertex_attributes
            if 'uv' in trimesh_mesh.visual.vertex_attributes:
                uv_coords = trimesh_mesh.visual.vertex_attributes['uv']
                print(f"DEBUG: Found UV coordinates in vertex_attributes")
        
        if texture_image is None:
            print(f"DEBUG: No texture found. Visual attrs: {visual_attrs[:5]}...")
        
        # Apply texture if we found one
        if texture_image is not None:
            try:
                # Convert PIL image to numpy
                if hasattr(texture_image, 'convert'):
                    img_array = np.asarray(texture_image.convert('RGB'))
                else:
                    img_array = np.asarray(texture_image)
                
                # Get UV coordinates (can be per-vertex or per-face)
                if uv_coords is not None:
                    if len(uv_coords) == len(trimesh_mesh.vertices):
                        # Per-vertex UVs: map to triangles
                        triangle_uvs = uv_coords[trimesh_mesh.faces].reshape(-1, 2)
                    elif len(uv_coords) == len(trimesh_mesh.faces):
                        # Per-face UVs: already correct
                        triangle_uvs = uv_coords.reshape(-1, 2)
                    else:
                        # Try to use face indices to map UVs
                        triangle_uvs = uv_coords[trimesh_mesh.faces].reshape(-1, 2)
                    
                    # Create texture
                    o3d_texture = o3d.geometry.Image(img_array.astype(np.uint8))
                    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs.astype(np.float64))
                    # Material IDs are not needed - Open3D will use the first texture automatically
                    # o3d_mesh.triangle_material_ids = ...  # Not needed, removed Vector1iVector which doesn't exist
                    o3d_mesh.textures = [o3d_texture]
                    has_texture = True
                    print(f"DEBUG: Applied texture to mesh (texture size: {img_array.shape}, UVs: {len(triangle_uvs)})")
            except Exception as e:
                print(f"DEBUG: Failed to apply texture: {e}")
                import traceback
                traceback.print_exc()
    
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
    
    # Make mesh double-sided to avoid backface culling issues
    # Duplicate faces with reversed winding order
    if len(o3d_mesh.triangles) > 0:
        triangles = np.asarray(o3d_mesh.triangles)
        reversed_triangles = triangles[:, [0, 2, 1]]  # Reverse winding
        all_triangles = np.vstack([triangles, reversed_triangles])
        o3d_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        o3d_mesh.compute_vertex_normals()  # Recompute normals
    
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


def clip_mesh_above_height(mesh: trimesh.Trimesh, transform: np.ndarray, 
                          max_height: float, up_axis: int = 1) -> Optional[trimesh.Trimesh]:
    """
    Clip mesh to remove parts above max_height (in world coordinates).
    Uses simple vertex/face filtering to avoid boolean operations.
    
    Args:
        mesh: Input trimesh mesh
        transform: 4x4 transformation matrix
        max_height: Maximum height coordinate along up_axis
        up_axis: Axis index for up direction (0=X, 1=Y, 2=Z, default=1 for Y-up)
        
    Returns:
        Clipped mesh or None if completely above threshold
    """
    # Transform vertices to world space
    if transform is not None and not np.allclose(transform, np.eye(4)):
        vertices_hom = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
        vertices_world = (transform @ vertices_hom.T).T[:, :3]
    else:
        vertices_world = mesh.vertices
    
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
    Render top-down RGB layout using Open3D with orthographic projection.
    Clips top portion of structure to make pathways visible.
    
    Args:
        trimesh_scene: trimesh scene
        hide_ceilings: If True, exclude ceiling meshes
        width: Image width
        height: Image height
        clip_top_meters: Height in meters to clip from top (default 1.0)
        
    Returns:
        RGB image array (H, W, 3) uint8
    """
    # Use metadata if available, otherwise compute bounds
    if scene_metadata is not None and 'scene_bounds' in scene_metadata:
        scene_bounds = scene_metadata['scene_bounds']
        if scene_bounds.get('min') and scene_bounds.get('max'):
            min_bounds = np.array(scene_bounds['min'])
            max_bounds = np.array(scene_bounds['max'])
        else:
            min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings)
    else:
        min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings)
    
    # Detect up direction BEFORE clipping calculation
    if scene_metadata is not None and 'up_direction' in scene_metadata:
        up_direction = scene_metadata['up_direction']
        up_axis = up_direction.get('up_axis', 1)  # Default to Y
        up_vector = np.array(up_direction.get('up_vector', [0, 1, 0]), dtype=np.float64)
    else:
        # Fallback: assume Y-up (3D-FRONT convention)
        up_axis = 1
        up_vector = np.array([0, 1, 0], dtype=np.float64)
    
    # Calculate clipping height (0.5 meters from top) - mandatory
    # Safety check: never clip below the bottom 1/3rd of room height to guarantee floor is never removed
    room_height = max_bounds[up_axis] - min_bounds[up_axis]
    min_safe_clip_height = min_bounds[up_axis] + (room_height / 3.0)  # Bottom 1/3rd is always safe
    clip_height = max_bounds[up_axis] - clip_top_meters  # Clip from top along up_axis
    if clip_height < min_safe_clip_height:
        clip_height = min_safe_clip_height
        print(f"WARNING: Clipping height adjusted to preserve floor. Using min_safe_clip_height = {clip_height:.2f}")
    
    # Calculate floor threshold - meshes below this are considered floor and should not be clipped
    floor_threshold = min_bounds[up_axis] + (room_height * 0.1)  # Bottom 10% is floor
    
    # Process geometry - handle both scene graph (GLB) and single mesh (PLY)
    all_vertices = []
    processed_geometries = []
    
    # Check if we have a scene graph (GLB) or just geometry (PLY)
    has_scene_graph = len(trimesh_scene.graph.nodes_geometry) > 0
    
    if has_scene_graph:
        # GLB format: use scene graph with transforms
        for node_name in trimesh_scene.graph.nodes_geometry:
            try:
                transform, geometry_name = trimesh_scene.graph.get(node_name)
                
                # Get geometry
                if geometry_name in trimesh_scene.geometry:
                    geometry = trimesh_scene.geometry[geometry_name]
                elif node_name in trimesh_scene.geometry:
                    geometry = trimesh_scene.geometry[node_name]
                else:
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    metadata = getattr(geometry, 'metadata', {})
                    if hide_ceilings and metadata.get('is_ceiling', False):
                        continue
                    
                    # Apply transform to vertices
                    vertices = geometry.vertices
                    if not np.allclose(transform, np.eye(4)):
                        vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                        vertices = (transform @ vertices_hom.T).T[:, :3]
                    
                    all_vertices.append(vertices)
                    processed_geometries.append((geometry, transform, metadata))
            except (KeyError, ValueError, IndexError, AttributeError):
                continue
    else:
        # PLY format: single merged mesh, already in world space
        for geometry_name, geometry in trimesh_scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                metadata = getattr(geometry, 'metadata', {})
                if hide_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                # PLY meshes are already in world space, no transform needed
                vertices = geometry.vertices
                all_vertices.append(vertices)
                processed_geometries.append((geometry, np.eye(4), metadata))
    
    if all_vertices:
        # Use actual geometry from scene graph (with transforms applied)
        all_vertices = np.vstack(all_vertices)
        actual_min_bounds = all_vertices.min(axis=0)
        actual_max_bounds = all_vertices.max(axis=0)
        center = (actual_min_bounds + actual_max_bounds) / 2
        actual_size = actual_max_bounds - actual_min_bounds
        base_max_size = max(actual_size[0], actual_size[2])  # X and Z dimensions for top-down (floor bounding box)
        # Apply margin to get target view size (margin is the driving parameter)
        margin = 0.03  # 3% margin
        max_size = base_max_size * (1.0 + margin)  # Target view size with margin
        print(f"DEBUG: Using scene graph meshes - center: {center}, size: {actual_size}, base_size: {base_max_size:.2f}, target_size: {max_size:.2f} (margin: {margin*100:.1f}%)")
    else:
        # Fallback to metadata bounds
        center = (min_bounds + max_bounds) / 2
        size = max_bounds - min_bounds
        base_max_size = max(size[0], size[2])  # X and Z dimensions for top-down (floor bounding box)
        # Apply margin to get target view size (margin is the driving parameter)
        margin = 0.03  # 3% margin
        max_size = base_max_size * (1.0 + margin)  # Target view size with margin
        print(f"DEBUG: No geometry found, using metadata bounds - center: {center}, size: {size}, base_size: {base_max_size:.2f}, target_size: {max_size:.2f} (margin: {margin*100:.1f}%)")
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes with clipping using scene graph method
    mesh_count = 0
    total_geometries = len(processed_geometries)
    print(f"DEBUG: Processing {total_geometries} geometries from scene graph")
    print(f"DEBUG: Clipping height set to: {clip_height:.2f} (room height: {room_height:.2f}, min_safe: {min_safe_clip_height:.2f})")
    
    for geometry, transform, metadata in processed_geometries:
        try:
            # Apply transform to vertices (already done above, but need for clipping)
            vertices = geometry.vertices
            if not np.allclose(transform, np.eye(4)):
                vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                vertices_world = (transform @ vertices_hom.T).T[:, :3]
            else:
                vertices_world = vertices
            
            # Clip mesh if clipping is enabled
            faces_to_use = geometry.faces
            if clip_height is not None:
                # Check if this is a floor mesh - if so, skip clipping
                mesh_center_height = vertices_world[:, up_axis].mean()
                is_floor_mesh = mesh_center_height < floor_threshold
                
                if not is_floor_mesh:
                    # Create temporary mesh with transformed vertices for clipping
                    temp_mesh = trimesh.Trimesh(vertices=vertices_world, faces=geometry.faces)
                    clipped_mesh = clip_mesh_above_height(temp_mesh, np.eye(4), clip_height, up_axis=up_axis)
                    if clipped_mesh is None:
                        continue
                    # Use clipped mesh vertices and faces
                    vertices_world = clipped_mesh.vertices
                    faces_to_use = clipped_mesh.faces
            
            # Create Open3D mesh with transformed vertices
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_world.astype(np.float64))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_to_use.astype(np.int32))
            
            # Handle textures/materials
            has_texture = False
            texture_image = None
            uv_coords = None
            
            # Try to get texture from material
            if hasattr(geometry.visual, 'material') and geometry.visual.material is not None:
                material = geometry.visual.material
                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                    texture_image = material.baseColorTexture
                elif hasattr(material, 'image') and material.image is not None:
                    texture_image = material.image
            
            # Get UV coordinates
            if hasattr(geometry.visual, 'uv') and geometry.visual.uv is not None:
                uv_coords = geometry.visual.uv
            
            # Apply texture if found
            if texture_image is not None and uv_coords is not None:
                try:
                    if hasattr(texture_image, 'convert'):
                        img_array = np.asarray(texture_image.convert('RGB'))
                    else:
                        img_array = np.asarray(texture_image)
                    
                    if len(uv_coords) == len(geometry.vertices):
                        triangle_uvs = uv_coords[geometry.faces].reshape(-1, 2)
                    else:
                        triangle_uvs = uv_coords.reshape(-1, 2)
                    
                    o3d_texture = o3d.geometry.Image(img_array.astype(np.uint8))
                    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs.astype(np.float64))
                    o3d_mesh.textures = [o3d_texture]
                    has_texture = True
                except Exception:
                    pass
            
            # Make double-sided
            if len(o3d_mesh.triangles) > 0:
                triangles = np.asarray(o3d_mesh.triangles)
                reversed_triangles = triangles[:, [0, 2, 1]]
                all_triangles = np.vstack([triangles, reversed_triangles])
                o3d_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
            
            o3d_mesh.compute_vertex_normals()
            
            # Check if mesh has vertices
            if len(o3d_mesh.vertices) == 0:
                continue
            
            vis.add_geometry(o3d_mesh)
            mesh_count += 1
        except (KeyError, ValueError, IndexError, AttributeError) as e:
            continue
    
    print(f"DEBUG: Added {mesh_count} meshes to visualizer (out of {total_geometries} geometries)")
    if mesh_count == 0:
        print(f"ERROR: No meshes added to visualizer!")
        print(f"  Scene bounds: min={min_bounds}, max={max_bounds}")
        print(f"  Clip height: {clip_height:.2f}")
        return np.full((height, width, 3), 128, dtype=np.uint8)  # Return gray image if no meshes
    
    # Set render options with lighting
    opt = vis.get_render_option()
    opt.background_color = np.array([0.2, 0.2, 0.25], dtype=np.float32)  # Slightly blue-tinted dark gray for better contrast
    opt.light_on = True  # Enable lighting for textures
    
    # Camera setup: orthographic projection (top-down)
    # up_axis and up_vector already determined above
    print(f"Using up direction: axis={up_axis}, vector={up_vector}")
    
    # Calculate focal length first (for orthographic projection)
    desired_fov_rad = np.radians(1.0)  # Very small FOV for orthographic projection
    fx = fy = (0.5 * min(width, height)) / np.tan(desired_fov_rad / 2.0)
    
    # Calculate camera distance to show exactly target_size (max_size) in the image
    # For pinhole camera: distance = (target_size / 2) * focal_length / (image_size / 2)
    # Simplified: distance = target_size * focal_length / image_size
    image_size = min(width, height)
    camera_distance = (max_size / 2.0) * fx / (image_size / 2.0)
    print(f"DEBUG: Camera distance: {camera_distance:.2f} (target_size: {max_size:.2f}, focal_length: {fx:.2f}, image_size: {image_size})")
    # Position camera along the up direction, looking down the negative up direction
    # Camera above scene looking down along -up_vector
    camera_offset = up_vector * camera_distance
    camera_height_pos = max_bounds[up_axis] + camera_distance
    eye = center.astype(np.float64) + camera_offset
    # Ensure camera is definitely above the scene
    eye[up_axis] = camera_height_pos
    center_point = center.astype(np.float64)
    
    # For top-down view, the camera's up vector in the image should be perpendicular to the world up
    # Find a horizontal axis to use as the camera's "up" in the image
    # Use the axis with the second-largest scene dimension
    scene_size_arr = np.array([max_bounds[0] - min_bounds[0], 
                               max_bounds[1] - min_bounds[1], 
                               max_bounds[2] - min_bounds[2]])
    # Set the up axis dimension to 0 so we don't pick it
    scene_size_arr[up_axis] = 0
    horizontal_axis = int(np.argmax(scene_size_arr))
    camera_up = np.zeros(3, dtype=np.float64)
    camera_up[horizontal_axis] = 1.0
    up = camera_up
    
    # Debug: print forward vector to verify direction
    forward = center_point - eye
    print(f"DEBUG: Scene bounds - min={min_bounds}, max={max_bounds}")
    print(f"DEBUG: Scene center={center}")
    print(f"DEBUG: Up direction - axis={up_axis}, vector={up_vector}")
    print(f"DEBUG: Camera height={camera_height_pos:.2f} (max_bounds[{up_axis}]={max_bounds[up_axis]:.2f} + distance={camera_distance:.2f})")
    print(f"DEBUG: Camera eye={eye}")
    print(f"DEBUG: Camera center_point={center_point}")
    print(f"DEBUG: Camera up (image orientation)={up}")
    print(f"DEBUG: Camera forward vector (should point DOWN): {forward}")
    print(f"DEBUG: Forward vector magnitude: {np.linalg.norm(forward):.2f}")
    print(f"DEBUG: Forward component along up axis (should be negative): {forward[up_axis]:.2f}")
    print(f"DEBUG: Camera is ABOVE scene: {eye[up_axis] > max_bounds[up_axis]} (eye[{up_axis}]={eye[up_axis]:.2f} > max_bounds[{up_axis}]={max_bounds[up_axis]:.2f})")
    
    # Focal length already calculated above for camera distance
    cx, cy = width / 2.0, height / 2.0
    
    print(f"DEBUG: Orthographic FOV: 1.0 degrees, focal length: {fx:.2f}")
    
    print(f"DEBUG: Camera intrinsics - fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        # Match old pipeline's look_at function exactly
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)  # Note: upn cross f (not f cross upn)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f  # Note: positive f (not -f)
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
                      width: int = 256, height: int = 256,
                      clip_top_meters: float = 0.5,
                      scene_metadata: Optional[Dict] = None) -> np.ndarray:
    """
    Render top-down segmentation layout with taxonomy colors.
    Uses orthographic projection and clips top portion.
    """
    # Use metadata if available, otherwise compute bounds
    if scene_metadata is not None and 'scene_bounds' in scene_metadata:
        scene_bounds = scene_metadata['scene_bounds']
        if scene_bounds.get('min') and scene_bounds.get('max'):
            min_bounds = np.array(scene_bounds['min'])
            max_bounds = np.array(scene_bounds['max'])
        else:
            min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings)
    else:
        min_bounds, max_bounds = get_scene_bounds(trimesh_scene, hide_ceilings)
    
    # Detect up direction BEFORE clipping calculation
    if scene_metadata is not None and 'up_direction' in scene_metadata:
        up_direction = scene_metadata['up_direction']
        up_axis = up_direction.get('up_axis', 1)  # Default to Y
        up_vector = np.array(up_direction.get('up_vector', [0, 1, 0]), dtype=np.float64)
    else:
        # Fallback: assume Y-up (3D-FRONT convention)
        up_axis = 1
        up_vector = np.array([0, 1, 0], dtype=np.float64)
    
    # Calculate clipping height (0.5 meters from top) - mandatory
    # Safety check: never clip below the bottom 1/3rd of room height to guarantee floor is never removed
    room_height = max_bounds[up_axis] - min_bounds[up_axis]
    min_safe_clip_height = min_bounds[up_axis] + (room_height / 3.0)  # Bottom 1/3rd is always safe
    clip_height = max_bounds[up_axis] - clip_top_meters  # Clip from top along up_axis
    if clip_height < min_safe_clip_height:
        clip_height = min_safe_clip_height
        print(f"WARNING: Clipping height adjusted to preserve floor. Using min_safe_clip_height = {clip_height:.2f}")
    
    # Calculate floor threshold - meshes below this are considered floor and should not be clipped
    floor_threshold = min_bounds[up_axis] + (room_height * 0.1)  # Bottom 10% is floor
    
    # Process geometry - handle both scene graph (GLB) and single mesh (PLY)
    all_vertices = []
    processed_geometries = []
    
    # Check if we have a scene graph (GLB) or just geometry (PLY)
    has_scene_graph = len(trimesh_scene.graph.nodes_geometry) > 0
    
    if has_scene_graph:
        # GLB format: use scene graph with transforms
        for node_name in trimesh_scene.graph.nodes_geometry:
            try:
                transform, geometry_name = trimesh_scene.graph.get(node_name)
                
                # Get geometry
                if geometry_name in trimesh_scene.geometry:
                    geometry = trimesh_scene.geometry[geometry_name]
                elif node_name in trimesh_scene.geometry:
                    geometry = trimesh_scene.geometry[node_name]
                else:
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    metadata = getattr(geometry, 'metadata', {})
                    if hide_ceilings and metadata.get('is_ceiling', False):
                        continue
                    
                    # Apply transform to vertices
                    vertices = geometry.vertices
                    if not np.allclose(transform, np.eye(4)):
                        vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                        vertices = (transform @ vertices_hom.T).T[:, :3]
                    
                    all_vertices.append(vertices)
                    processed_geometries.append((geometry, transform, metadata))
            except (KeyError, ValueError, IndexError, AttributeError):
                continue
    else:
        # PLY format: single merged mesh, already in world space
        for geometry_name, geometry in trimesh_scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                metadata = getattr(geometry, 'metadata', {})
                if hide_ceilings and metadata.get('is_ceiling', False):
                    continue
                
                # PLY meshes are already in world space, no transform needed
                vertices = geometry.vertices
                all_vertices.append(vertices)
                processed_geometries.append((geometry, np.eye(4), metadata))
    
    if all_vertices:
        # Use actual geometry from scene graph (with transforms applied)
        all_vertices = np.vstack(all_vertices)
        actual_min_bounds = all_vertices.min(axis=0)
        actual_max_bounds = all_vertices.max(axis=0)
        center = (actual_min_bounds + actual_max_bounds) / 2
        actual_size = actual_max_bounds - actual_min_bounds
        base_max_size = max(actual_size[0], actual_size[2])  # X and Z dimensions for top-down (floor bounding box)
        # Apply margin to get target view size (margin is the driving parameter)
        margin = 0.03  # 3% margin
        max_size = base_max_size * (1.0 + margin)  # Target view size with margin
        print(f"DEBUG: Using scene graph meshes - center: {center}, size: {actual_size}, base_size: {base_max_size:.2f}, target_size: {max_size:.2f} (margin: {margin*100:.1f}%)")
    else:
        # Fallback to metadata bounds
        center = (min_bounds + max_bounds) / 2
        size = max_bounds - min_bounds
        base_max_size = max(size[0], size[2])  # X and Z dimensions for top-down (floor bounding box)
        # Apply margin to get target view size (margin is the driving parameter)
        margin = 0.03  # 3% margin
        max_size = base_max_size * (1.0 + margin)  # Target view size with margin
        print(f"DEBUG: No geometry found, using metadata bounds - center: {center}, size: {size}, base_size: {base_max_size:.2f}, target_size: {max_size:.2f} (margin: {margin*100:.1f}%)")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add meshes with taxonomy colors and clipping using scene graph method
    mesh_count = 0
    total_geometries = len(processed_geometries)
    print(f"DEBUG: Processing {total_geometries} geometries from scene graph")
    print(f"DEBUG: Clipping height set to: {clip_height:.2f} (room height: {room_height:.2f}, min_safe: {min_safe_clip_height:.2f})")
    
    for geometry, transform, metadata in processed_geometries:
        try:
            # Apply transform to vertices
            vertices = geometry.vertices
            if not np.allclose(transform, np.eye(4)):
                vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                vertices_world = (transform @ vertices_hom.T).T[:, :3]
            else:
                vertices_world = vertices
            
            # Clip mesh if clipping is enabled
            faces_to_use = geometry.faces
            if clip_height is not None:
                # Check if this is a floor mesh - if so, skip clipping
                mesh_center_height = vertices_world[:, up_axis].mean()
                is_floor_mesh = mesh_center_height < floor_threshold
                
                if not is_floor_mesh:
                    # Create temporary mesh with transformed vertices for clipping
                    temp_mesh = trimesh.Trimesh(vertices=vertices_world, faces=geometry.faces)
                    clipped_mesh = clip_mesh_above_height(temp_mesh, np.eye(4), clip_height, up_axis=up_axis)
                    if clipped_mesh is None:
                        continue
                    # Use clipped mesh vertices and faces
                    vertices_world = clipped_mesh.vertices
                    faces_to_use = clipped_mesh.faces
            
            # Create Open3D mesh with transformed vertices
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_world.astype(np.float64))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_to_use.astype(np.int32))
            
            # Apply segmentation colors
            category_id = metadata.get('category_id', 0)
            color_rgb = taxonomy.get_color(category_id, mode="category")
            if color_rgb is None:
                color_rgb = (127, 127, 127)
            
            num_vertices = len(vertices_world)
            seg_color = np.array([c/255.0 for c in color_rgb], dtype=np.float64)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(seg_color, (num_vertices, 1))
            )
            
            # Make double-sided
            if len(o3d_mesh.triangles) > 0:
                triangles = np.asarray(o3d_mesh.triangles)
                reversed_triangles = triangles[:, [0, 2, 1]]
                all_triangles = np.vstack([triangles, reversed_triangles])
                o3d_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
            
            o3d_mesh.compute_vertex_normals()
            
            # Check if mesh has vertices
            if len(o3d_mesh.vertices) == 0:
                continue
            
            vis.add_geometry(o3d_mesh)
            mesh_count += 1
        except (KeyError, ValueError, IndexError, AttributeError):
            continue
    
    print(f"DEBUG: Added {mesh_count} meshes to visualizer (out of {total_geometries} geometries)")
    if mesh_count == 0:
        print(f"ERROR: No meshes added to visualizer!")
        print(f"  Scene bounds: min={min_bounds}, max={max_bounds}")
        print(f"  Clip height: {clip_height:.2f}")
        return np.full((height, width, 3), 255, dtype=np.uint8)  # Return white image if no meshes
    
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White background for segmentation
    opt.light_on = True
    
    # Camera setup: orthographic projection (same as RGB)
    # up_axis and up_vector already determined above
    print(f"Using up direction: axis={up_axis}, vector={up_vector}")
    
    # Calculate focal length first (for orthographic projection)
    desired_fov_rad = np.radians(1.0)  # Very small FOV for orthographic projection
    fx = fy = (0.5 * min(width, height)) / np.tan(desired_fov_rad / 2.0)
    
    # Calculate camera distance to show exactly target_size (max_size) in the image
    # For pinhole camera: distance = (target_size / 2) * focal_length / (image_size / 2)
    # Simplified: distance = target_size * focal_length / image_size
    image_size = min(width, height)
    camera_distance = (max_size / 2.0) * fx / (image_size / 2.0)
    print(f"DEBUG: Camera distance: {camera_distance:.2f} (target_size: {max_size:.2f}, focal_length: {fx:.2f}, image_size: {image_size})")
    # Position camera along the up direction, looking down the negative up direction
    camera_offset = up_vector * camera_distance
    camera_height_pos = max_bounds[up_axis] + camera_distance
    eye = center.astype(np.float64) + camera_offset
    # Ensure camera is definitely above the scene
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
    up = camera_up
    
    # Debug: print forward vector to verify direction
    forward = center_point - eye
    print(f"DEBUG: Camera forward vector (should point DOWN): {forward}")
    print(f"DEBUG: Forward vector magnitude: {np.linalg.norm(forward):.2f}")
    print(f"DEBUG: Forward Y component (should be negative for looking down): {forward[1]:.2f}")
    
    # Orthographic-like: calculate focal length
    # For top-down view, we want the entire scene to fit in the image
    # Focal length already calculated above for camera distance
    cx, cy = width / 2.0, height / 2.0
    
    print(f"DEBUG: Orthographic FOV: 1.0 degrees, focal length: {fx:.2f}")
    
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def look_at(eye_, center_, up_):
        # Match old pipeline's look_at function exactly
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)  # Note: upn cross f (not f cross upn)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f  # Note: positive f (not -f)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T
    
    pin.extrinsic = look_at(eye, center_point, up)
    
    print(f"DEBUG: Camera setup - eye={eye}, center={center_point}, up={up}")
    print(f"DEBUG: Scene center={center}, max_size={max_size:.2f}, camera_distance={camera_distance:.2f}")
    print(f"DEBUG: Camera intrinsics - fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
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
        # Match old pipeline's look_at function exactly
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)  # Note: upn cross f (not f cross upn)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f  # Note: positive f (not -f)
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
        # Match old pipeline's look_at function exactly
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)  # Note: upn cross f (not f cross upn)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f  # Note: positive f (not -f)
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

