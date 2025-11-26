#!/usr/bin/env python3
"""
Headless 3D scene renderer using Open3D OffscreenRenderer.
Generates top-down layout views and POV images from GLB/OBJ meshes.
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import open3d as o3d
import trimesh
from PIL import Image, ImageDraw

# Configure Open3D for headless rendering
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class SceneRenderer:
    """Headless 3D scene renderer using Open3D OffscreenRenderer."""
    
    def __init__(self, width: int = 1024, height: int = 1024, background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize the renderer.
        
        Args:
            width: Render width in pixels
            height: Render height in pixels
            background_color: RGB background color (0-1 range)
        """
        self.width = width
        self.height = height
        self.background_color = np.array(background_color, dtype=np.float32)
        
        # Initialize OffscreenRenderer
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.scene = self.renderer.scene
        
        # Set up scene
        self.scene.set_background(background_color)
        
        # Set up lighting (try different API approaches)
        try:
            # Method 1: Direct scene lighting (Open3D 0.13+)
            if hasattr(self.scene, 'set_lighting'):
                self.scene.set_lighting(
                    self.scene.LightingProfile.SOFT_SHADOWS,
                    (0.577, -0.577, 0.577)
                )
        except (AttributeError, TypeError):
            try:
                # Method 2: Via scene.scene
                if hasattr(self.scene, 'scene') and hasattr(self.scene.scene, 'set_lighting'):
                    self.scene.scene.set_lighting(
                        self.scene.scene.LightingProfile.SOFT_SHADOWS,
                        (0.577, -0.577, 0.577)
                    )
            except (AttributeError, TypeError):
                pass
        
        # Add directional light (try different API approaches)
        try:
            if hasattr(self.scene, 'add_directional_light'):
                light = self.scene.add_directional_light("sun", (0.577, -0.577, 0.577), (1.0, 1.0, 1.0))
                if hasattr(light, 'set_shadow_map_size'):
                    light.set_shadow_map_size(2048)
        except (AttributeError, TypeError):
            try:
                if hasattr(self.scene, 'scene') and hasattr(self.scene.scene, 'add_directional_light'):
                    light = self.scene.scene.add_directional_light("sun", (0.577, -0.577, 0.577), (1.0, 1.0, 1.0))
                    if hasattr(light, 'set_shadow_map_size'):
                        light.set_shadow_map_size(2048)
            except (AttributeError, TypeError):
                pass
    
    def load_mesh_to_o3d(self, mesh_path: Path, debug: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Load GLB/OBJ mesh using trimesh and convert to Open3D TriangleMesh.
        
        Args:
            mesh_path: Path to GLB or OBJ file
            debug: Enable debug output
            
        Returns:
            Combined Open3D TriangleMesh
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        if debug:
            print(f"Loading mesh: {mesh_path}")
        
        # Load with trimesh
        scene = trimesh.load(str(mesh_path), force="scene", process=False)
        
        o3d_meshes = []
        
        if isinstance(scene, trimesh.Scene):
            if debug:
                print(f"  Scene has {len(scene.geometry)} geometries")
            
            for node_name in scene.graph.nodes_geometry:
                try:
                    transform, geometry_name = scene.graph.get(node_name)
                    
                    geometry = None
                    if geometry_name in scene.geometry:
                        geometry = scene.geometry[geometry_name]
                    elif node_name in scene.geometry:
                        geometry = scene.geometry[node_name]
                    
                    if geometry is None or not isinstance(geometry, trimesh.Trimesh):
                        continue
                    
                    mesh_world = geometry.copy()
                    if not np.allclose(transform, np.eye(4)):
                        mesh_world.apply_transform(transform)
                    
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_world.vertices.astype(np.float64))
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_world.faces.astype(np.int32))
                    
                    # Handle vertex colors (for segmented meshes)
                    colors_found = False
                    if hasattr(mesh_world.visual, 'vertex_colors') and mesh_world.visual.vertex_colors is not None:
                        vcolors = np.array(mesh_world.visual.vertex_colors)
                        if len(vcolors) == len(mesh_world.vertices) and vcolors.shape[1] >= 3:
                            colors = vcolors[:, :3].astype(np.float64)
                            if colors.max() > 1.0:
                                colors = colors / 255.0
                            if not np.allclose(colors[0], colors).all():
                                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
                                colors_found = True
                    
                    if not colors_found:
                        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                    
                    o3d_mesh.compute_vertex_normals()
                    o3d_meshes.append(o3d_mesh)
                    
                except Exception as e:
                    if debug:
                        print(f"  Warning: Failed to process geometry {node_name}: {e}")
                    continue
        
        elif isinstance(scene, trimesh.Trimesh):
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(scene.vertices.astype(np.float64))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(scene.faces.astype(np.int32))
            
            if hasattr(scene.visual, 'vertex_colors') and scene.visual.vertex_colors is not None:
                colors = np.array(scene.visual.vertex_colors)[:, :3].astype(np.float64)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            else:
                o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            
            o3d_mesh.compute_vertex_normals()
            o3d_meshes.append(o3d_mesh)
        
        if not o3d_meshes:
            raise ValueError(f"No valid meshes found in {mesh_path}")
        
        # Combine all meshes
        combined = o3d_meshes[0]
        for m in o3d_meshes[1:]:
            combined += m
        
        combined.compute_vertex_normals()
        
        if debug:
            print(f"  Total: {len(combined.vertices)} vertices, {len(combined.triangles)} triangles")
        
        return combined
    
    def render_layout_orthographic(
        self,
        mesh: o3d.geometry.TriangleMesh,
        scene_bbox: Dict,
        output_path: Path,
        margin_percent: float = 5.0
    ) -> str:
        """
        Render orthographic top-down layout view.
        
        Args:
            mesh: Open3D TriangleMesh
            scene_bbox: Scene bounding box dict with 'min' and 'max' keys
            output_path: Output image path
            margin_percent: Margin percentage around scene
            
        Returns:
            Path to saved image
        """
        if len(mesh.triangles) == 0:
            print(f"    Warning: No triangles in mesh")
            return None
        
        # Calculate view bounds
        min_bound = np.array(scene_bbox['min'])
        max_bound = np.array(scene_bbox['max'])
        target_center = (min_bound + max_bound) / 2
        target_size = max_bound - min_bound
        
        # Calculate camera position (above scene, looking down)
        max_extent = max(target_size[0], target_size[2])
        margin_factor = 1.0 + (margin_percent / 100.0)
        view_size = max_extent * margin_factor
        
        # Camera setup: orthographic top-down
        camera_height = max(target_size[1], view_size) * 1.5  # Height above scene
        eye = target_center + np.array([0, camera_height, 0])  # Above scene
        center = target_center  # Look at scene center
        up = np.array([0, 0, -1])  # Up direction (negative Z)
        
        # Set up orthographic camera using PinholeCameraParameters
        camera_params = o3d.camera.PinholeCameraParameters()
        
        # Orthographic projection: use very large focal length
        fx = fy = 100000.0  # Large value for orthographic
        cx = self.width / 2.0
        cy = self.height / 2.0
        camera_params.intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)
        
        # Set camera extrinsic (look-at matrix)
        look_at = self._look_at_matrix(eye, center, up)
        camera_params.extrinsic = look_at
        
        # Apply camera to scene (try different API methods for compatibility)
        camera_set = False
        try:
            # Method 1: Direct camera setup (Open3D 0.13+)
            if hasattr(self.scene.camera, 'set_projection'):
                self.scene.camera.set_projection(camera_params.intrinsic, camera_params.extrinsic, self.width, self.height)
                camera_set = True
        except (AttributeError, TypeError):
            pass
        
        if not camera_set:
            try:
                # Method 2: Setup camera via renderer (alternative API)
                if hasattr(self.renderer, 'setup_camera'):
                    self.renderer.setup_camera(camera_params.intrinsic, camera_params.extrinsic)
                    camera_set = True
            except (AttributeError, TypeError):
                pass
        
        if not camera_set:
            # Method 3: Fallback - may not work perfectly
            print("    Warning: Using fallback camera setup for layout - results may vary")
            try:
                if hasattr(self.scene.camera, 'set_model_matrix'):
                    self.scene.camera.set_model_matrix(look_at)
            except AttributeError:
                pass
        
        # Clear scene and add mesh
        self.scene.clear_geometries()
        self.scene.add_geometry("mesh", mesh, self._get_material())
        
        # Render
        image = self.renderer.render_to_image()
        
        # Save image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_image(str(output_path), image)
        
        print(f"  Saved layout: {output_path}")
        return str(output_path)
    
    def render_pov(
        self,
        mesh: o3d.geometry.TriangleMesh,
        eye: np.ndarray,
        center: np.ndarray,
        up: np.ndarray,
        output_path: Path,
        fov_deg: float = 70.0
    ) -> str:
        """
        Render perspective POV image.
        
        Args:
            mesh: Open3D TriangleMesh
            eye: Camera eye position (3D point)
            center: Camera look-at point (3D point)
            up: Camera up vector
            output_path: Output image path
            fov_deg: Field of view in degrees
            
        Returns:
            Path to saved image
        """
        if len(mesh.triangles) == 0:
            print(f"    Warning: No triangles in mesh")
            return None
        
        # Calculate perspective projection
        fov_rad = math.radians(fov_deg)
        fx = (0.5 * self.width) / math.tan(fov_rad / 2.0)
        fy = (0.5 * self.height) / math.tan(fov_rad / 2.0)
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        # Set up perspective camera using PinholeCameraParameters
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)
        
        # Set camera extrinsic (look-at matrix)
        look_at = self._look_at_matrix(eye, center, up)
        camera_params.extrinsic = look_at
        
        # Apply camera to scene (try different API methods for compatibility)
        camera_set = False
        try:
            # Method 1: Direct camera setup (Open3D 0.13+)
            if hasattr(self.scene.camera, 'set_projection'):
                self.scene.camera.set_projection(camera_params.intrinsic, camera_params.extrinsic, self.width, self.height)
                camera_set = True
        except (AttributeError, TypeError):
            pass
        
        if not camera_set:
            try:
                # Method 2: Setup camera via renderer (alternative API)
                if hasattr(self.renderer, 'setup_camera'):
                    self.renderer.setup_camera(camera_params.intrinsic, camera_params.extrinsic)
                    camera_set = True
            except (AttributeError, TypeError):
                pass
        
        if not camera_set:
            # Method 3: Fallback - may not work perfectly
            try:
                if hasattr(self.scene.camera, 'set_model_matrix'):
                    self.scene.camera.set_model_matrix(look_at)
            except AttributeError:
                pass
        
        # Clear scene and add mesh
        self.scene.clear_geometries()
        self.scene.add_geometry("mesh", mesh, self._get_material())
        
        # Render
        image = self.renderer.render_to_image()
        
        # Save image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_image(str(output_path), image)
        
        return str(output_path)
    
    def _look_at_matrix(self, eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Compute look-at transformation matrix."""
        eye = np.array(eye, dtype=np.float64)
        center = np.array(center, dtype=np.float64)
        up = np.array(up, dtype=np.float64)
        
        f = center - eye
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up / (np.linalg.norm(up) + 1e-12)
        
        s = np.cross(f, upn)
        s = s / (np.linalg.norm(s) + 1e-12)
        u = np.cross(s, f)
        
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        M[:3, 3] = -M[:3, :3] @ eye
        
        return M
    
    def _get_material(self) -> o3d.visualization.rendering.MaterialRecord:
        """Get material for rendering."""
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        mat.base_metallic = 0.0
        mat.base_roughness = 0.5
        return mat
    
    def draw_doors_on_layout(
        self,
        image_path: Path,
        doors: List[Dict],
        scene_bbox: Dict,
        margin_percent: float = 5.0
    ) -> None:
        """
        Draw door rectangles on layout image.
        
        Args:
            image_path: Path to layout PNG image
            doors: List of door metadata dicts with 'bbox' keys
            scene_bbox: Scene bounding box dict
            margin_percent: Margin percentage used in rendering
        """
        if not doors:
            return
        
        try:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            draw = ImageDraw.Draw(img)
            
            # Door colors
            fill_color = (194, 178, 128)  # Beige
            outline_color = (100, 90, 60)  # Dark brown
            
            # Calculate view bounds (same as in render_layout_orthographic)
            min_bound = np.array(scene_bbox['min'])
            max_bound = np.array(scene_bbox['max'])
            target_center = (min_bound + max_bound) / 2
            target_size = max_bound - min_bound
            max_extent = max(target_size[0], target_size[2])
            margin_factor = 1.0 + (margin_percent / 100.0)
            view_size = max_extent * margin_factor
            
            view_min_x = target_center[0] - (view_size / 2)
            view_max_x = target_center[0] + (view_size / 2)
            view_min_z = target_center[2] - (view_size / 2)
            view_max_z = target_center[2] + (view_size / 2)
            
            doors_drawn = 0
            for door in doors:
                bbox = door.get('bbox', {})
                if not bbox:
                    continue
                
                d_min = np.array(bbox['min'])
                d_max = np.array(bbox['max'])
                
                # Check if door is in view
                if (d_max[0] < view_min_x or d_min[0] > view_max_x or
                    d_max[2] < view_min_z or d_min[2] > view_max_z):
                    continue
                
                # Project 3D bbox to 2D image coordinates
                # X maps to image X, Z maps to image Y (top-down view)
                x0 = int((d_min[0] - view_min_x) / (view_max_x - view_min_x) * width)
                x1 = int((d_max[0] - view_min_x) / (view_max_x - view_min_x) * width)
                y0 = int((d_min[2] - view_min_z) / (view_max_z - view_min_z) * height)
                y1 = int((d_max[2] - view_min_z) / (view_max_z - view_min_z) * height)
                
                # Draw rectangle
                draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=2)
                doors_drawn += 1
            
            img.save(image_path)
            if doors_drawn > 0:
                print(f"    Overlaid {doors_drawn} doors on layout")
        
        except Exception as e:
            print(f"    Error drawing doors: {e}")
    
    def calculate_pov_positions(
        self,
        room_bbox: Dict,
        room_center: List[float],
        floor_origin: List[float],
        up_direction: List[float],
        num_views: int = 6,
        eye_height: float = 1.6
    ) -> List[Dict]:
        """
        Calculate strategic POV camera positions for a room.
        
        Args:
            room_bbox: Room bounding box dict with 'min' and 'max' keys
            room_center: Room center point [x, y, z]
            floor_origin: Floor plane origin [x, y, z]
            up_direction: Up direction vector [x, y, z]
            num_views: Number of POV views to generate
            eye_height: Eye height above floor in meters
            
        Returns:
            List of POV camera pose dicts with 'eye', 'center', 'up' keys
        """
        min_bound = np.array(room_bbox['min'])
        max_bound = np.array(room_bbox['max'])
        center = np.array(room_center)
        up = np.array(up_direction)
        floor_origin = np.array(floor_origin)
        
        # Calculate room corners in XY plane (top-down)
        corners_2d = [
            [min_bound[0], min_bound[2]],  # Bottom-left
            [max_bound[0], min_bound[2]],  # Bottom-right
            [max_bound[0], max_bound[2]],  # Top-right
            [min_bound[0], max_bound[2]],  # Top-left
        ]
        
        # Add center point
        center_2d = [center[0], center[2]]
        
        # Select positions: corners + center (up to num_views)
        positions_2d = corners_2d + [center_2d]
        if len(positions_2d) > num_views:
            # Select evenly spaced positions
            indices = np.linspace(0, len(positions_2d) - 1, num_views, dtype=int)
            positions_2d = [positions_2d[i] for i in indices]
        
        povs = []
        for idx, pos_2d in enumerate(positions_2d[:num_views]):
            # Calculate eye position: pos_2d at floor level + eye_height * up
            eye_floor = np.array([pos_2d[0], floor_origin[1], pos_2d[1]])
            eye = eye_floor + eye_height * up
            
            # Look toward room center (slightly above floor)
            center_point = center + 0.5 * eye_height * up
            
            # Calculate forward direction
            forward = center_point - eye
            forward = forward / (np.linalg.norm(forward) + 1e-12)
            
            # Up vector is room's up direction
            povs.append({
                'eye': eye.tolist(),
                'center': center_point.tolist(),
                'up': up.tolist(),
                'name': f'pov_{idx:03d}'
            })
        
        return povs


def process_scene(
    scene_id: str,
    metadata_path: Path,
    seg_mesh_path: Path,
    tex_mesh_path: Path,
    output_dir: Path,
    layout_resolution: int = 1024,
    pov_resolution: Tuple[int, int] = (1280, 800),
    num_pov_views: int = 6,
    eye_height: float = 1.6,
    fov_deg: float = 70.0,
    margin_percent: float = 5.0,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> bool:
    """
    Process a single scene: render layout and POV images.
    
    Args:
        scene_id: Scene identifier
        metadata_path: Path to scene metadata JSON
        seg_mesh_path: Path to segmented mesh (GLB/OBJ)
        tex_mesh_path: Path to textured mesh (GLB/OBJ)
        output_dir: Output directory for images
        layout_resolution: Layout image resolution (square)
        pov_resolution: POV image resolution (width, height)
        num_pov_views: Number of POV views per room
        eye_height: Eye height above floor in meters
        fov_deg: Field of view for POV images
        margin_percent: Margin percentage for layout view
        background_color: Background color (RGB, 0-1 range)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        scene_bbox = metadata.get('scene_bbox', {})
        doors = metadata.get('doors', [])
        rooms = metadata.get('rooms', [])
        up_direction = metadata.get('up_direction', [0, 0, 1])
        
        print(f"\nProcessing scene: {scene_id}")
        print(f"  Rooms: {len(rooms)}")
        print(f"  Doors: {len(doors)}")
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        layout_dir = output_dir / "layout"
        pov_dir = output_dir / "pov"
        layout_dir.mkdir(parents=True, exist_ok=True)
        pov_dir.mkdir(parents=True, exist_ok=True)
        
        # Load meshes
        print("  Loading meshes...")
        renderer_seg = SceneRenderer(layout_resolution, layout_resolution, background_color)
        renderer_tex = SceneRenderer(layout_resolution, layout_resolution, background_color)
        renderer_pov_seg = SceneRenderer(pov_resolution[0], pov_resolution[1], background_color)
        renderer_pov_tex = SceneRenderer(pov_resolution[0], pov_resolution[1], background_color)
        
        mesh_seg = renderer_seg.load_mesh_to_o3d(seg_mesh_path, debug=True)
        mesh_tex = renderer_tex.load_mesh_to_o3d(tex_mesh_path, debug=True)
        
        # Render layout views
        print("  Rendering layout views...")
        layout_seg_path = layout_dir / f"{scene_id}_layout_seg.png"
        layout_tex_path = layout_dir / f"{scene_id}_layout_tex.png"
        
        renderer_seg.render_layout_orthographic(mesh_seg, scene_bbox, layout_seg_path, margin_percent)
        renderer_tex.render_layout_orthographic(mesh_tex, scene_bbox, layout_tex_path, margin_percent)
        
        # Draw doors on layout images
        if doors:
            print("  Annotating doors on layout...")
            renderer_seg.draw_doors_on_layout(layout_seg_path, doors, scene_bbox, margin_percent)
            renderer_tex.draw_doors_on_layout(layout_tex_path, doors, scene_bbox, margin_percent)
        
        # Render POV views for each room
        print("  Rendering POV views...")
        pov_count = 0
        
        for room_idx, room in enumerate(rooms):
            room_id = room.get('room_id', room_idx)
            room_type = room.get('room_type', 'UnknownRoom')
            room_bbox = room.get('room_bbox', {})
            room_center = room.get('room_center', [0, 0, 0])
            floor_origin = room.get('floor_origin', [0, 0, 0])
            
            # Calculate POV positions
            povs = renderer_pov_seg.calculate_pov_positions(
                room_bbox, room_center, floor_origin, up_direction,
                num_views=num_pov_views, eye_height=eye_height
            )
            
            for pov in povs:
                pov_name = pov['name']
                eye = np.array(pov['eye'])
                center = np.array(pov['center'])
                up = np.array(pov['up'])
                
                # Render segmented POV
                pov_seg_path = pov_dir / f"{scene_id}_room{room_id}_{pov_name}_seg.png"
                renderer_pov_seg.render_pov(mesh_seg, eye, center, up, pov_seg_path, fov_deg)
                
                # Render textured POV
                pov_tex_path = pov_dir / f"{scene_id}_room{room_id}_{pov_name}_tex.png"
                renderer_pov_tex.render_pov(mesh_tex, eye, center, up, pov_tex_path, fov_deg)
                
                pov_count += 1
        
        print(f"\n  Completed: {pov_count} POV images rendered")
        return True
        
    except Exception as e:
        print(f"  Error processing scene {scene_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Headless 3D scene renderer using Open3D OffscreenRenderer"
    )
    parser.add_argument("--scene-id", type=str, help="Scene ID (if processing single scene)")
    parser.add_argument("--metadata", type=str, help="Path to scene metadata JSON")
    parser.add_argument("--seg-mesh", type=str, help="Path to segmented mesh (GLB/OBJ)")
    parser.add_argument("--tex-mesh", type=str, help="Path to textured mesh (GLB/OBJ)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for images")
    parser.add_argument("--stage1-output", type=str, help="Stage1 output directory (auto-detect meshes/metadata)")
    
    # Rendering parameters
    parser.add_argument("--layout-resolution", type=int, default=1024, help="Layout image resolution (square)")
    parser.add_argument("--pov-width", type=int, default=1280, help="POV image width")
    parser.add_argument("--pov-height", type=int, default=800, help="POV image height")
    parser.add_argument("--num-pov-views", type=int, default=6, help="Number of POV views per room")
    parser.add_argument("--eye-height", type=float, default=1.6, help="Eye height above floor (meters)")
    parser.add_argument("--fov", type=float, default=70.0, help="Field of view for POV images (degrees)")
    parser.add_argument("--margin", type=float, default=5.0, help="Margin percentage for layout view")
    parser.add_argument("--background", type=str, default="1.0,1.0,1.0", help="Background color (RGB, 0-1 range)")
    
    args = parser.parse_args()
    
    # Parse background color
    try:
        bg_color = tuple(float(x) for x in args.background.split(','))
        if len(bg_color) != 3:
            raise ValueError("Background color must have 3 components")
    except:
        bg_color = (1.0, 1.0, 1.0)
        print(f"Warning: Invalid background color, using default: {bg_color}")
    
    # Auto-detect from stage1 output if provided
    if args.stage1_output:
        stage1_dir = Path(args.stage1_output)
        if not stage1_dir.exists():
            print(f"Error: Stage1 output directory not found: {stage1_dir}")
            return 1
        
        # Find all scene metadata files
        metadata_dir = stage1_dir / "metadatas"
        if not metadata_dir.exists():
            print(f"Error: Metadata directory not found: {metadata_dir}")
            return 1
        
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        if not metadata_files:
            print(f"Warning: No metadata files found in {metadata_dir}")
            return 0
        
        print(f"Found {len(metadata_files)} scenes to process")
        
        # Process each scene
        success_count = 0
        for metadata_path in metadata_files:
            scene_id = metadata_path.stem.replace("_metadata", "")
            
            # Find corresponding mesh files
            seg_mesh_path = stage1_dir / "geometry" / "seg" / f"{scene_id}.glb"
            if not seg_mesh_path.exists():
                seg_mesh_path = stage1_dir / "geometry" / "seg" / f"{scene_id}.obj"
            
            tex_mesh_path = stage1_dir / "geometry" / "tex" / f"{scene_id}.glb"
            if not tex_mesh_path.exists():
                tex_mesh_path = stage1_dir / "geometry" / "tex" / f"{scene_id}.obj"
            
            if not seg_mesh_path.exists() or not tex_mesh_path.exists():
                print(f"  Skipping {scene_id}: mesh files not found")
                continue
            
            output_dir = Path(args.output_dir) / scene_id
            
            success = process_scene(
                scene_id, metadata_path, seg_mesh_path, tex_mesh_path, output_dir,
                args.layout_resolution, (args.pov_width, args.pov_height),
                args.num_pov_views, args.eye_height, args.fov, args.margin, bg_color
            )
            
            if success:
                success_count += 1
        
        print(f"\nSuccessfully processed {success_count}/{len(metadata_files)} scenes")
        return 0
    
    # Process single scene
    if not args.metadata or not args.seg_mesh or not args.tex_mesh:
        parser.error("Either --stage1-output or --metadata/--seg-mesh/--tex-mesh must be provided")
    
    scene_id = args.scene_id or Path(args.metadata).stem.replace("_metadata", "")
    output_dir = Path(args.output_dir)
    
    success = process_scene(
        scene_id, Path(args.metadata), Path(args.seg_mesh), Path(args.tex_mesh), output_dir,
        args.layout_resolution, (args.pov_width, args.pov_height),
        args.num_pov_views, args.eye_height, args.fov, args.margin, bg_color
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

