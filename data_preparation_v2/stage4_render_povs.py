#!/usr/bin/env python3
"""
Stage 4: POV Rendering (HPC-Fixed Version)

Key fixes for HPC headless rendering:
1. Set PYOPENGL_PLATFORM and DISPLAY environment variables BEFORE any OpenGL imports
2. Suppress XDG_RUNTIME_DIR warnings
3. Better error handling for rendering backends
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CRITICAL: Set environment variables BEFORE any OpenGL-related imports
# This must happen at module load time, before numpy/trimesh/pyrender
# ============================================================================

def setup_environment_early(hpc_mode: bool = False, backend: str = "auto"):
    """
    Set up environment variables before importing OpenGL-dependent libraries.
    Must be called BEFORE importing trimesh, pyrender, etc.
    """
    if not hpc_mode:
        return
    
    # Suppress the XDG_RUNTIME_DIR warning by setting a valid directory
    if "XDG_RUNTIME_DIR" not in os.environ or not os.environ["XDG_RUNTIME_DIR"]:
        runtime_dir = os.path.expanduser("~/.cache/xdg-runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        os.environ["XDG_RUNTIME_DIR"] = runtime_dir
    
    # Set Mesa to use software rendering as fallback
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
    
    # Disable GPU for EGL (use software)
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ""


# Parse args early to check for --hpc flag
_early_args = sys.argv[1:]
_hpc_mode = "--hpc" in _early_args
_backend = "auto"
for i, arg in enumerate(_early_args):
    if arg == "--backend" and i + 1 < len(_early_args):
        _backend = _early_args[i + 1]

setup_environment_early(_hpc_mode, _backend)

# Now safe to import the rest
import numpy as np
import trimesh
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global Xvfb display reference
_xvfb_display = None
_renderer_initialized = False


def setup_hpc_rendering(backend: str = "auto") -> bool:
    """
    Set up rendering backend for HPC headless rendering.
    
    Args:
        backend: One of "auto", "egl", "osmesa", "xvfb"
    
    Returns True if successful, False otherwise.
    """
    global _xvfb_display, _renderer_initialized
    
    if _renderer_initialized:
        return True
    
    if backend == "auto":
        # Try backends in order: osmesa (most reliable CPU), then xvfb, then egl
        for try_backend in ["osmesa", "xvfb", "egl"]:
            if setup_hpc_rendering(try_backend):
                return True
        return False
    
    elif backend == "osmesa":
        try:
            # OSMesa is the most reliable for CPU-only rendering
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            
            # Force reimport of pyrender with new settings
            for mod in list(sys.modules.keys()):
                if "pyrender" in mod or "OpenGL" in mod:
                    del sys.modules[mod]
            
            import pyrender
            renderer = pyrender.OffscreenRenderer(64, 64)
            renderer.delete()
            logger.info("Using OSMesa backend (CPU software rendering)")
            _renderer_initialized = True
            return True
        except Exception as e:
            logger.debug(f"OSMesa backend failed: {e}")
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
            return False
    
    elif backend == "xvfb":
        try:
            from xvfbwrapper import Xvfb
            _xvfb_display = Xvfb(width=1920, height=1080, colordepth=24)
            _xvfb_display.start()
            
            # Set DISPLAY environment variable
            os.environ["DISPLAY"] = f":{_xvfb_display.new_display}"
            logger.info(f"Xvfb started on display :{_xvfb_display.new_display}")
            
            # Don't set PYOPENGL_PLATFORM for Xvfb - let it use GLX
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
            
            # Force reimport
            for mod in list(sys.modules.keys()):
                if "pyrender" in mod or "OpenGL" in mod:
                    del sys.modules[mod]
            
            # Test if pyrender works
            try:
                import pyrender
                renderer = pyrender.OffscreenRenderer(64, 64)
                renderer.delete()
                logger.info("Xvfb backend working with pyrender")
                _renderer_initialized = True
                return True
            except Exception as e:
                logger.warning(f"Xvfb started but pyrender failed: {e}")
                _xvfb_display.stop()
                _xvfb_display = None
                return False
                
        except ImportError:
            logger.debug("xvfbwrapper not installed")
            return False
        except Exception as e:
            logger.debug(f"Xvfb backend failed: {e}")
            return False
    
    elif backend == "egl":
        try:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            
            # Force reimport
            for mod in list(sys.modules.keys()):
                if "pyrender" in mod or "OpenGL" in mod:
                    del sys.modules[mod]
            
            import pyrender
            renderer = pyrender.OffscreenRenderer(64, 64)
            renderer.delete()
            logger.info("Using EGL backend (GPU headless)")
            _renderer_initialized = True
            return True
        except Exception as e:
            logger.debug(f"EGL backend failed: {e}")
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
            return False
    
    return False


def cleanup_hpc_rendering():
    """Clean up any resources used by the rendering backend."""
    global _xvfb_display, _renderer_initialized
    if _xvfb_display is not None:
        try:
            _xvfb_display.stop()
            logger.info("Xvfb stopped")
        except Exception:
            pass
        _xvfb_display = None
    _renderer_initialized = False


def load_scene_list(scene_list_path: Path) -> List[str]:
    """Load scene IDs from a text file (one per line)."""
    scenes = []
    with open(scene_list_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scenes.append(scene_id)
    return scenes


def check_scene_povs_exist(scene_id: str, output_dir: Path, rooms_metadata: list) -> bool:
    """Check if POV outputs already exist for a scene."""
    if not rooms_metadata:
        return False
    
    for room_meta in rooms_metadata:
        room_id = room_meta.get("room_id", room_meta.get("room_type", "Unknown"))
        
        # Check for at least one POV per room (door0 or window0)
        tex_pov = output_dir / "tex" / f"{scene_id}_{room_id}_door0_tex_pov.png"
        if not tex_pov.exists():
            tex_pov = output_dir / "tex" / f"{scene_id}_{room_id}_window0_tex_pov.png"
        
        if not tex_pov.exists():
            return False
    
    return True


# ============================================================================
# GLB Loading
# ============================================================================

def load_glb_with_transforms(glb_path: Path) -> List[trimesh.Trimesh]:
    """Load GLB and return meshes with transforms applied."""
    scene = trimesh.load(str(glb_path), process=False, force="scene")
    
    if not isinstance(scene, trimesh.Scene):
        return [scene] if isinstance(scene, trimesh.Trimesh) else []
    
    try:
        meshes = scene.dump(concatenate=False)
        if meshes is None:
            meshes = []
        meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        return meshes
    except Exception as e:
        logger.warning(f"scene.dump() failed: {e}")
        return []


# ============================================================================
# Camera Computation
# ============================================================================

def compute_door_center(door: Dict) -> Optional[np.ndarray]:
    """Get the center position of a door from its bbox."""
    bbox = door.get("bbox")
    if not bbox or "min" not in bbox or "max" not in bbox:
        pos = door.get("position")
        if pos and len(pos) >= 3:
            return np.array(pos, dtype=np.float64)
        return None
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    return (bbox_min + bbox_max) / 2.0


def compute_door_normal(door: Dict) -> np.ndarray:
    """Compute the normal direction of a door."""
    bbox = door.get("bbox")
    if not bbox:
        return np.array([0.0, 0.0, 1.0])
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    size = bbox_max - bbox_min
    
    if size[0] < size[2]:
        return np.array([1.0, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, 1.0])


def find_room_door(room_meta: Dict, scene_doors: List[Dict]) -> Optional[Dict]:
    """Find a door that belongs to this room - must be ON the room boundary."""
    room_doors = room_meta.get("doors", [])
    if room_doors:
        return room_doors[0]
    
    bbox = room_meta.get("bbox", {})
    if not bbox or "min" not in bbox:
        return None
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    room_center = (bbox_min + bbox_max) / 2.0
    
    best_door = None
    best_score = float('inf')
    
    for door in scene_doors:
        door_center = compute_door_center(door)
        if door_center is None:
            continue
        
        # Check if door is near room boundary
        on_x_min = abs(door_center[0] - bbox_min[0]) < 0.5
        on_x_max = abs(door_center[0] - bbox_max[0]) < 0.5
        in_z_range = bbox_min[2] - 0.5 <= door_center[2] <= bbox_max[2] + 0.5
        
        on_z_min = abs(door_center[2] - bbox_min[2]) < 0.5
        on_z_max = abs(door_center[2] - bbox_max[2]) < 0.5
        in_x_range = bbox_min[0] - 0.5 <= door_center[0] <= bbox_max[0] + 0.5
        
        is_on_boundary = ((on_x_min or on_x_max) and in_z_range) or \
                         ((on_z_min or on_z_max) and in_x_range)
        
        if is_on_boundary:
            dist = np.linalg.norm(door_center[[0,2]] - room_center[[0,2]])
            if dist < best_score:
                best_score = dist
                best_door = door
    
    return best_door


def compute_pov_camera(
    room_meta: Dict,
    door: Optional[Dict],
    camera_height: float = 1.5,
    fov: float = 60.0
) -> Optional[Dict]:
    """
    POV camera at door, looking at room center.
    """
    bbox = room_meta.get("bbox", {})
    if not bbox or "min" not in bbox:
        return None
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    room_center = (bbox_min + bbox_max) / 2.0
    
    if door is not None:
        door_center = compute_door_center(door)
        if door_center is not None:
            eye = door_center.copy()
            eye[1] = camera_height
            
            look_at = room_center.copy()
            look_at[1] = 1.0
            
            return {
                "eye": eye.tolist(),
                "center": look_at.tolist(),
                "up": [0.0, 1.0, 0.0],
                "fov": fov
            }
    
    # Fallback
    eye = np.array([bbox_min[0], camera_height, (bbox_min[2] + bbox_max[2]) / 2])
    look_at = room_center.copy()
    look_at[1] = 1.0
    
    return {
        "eye": eye.tolist(),
        "center": look_at.tolist(),
        "up": [0.0, 1.0, 0.0],
        "fov": fov
    }


# ============================================================================
# Rendering
# ============================================================================

def get_mesh_color(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract vertex colors from mesh."""
    n_vertices = len(mesh.vertices)
    default_color = np.full((n_vertices, 3), 180, dtype=np.uint8)
    
    if not hasattr(mesh, 'visual'):
        return default_color
    
    visual = mesh.visual
    
    if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
        colors = np.array(visual.vertex_colors)
        if len(colors) == n_vertices and colors.shape[-1] >= 3:
            return colors[:, :3].astype(np.uint8)
    
    try:
        if hasattr(visual, 'to_color'):
            color_visual = visual.to_color()
            if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                colors = np.array(color_visual.vertex_colors)
                if len(colors) == n_vertices and colors.shape[-1] >= 3:
                    return colors[:, :3].astype(np.uint8)
    except Exception:
        pass
    
    if hasattr(visual, 'material') and visual.material is not None:
        mat = visual.material
        for attr in ['diffuse', 'baseColorFactor', 'main_color']:
            if hasattr(mat, attr):
                color = getattr(mat, attr)
                if color is not None:
                    color = np.array(color).flatten()
                    if len(color) >= 3:
                        if color.max() <= 1.0:
                            color = (color * 255).astype(np.uint8)
                        return np.tile(color[:3].astype(np.uint8), (n_vertices, 1))
    
    return default_color


def try_render_pyrender(
    meshes: List[trimesh.Trimesh],
    camera: Dict,
    width: int,
    height: int
) -> Optional[Image.Image]:
    """Try to render using pyrender."""
    try:
        import pyrender
        from pyrender import OffscreenRenderer
    except ImportError:
        logger.warning("pyrender not installed")
        return None
    
    try:
        eye = np.array(camera["eye"])
        center = np.array(camera["center"])
        up = np.array(camera["up"])
        fov = camera.get("fov", 60.0)
        
        scene = pyrender.Scene(bg_color=[0.53, 0.81, 0.92, 1.0], ambient_light=[0.4, 0.4, 0.4])
        
        for mesh in meshes:
            if len(mesh.vertices) == 0:
                continue
            try:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene.add(pr_mesh)
            except Exception:
                continue
        
        cam = pyrender.PerspectiveCamera(yfov=np.radians(fov), aspectRatio=width/height)
        
        forward = center - eye
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-9)
        cam_up = np.cross(right, forward)
        
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = cam_up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = eye
        
        scene.add(cam, pose=cam_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=cam_pose)
        
        renderer = OffscreenRenderer(width, height)
        color, _ = renderer.render(scene)
        renderer.delete()
        
        return Image.fromarray(color)
        
    except Exception as e:
        logger.warning(f"pyrender failed: {e}")
        return None


def render_fast_fallback(
    meshes: List[trimesh.Trimesh],
    camera: Dict,
    width: int,
    height: int
) -> Image.Image:
    """Fast software fallback renderer using trimesh's built-in rendering."""
    
    eye = np.array(camera["eye"])
    center = np.array(camera["center"])
    up = np.array(camera["up"])
    fov = camera.get("fov", 60.0)
    
    # Build view matrix
    forward = center - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    cam_up = np.cross(right, forward)
    
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = cam_up
    view_matrix[:3, 2] = -forward
    view_matrix[:3, 3] = eye
    view_matrix = np.linalg.inv(view_matrix)
    
    # Build projection matrix
    aspect = width / height
    fov_rad = np.radians(fov)
    f = 1.0 / np.tan(fov_rad / 2.0)
    near, far = 0.1, 100.0
    
    proj_matrix = np.zeros((4, 4))
    proj_matrix[0, 0] = f / aspect
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0
    
    # Create output image
    img = Image.new("RGB", (width, height), (135, 206, 235))  # Sky blue background
    draw = ImageDraw.Draw(img)
    z_buffer = np.full((height, width), float('inf'))
    
    for mesh in meshes:
        if len(mesh.vertices) == 0:
            continue
        
        vertices = mesh.vertices
        colors = get_mesh_color(mesh)
        
        # Transform vertices
        ones = np.ones((len(vertices), 1))
        verts_h = np.hstack([vertices, ones])
        
        # View transform
        verts_view = (view_matrix @ verts_h.T).T
        
        # Projection
        verts_proj = (proj_matrix @ verts_view.T).T
        
        # Perspective divide
        w = verts_proj[:, 3:4]
        w[w == 0] = 1e-9
        verts_ndc = verts_proj[:, :3] / w
        
        # To screen coordinates
        screen_x = ((verts_ndc[:, 0] + 1) * 0.5 * width).astype(int)
        screen_y = ((1 - verts_ndc[:, 1]) * 0.5 * height).astype(int)
        depth = verts_ndc[:, 2]
        
        # Draw triangles
        for face in mesh.faces:
            i0, i1, i2 = face
            
            # Skip if behind camera
            if depth[i0] < -1 or depth[i1] < -1 or depth[i2] < -1:
                continue
            if depth[i0] > 1 or depth[i1] > 1 or depth[i2] > 1:
                continue
            
            x0, y0 = screen_x[i0], screen_y[i0]
            x1, y1 = screen_x[i1], screen_y[i1]
            x2, y2 = screen_x[i2], screen_y[i2]
            
            # Skip degenerate or off-screen
            if (x0 == x1 == x2) or (y0 == y1 == y2):
                continue
            if max(x0, x1, x2) < 0 or min(x0, x1, x2) >= width:
                continue
            if max(y0, y1, y2) < 0 or min(y0, y1, y2) >= height:
                continue
            
            # Average color
            color = tuple(int(c) for c in colors[i0])
            
            # Simple depth test
            avg_depth = (depth[i0] + depth[i1] + depth[i2]) / 3
            cx = (x0 + x1 + x2) // 3
            cy = (y0 + y1 + y2) // 3
            
            if 0 <= cx < width and 0 <= cy < height:
                if avg_depth < z_buffer[cy, cx]:
                    z_buffer[cy, cx] = avg_depth
                    try:
                        draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill=color)
                    except Exception:
                        pass
    
    return img


def render_pov(
    meshes: List[trimesh.Trimesh],
    camera: Dict,
    width: int,
    height: int,
    use_pyrender: bool = True
) -> Image.Image:
    """Render POV using available renderer."""
    if use_pyrender:
        result = try_render_pyrender(meshes, camera, width, height)
        if result is not None:
            return result
    
    return render_fast_fallback(meshes, camera, width, height)


def process_one_scene(
    scene_id: str,
    tex_glb_path: Path,
    seg_glb_path: Path,
    scene_meta: Dict,
    rooms_metadata: List[Dict],
    output_dir: Path,
    width: int,
    height: int,
    fov: float,
    use_pyrender: bool
) -> Tuple[bool, Optional[str]]:
    """Process one scene and generate POV renders."""
    
    try:
        # Load meshes
        tex_meshes = load_glb_with_transforms(tex_glb_path)
        seg_meshes = load_glb_with_transforms(seg_glb_path)
        
        if not tex_meshes:
            return False, "No textured meshes loaded"
        
        scene_doors = scene_meta.get("doors", [])
        scene_windows = scene_meta.get("windows", [])
        
        # Process each room's door
        for room_meta in rooms_metadata:
            room_id = room_meta.get("room_id", room_meta.get("room_type", "Unknown"))
            
            door = find_room_door(room_meta, scene_doors)
            if door is None:
                continue
            
            logger.info(f"  {room_id} - rendering door POV")
            
            camera = compute_pov_camera(room_meta, door, fov=fov)
            if camera is None:
                continue
            
            tex_pov = render_pov(tex_meshes, camera, width, height, use_pyrender)
            seg_pov = render_pov(seg_meshes, camera, width, height, use_pyrender)
            
            tex_output = output_dir / "tex" / f"{scene_id}_{room_id}_door0_tex_pov.png"
            seg_output = output_dir / "seg" / f"{scene_id}_{room_id}_door0_seg_pov.png"
            tex_output.parent.mkdir(parents=True, exist_ok=True)
            seg_output.parent.mkdir(parents=True, exist_ok=True)
            tex_pov.save(tex_output)
            seg_pov.save(seg_output)
        
        # Process all windows
        for win_idx, window in enumerate(scene_windows):
            window_center = compute_door_center(window)
            if window_center is None:
                continue
            
            # Find closest room
            best_room = None
            best_dist = float('inf')
            for room_meta in rooms_metadata:
                bbox = room_meta.get("bbox", {})
                if not bbox or "min" not in bbox:
                    continue
                bbox_min = np.array(bbox["min"])
                bbox_max = np.array(bbox["max"])
                room_center = (bbox_min + bbox_max) / 2.0
                dist = np.linalg.norm(window_center[[0,2]] - room_center[[0,2]])
                if dist < best_dist:
                    best_dist = dist
                    best_room = room_meta
            
            if best_room is None:
                continue
            
            room_id = best_room.get("room_id", best_room.get("room_type", "Unknown"))
            logger.info(f"  Window {win_idx} -> {room_id}")
            
            camera = compute_pov_camera(best_room, window, fov=fov)
            if camera is None:
                continue
            
            tex_pov = render_pov(tex_meshes, camera, width, height, use_pyrender)
            seg_pov = render_pov(seg_meshes, camera, width, height, use_pyrender)
            
            tex_output = output_dir / "tex" / f"{scene_id}_{room_id}_window{win_idx}_tex_pov.png"
            seg_output = output_dir / "seg" / f"{scene_id}_{room_id}_window{win_idx}_seg_pov.png"
            tex_output.parent.mkdir(parents=True, exist_ok=True)
            seg_output.parent.mkdir(parents=True, exist_ok=True)
            tex_pov.save(tex_output)
            seg_pov.save(seg_output)
        
        return True, None
    
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Render POVs")
    parser.add_argument("--dataset-root", default=None, help="Root directory of dataset")
    parser.add_argument("--geometry-dir", default=None, help="Directory containing geometry GLB files")
    parser.add_argument("--metadata-dir", default=None, help="Directory containing metadata JSON files")
    parser.add_argument("--output-dir", default=None, help="Output directory for POV images")
    parser.add_argument("--scene-list", type=str, default=None, help="Path to text file with scene IDs")
    parser.add_argument("--width", type=int, default=1280, help="Output image width")
    parser.add_argument("--height", type=int, default=720, help="Output image height")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera field of view in degrees")
    parser.add_argument("--no-pyrender", action="store_true", help="Disable pyrender, use software fallback")
    parser.add_argument("--hpc", action="store_true", help="Enable HPC mode with headless rendering")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes that already have output files")
    parser.add_argument("--backend", type=str, default="auto", 
                        choices=["auto", "egl", "osmesa", "xvfb"],
                        help="Rendering backend: auto, egl, osmesa, or xvfb")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes to process")
    args = parser.parse_args()
    
    # Derive paths
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        geometry_dir = Path(args.geometry_dir) if args.geometry_dir else dataset_root / "geometry"
        metadata_dir = Path(args.metadata_dir) if args.metadata_dir else dataset_root / "metadata"
        output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "povs"
    else:
        if not args.geometry_dir:
            parser.error("--geometry-dir is required when --dataset-root is not provided")
        if not args.metadata_dir:
            parser.error("--metadata-dir is required when --dataset-root is not provided")
        if not args.output_dir:
            parser.error("--output-dir is required when --dataset-root is not provided")
        geometry_dir = Path(args.geometry_dir)
        metadata_dir = Path(args.metadata_dir)
        output_dir = Path(args.output_dir)
    
    use_pyrender = not args.no_pyrender
    
    # Set up HPC rendering if requested
    if args.hpc and use_pyrender:
        if not setup_hpc_rendering(args.backend):
            logger.warning("No HPC rendering backend available, falling back to software renderer")
            use_pyrender = False
    
    try:
        # Determine which scenes to process
        if args.scene_list:
            scene_list_path = Path(args.scene_list)
            if not scene_list_path.exists():
                logger.error(f"Scene list not found: {scene_list_path}")
                return
            
            scene_ids = load_scene_list(scene_list_path)
            logger.info(f"Loaded {len(scene_ids)} scene IDs from {scene_list_path}")
        else:
            scene_meta_files = list((metadata_dir / "scenes").glob("*.json"))
            if not scene_meta_files:
                logger.error(f"No scene metadata found in {metadata_dir / 'scenes'}")
                return
            scene_ids = [f.stem for f in scene_meta_files]
            logger.info(f"Discovered {len(scene_ids)} scenes from metadata directory")
        
        if args.limit:
            scene_ids = scene_ids[:args.limit]
        
        total_scenes = len(scene_ids)
        logger.info(f"Processing {total_scenes} scenes...")
        
        import time
        start_time = time.time()
        success_count = 0
        skip_count = 0
        
        for i, scene_id in enumerate(scene_ids, 1):
            scene_meta_path = metadata_dir / "scenes" / f"{scene_id}.json"
            if not scene_meta_path.exists():
                logger.warning(f"Scene metadata not found: {scene_meta_path}")
                skip_count += 1
                continue
            
            with open(scene_meta_path, "r") as f:
                scene_meta = json.load(f)
            
            tex_glb = geometry_dir / "tex" / f"{scene_id}_tex.glb"
            seg_glb = geometry_dir / "seg" / f"{scene_id}_seg.glb"
            
            if not tex_glb.exists() or not seg_glb.exists():
                logger.warning(f"GLB not found for {scene_id}")
                skip_count += 1
                continue
            
            # Load room metadata
            rooms_metadata = []
            rooms_dir = metadata_dir / "rooms"
            if rooms_dir.exists():
                for room_meta_path in rooms_dir.glob(f"{scene_id}_*.json"):
                    with open(room_meta_path, "r") as f:
                        room_data = json.load(f)
                        if room_data.get("scene_id") == scene_id:
                            rooms_metadata.append(room_data)
            
            if not rooms_metadata:
                logger.warning(f"No room metadata for {scene_id}")
                skip_count += 1
                continue
            
            # Skip if outputs exist
            if args.skip_existing:
                if check_scene_povs_exist(scene_id, output_dir, rooms_metadata):
                    logger.info(f"[{i}/{len(scene_ids)}] ⏭ {scene_id} (exists)")
                    skip_count += 1
                    continue
            
            success, error = process_one_scene(
                scene_id, tex_glb, seg_glb, scene_meta, rooms_metadata,
                output_dir, args.width, args.height, args.fov, use_pyrender
            )
            
            if success:
                success_count += 1
            
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_scenes - i) * avg_time
            eta_hours = remaining / 3600
            
            status = '✓' if success else '✗'
            logger.info(f"[{i}/{total_scenes}] {status} {scene_id} | ETA: {eta_hours:.1f}h")
        
        total_time = time.time() - start_time
        logger.info(f"\nDone: {success_count}/{total_scenes} succeeded, {skip_count} skipped in {total_time/3600:.1f} hours")
    
    finally:
        if args.hpc:
            cleanup_hpc_rendering()


if __name__ == "__main__":
    main()