#!/usr/bin/env python3
"""
Stage 3+4 Combined: POV-Oriented Layout Rendering

Instead of rendering layouts in world orientation and then rotating,
this script renders layouts directly from each POV camera orientation.

For each door/window:
1. Determine which wall it's on
2. Render layout with rotation so that door/window is at bottom-center
3. Camera is always at bottom-center, looking up into the room

Benefits:
- No image quality loss from rotation
- No complex post-processing
- Layout is rendered exactly as the camera sees it
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import trimesh
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_xvfb_display = None


# ============================================================================
# HPC Setup
# ============================================================================

def setup_hpc_rendering(backend: str = "auto") -> bool:
    """Set up rendering backend for HPC headless rendering."""
    global _xvfb_display
    
    if backend == "auto":
        for try_backend in ["xvfb", "osmesa", "egl"]:
            if setup_hpc_rendering(try_backend):
                return True
        return False
    
    elif backend == "xvfb":
        try:
            from xvfbwrapper import Xvfb
            _xvfb_display = Xvfb(width=1280, height=720)
            _xvfb_display.start()
            logger.info(f"Xvfb started on display :{_xvfb_display.new_display}")
            return True
        except ImportError:
            logger.debug("xvfbwrapper not installed")
            return False
        except Exception as e:
            logger.debug(f"Xvfb backend failed: {e}")
            return False
    
    elif backend == "egl":
        try:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            import pyrender
            renderer = pyrender.OffscreenRenderer(64, 64)
            renderer.delete()
            logger.info("Using EGL backend")
            return True
        except Exception as e:
            logger.debug(f"EGL backend failed: {e}")
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
            return False
    
    elif backend == "osmesa":
        try:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            import pyrender
            renderer = pyrender.OffscreenRenderer(64, 64)
            renderer.delete()
            logger.info("Using OSMesa backend")
            return True
        except Exception as e:
            logger.debug(f"OSMesa backend failed: {e}")
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
            return False
    
    return False


def cleanup_hpc_rendering():
    global _xvfb_display
    if _xvfb_display is not None:
        try:
            _xvfb_display.stop()
        except Exception:
            pass
        _xvfb_display = None


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
        return [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
    except Exception:
        return []


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


def clip_mesh_to_bbox(
    mesh: trimesh.Trimesh,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray
) -> Optional[trimesh.Trimesh]:
    """Clip mesh to bounding box (XZ plane only)."""
    if len(mesh.vertices) == 0:
        return None
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    margin = 0.1
    inside_x = (vertices[:, 0] >= bbox_min[0] - margin) & (vertices[:, 0] <= bbox_max[0] + margin)
    inside_z = (vertices[:, 2] >= bbox_min[2] - margin) & (vertices[:, 2] <= bbox_max[2] + margin)
    inside = inside_x & inside_z
    
    face_mask = inside[faces].any(axis=1)
    
    if not face_mask.any():
        return None
    
    kept_faces = faces[face_mask]
    unique_verts = np.unique(kept_faces.flatten())
    
    vert_map = np.zeros(len(vertices), dtype=np.int64)
    vert_map[unique_verts] = np.arange(len(unique_verts))
    
    new_vertices = vertices[unique_verts]
    new_faces = vert_map[kept_faces]
    
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    if hasattr(mesh, 'visual'):
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            old_colors = np.array(mesh.visual.vertex_colors)
            if len(old_colors) == len(vertices):
                new_mesh.visual.vertex_colors = old_colors[unique_verts]
        elif hasattr(mesh.visual, 'material'):
            new_mesh.visual.material = mesh.visual.material
    
    return new_mesh


def filter_meshes_to_bbox(
    meshes: List[trimesh.Trimesh],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray
) -> List[trimesh.Trimesh]:
    """Filter and clip meshes to bounding box."""
    filtered = []
    for mesh in meshes:
        clipped = clip_mesh_to_bbox(mesh, bbox_min, bbox_max)
        if clipped is not None and len(clipped.vertices) > 0:
            filtered.append(clipped)
    return filtered


# ============================================================================
# Opening / Wall Detection
# ============================================================================

def compute_opening_center(opening: Dict) -> Optional[np.ndarray]:
    """Get center position of door/window."""
    bbox = opening.get("bbox")
    if bbox and "min" in bbox and "max" in bbox:
        bbox_min = np.array(bbox["min"])
        bbox_max = np.array(bbox["max"])
        return (bbox_min + bbox_max) / 2.0
    
    pos = opening.get("position")
    if pos and len(pos) >= 3:
        return np.array(pos, dtype=np.float64)
    return None


def determine_opening_wall(
    opening: Dict,
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray
) -> str:
    """
    Determine which wall the opening is on.
    Returns: "min_x", "max_x", "min_z", or "max_z"
    """
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return "min_z"
    
    dist_to_min_x = abs(opening_center[0] - room_bbox_min[0])
    dist_to_max_x = abs(opening_center[0] - room_bbox_max[0])
    dist_to_min_z = abs(opening_center[2] - room_bbox_min[2])
    dist_to_max_z = abs(opening_center[2] - room_bbox_max[2])
    
    min_dist = min(dist_to_min_x, dist_to_max_x, dist_to_min_z, dist_to_max_z)
    
    if min_dist == dist_to_min_x:
        return "min_x"
    elif min_dist == dist_to_max_x:
        return "max_x"
    elif min_dist == dist_to_min_z:
        return "min_z"
    else:
        return "max_z"


def get_wall_rotation(wall: str) -> float:
    """
    Get rotation angle (degrees) to bring wall to bottom of image.
    
    In the standard top-down projection:
    - World +X → Image +X (right)
    - World +Z → Image -Y (up)
    
    So currently:
    - min_z wall → bottom of image (no rotation)
    - max_z wall → top of image (180° rotation)
    - min_x wall → left of image (90° rotation)
    - max_x wall → right of image (270° rotation)
    """
    rotation_map = {
        "min_z": 0.0,
        "max_z": 180.0,
        "min_x": 90.0,
        "max_x": 270.0,
    }
    return rotation_map.get(wall, 0.0)


def get_inside_direction(wall: str) -> np.ndarray:
    """Get unit vector pointing from wall into room."""
    directions = {
        "min_x": np.array([1.0, 0.0, 0.0]),
        "max_x": np.array([-1.0, 0.0, 0.0]),
        "min_z": np.array([0.0, 0.0, 1.0]),
        "max_z": np.array([0.0, 0.0, -1.0]),
    }
    return directions.get(wall, np.array([0.0, 0.0, 1.0]))


# ============================================================================
# POV-Oriented Layout Rendering
# ============================================================================

def render_pov_layout(
    meshes: List[trimesh.Trimesh],
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray,
    opening: Dict,
    resolution: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    door_color: Tuple[int, int, int] = (255, 100, 100),
    window_color: Tuple[int, int, int] = (100, 200, 255),
    doors: List[Dict] = None,
    windows: List[Dict] = None,
) -> Tuple[Image.Image, Dict]:
    """
    Render layout from POV orientation.
    
    The opening (door/window where camera is) will be at bottom-center of image.
    Camera looks "up" into the room.
    
    Returns: (image, metadata dict with camera info)
    """
    doors = doors or []
    windows = windows or []
    
    # Determine wall and rotation
    wall = determine_opening_wall(opening, room_bbox_min, room_bbox_max)
    rotation_deg = get_wall_rotation(wall)
    inside_dir = get_inside_direction(wall)
    
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        opening_center = (room_bbox_min + room_bbox_max) / 2.0
    
    # Use opening as the pivot point for rotation
    pivot_x = opening_center[0]
    pivot_z = opening_center[2]
    
    # Room extent for scaling
    extent_x = room_bbox_max[0] - room_bbox_min[0]
    extent_z = room_bbox_max[2] - room_bbox_min[2]
    extent = max(extent_x, extent_z) * 1.15
    
    if extent < 1e-6:
        extent = 1.0
    
    margin = 15
    scale = (resolution - 2 * margin) / extent
    
    # Rotation in radians
    angle_rad = math.radians(rotation_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Camera position in image: bottom-center
    camera_img_x = resolution / 2
    camera_img_y = resolution - margin
    
    def world_to_img(wx, wz):
        """Transform world XZ to image XY with rotation around opening."""
        # Translate so opening is at origin
        rx = wx - pivot_x
        rz = wz - pivot_z
        
        # Rotate around opening
        rotated_x = rx * cos_a - rz * sin_a
        rotated_z = rx * sin_a + rz * cos_a
        
        # Project to image with opening at bottom-center
        ix = rotated_x * scale + camera_img_x
        iy = camera_img_y - rotated_z * scale  # Camera at bottom, +Z goes up
        
        return ix, iy
    
    # Create image
    img = Image.new("RGB", (resolution, resolution), bg_color)
    
    # Collect and sort triangles by height (painter's algorithm)
    triangles = []
    
    for mesh in meshes:
        if len(mesh.vertices) == 0:
            continue
        
        colors = get_mesh_color(mesh)
        
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            c0, c1, c2 = colors[face]
            
            avg_y = (v0[1] + v1[1] + v2[1]) / 3
            avg_color = tuple(int((int(c0[i]) + int(c1[i]) + int(c2[i])) / 3) for i in range(3))
            
            p0 = world_to_img(v0[0], v0[2])
            p1 = world_to_img(v1[0], v1[2])
            p2 = world_to_img(v2[0], v2[2])
            
            triangles.append((avg_y, [p0, p1, p2], avg_color))
    
    # Sort by height (lower first)
    triangles.sort(key=lambda x: x[0])
    
    # Draw triangles
    draw = ImageDraw.Draw(img)
    for _, points, color in triangles:
        flat_points = [coord for point in points for coord in point]
        try:
            draw.polygon(flat_points, fill=color)
        except Exception:
            pass
    
    # Draw doors and windows
    def draw_opening_rect(opening_item: Dict, color: Tuple[int, int, int]):
        bbox = opening_item.get("bbox")
        if not bbox:
            return
        
        corners_world = [
            (bbox["min"][0], bbox["min"][2]),
            (bbox["max"][0], bbox["min"][2]),
            (bbox["max"][0], bbox["max"][2]),
            (bbox["min"][0], bbox["max"][2]),
        ]
        
        corners_img = [world_to_img(wx, wz) for wx, wz in corners_world]
        flat = [coord for point in corners_img for coord in point]
        
        try:
            draw.polygon(flat, fill=color)
        except Exception:
            pass
    
    for door in doors:
        draw_opening_rect(door, door_color)
    
    for window in windows:
        draw_opening_rect(window, window_color)
    
    # Build metadata
    camera_info = {
        "wall": wall,
        "rotation_deg": rotation_deg,
        "opening_center": opening_center.tolist(),
        "inside_direction": inside_dir.tolist(),
        "eye": [opening_center[0], 1.6, opening_center[2]],
        "look_at": (opening_center + inside_dir * 3.0).tolist(),
    }
    
    return img, camera_info


def create_debug_layout(layout: Image.Image, margin: int = 15) -> Image.Image:
    """Add camera marker at bottom-center pointing up."""
    debug_img = layout.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    width, height = debug_img.size
    
    cam_x = width // 2
    cam_y = height - margin  # Same margin as rendering
    
    tri_size = min(width, height) * 0.06
    
    # Triangle pointing up
    tip_x, tip_y = cam_x, cam_y - tri_size
    base1_x, base1_y = cam_x - tri_size * 0.5, cam_y + tri_size * 0.3
    base2_x, base2_y = cam_x + tri_size * 0.5, cam_y + tri_size * 0.3
    
    draw.polygon([(tip_x, tip_y), (base1_x, base1_y), (base2_x, base2_y)],
                 fill=(0, 0, 0), outline=(255, 255, 255))
    draw.ellipse([cam_x - 5, cam_y - 5, cam_x + 5, cam_y + 5],
                 fill=(255, 255, 255), outline=(0, 0, 0))
    
    return debug_img


# ============================================================================
# POV Graph Generation
# ============================================================================

def world_to_pov_coords(pos: np.ndarray, pov_origin: np.ndarray, rotation_rad: float) -> np.ndarray:
    """Transform world position to POV-relative coordinates."""
    rel_pos = pos - pov_origin
    
    cos_a = math.cos(-rotation_rad)
    sin_a = math.sin(-rotation_rad)
    
    new_x = rel_pos[0] * cos_a - rel_pos[2] * sin_a
    new_z = rel_pos[0] * sin_a + rel_pos[2] * cos_a
    
    return np.array([new_x, rel_pos[1], new_z])


def get_pov_direction(pov_pos: np.ndarray) -> str:
    """Get direction description relative to POV."""
    x, z = pov_pos[0], pov_pos[2]
    threshold = 0.3
    
    if abs(x) < threshold and abs(z) < threshold:
        return "nearby"
    
    if abs(z) > abs(x):
        return "ahead" if z > threshold else "behind you"
    else:
        return "to your right" if x > threshold else "to your left"


def build_pov_graph(room_meta: Dict, camera_info: Dict, pov_id: str) -> Dict:
    """Build POV-normalized scene graph."""
    room_type = room_meta.get("room_type", "Unknown")
    room_id = room_meta.get("room_id", room_type)
    
    opening_center = np.array(camera_info["opening_center"])
    rotation_rad = math.radians(camera_info["rotation_deg"])
    
    pov_origin = opening_center.copy()
    pov_origin[1] = 0
    
    objects = []
    
    for item in room_meta.get("furniture", []):
        category = item.get("category", "object").replace("_", " ").lower()
        if any(skip in category for skip in ["wall", "floor", "ceiling"]):
            continue
        
        transform = item.get("transform", {})
        pos = transform.get("pos", [0, 0, 0])
        
        objects.append({
            "uid": item.get("uid", f"obj_{len(objects)}"),
            "category": category,
            "position": np.array(pos),
        })
    
    # Add doors/windows
    for i, door in enumerate(room_meta.get("doors", [])):
        door_center = compute_opening_center(door)
        if door_center is not None:
            objects.append({
                "uid": f"door_{i}",
                "category": "door",
                "position": door_center,
            })
    
    for i, window in enumerate(room_meta.get("windows", [])):
        window_center = compute_opening_center(window)
        if window_center is not None:
            objects.append({
                "uid": f"window_{i}",
                "category": "window",
                "position": window_center,
            })
    
    # Compute POV positions
    pov_positions = {}
    for obj in objects:
        pov_pos = world_to_pov_coords(obj["position"], pov_origin, rotation_rad)
        pov_positions[obj["uid"]] = pov_pos
    
    # Group by category for naming
    category_positions = defaultdict(list)
    for obj in objects:
        category_positions[obj["category"]].append(pov_positions[obj["uid"]])
    
    object_locations = []
    for obj in objects:
        pov_pos = pov_positions[obj["uid"]]
        direction = get_pov_direction(pov_pos)
        dist = np.sqrt(pov_pos[0]**2 + pov_pos[2]**2)
        
        object_locations.append({
            "uid": obj["uid"],
            "category": obj["category"],
            "direction": direction,
            "distance": round(dist, 2),
        })
    
    object_locations.sort(key=lambda x: x["distance"])
    
    return {
        "room_id": room_id,
        "room_type": room_type,
        "pov_id": pov_id,
        "pov_type": "door" if pov_id.startswith("door") else "window",
        "camera": camera_info,
        "num_objects": len(objects),
        "objects": object_locations,
    }


def graph_to_text(graph: Dict) -> str:
    """Convert POV graph to natural language."""
    lines = []
    
    room_type = graph["room_type"]
    pov_type = graph["pov_type"]
    objects = graph["objects"]
    
    if pov_type == "door":
        lines.append(f"You are standing at the doorway, looking into the {room_type}.")
    else:
        lines.append(f"You are looking into the {room_type} from the window.")
    
    # Group by direction
    direction_objects = defaultdict(list)
    for obj in objects:
        if obj["category"] not in ["door", "window"]:
            direction_objects[obj["direction"]].append(obj["category"])
    
    for direction in ["ahead", "to your left", "to your right", "behind you", "nearby"]:
        items = direction_objects.get(direction, [])
        if not items:
            continue
        
        prefix = {
            "ahead": "Directly ahead,",
            "to your left": "To your left,",
            "to your right": "To your right,",
            "behind you": "Behind you,",
            "nearby": "Nearby,",
        }[direction]
        
        if len(items) == 1:
            lines.append(f"{prefix} you see a {items[0]}.")
        else:
            lines.append(f"{prefix} you see: {', '.join(items)}.")
    
    return "\n".join(lines)


# ============================================================================
# Main Processing
# ============================================================================

def process_one_room(
    scene_id: str,
    room_meta: Dict,
    tex_meshes: List[trimesh.Trimesh],
    seg_meshes: List[trimesh.Trimesh],
    output_dir: Path,
    resolution: int,
    door_color: Tuple[int, int, int],
    window_color: Tuple[int, int, int],
    generate_debug: bool = True,
    generate_graphs: bool = True,
) -> List[Dict]:
    """Process one room, generating POV layouts for each door/window."""
    
    room_id = room_meta.get("room_id", room_meta.get("room_type", "Unknown"))
    bbox = room_meta.get("bbox", {})
    
    if not bbox or "min" not in bbox:
        logger.warning(f"  Room {room_id} has no valid bbox")
        return []
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    
    # Filter meshes to room
    room_tex_meshes = filter_meshes_to_bbox(tex_meshes, bbox_min, bbox_max)
    room_seg_meshes = filter_meshes_to_bbox(seg_meshes, bbox_min, bbox_max)
    
    # Get doors and windows
    room_doors = room_meta.get("doors", [])
    room_windows = room_meta.get("windows", [])
    
    # Collect all openings
    openings = []
    for i, door in enumerate(room_doors):
        openings.append((f"door{i}", "door", door))
    for i, window in enumerate(room_windows):
        openings.append((f"window{i}", "window", window))
    
    if not openings:
        logger.debug(f"  Room {room_id} has no doors or windows")
        return []
    
    pov_info_list = []
    
    for pov_id, pov_type, opening in openings:
        # Render tex layout
        tex_img, camera_info = render_pov_layout(
            room_tex_meshes, bbox_min, bbox_max, opening,
            resolution=resolution,
            bg_color=(255, 255, 255),
            door_color=door_color,
            window_color=window_color,
            doors=room_doors,
            windows=room_windows,
        )
        
        # Render seg layout
        seg_img, _ = render_pov_layout(
            room_seg_meshes, bbox_min, bbox_max, opening,
            resolution=resolution,
            bg_color=(255, 255, 255),
            door_color=door_color,
            window_color=window_color,
            doors=room_doors,
            windows=room_windows,
        )
        
        # Save layouts
        tex_dir = output_dir / "layouts_pov" / "tex"
        seg_dir = output_dir / "layouts_pov" / "seg"
        tex_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        tex_path = tex_dir / f"{scene_id}_{room_id}_{pov_id}_tex_layout.png"
        seg_path = seg_dir / f"{scene_id}_{room_id}_{pov_id}_seg_layout.png"
        
        tex_img.save(tex_path)
        seg_img.save(seg_path)
        
        # Debug layout
        if generate_debug:
            debug_dir = output_dir / "layouts_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_img = create_debug_layout(tex_img)
            debug_path = debug_dir / f"{scene_id}_{room_id}_{pov_id}_debug_layout.png"
            debug_img.save(debug_path)
        
        # Generate graph
        graph = None
        if generate_graphs:
            graph = build_pov_graph(room_meta, camera_info, pov_id)
            graph_text = graph_to_text(graph)
            
            graphs_dir = output_dir / "pov_graphs"
            (graphs_dir / "jsons").mkdir(parents=True, exist_ok=True)
            (graphs_dir / "texts").mkdir(parents=True, exist_ok=True)
            
            graph_json_path = graphs_dir / "jsons" / f"{scene_id}_{room_id}_{pov_id}_room_graph.json"
            graph_text_path = graphs_dir / "texts" / f"{scene_id}_{room_id}_{pov_id}_room_description.txt"
            
            with open(graph_json_path, "w") as f:
                json.dump(graph, f, indent=2)
            with open(graph_text_path, "w") as f:
                f.write(graph_text)
        
        # POV info
        pov_info = {
            "scene_id": scene_id,
            "room_id": room_id,
            "room_type": room_meta.get("room_type", "Unknown"),
            "pov_id": pov_id,
            "pov_type": pov_type,
            "camera": camera_info,
            "layout_path_tex": str(tex_path.relative_to(output_dir)),
            "layout_path_seg": str(seg_path.relative_to(output_dir)),
            "graph_path": f"pov_graphs/jsons/{scene_id}_{room_id}_{pov_id}_room_graph.json" if graph else "",
        }
        pov_info_list.append(pov_info)
        
        logger.debug(f"    {pov_id}: wall={camera_info['wall']}, rotation={camera_info['rotation_deg']}°")
    
    return pov_info_list


def process_one_scene(
    scene_id: str,
    tex_glb: Path,
    seg_glb: Path,
    rooms_metadata: List[Dict],
    output_dir: Path,
    resolution: int,
    door_color: Tuple[int, int, int],
    window_color: Tuple[int, int, int],
) -> Tuple[bool, Optional[str], List[Dict]]:
    """Process one scene."""
    try:
        tex_meshes = load_glb_with_transforms(tex_glb)
        seg_meshes = load_glb_with_transforms(seg_glb)
        
        all_pov_info = []
        
        for room_meta in rooms_metadata:
            pov_info = process_one_room(
                scene_id, room_meta, tex_meshes, seg_meshes,
                output_dir, resolution, door_color, window_color
            )
            all_pov_info.extend(pov_info)
        
        return True, None, all_pov_info
    
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return False, str(e), []


def load_taxonomy(taxonomy_path: Optional[Path]) -> Dict:
    """Load taxonomy JSON with flexible path discovery."""
    if taxonomy_path is None:
        logger.warning("No taxonomy path provided, using default colors")
        return {}
    
    # Try exact path first
    if taxonomy_path.exists() and taxonomy_path.is_file():
        try:
            with open(taxonomy_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load taxonomy from {taxonomy_path}: {e}")
            return {}
    
    # If path is a directory, look for json files inside
    if taxonomy_path.exists() and taxonomy_path.is_dir():
        for json_file in taxonomy_path.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    logger.info(f"Loaded taxonomy from {json_file}")
                    return json.load(f)
            except Exception:
                continue
    
    # Try parent directory / taxonomy / taxonomy.json
    if taxonomy_path.parent.exists():
        alt_path = taxonomy_path.parent / "taxonomy.json"
        if alt_path.exists():
            try:
                with open(alt_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load taxonomy from {alt_path}: {e}")
    
    logger.warning(f"Taxonomy not found at {taxonomy_path}, using default colors")
    return {}


def get_door_window_colors(taxonomy: Dict) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get door and window colors from taxonomy."""
    door_color = (255, 100, 100)  # Default red-ish
    window_color = (100, 200, 255)  # Default blue-ish
    
    if not taxonomy:
        return door_color, window_color
    
    try:
        # Primary format: {"category_to_color": {"Door": [R,G,B], "Window": [R,G,B]}}
        cat_to_color = taxonomy.get("category_to_color", {})
        if isinstance(cat_to_color, dict):
            for cat_name, color in cat_to_color.items():
                if isinstance(color, (list, tuple)) and len(color) >= 3:
                    rgb = tuple(int(c) for c in color[:3])
                    if cat_name.lower() == "door":
                        door_color = rgb
                    elif cat_name.lower() == "window":
                        window_color = rgb
        
        # If not found, check if category list contains color info
        if door_color == (255, 100, 100) or window_color == (100, 200, 255):
            categories = taxonomy.get("categories", [])
            if isinstance(categories, list):
                for cat in categories:
                    if isinstance(cat, dict):
                        name = str(cat.get("name", "")).lower()
                        color = cat.get("color")
                        if color and len(color) >= 3:
                            rgb = tuple(int(c) for c in color[:3])
                            if name == "door":
                                door_color = rgb
                            elif name == "window":
                                window_color = rgb
    
    except Exception as e:
        logger.warning(f"Error parsing taxonomy: {e}, using default colors")
    
    return door_color, window_color


def load_scene_list(path: Path) -> List[str]:
    """Load scene IDs from file."""
    scenes = []
    with open(path, "r") as f:
        for line in f:
            sid = line.strip()
            if sid and not sid.startswith("#"):
                scenes.append(sid)
    return scenes


def main():
    parser = argparse.ArgumentParser(description="Combined POV-Oriented Layout Rendering")
    parser.add_argument("--dataset-root", required=True, help="Root directory of dataset")
    parser.add_argument("--scene-list", default=None, help="File with scene IDs")
    parser.add_argument("--room-list", default=None, help="File with room metadata paths")
    parser.add_argument("--resolution", type=int, default=512, help="Output image resolution")
    parser.add_argument("--hpc", action="store_true", help="Enable HPC mode")
    parser.add_argument("--backend", default="auto", choices=["auto", "egl", "osmesa", "xvfb"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-id", type=str, default=None)
    parser.add_argument("--no-debug", action="store_true", help="Skip debug layout generation")
    parser.add_argument("--no-graphs", action="store_true", help="Skip graph generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    dataset_root = Path(args.dataset_root)
    geometry_dir = dataset_root / "geometry"
    metadata_dir = dataset_root / "metadata"
    output_dir = dataset_root
    
    # Try multiple taxonomy paths
    taxonomy_candidates = [
        dataset_root / "taxonomy" / "taxonomy.json",
        dataset_root / "taxonomy.json",
        dataset_root / "taxonomy",  # Will search for *.json inside
    ]
    
    taxonomy_path = None
    for candidate in taxonomy_candidates:
        if candidate.exists():
            taxonomy_path = candidate
            break
    
    if args.hpc:
        if not setup_hpc_rendering(args.backend):
            logger.error("Failed to set up HPC rendering")
            return
    
    try:
        # Load taxonomy for colors
        taxonomy = load_taxonomy(taxonomy_path) if taxonomy_path else {}
        door_color, window_color = get_door_window_colors(taxonomy)
        logger.info(f"Door color: {door_color}, Window color: {window_color}")
        
        all_pov_info = []
        success_count = 0
        
        scene_rooms: Dict[str, List[Dict]] = {}
        
        if args.room_list:
            logger.info(f"Loading rooms from: {args.room_list}")
            with open(args.room_list, "r") as f:
                room_paths = [line.strip() for line in f if line.strip()]
            
            if args.limit:
                room_paths = room_paths[:args.limit]
            
            for room_path in room_paths:
                if not Path(room_path).is_absolute():
                    room_path = dataset_root / room_path
                else:
                    room_path = Path(room_path)
                
                if not room_path.exists():
                    continue
                
                with open(room_path, "r") as f:
                    room_meta = json.load(f)
                
                scene_id = room_meta.get("scene_id")
                if not scene_id:
                    continue
                
                if scene_id not in scene_rooms:
                    scene_rooms[scene_id] = []
                scene_rooms[scene_id].append(room_meta)
        else:
            if args.scene_list:
                scene_ids = load_scene_list(Path(args.scene_list))
            else:
                scene_ids = [f.stem for f in (metadata_dir / "scenes").glob("*.json")]
            
            if args.limit:
                scene_ids = scene_ids[:args.limit]
            
            for scene_id in scene_ids:
                rooms_metadata = []
                for p in (metadata_dir / "rooms").glob(f"{scene_id}_*.json"):
                    with open(p) as f:
                        data = json.load(f)
                        if data.get("scene_id") == scene_id:
                            rooms_metadata.append(data)
                if rooms_metadata:
                    scene_rooms[scene_id] = rooms_metadata
        
        logger.info(f"Processing {len(scene_rooms)} scenes...")
        
        total_scenes = len(scene_rooms)
        for i, (scene_id, rooms_metadata) in enumerate(scene_rooms.items(), 1):
            tex_glb = geometry_dir / "tex" / f"{scene_id}_tex.glb"
            seg_glb = geometry_dir / "seg" / f"{scene_id}_seg.glb"
            
            if not tex_glb.exists() or not seg_glb.exists():
                logger.warning(f"GLB not found for {scene_id}")
                continue
            
            success, error, pov_info = process_one_scene(
                scene_id, tex_glb, seg_glb, rooms_metadata,
                output_dir, args.resolution, door_color, window_color
            )
            
            if success:
                success_count += 1
                all_pov_info.extend(pov_info)
                logger.info(f"[{i}/{total_scenes}] ✓ {scene_id} ({len(rooms_metadata)} rooms, {len(pov_info)} POVs)")
            else:
                logger.warning(f"[{i}/{total_scenes}] ✗ {scene_id}: {error}")
        
        # Save POV info
        if args.shard_id:
            info_path = output_dir / f"pov_info_shard_{args.shard_id}.json"
        else:
            info_path = output_dir / "pov_info.json"
        
        with open(info_path, "w") as f:
            json.dump(all_pov_info, f, indent=2)
        
        logger.info(f"\nDone: {success_count} scenes, {len(all_pov_info)} POVs")
        logger.info(f"POV info saved to: {info_path}")
    
    finally:
        if args.hpc:
            cleanup_hpc_rendering()


if __name__ == "__main__":
    main()