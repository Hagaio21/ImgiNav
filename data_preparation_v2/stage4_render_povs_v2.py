#!/usr/bin/env python3
"""
Stage 4 v2: Improved POV Rendering with Layout Rotation

Key improvements:
1. Camera stepped BACK from door/window (outside room, looking in)
2. Wider FOV (80°) for better room coverage  
3. Smart outside detection using door normal + room center
4. Furniture-weighted look-at target
5. Rotates EXISTING layout images (no re-rendering from 3D)
6. Generates POV-normalized graphs with descriptive naming

Camera positioning strategy:
- Find door/window normal direction
- Determine which side is "outside" the room (away from room center)
- Step back from opening by configurable distance
- Look at furniture center-of-mass (or room center if empty)
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
# Camera Configuration
# ============================================================================

class CameraConfig:
    """Configuration for POV camera parameters."""
    
    def __init__(
        self,
        step_back_distance: float = 0.8,   # How far to step back from door/window
        camera_height: float = 1.6,         # Eye level height
        fov: float = 80.0,                  # Field of view in degrees (wider!)
        look_at_height: float = 0.9,        # Height of look-at target
        downward_tilt_deg: float = 5.0,     # Degrees to tilt camera down
    ):
        self.step_back_distance = step_back_distance
        self.camera_height = camera_height
        self.fov = fov
        self.look_at_height = look_at_height
        self.downward_tilt_deg = downward_tilt_deg


DEFAULT_CONFIG = CameraConfig()


# ============================================================================
# HPC Setup
# ============================================================================

def setup_hpc_rendering(backend: str = "auto") -> bool:
    """
    Set up rendering backend for HPC headless rendering.
    
    Args:
        backend: One of "auto", "egl", "osmesa", "xvfb"
    
    Returns True if successful, False otherwise.
    """
    global _xvfb_display
    
    if backend == "auto":
        # Try backends in order: xvfb first (most reliable on CPU nodes), then others
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
            
            # Test if pyrender works
            try:
                import pyrender
                renderer = pyrender.OffscreenRenderer(64, 64)
                renderer.delete()
                logger.info("Xvfb backend working with pyrender")
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
            import pyrender
            renderer = pyrender.OffscreenRenderer(64, 64)
            renderer.delete()
            logger.info("Using EGL backend (GPU headless)")
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
            logger.info("Using OSMesa backend (CPU software)")
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


# ============================================================================
# Improved Camera Computation
# ============================================================================

def compute_opening_center(opening: Dict) -> Optional[np.ndarray]:
    """Get the center position of a door/window from its bbox."""
    bbox = opening.get("bbox")
    if bbox and "min" in bbox and "max" in bbox:
        bbox_min = np.array(bbox["min"])
        bbox_max = np.array(bbox["max"])
        return (bbox_min + bbox_max) / 2.0
    
    pos = opening.get("position")
    if pos and len(pos) >= 3:
        return np.array(pos, dtype=np.float64)
    return None


def compute_opening_normal(opening: Dict) -> np.ndarray:
    """
    Compute the normal direction of a door/window.
    Normal is perpendicular to the thin dimension.
    """
    bbox = opening.get("bbox")
    if not bbox or "min" not in bbox:
        return np.array([0.0, 0.0, 1.0])
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    size = bbox_max - bbox_min
    
    # The thin dimension indicates the normal direction
    # Door/window is thin in one horizontal direction
    if size[0] < size[2]:
        # Thin in X, so normal points in X direction
        return np.array([1.0, 0.0, 0.0])
    else:
        # Thin in Z, so normal points in Z direction
        return np.array([0.0, 0.0, 1.0])


def compute_furniture_centroid(room_meta: Dict) -> Optional[np.ndarray]:
    """Compute center-of-mass of furniture in the room."""
    furniture = room_meta.get("furniture", [])
    if not furniture:
        return None
    
    positions = []
    for item in furniture:
        transform = item.get("transform", {})
        pos = transform.get("pos")
        if pos and len(pos) >= 3:
            positions.append(pos)
    
    if not positions:
        return None
    
    positions = np.array(positions)
    return positions.mean(axis=0)


def compute_improved_pov_camera(
    room_meta: Dict,
    opening: Dict,
    config: CameraConfig = DEFAULT_CONFIG
) -> Optional[Dict]:
    """
    Compute POV camera position.
    
    Strategy:
    1. Camera positioned AT the door/window opening
    2. Look straight forward (perpendicular to door, into room)
    """
    # Get room bounds
    bbox = room_meta.get("bbox", {})
    if not bbox or "min" not in bbox:
        return None
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    room_center = (bbox_min + bbox_max) / 2.0
    
    # Get opening position and normal
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return None
    
    opening_normal = compute_opening_normal(opening)
    
    # Determine which direction points INTO the room
    # Test both directions, pick the one closer to room center
    test_pos_positive = opening_center + opening_normal * 0.5
    test_pos_negative = opening_center - opening_normal * 0.5
    
    dist_positive = np.linalg.norm(test_pos_positive[[0, 2]] - room_center[[0, 2]])
    dist_negative = np.linalg.norm(test_pos_negative[[0, 2]] - room_center[[0, 2]])
    
    # Inside direction is the one closer to room center
    if dist_positive < dist_negative:
        inside_normal = opening_normal
    else:
        inside_normal = -opening_normal
    
    # Camera AT the door opening (not stepped back)
    eye = opening_center.copy()
    eye[1] = config.camera_height
    
    # Look STRAIGHT FORWARD (into the room, perpendicular to door)
    look_at = opening_center + inside_normal * 5.0  # 5m ahead
    look_at[1] = config.look_at_height
    
    # Apply slight downward tilt
    view_dist = np.linalg.norm(look_at[[0, 2]] - eye[[0, 2]])
    tilt_offset = view_dist * math.tan(math.radians(config.downward_tilt_deg))
    look_at[1] -= tilt_offset
    
    # Compute view direction for metadata
    view_direction = look_at - eye
    view_direction = view_direction / (np.linalg.norm(view_direction) + 1e-9)
    
    return {
        "eye": eye.tolist(),
        "center": look_at.tolist(),
        "up": [0.0, 1.0, 0.0],
        "fov": config.fov,
        "view_direction": view_direction.tolist(),
        "inside_normal": inside_normal.tolist(),  # Direction into room
        "opening_center": opening_center.tolist(),
    }


# ============================================================================
# Layout Rotation
# ============================================================================

def compute_pov_rotation_angle(room_meta: Dict, opening: Dict) -> float:
    """
    Compute rotation angle for layout based on where the door/window is in the layout IMAGE.
    
    Returns angle in degrees (0, 90, 180, or 270) to rotate layout
    so that the door/window is at the BOTTOM of the layout.
    
    Layout image coordinates:
    - Top of image corresponds to -Z in world
    - Bottom of image corresponds to +Z in world  
    - Left of image corresponds to -X in world
    - Right of image corresponds to +X in world
    """
    bbox = room_meta.get("bbox", {})
    if not bbox or "min" not in bbox:
        return 0.0
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    room_size = bbox_max - bbox_min
    
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return 0.0
    
    # Convert opening position to normalized layout coordinates [0, 1]
    # layout_x = (world_x - min_x) / size_x
    # layout_y = (max_z - world_z) / size_z  (inverted because image Y is flipped)
    rel_x = (opening_center[0] - bbox_min[0]) / (room_size[0] + 1e-9)
    rel_y = (bbox_max[2] - opening_center[2]) / (room_size[2] + 1e-9)  # Inverted Z
    
    # Determine which edge the opening is closest to in layout image
    dist_to_top = rel_y
    dist_to_bottom = 1.0 - rel_y
    dist_to_left = rel_x
    dist_to_right = 1.0 - rel_x
    
    min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)
    
    # Rotate so that edge becomes the bottom
    if min_dist == dist_to_bottom:
        # Already at bottom → no rotation
        rotation = 0.0
    elif min_dist == dist_to_top:
        # At top → rotate 180°
        rotation = 180.0
    elif min_dist == dist_to_right:
        # At right → rotate 90° CCW (right becomes bottom)
        rotation = 90.0
    elif min_dist == dist_to_left:
        # At left → rotate 270° CCW (left becomes bottom)
        rotation = 270.0
    else:
        rotation = 0.0
    
    return rotation


def get_opening_position_in_rotated_layout(
    room_meta: Dict, 
    opening: Dict, 
    layout_size: int,
    rotation_angle_deg: float
) -> Optional[Tuple[int, int]]:
    """
    Get the center position of an opening in the ROTATED layout image coordinates.
    
    Returns (x, y) in pixels, or None if can't compute.
    """
    bbox = room_meta.get("bbox", {})
    if not bbox or "min" not in bbox:
        return None
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    room_size = bbox_max - bbox_min
    
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return None
    
    # Convert to original layout image coordinates
    rel_x = (opening_center[0] - bbox_min[0]) / (room_size[0] + 1e-9)
    rel_y = (bbox_max[2] - opening_center[2]) / (room_size[2] + 1e-9)  # Inverted Z
    
    ox = rel_x * layout_size
    oy = rel_y * layout_size
    
    # Apply rotation transform
    cx, cy = layout_size / 2, layout_size / 2
    
    if rotation_angle_deg == 0:
        rx, ry = ox, oy
    elif rotation_angle_deg == 90:
        # 90° CCW: (x, y) -> (y, size - x)
        rx = oy
        ry = layout_size - ox
    elif rotation_angle_deg == 180:
        # 180°: (x, y) -> (size - x, size - y)
        rx = layout_size - ox
        ry = layout_size - oy
    elif rotation_angle_deg == 270:
        # 270° CCW: (x, y) -> (size - y, x)
        rx = layout_size - oy
        ry = ox
    else:
        rx, ry = ox, oy
    
    return (int(rx), int(ry))




def rotate_layout_image(layout_path: Path, rotation_angle_deg: float) -> Optional[Image.Image]:
    """
    Rotate an existing layout image by 0, 90, 180, or 270 degrees.
    
    Args:
        layout_path: Path to the layout image
        rotation_angle_deg: Rotation angle in degrees (0, 90, 180, or 270)
    
    Returns:
        Rotated PIL Image
    """
    if not layout_path.exists():
        return None
    
    try:
        img = Image.open(layout_path)
        
        # Skip if no rotation needed
        if rotation_angle_deg == 0:
            return img.copy()
        
        # For 90° multiples, use transpose which is exact (no interpolation)
        if rotation_angle_deg == 90:
            return img.transpose(Image.ROTATE_90)
        elif rotation_angle_deg == 180:
            return img.transpose(Image.ROTATE_180)
        elif rotation_angle_deg == 270:
            return img.transpose(Image.ROTATE_270)
        else:
            # Fallback for other angles (shouldn't happen)
            return img.rotate(rotation_angle_deg, resample=Image.BILINEAR, expand=False)
    except Exception as e:
        logger.warning(f"Failed to rotate layout {layout_path}: {e}")
        return None


def create_debug_layout(
    rotated_layout: Image.Image,
    room_meta: Dict,
    opening: Dict,
    rotation_angle_deg: float
) -> Image.Image:
    """
    Create debug layout with black triangle showing camera position and orientation.
    
    The triangle is at the door/window position, pointing into the room (upward).
    """
    debug_img = rotated_layout.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    width, height = debug_img.size
    
    # Get opening position in rotated layout
    pos = get_opening_position_in_rotated_layout(room_meta, opening, width, rotation_angle_deg)
    
    if pos is None:
        # Fallback: draw triangle at bottom center
        cx = width // 2
        cy = height - 20
    else:
        cx, cy = pos
        # Clamp to image bounds
        cx = max(20, min(width - 20, cx))
        cy = max(20, min(height - 20, cy))
    
    # Draw black triangle pointing UP (into the room)
    # Triangle size
    tri_size = 25
    
    # Triangle vertices: tip at top, base at bottom
    tip = (cx, cy - tri_size)
    left = (cx - tri_size // 2, cy + tri_size // 3)
    right = (cx + tri_size // 2, cy + tri_size // 3)
    
    # Draw filled black triangle
    draw.polygon([tip, left, right], fill=(0, 0, 0), outline=(0, 0, 0))
    
    # Draw a small white circle at camera position for visibility
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 255, 255), outline=(0, 0, 0))
    
    return debug_img


# ============================================================================
# POV-Normalized Graph Generation
# ============================================================================

def world_to_pov_coords(pos: np.ndarray, pov_origin: np.ndarray, rotation_angle_rad: float) -> np.ndarray:
    """Transform world position to POV-relative coordinates."""
    rel_pos = pos - pov_origin
    
    cos_a = math.cos(-rotation_angle_rad)
    sin_a = math.sin(-rotation_angle_rad)
    
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


def get_position_descriptor(pov_pos: np.ndarray, same_category_positions: List[np.ndarray], category: str) -> str:
    """Generate descriptive name for an object based on POV-relative position."""
    if len(same_category_positions) == 1:
        return f"the {category}"
    
    x, z = pov_pos[0], pov_pos[2]
    
    all_x = [p[0] for p in same_category_positions]
    all_z = [p[2] for p in same_category_positions]
    
    x_range = max(all_x) - min(all_x)
    z_range = max(all_z) - min(all_z)
    
    descriptors = []
    
    if x_range > 0.5:
        if x < min(all_x) + x_range * 0.33:
            descriptors.append("left")
        elif x > max(all_x) - x_range * 0.33:
            descriptors.append("right")
    
    if z_range > 0.5:
        if z < min(all_z) + z_range * 0.33:
            descriptors.append("near")
        elif z > max(all_z) - z_range * 0.33:
            descriptors.append("far")
    
    if descriptors:
        return f"the {' '.join(descriptors)} {category}"
    
    # Fallback
    direction = get_pov_direction(pov_pos)
    if direction == "ahead":
        return f"the {category} ahead"
    elif direction == "to your left":
        return f"the left {category}"
    elif direction == "to your right":
        return f"the right {category}"
    else:
        return f"the {category}"


def build_pov_graph(room_meta: Dict, camera: Dict, pov_id: str) -> Dict:
    """Build POV-normalized graph with descriptive object naming."""
    room_type = room_meta.get("room_type", "Unknown")
    room_id = room_meta.get("room_id", room_type)
    
    eye = np.array(camera["eye"])
    center = np.array(camera["center"])
    
    # Compute rotation
    view_dir = np.array([center[0] - eye[0], center[2] - eye[2]])
    if np.linalg.norm(view_dir) > 1e-6:
        view_dir = view_dir / np.linalg.norm(view_dir)
        rotation_angle_rad = math.atan2(view_dir[0], view_dir[1])
    else:
        rotation_angle_rad = 0.0
    
    pov_origin = eye.copy()
    pov_origin[1] = 0  # Project to floor
    
    # Collect objects
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
    
    # Compute POV positions and generate names
    pov_positions = {}
    for obj in objects:
        pov_pos = world_to_pov_coords(obj["position"], pov_origin, rotation_angle_rad)
        pov_positions[obj["uid"]] = pov_pos
    
    # Group by category
    category_positions = defaultdict(list)
    for obj in objects:
        category_positions[obj["category"]].append(pov_positions[obj["uid"]])
    
    # Generate names
    object_names = {}
    object_locations = []
    
    for obj in objects:
        pov_pos = pov_positions[obj["uid"]]
        same_cat_positions = category_positions[obj["category"]]
        
        name = get_position_descriptor(pov_pos, same_cat_positions, obj["category"])
        object_names[obj["uid"]] = name
        
        direction = get_pov_direction(pov_pos)
        dist = np.sqrt(pov_pos[0]**2 + pov_pos[2]**2)
        
        object_locations.append({
            "name": name,
            "uid": obj["uid"],
            "category": obj["category"],
            "direction": direction,
            "distance": round(dist, 2),
        })
    
    object_locations.sort(key=lambda x: x["distance"])
    
    # Build relations
    relations = []
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            
            dist = np.linalg.norm(obj1["position"] - obj2["position"])
            if dist > 8.0:
                continue
            
            pov1 = pov_positions[obj1["uid"]]
            pov2 = pov_positions[obj2["uid"]]
            
            diff = pov2 - pov1
            if abs(diff[2]) > abs(diff[0]):
                relation = "ahead of" if diff[2] > 0.3 else "behind" if diff[2] < -0.3 else "next to"
            else:
                relation = "to the right of" if diff[0] > 0.3 else "to the left of" if diff[0] < -0.3 else "next to"
            
            relations.append({
                "from": object_names[obj1["uid"]],
                "to": object_names[obj2["uid"]],
                "relation": relation,
                "distance": round(dist, 2),
            })
    
    return {
        "room_id": room_id,
        "room_type": room_type,
        "pov_id": pov_id,
        "pov_type": "door" if pov_id.startswith("door") else "window",
        "camera": camera,
        "rotation_angle_deg": math.degrees(rotation_angle_rad),
        "num_objects": len(objects),
        "is_empty": len(objects) == 0,
        "objects": object_locations,
        "relations": relations,
    }


def describe_room_shape(room_meta: Dict) -> str:
    """Describe room shape based on bounding box aspect ratio."""
    bbox = room_meta.get("bbox", {})
    if not bbox or "min" not in bbox:
        return "rectangular"
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    size = bbox_max - bbox_min
    
    width = size[0]  # X dimension
    depth = size[2]  # Z dimension
    
    aspect = max(width, depth) / (min(width, depth) + 0.01)
    
    if aspect > 2.5:
        return "long and narrow"
    elif aspect > 1.8:
        return "narrow"
    elif aspect < 1.2:
        return "roughly square"
    else:
        return "rectangular"


def describe_room_size(room_meta: Dict) -> str:
    """Describe room size based on floor area."""
    bbox = room_meta.get("bbox", {})
    if not bbox or "min" not in bbox:
        return ""
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    size = bbox_max - bbox_min
    
    area = size[0] * size[2]  # Floor area in m²
    
    if area < 6:
        return "small"
    elif area < 12:
        return "modest-sized"
    elif area < 25:
        return "spacious"
    else:
        return "large"


def describe_openings_layout(objects: List[Dict]) -> str:
    """Describe the layout of doors and windows."""
    doors = [obj for obj in objects if obj["category"] == "door"]
    windows = [obj for obj in objects if obj["category"] == "window"]
    
    sentences = []
    
    # Describe doors
    if len(doors) == 1:
        sentences.append(f"There is a door {doors[0]['direction']}.")
    elif len(doors) == 2:
        dirs = [d["direction"] for d in doors]
        if dirs[0] == dirs[1]:
            sentences.append(f"There are two doors {dirs[0]}.")
        else:
            sentences.append(f"There are doors {dirs[0]} and {dirs[1]}.")
    elif len(doors) > 2:
        sentences.append(f"There are {len(doors)} doors around the room.")
    
    # Describe windows
    if len(windows) == 1:
        sentences.append(f"There is a window {windows[0]['direction']}.")
    elif len(windows) == 2:
        dirs = [w["direction"] for w in windows]
        if dirs[0] == dirs[1]:
            sentences.append(f"There are two windows {dirs[0]}.")
        else:
            sentences.append(f"There are windows {dirs[0]} and {dirs[1]}.")
    elif len(windows) > 2:
        sentences.append(f"There are {len(windows)} windows.")
    
    return " ".join(sentences)


def graph_to_text(graph: Dict, room_meta: Optional[Dict] = None) -> str:
    """Convert POV graph to natural language."""
    lines = []
    
    room_type = graph["room_type"]
    pov_type = graph["pov_type"]
    objects = graph["objects"]
    
    if pov_type == "door":
        lines.append(f"You are standing at the doorway, looking into the {room_type}.")
    else:
        lines.append(f"You are looking into the {room_type} from the window.")
    
    # Handle empty rooms with richer description
    if graph["is_empty"]:
        # Get room shape/size if metadata available
        if room_meta:
            shape = describe_room_shape(room_meta)
            size = describe_room_size(room_meta)
            if size:
                lines.append(f"The room is empty. It is a {size}, {shape} space.")
            else:
                lines.append(f"The room is empty. It is a {shape} space.")
        else:
            lines.append("The room is empty.")
        
        # Describe openings layout
        openings_desc = describe_openings_layout(objects)
        if openings_desc:
            lines.append(openings_desc)
        
        return "\n".join(lines)
    
    # Group furniture by direction (exclude doors/windows for main description)
    direction_objects = defaultdict(list)
    for obj in objects:
        if obj["category"] not in ["door", "window"]:
            direction_objects[obj["direction"]].append(obj["name"])
    
    for direction in ["ahead", "to your left", "to your right", "behind you", "nearby"]:
        names = direction_objects.get(direction, [])
        if not names:
            continue
        
        prefix = {
            "ahead": "Directly ahead,",
            "to your left": "To your left,",
            "to your right": "To your right,",
            "behind you": "Behind you,",
            "nearby": "Nearby,",
        }[direction]
        
        if len(names) == 1:
            lines.append(f"{prefix} you see {names[0]}.")
        elif len(names) == 2:
            lines.append(f"{prefix} you see {names[0]} and {names[1]}.")
        else:
            lines.append(f"{prefix} you see {', '.join(names[:-1])}, and {names[-1]}.")
    
    return "\n".join(lines)


# ============================================================================
# POV Rendering
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
    
    return default_color


def try_render_pyrender(meshes: List[trimesh.Trimesh], camera: Dict, width: int, height: int) -> Optional[Image.Image]:
    """Try to render using pyrender."""
    try:
        import pyrender
    except ImportError:
        return None
    
    try:
        eye = np.array(camera["eye"])
        center = np.array(camera["center"])
        up = np.array(camera["up"])
        fov = camera.get("fov", 80.0)
        
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
        
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(scene)
        renderer.delete()
        
        return Image.fromarray(color)
    except Exception as e:
        logger.warning(f"pyrender failed: {e}")
        return None


def render_pov(meshes: List[trimesh.Trimesh], camera: Dict, width: int, height: int) -> Image.Image:
    """Render POV image."""
    img = try_render_pyrender(meshes, camera, width, height)
    if img is not None:
        return img
    
    # Fallback to simple rendering
    return Image.new("RGB", (width, height), (135, 206, 235))


# ============================================================================
# Main Processing
# ============================================================================

def process_one_scene(
    scene_id: str,
    tex_glb: Path,
    seg_glb: Path,
    scene_meta: Dict,
    rooms_metadata: List[Dict],
    layouts_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    config: CameraConfig,
    generate_graphs: bool = True,
    render_povs: bool = True,
    rotate_layouts: bool = True
) -> Tuple[bool, Optional[str], List[Dict]]:
    """Process one scene."""
    try:
        # Create output directories
        if render_povs:
            (output_dir / "tex").mkdir(parents=True, exist_ok=True)
            (output_dir / "seg").mkdir(parents=True, exist_ok=True)
        
        # Load meshes only if rendering POVs
        tex_meshes = []
        seg_meshes = []
        if render_povs:
            tex_meshes = load_glb_with_transforms(tex_glb)
            seg_meshes = load_glb_with_transforms(seg_glb)
        
        all_doors = scene_meta.get("doors", [])
        all_windows = scene_meta.get("windows", [])
        
        pov_info_list = []
        
        for room_meta in rooms_metadata:
            room_id = room_meta.get("room_id", room_meta.get("room_type", "Unknown"))
            bbox = room_meta.get("bbox", {})
            
            if not bbox or "min" not in bbox:
                continue
            
            # Get room's doors and windows
            room_doors = room_meta.get("doors", [])
            room_windows = room_meta.get("windows", [])
            
            # Collect all openings for this room
            openings = []
            for i, door in enumerate(room_doors):
                openings.append((f"door{i}", "door", door))
            for i, window in enumerate(room_windows):
                openings.append((f"window{i}", "window", window))
            
            if not openings:
                continue
            
            # Find room layout paths
            tex_layout_path = layouts_dir / "tex" / f"{scene_id}_{room_id}_tex_layout.png"
            seg_layout_path = layouts_dir / "seg" / f"{scene_id}_{room_id}_seg_layout.png"
            
            for pov_id, pov_type, opening in openings:
                # Compute improved camera
                camera = compute_improved_pov_camera(room_meta, opening, config)
                if camera is None:
                    continue
                
                # Render POV images
                tex_pov_path = output_dir / "tex" / f"{scene_id}_{room_id}_{pov_id}_tex_pov.png"
                seg_pov_path = output_dir / "seg" / f"{scene_id}_{room_id}_{pov_id}_seg_pov.png"
                
                if render_povs:
                    tex_pov = render_pov(tex_meshes, camera, width, height)
                    seg_pov = render_pov(seg_meshes, camera, width, height)
                    tex_pov.save(tex_pov_path)
                    seg_pov.save(seg_pov_path)
                
                # Compute rotation for layout (based on door/window position, not viewing direction)
                rotation_angle = compute_pov_rotation_angle(room_meta, opening)
                
                # Rotate existing layouts
                if rotate_layouts:
                    rotated_layouts_dir = output_dir.parent / "layouts_pov"
                    (rotated_layouts_dir / "tex").mkdir(parents=True, exist_ok=True)
                    (rotated_layouts_dir / "seg").mkdir(parents=True, exist_ok=True)
                    
                    tex_rotated = rotate_layout_image(tex_layout_path, rotation_angle)
                    seg_rotated = rotate_layout_image(seg_layout_path, rotation_angle)
                    
                    if tex_rotated:
                        tex_rotated_path = rotated_layouts_dir / "tex" / f"{scene_id}_{room_id}_{pov_id}_tex_layout.png"
                        tex_rotated.save(tex_rotated_path)
                        
                        # Create debug layout with red door/window highlight
                        debug_layouts_dir = output_dir.parent / "layouts_debug"
                        debug_layouts_dir.mkdir(parents=True, exist_ok=True)
                        debug_layout = create_debug_layout(tex_rotated, room_meta, opening, rotation_angle)
                        debug_path = debug_layouts_dir / f"{scene_id}_{room_id}_{pov_id}_debug_layout.png"
                        debug_layout.save(debug_path)
                    
                    if seg_rotated:
                        seg_rotated_path = rotated_layouts_dir / "seg" / f"{scene_id}_{room_id}_{pov_id}_seg_layout.png"
                        seg_rotated.save(seg_rotated_path)
                
                # Build POV graph
                graph = None
                graph_text = None
                if generate_graphs:
                    graph = build_pov_graph(room_meta, camera, pov_id)
                    graph_text = graph_to_text(graph, room_meta)  # Pass room_meta for shape info
                    
                    graphs_dir = output_dir.parent / "pov_graphs"
                    (graphs_dir / "jsons").mkdir(parents=True, exist_ok=True)
                    (graphs_dir / "texts").mkdir(parents=True, exist_ok=True)
                    
                    graph_json_path = graphs_dir / "jsons" / f"{scene_id}_{room_id}_{pov_id}_room_graph.json"
                    graph_text_path = graphs_dir / "texts" / f"{scene_id}_{room_id}_{pov_id}_room_description.txt"
                    
                    with open(graph_json_path, "w") as f:
                        json.dump(graph, f, indent=2)
                    with open(graph_text_path, "w") as f:
                        f.write(graph_text)
                
                # Collect POV info
                pov_info = {
                    "scene_id": scene_id,
                    "room_id": room_id,
                    "room_type": room_meta.get("room_type", "Unknown"),
                    "pov_id": pov_id,
                    "pov_type": pov_type,
                    "camera": camera,
                    "rotation_angle_deg": rotation_angle,
                    "pov_path_tex": str(tex_pov_path.relative_to(output_dir.parent)),
                    "pov_path_seg": str(seg_pov_path.relative_to(output_dir.parent)),
                    "layout_path_tex": f"layouts_pov/tex/{scene_id}_{room_id}_{pov_id}_tex_layout.png",
                    "layout_path_seg": f"layouts_pov/seg/{scene_id}_{room_id}_{pov_id}_seg_layout.png",
                    "graph_path": f"pov_graphs/jsons/{scene_id}_{room_id}_{pov_id}_room_graph.json" if graph else "",
                    "graph_text_path": f"pov_graphs/texts/{scene_id}_{room_id}_{pov_id}_room_description.txt" if graph else "",
                    "furniture_count": room_meta.get("furniture_count", 0),
                    "is_empty": room_meta.get("is_empty", False),
                }
                pov_info_list.append(pov_info)
                
                logger.debug(f"  {room_id}/{pov_id}: rotation={rotation_angle:.1f}°")
        
        return True, None, pov_info_list
    
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return False, str(e), []


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
    parser = argparse.ArgumentParser(description="Stage 4 v2: Improved POV rendering")
    parser.add_argument("--dataset-root", required=True, help="Root directory of dataset")
    parser.add_argument("--scene-list", default=None, help="File with scene IDs (one per line)")
    parser.add_argument("--room-list", default=None, help="File with room metadata paths (one per line)")
    parser.add_argument("--width", type=int, default=1280, help="POV image width")
    parser.add_argument("--height", type=int, default=720, help="POV image height")
    parser.add_argument("--fov", type=float, default=80.0, help="Camera FOV in degrees")
    parser.add_argument("--step-back", type=float, default=0.8, help="Step back distance from opening")
    parser.add_argument("--camera-height", type=float, default=1.6, help="Camera height")
    parser.add_argument("--hpc", action="store_true", help="Enable HPC mode")
    parser.add_argument("--backend", default="auto", choices=["auto", "egl", "osmesa", "xvfb"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-id", type=str, default=None, help="Shard ID for output file naming")
    parser.add_argument("--no-graphs", action="store_true", help="Skip graph generation")
    parser.add_argument("--no-layouts", action="store_true", help="Skip layout rotation (just render POVs)")
    parser.add_argument("--only-graphs", action="store_true", help="Only generate graphs (skip POV and layout rendering)")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    geometry_dir = dataset_root / "geometry"
    metadata_dir = dataset_root / "metadata"
    layouts_dir = dataset_root / "layouts"
    output_dir = dataset_root / "povs"
    
    config = CameraConfig(
        step_back_distance=args.step_back,
        camera_height=args.camera_height,
        fov=args.fov,
    )
    
    if args.hpc:
        if not setup_hpc_rendering(args.backend):
            logger.error("Failed to set up HPC rendering")
            return
    
    try:
        # Log mode
        mode_parts = []
        if not args.only_graphs:
            mode_parts.append("POVs")
        if not args.no_layouts and not args.only_graphs:
            mode_parts.append("layouts")
        if not args.no_graphs:
            mode_parts.append("graphs")
        mode_str = " + ".join(mode_parts) if mode_parts else "nothing"
        logger.info(f"Mode: {mode_str}")
        
        all_pov_info = []
        success_count = 0
        
        # Group rooms by scene for processing
        # Dict: scene_id -> list of room metadata
        scene_rooms: Dict[str, List[Dict]] = {}
        
        if args.room_list:
            # Load room metadata files from list
            logger.info(f"Loading rooms from: {args.room_list}")
            with open(args.room_list, "r") as f:
                room_paths = [line.strip() for line in f if line.strip()]
            
            if args.limit:
                room_paths = room_paths[:args.limit]
            
            logger.info(f"Processing {len(room_paths)} room files...")
            
            for room_path in room_paths:
                # Handle both absolute and relative paths
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
            # Use scene list or discover all scenes
            if args.scene_list:
                scene_ids = load_scene_list(Path(args.scene_list))
            else:
                scene_ids = [f.stem for f in (metadata_dir / "scenes").glob("*.json")]
            
            if args.limit:
                scene_ids = scene_ids[:args.limit]
            
            logger.info(f"Processing {len(scene_ids)} scenes...")
            
            for scene_id in scene_ids:
                rooms_metadata = []
                for p in (metadata_dir / "rooms").glob(f"{scene_id}_*.json"):
                    with open(p) as f:
                        data = json.load(f)
                        if data.get("scene_id") == scene_id:
                            rooms_metadata.append(data)
                if rooms_metadata:
                    scene_rooms[scene_id] = rooms_metadata
        
        # Process each scene
        total_scenes = len(scene_rooms)
        for i, (scene_id, rooms_metadata) in enumerate(scene_rooms.items(), 1):
            scene_meta_path = metadata_dir / "scenes" / f"{scene_id}.json"
            if not scene_meta_path.exists():
                continue
            
            with open(scene_meta_path) as f:
                scene_meta = json.load(f)
            
            tex_glb = geometry_dir / "tex" / f"{scene_id}_tex.glb"
            seg_glb = geometry_dir / "seg" / f"{scene_id}_seg.glb"
            
            if not tex_glb.exists() or not seg_glb.exists():
                continue
            
            # Determine what to generate
            render_povs = not args.only_graphs
            rotate_layouts = not args.no_layouts and not args.only_graphs
            generate_graphs = not args.no_graphs
            
            success, error, pov_info = process_one_scene(
                scene_id, tex_glb, seg_glb, scene_meta, rooms_metadata,
                layouts_dir, output_dir, args.width, args.height, config,
                generate_graphs=generate_graphs,
                render_povs=render_povs,
                rotate_layouts=rotate_layouts
            )
            
            if success:
                success_count += 1
                all_pov_info.extend(pov_info)
                logger.info(f"[{i}/{total_scenes}] ✓ {scene_id} ({len(rooms_metadata)} rooms, {len(pov_info)} POVs)")
            else:
                logger.warning(f"[{i}/{total_scenes}] ✗ {scene_id}: {error}")
        
        # Save POV info (per-shard if shard-id provided)
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