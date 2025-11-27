#!/usr/bin/env python3
"""
Stage 2: Metadata Compiler - FIXED VERSION

Key Fix: Computes room bounding boxes from floor geometry vertices,
not from placeholder values.

Generates comprehensive metadata for scenes and rooms.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy JSON file."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_bbox_from_vertices(vertices: np.ndarray) -> Dict:
    """Compute bounding box from vertices array."""
    if len(vertices) == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    
    return {
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist()
    }


def get_room_floor_vertices(room: Dict, arch_map: Dict) -> np.ndarray:
    """
    Get vertices from the floor geometry of a room.
    
    In 3D-FRONT, rooms reference floor meshes via children -> ref -> mesh.
    """
    all_vertices = []
    
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if not ref_id or ref_id not in arch_map:
            continue
        
        arch = arch_map[ref_id]
        arch_type = arch.get("type", "")
        
        # We want floor geometry to define the room bounds
        if "Floor" in arch_type:
            xyz = arch.get("xyz", [])
            if xyz:
                vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
                all_vertices.append(vertices)
    
    if all_vertices:
        return np.vstack(all_vertices)
    
    return np.array([])


def get_room_furniture_positions(room: Dict, furniture_map: Dict) -> np.ndarray:
    """Get furniture positions in a room as fallback for bbox."""
    positions = []
    
    for child in room.get("children", []):
        ref_id = child.get("ref")
        # Check if this child references furniture
        if ref_id and ref_id in furniture_map:
            pos = child.get("pos", [0, 0, 0])
            if isinstance(pos, list) and len(pos) >= 3:
                positions.append(pos)
    
    if positions:
        return np.array(positions, dtype=np.float64)
    
    return np.array([])


def get_room_wall_vertices(room: Dict, arch_map: Dict) -> np.ndarray:
    """Get vertices from wall geometry of a room as fallback."""
    all_vertices = []
    
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if not ref_id or ref_id not in arch_map:
            continue
        
        arch = arch_map[ref_id]
        arch_type = arch.get("type", "")
        
        # Use wall geometry
        if "Wall" in arch_type:
            xyz = arch.get("xyz", [])
            if xyz:
                vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
                all_vertices.append(vertices)
    
    if all_vertices:
        return np.vstack(all_vertices)
    
    return np.array([])


def compute_room_bbox(room: Dict, arch_map: Dict, furniture_map: Dict) -> Optional[Dict]:
    """
    Compute bounding box for a room.
    
    Priority:
    1. Floor geometry vertices
    2. Wall geometry vertices  
    3. Furniture positions + padding
    4. Doors/windows from arch_map + padding
    5. None (invalid room - should be excluded)
    
    Returns None if no valid bbox can be computed.
    """
    room_type = room.get("type", "Unknown")
    
    # Try floor geometry first
    floor_vertices = get_room_floor_vertices(room, arch_map)
    if len(floor_vertices) > 0:
        bbox = compute_bbox_from_vertices(floor_vertices)
        logger.debug(f"Room '{room_type}' bbox from floor geometry")
        return bbox
    
    # Try wall geometry second
    wall_vertices = get_room_wall_vertices(room, arch_map)
    if len(wall_vertices) > 0:
        bbox = compute_bbox_from_vertices(wall_vertices)
        logger.debug(f"Room '{room_type}' bbox from wall geometry")
        return bbox
    
    # Fall back to furniture positions
    furniture_positions = get_room_furniture_positions(room, furniture_map)
    if len(furniture_positions) > 0:
        # Add padding around furniture
        padding = 1.0
        bbox_min = furniture_positions.min(axis=0) - padding
        bbox_max = furniture_positions.max(axis=0) + padding
        bbox = {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist()
        }
        logger.debug(f"Room '{room_type}' bbox from furniture positions")
        return bbox
    
    # Try doors/windows referenced by the room
    door_window_positions = []
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if ref_id and ref_id in arch_map:
            arch = arch_map[ref_id]
            arch_type = arch.get("type", "").lower()
            if "door" in arch_type or "window" in arch_type:
                xyz = arch.get("xyz", [])
                if xyz:
                    vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
                    door_window_positions.append(vertices)
    
    if door_window_positions:
        all_verts = np.vstack(door_window_positions)
        padding = 2.0  # Larger padding since doors/windows are on boundaries
        bbox = {
            "min": (all_verts.min(axis=0) - padding).tolist(),
            "max": (all_verts.max(axis=0) + padding).tolist()
        }
        logger.debug(f"Room '{room_type}' bbox from doors/windows")
        return bbox
    
    # No valid bbox - return None to mark room as invalid
    logger.warning(f"Room '{room_type}' has no geometry, marking as invalid")
    return None


def compute_room_centroid(bbox: Dict) -> List[float]:
    """Compute room centroid from bbox."""
    min_pt = np.array(bbox["min"])
    max_pt = np.array(bbox["max"])
    centroid = (min_pt + max_pt) / 2.0
    return centroid.tolist()


def extract_door_window_bbox(element: Dict, arch_map: Dict) -> Optional[Dict]:
    """Extract bounding box for a door or window."""
    uid = element.get("uid") or element.get("ref")
    
    # First check if element has vertices directly
    xyz = element.get("xyz", [])
    if xyz:
        vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        return compute_bbox_from_vertices(vertices)
    
    # Check if it references a mesh in arch_map
    if uid and uid in arch_map:
        arch = arch_map[uid]
        xyz = arch.get("xyz", [])
        if xyz:
            vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
            return compute_bbox_from_vertices(vertices)
    
    return None


def quaternion_to_matrix(quat: List[float]) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = quat
    
    # Normalize quaternion
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-10:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    
    # Build rotation matrix
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def compute_rotated_bbox(pos: List[float], rot: List[float], base_size: List[float]) -> Dict:
    """
    Compute axis-aligned bounding box for a rotated object.
    
    Args:
        pos: [x, y, z] position
        rot: [x, y, z, w] quaternion rotation
        base_size: [width, height, depth] unrotated size
    
    Returns:
        bbox dict with min/max
    """
    # Create corners of unrotated box centered at origin
    hw, hh, hd = base_size[0]/2, base_size[1]/2, base_size[2]/2
    corners = np.array([
        [-hw, 0, -hd],
        [hw, 0, -hd],
        [-hw, 0, hd],
        [hw, 0, hd],
        [-hw, hh*2, -hd],
        [hw, hh*2, -hd],
        [-hw, hh*2, hd],
        [hw, hh*2, hd],
    ])
    
    # Rotate corners
    rot_matrix = quaternion_to_matrix(rot)
    rotated_corners = corners @ rot_matrix.T
    
    # Translate to position
    rotated_corners += np.array(pos)
    
    # Compute AABB
    bbox_min = rotated_corners.min(axis=0)
    bbox_max = rotated_corners.max(axis=0)
    
    return {
        "min": bbox_min.tolist(),
        "max": bbox_max.tolist()
    }


def boxes_overlap_2d(box1: Dict, box2: Dict, margin: float = 0.3) -> bool:
    """Check if two bboxes overlap in XZ plane (top-down view)."""
    if box1 is None or box2 is None:
        return False
    
    # Check X overlap
    if box1["max"][0] + margin < box2["min"][0] or box2["max"][0] + margin < box1["min"][0]:
        return False
    
    # Check Z overlap
    if box1["max"][2] + margin < box2["min"][2] or box2["max"][2] + margin < box1["min"][2]:
        return False
    
    return True


def is_duplicate_opening(new_bbox: Dict, existing_bboxes: List[Dict], margin: float = 0.3) -> bool:
    """Check if a door/window bbox overlaps with any existing ones."""
    for existing in existing_bboxes:
        if boxes_overlap_2d(new_bbox, existing, margin):
            return True
    return False


def extract_doors_windows(scene_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract door and window information from scene.
    
    Strategy:
    1. First extract from mesh array (these have correct geometry)
    2. Then extract from furniture array, but only if no overlap with mesh ones
    3. Apply rotation to furniture items to get correct bbox orientation
    4. Deduplicate overlapping items within each source
    """
    doors = []
    windows = []
    
    # Track bboxes for deduplication
    doors_bboxes = []
    windows_bboxes = []
    
    # ========================================
    # Step 1: Extract from mesh array (priority)
    # ========================================
    
    for mesh_item in scene_data.get("mesh", []):
        mesh_type = mesh_item.get("type", "").lower()
        uid = mesh_item.get("uid", "")
        
        # Get bbox from mesh vertices
        xyz = mesh_item.get("xyz", [])
        if not xyz:
            continue
            
        vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        bbox = compute_bbox_from_vertices(vertices)
        
        # Compute center position
        position = [
            (bbox["min"][0] + bbox["max"][0]) / 2,
            (bbox["min"][1] + bbox["max"][1]) / 2,
            (bbox["min"][2] + bbox["max"][2]) / 2
        ]
        
        if "door" in mesh_type:
            # Check for duplicates within mesh doors
            if is_duplicate_opening(bbox, doors_bboxes, margin=0.3):
                logger.debug(f"Skipping duplicate mesh door at {position}")
                continue
            
            doors_bboxes.append(bbox)
            doors.append({
                "uid": uid,
                "type": "door",
                "source": "mesh",
                "title": mesh_type,
                "room_type": None,
                "position": position,
                "bbox": bbox,
                "transform": {"pos": position, "rot": [0, 0, 0, 1], "scale": [1, 1, 1]}
            })
        
        elif "window" in mesh_type:
            # Check for duplicates within mesh windows
            if is_duplicate_opening(bbox, windows_bboxes, margin=0.3):
                logger.debug(f"Skipping duplicate mesh window at {position}")
                continue
            
            windows_bboxes.append(bbox)
            windows.append({
                "uid": uid,
                "type": "window",
                "source": "mesh",
                "title": mesh_type,
                "room_type": None,
                "position": position,
                "bbox": bbox,
                "transform": {"pos": position, "rot": [0, 0, 0, 1], "scale": [1, 1, 1]}
            })
    
    logger.debug(f"Found {len(doors)} unique doors, {len(windows)} unique windows in mesh array")
    
    # ========================================
    # Step 2: Extract from furniture array
    # ========================================
    # Only add if not overlapping with existing items
    
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    
    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "Unknown")
        
        for child in room.get("children", []):
            ref_id = child.get("ref")
            if ref_id not in furniture_map:
                continue
            
            furn = furniture_map[ref_id]
            title = furn.get("title", "").lower()
            jid = furn.get("jid", "")
            
            # Check if this is a door or window
            is_door = title.startswith("door/") or title.startswith("door\\")
            is_window = title.startswith("window/") or title.startswith("window\\")
            
            if not is_door and not is_window:
                continue
            
            # Get transform from child
            pos = child.get("pos", [0, 0, 0])
            rot = child.get("rot", [0, 0, 0, 1])
            scale = child.get("scale", [1, 1, 1])
            
            # Get size from furniture item if available
            furn_size = furn.get("size") or furn.get("bbox")
            if furn_size and len(furn_size) >= 3:
                # size is [width, depth, height] typically
                base_size = [furn_size[0] * scale[0], furn_size[2] * scale[1], furn_size[1] * scale[2]]
            else:
                # Default sizes
                if is_door:
                    base_size = [1.0, 2.2, 0.1]  # width, height, depth
                else:
                    base_size = [1.0, 1.5, 0.1]
            
            # Compute rotated bbox
            bbox = compute_rotated_bbox(pos, rot, base_size)
            
            # Check for overlap with ALL existing items (mesh + previously added furniture)
            existing_bboxes = doors_bboxes if is_door else windows_bboxes
            if is_duplicate_opening(bbox, existing_bboxes, margin=0.3):
                logger.debug(f"Skipping overlapping furniture {title} at {pos}")
                continue
            
            item_data = {
                "uid": ref_id,
                "jid": jid,
                "type": "door" if is_door else "window",
                "source": "furniture",
                "title": furn.get("title", ""),
                "room_type": room_type,
                "position": pos,
                "bbox": bbox,
                "transform": {"pos": pos, "rot": rot, "scale": scale}
            }
            
            if is_door:
                doors.append(item_data)
                doors_bboxes.append(bbox)
            else:
                windows.append(item_data)
                windows_bboxes.append(bbox)
    
    logger.debug(f"Final count: {len(doors)} unique doors, {len(windows)} unique windows")
    
    return doors, windows


def extract_furniture_info(
    scene_data: Dict,
    model_info_map: Dict,
    taxonomy: Dict
) -> List[Dict]:
    """Extract furniture information from scene."""
    furniture_list = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    
    for room in scene_data.get("scene", {}).get("room", []):
        room_id = room.get("roomId") or room.get("id")
        room_type = room.get("type", "UnknownRoom")
        
        for child in room.get("children", []):
            ref_id = child.get("ref")
            if ref_id not in furniture_map:
                continue
            
            item_info = furniture_map[ref_id]
            jid = item_info.get('jid')
            if not jid:
                continue
            
            model_info = model_info_map.get(jid, {})
            category = model_info.get('category') or item_info.get('title') or "UnknownCategory"
            supercategory = model_info.get('super-category') or "UnknownSuper"
            
            pos = child.get("pos", [0, 0, 0])
            rot = child.get("rot", [0, 0, 0, 1])
            scale = child.get("scale", [1, 1, 1])
            
            furniture_list.append({
                "uid": ref_id,
                "jid": jid,
                "title": item_info.get('title', ''),
                "category": category,
                "supercategory": supercategory,
                "room_id": room_id,
                "room_type": room_type,
                "transform": {
                    "pos": pos,
                    "rot": rot,
                    "scale": scale
                }
            })
    
    return furniture_list


def compute_pov_camera(room_bbox: Dict, door_info: Optional[Dict] = None) -> Optional[Dict]:
    """Compute POV camera position for a room."""
    bbox_min = np.array(room_bbox["min"])
    bbox_max = np.array(room_bbox["max"])
    center = (bbox_min + bbox_max) / 2.0
    
    # Default: place camera at one corner looking toward center
    # Use the corner that's furthest from the door if available
    size = bbox_max - bbox_min
    
    # Camera at corner, raised to eye level (Y=1.6 in Y-up system)
    eye = bbox_min.copy()
    eye[1] = 1.6  # Eye level
    
    return {
        "eye": eye.tolist(),
        "center": center.tolist(),
        "up": [0.0, 1.0, 0.0]  # Y-up
    }


# Room types to exclude (outdoor/balcony areas)
EXCLUDED_ROOM_TYPES = {
    "balcony", "garden", "terrace", "outdoor", "patio", 
    "yard", "deck", "porch", "veranda", "loggia"
}


def is_excluded_room(room_type: str) -> bool:
    """Check if room type should be excluded from processing."""
    room_lower = room_type.lower()
    return any(excluded in room_lower for excluded in EXCLUDED_ROOM_TYPES)


def compile_scene_metadata(
    scene_data: Dict,
    scene_id: str,
    model_info_map: Dict,
    taxonomy: Dict
) -> Dict:
    """Compile scene-level metadata."""
    arch_map = {m["uid"]: m for m in scene_data.get("mesh", [])}
    furniture_map = {f["uid"]: f for f in scene_data.get("furniture", [])}
    
    # Extract rooms with proper bboxes
    rooms = []
    invalid_rooms = 0
    excluded_rooms = 0
    
    for room_index, room in enumerate(scene_data.get("scene", {}).get("room", [])):
        room_type = room.get("type", "UnknownRoom")
        
        # Skip balconies and outdoor areas
        if is_excluded_room(room_type):
            excluded_rooms += 1
            logger.debug(f"Excluding outdoor/balcony room: {room_type}")
            continue
        
        # Compute bbox from floor geometry
        bbox = compute_room_bbox(room, arch_map, furniture_map)
        
        # Skip rooms with no valid bbox
        if bbox is None:
            invalid_rooms += 1
            continue
        
        centroid = compute_room_centroid(bbox)
        
        rooms.append({
            "room_type": room_type,
            "room_index": room_index,
            "bbox": bbox,
            "centroid": centroid
        })
        
        logger.debug(f"Room {room_type}: bbox={bbox}")
    
    if invalid_rooms > 0:
        logger.debug(f"Skipped {invalid_rooms} invalid rooms (no geometry)")
    
    # Compute scene bbox from all valid rooms
    if rooms:
        all_mins = np.array([r["bbox"]["min"] for r in rooms])
        all_maxs = np.array([r["bbox"]["max"] for r in rooms])
        scene_bbox = {
            "min": all_mins.min(axis=0).tolist(),
            "max": all_maxs.max(axis=0).tolist()
        }
    else:
        scene_bbox = {"min": [0, 0, 0], "max": [1, 1, 1]}
    
    # Extract doors and windows
    doors, windows = extract_doors_windows(scene_data)
    
    # Extract furniture
    furniture = extract_furniture_info(scene_data, model_info_map, taxonomy)
    
    return {
        "scene_id": scene_id,
        "bbox": scene_bbox,
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        "furniture_count": len(furniture)
    }


def compile_room_metadata(
    scene_data: Dict,
    scene_id: str,
    room: Dict,
    room_index: int,
    model_info_map: Dict,
    taxonomy: Dict,
    all_doors: List[Dict],
    all_windows: List[Dict]
) -> Optional[Dict]:
    """
    Compile room-level metadata.
    
    Returns None if:
    - Room has no valid bbox
    - Room is a balcony/outdoor area
    - Room has no doors AND no windows (can't render POVs)
    """
    arch_map = {m["uid"]: m for m in scene_data.get("mesh", [])}
    furniture_map = {f["uid"]: f for f in scene_data.get("furniture", [])}
    
    room_type = room.get("type", "UnknownRoom")
    
    # Skip balconies and outdoor areas
    if is_excluded_room(room_type):
        return None
    
    # Compute proper bbox
    bbox = compute_room_bbox(room, arch_map, furniture_map)
    
    # Skip rooms with no valid bbox
    if bbox is None:
        return None
    
    centroid = compute_room_centroid(bbox)
    
    # Get furniture in this room
    room_furniture = []
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if ref_id not in furniture_map:
            continue
        
        item_info = furniture_map[ref_id]
        jid = item_info.get('jid')
        if not jid:
            continue
        
        model_info = model_info_map.get(jid, {})
        category = model_info.get('category') or item_info.get('title') or "UnknownCategory"
        supercategory = model_info.get('super-category') or "UnknownSuper"
        
        room_furniture.append({
            "uid": ref_id,
            "jid": jid,
            "title": item_info.get('title', ''),
            "category": category,
            "supercategory": supercategory,
            "transform": {
                "pos": child.get("pos", [0, 0, 0]),
                "rot": child.get("rot", [0, 0, 0, 1]),
                "scale": child.get("scale", [1, 1, 1])
            }
        })
    
    # Filter doors/windows to this room
    room_doors = []
    room_windows = []
    
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    
    for door in all_doors:
        door_bbox = door.get("bbox")
        if door_bbox:
            door_center = np.array([
                (door_bbox["min"][0] + door_bbox["max"][0]) / 2,
                (door_bbox["min"][1] + door_bbox["max"][1]) / 2,
                (door_bbox["min"][2] + door_bbox["max"][2]) / 2
            ])
            # Check if door is near room (with margin)
            margin = 0.5
            if (bbox_min[0] - margin <= door_center[0] <= bbox_max[0] + margin and
                bbox_min[2] - margin <= door_center[2] <= bbox_max[2] + margin):
                room_doors.append(door)
    
    for window in all_windows:
        window_bbox = window.get("bbox")
        if window_bbox:
            window_center = np.array([
                (window_bbox["min"][0] + window_bbox["max"][0]) / 2,
                (window_bbox["min"][1] + window_bbox["max"][1]) / 2,
                (window_bbox["min"][2] + window_bbox["max"][2]) / 2
            ])
            margin = 0.5
            if (bbox_min[0] - margin <= window_center[0] <= bbox_max[0] + margin and
                bbox_min[2] - margin <= window_center[2] <= bbox_max[2] + margin):
                room_windows.append(window)
    
    # Skip rooms with no doors AND no windows (can't render POVs)
    if len(room_doors) == 0 and len(room_windows) == 0:
        logger.debug(f"Skipping room '{room_type}' - no doors or windows")
        return None
    
    # Compute POV camera
    pov_camera = compute_pov_camera(bbox, room_doors[0] if room_doors else None)
    
    # Generate room_id
    room_id = f"{room_type}_{room_index + 1}" if room_index > 0 else room_type
    
    # Check if room is empty (no furniture)
    is_empty = len(room_furniture) == 0
    
    return {
        "scene_id": scene_id,
        "room_type": room_type,
        "room_index": room_index,
        "room_id": room_id,
        "bbox": bbox,
        "centroid": centroid,
        "doors": room_doors,
        "windows": room_windows,
        "furniture": room_furniture,
        "furniture_count": len(room_furniture),
        "is_empty": is_empty,
        "pov_camera": pov_camera,
        "adjacent_rooms": []
    }


def process_one_scene(
    scene_path: Path,
    model_info_file: Path,
    taxonomy_path: Path,
    output_dir: Path
) -> Tuple[bool, Optional[str]]:
    """Process a single scene."""
    scene_id = scene_path.stem
    
    with open(scene_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)
    
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}
    
    taxonomy = load_taxonomy(taxonomy_path)
    
    try:
        # Compile scene metadata
        scene_meta = compile_scene_metadata(scene_data, scene_id, model_info_map, taxonomy)
        
        # Get doors/windows for room assignment
        all_doors = scene_meta["doors"]
        all_windows = scene_meta["windows"]
        
        # Validate: scene must have at least one door or window
        if len(all_doors) == 0 and len(all_windows) == 0:
            logger.warning(f"Scene {scene_id} has no doors or windows - skipping")
            return False, "No doors or windows in scene"
        
        # Save scene metadata
        scene_output = output_dir / "scenes" / f"{scene_id}.json"
        scene_output.parent.mkdir(parents=True, exist_ok=True)
        with open(scene_output, "w", encoding="utf-8") as f:
            json.dump(scene_meta, f, indent=2)
        
        # Compile room metadata
        rooms_output_dir = output_dir / "rooms"
        rooms_output_dir.mkdir(parents=True, exist_ok=True)
        
        room_type_counts = {}
        valid_rooms = 0
        invalid_rooms = 0
        
        for room_index, room in enumerate(scene_data.get("scene", {}).get("room", [])):
            room_type = room.get("type", "UnknownRoom")
            
            room_meta = compile_room_metadata(
                scene_data, scene_id, room, room_index,
                model_info_map, taxonomy, all_doors, all_windows
            )
            
            # Skip invalid rooms (no bbox, outdoor/balcony, or no doors/windows)
            if room_meta is None:
                invalid_rooms += 1
                continue
            
            valid_rooms += 1
            
            # Track count for room naming
            if room_type not in room_type_counts:
                room_type_counts[room_type] = 0
            room_type_counts[room_type] += 1
            
            # Generate unique room filename
            count = room_type_counts[room_type]
            if count > 1:
                room_filename = f"{scene_id}_{room_type}_{count}.json"
            else:
                room_filename = f"{scene_id}_{room_type}.json"
            
            room_output = rooms_output_dir / room_filename
            with open(room_output, "w", encoding="utf-8") as f:
                json.dump(room_meta, f, indent=2)
        
        # Check if any valid rooms remain
        if valid_rooms == 0:
            logger.warning(f"Scene {scene_id} has no valid rooms after filtering - marking as invalid")
            # Clean up the scene metadata file we created
            if scene_output.exists():
                scene_output.unlink()
            return False, "No valid rooms in scene"
        
        if invalid_rooms > 0:
            logger.debug(f"Scene {scene_id}: skipped {invalid_rooms} invalid rooms, kept {valid_rooms}")
        
        return True, None
    
    except Exception as e:
        logger.exception(f"Failed {scene_id}: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Compile metadata (fixed)")
    parser.add_argument("--scenes-dir", required=True, help="3D-FRONT scenes directory")
    parser.add_argument("--model-info", required=True, help="model_info.json path")
    parser.add_argument("--taxonomy", required=True, help="taxonomy.json path")
    parser.add_argument("--output-dir", required=True, help="Output metadata directory")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    scenes_dir = Path(args.scenes_dir)
    model_info_file = Path(args.model_info)
    taxonomy_path = Path(args.taxonomy)
    output_dir = Path(args.output_dir)
    
    scene_files = list(scenes_dir.glob("*.json"))
    if not scene_files:
        logger.error(f"No scene files found in {scenes_dir}")
        return
    
    if args.limit:
        scene_files = scene_files[:args.limit]
    
    logger.info(f"Processing {len(scene_files)} scenes...")
    
    success_count = 0
    for i, scene_path in enumerate(scene_files, 1):
        success, error = process_one_scene(scene_path, model_info_file, taxonomy_path, output_dir)
        
        if success:
            success_count += 1
            logger.info(f"[{i}/{len(scene_files)}] ✓ {scene_path.name}")
        else:
            logger.warning(f"[{i}/{len(scene_files)}] ✗ {scene_path.name}: {error}")
    
    logger.info(f"\nDone: {success_count}/{len(scene_files)}")


if __name__ == "__main__":
    main()