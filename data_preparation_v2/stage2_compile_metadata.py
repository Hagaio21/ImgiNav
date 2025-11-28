#!/usr/bin/env python3
"""
Stage 2: Metadata Compiler

Generates comprehensive metadata for scenes and rooms from 3D-FRONT data.

Usage:
    python stage2_compile_metadata.py --config paths.yaml --scene-list shard_1.txt
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # base_dir is used for relative pipeline paths only
    base_dir = Path(config.get("base_dir", ".")).expanduser()
    
    # These pipeline paths can be relative to base_dir
    for key in ["shards_dir", "log_dir", "python_scripts_dir", "hpc_scripts_dir"]:
        val = config.get(key, "")
        if val and not Path(val).is_absolute():
            config[key] = str(base_dir / val)
    
    # 3D-FRONT source paths are ALWAYS absolute - don't modify them
    # output_dataset_root is ALWAYS absolute - don't modify it
    # front3d_scenes_dir, front3d_model_info, front3d_model_dir are ALWAYS absolute
    
    # Set default taxonomy path if not specified
    if not config.get("taxonomy_path"):
        config["taxonomy_path"] = str(Path(config["output_dataset_root"]) / "taxonomy" / "taxonomy.json")
    
    config["base_dir"] = str(base_dir)
    return config


# ==============================================================================
# Geometry Utilities
# ==============================================================================

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
    """Get vertices from the floor geometry of a room."""
    all_vertices = []
    
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if not ref_id or ref_id not in arch_map:
            continue
        
        arch = arch_map[ref_id]
        arch_type = arch.get("type", "")
        
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
        
        if "Wall" in arch_type:
            xyz = arch.get("xyz", [])
            if xyz:
                vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
                all_vertices.append(vertices)
    
    if all_vertices:
        return np.vstack(all_vertices)
    return np.array([])


def compute_room_bbox(room: Dict, arch_map: Dict, furniture_map: Dict) -> Optional[Dict]:
    """Compute bounding box for a room with multiple fallback strategies."""
    room_type = room.get("type", "Unknown")
    
    # Try floor geometry first
    floor_vertices = get_room_floor_vertices(room, arch_map)
    if len(floor_vertices) > 0:
        return compute_bbox_from_vertices(floor_vertices)
    
    # Try wall geometry
    wall_vertices = get_room_wall_vertices(room, arch_map)
    if len(wall_vertices) > 0:
        return compute_bbox_from_vertices(wall_vertices)
    
    # Fall back to furniture positions
    furniture_positions = get_room_furniture_positions(room, furniture_map)
    if len(furniture_positions) > 0:
        padding = 1.0
        bbox_min = furniture_positions.min(axis=0) - padding
        bbox_max = furniture_positions.max(axis=0) + padding
        return {"min": bbox_min.tolist(), "max": bbox_max.tolist()}
    
    # Try doors/windows
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
        padding = 2.0
        return {
            "min": (all_verts.min(axis=0) - padding).tolist(),
            "max": (all_verts.max(axis=0) + padding).tolist()
        }
    
    logger.warning(f"Room '{room_type}' has no geometry")
    return None


def compute_room_centroid(bbox: Dict) -> List[float]:
    """Compute room centroid from bbox."""
    min_pt = np.array(bbox["min"])
    max_pt = np.array(bbox["max"])
    return ((min_pt + max_pt) / 2.0).tolist()


# ==============================================================================
# Door/Window Extraction
# ==============================================================================

def quaternion_to_matrix(quat: List[float]) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = quat
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-10:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def compute_rotated_bbox(pos: List[float], rot: List[float], base_size: List[float]) -> Dict:
    """Compute axis-aligned bounding box for a rotated object."""
    hw, hh, hd = base_size[0]/2, base_size[1]/2, base_size[2]/2
    corners = np.array([
        [-hw, 0, -hd], [hw, 0, -hd], [-hw, 0, hd], [hw, 0, hd],
        [-hw, hh*2, -hd], [hw, hh*2, -hd], [-hw, hh*2, hd], [hw, hh*2, hd],
    ])
    
    rot_matrix = quaternion_to_matrix(rot)
    rotated_corners = corners @ rot_matrix.T + np.array(pos)
    
    return {
        "min": rotated_corners.min(axis=0).tolist(),
        "max": rotated_corners.max(axis=0).tolist()
    }


def boxes_overlap_2d(box1: Dict, box2: Dict, margin: float = 0.3) -> bool:
    """Check if two bboxes overlap in XZ plane."""
    if box1 is None or box2 is None:
        return False
    if box1["max"][0] + margin < box2["min"][0] or box2["max"][0] + margin < box1["min"][0]:
        return False
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
    """Extract door and window information from scene."""
    doors, windows = [], []
    doors_bboxes, windows_bboxes = [], []
    
    # Extract from mesh array
    for mesh_item in scene_data.get("mesh", []):
        mesh_type = mesh_item.get("type", "").lower()
        uid = mesh_item.get("uid", "")
        
        xyz = mesh_item.get("xyz", [])
        if not xyz:
            continue
        
        vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        bbox = compute_bbox_from_vertices(vertices)
        position = [
            (bbox["min"][0] + bbox["max"][0]) / 2,
            (bbox["min"][1] + bbox["max"][1]) / 2,
            (bbox["min"][2] + bbox["max"][2]) / 2
        ]
        
        if "door" in mesh_type:
            if not is_duplicate_opening(bbox, doors_bboxes):
                doors_bboxes.append(bbox)
                doors.append({
                    "uid": uid, "type": "door", "source": "mesh",
                    "title": mesh_type, "position": position, "bbox": bbox,
                    "transform": {"pos": position, "rot": [0, 0, 0, 1], "scale": [1, 1, 1]}
                })
        elif "window" in mesh_type:
            if not is_duplicate_opening(bbox, windows_bboxes):
                windows_bboxes.append(bbox)
                windows.append({
                    "uid": uid, "type": "window", "source": "mesh",
                    "title": mesh_type, "position": position, "bbox": bbox,
                    "transform": {"pos": position, "rot": [0, 0, 0, 1], "scale": [1, 1, 1]}
                })
    
    # Extract from furniture array
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    
    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "Unknown")
        
        for child in room.get("children", []):
            ref_id = child.get("ref")
            if ref_id not in furniture_map:
                continue
            
            furn = furniture_map[ref_id]
            title = furn.get("title", "").lower()
            
            is_door = title.startswith("door/") or title.startswith("door\\")
            is_window = title.startswith("window/") or title.startswith("window\\")
            
            if not is_door and not is_window:
                continue
            
            pos = child.get("pos", [0, 0, 0])
            rot = child.get("rot", [0, 0, 0, 1])
            scale = child.get("scale", [1, 1, 1])
            
            furn_size = furn.get("size") or furn.get("bbox")
            if furn_size and len(furn_size) >= 3:
                base_size = [furn_size[0] * scale[0], furn_size[2] * scale[1], furn_size[1] * scale[2]]
            else:
                base_size = [1.0, 2.2, 0.1] if is_door else [1.0, 1.5, 0.1]
            
            bbox = compute_rotated_bbox(pos, rot, base_size)
            existing_bboxes = doors_bboxes if is_door else windows_bboxes
            
            if not is_duplicate_opening(bbox, existing_bboxes):
                item = {
                    "uid": ref_id, "jid": furn.get("jid", ""),
                    "type": "door" if is_door else "window", "source": "furniture",
                    "title": furn.get("title", ""), "room_type": room_type,
                    "position": pos, "bbox": bbox,
                    "transform": {"pos": pos, "rot": rot, "scale": scale}
                }
                if is_door:
                    doors.append(item)
                    doors_bboxes.append(bbox)
                else:
                    windows.append(item)
                    windows_bboxes.append(bbox)
    
    return doors, windows


# ==============================================================================
# Furniture Extraction
# ==============================================================================

def extract_furniture_info(scene_data: Dict, model_info_map: Dict, taxonomy: Dict) -> List[Dict]:
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
            
            furniture_list.append({
                "uid": ref_id, "jid": jid,
                "title": item_info.get('title', ''),
                "category": category, "supercategory": supercategory,
                "room_id": room_id, "room_type": room_type,
                "transform": {
                    "pos": child.get("pos", [0, 0, 0]),
                    "rot": child.get("rot", [0, 0, 0, 1]),
                    "scale": child.get("scale", [1, 1, 1])
                }
            })
    
    return furniture_list


# ==============================================================================
# Room Processing
# ==============================================================================

EXCLUDED_ROOM_TYPES = {"balcony", "garden", "terrace", "outdoor", "patio", "yard", "deck", "porch", "veranda", "loggia"}


def is_excluded_room(room_type: str) -> bool:
    """Check if room type should be excluded."""
    room_lower = room_type.lower()
    return any(excluded in room_lower for excluded in EXCLUDED_ROOM_TYPES)


def compute_pov_camera(room_bbox: Dict, door_info: Optional[Dict] = None) -> Optional[Dict]:
    """Compute POV camera position for a room."""
    bbox_min = np.array(room_bbox["min"])
    bbox_max = np.array(room_bbox["max"])
    center = (bbox_min + bbox_max) / 2.0
    
    eye = bbox_min.copy()
    eye[1] = 1.6  # Eye level
    
    return {"eye": eye.tolist(), "center": center.tolist(), "up": [0.0, 1.0, 0.0]}


def compile_scene_metadata(scene_data: Dict, scene_id: str, model_info_map: Dict, taxonomy: Dict) -> Dict:
    """Compile scene-level metadata."""
    arch_map = {m["uid"]: m for m in scene_data.get("mesh", [])}
    furniture_map = {f["uid"]: f for f in scene_data.get("furniture", [])}
    
    rooms = []
    invalid_rooms = 0
    excluded_rooms = 0
    
    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "UnknownRoom")
        
        if is_excluded_room(room_type):
            excluded_rooms += 1
            continue
        
        bbox = compute_room_bbox(room, arch_map, furniture_map)
        if bbox is None:
            invalid_rooms += 1
            continue
        
        rooms.append({
            "room_id": room.get("roomId") or room.get("id"),
            "room_type": room_type,
            "bbox": bbox,
            "centroid": compute_room_centroid(bbox)
        })
    
    doors, windows = extract_doors_windows(scene_data)
    furniture = extract_furniture_info(scene_data, model_info_map, taxonomy)
    
    if rooms:
        all_mins = np.array([r["bbox"]["min"] for r in rooms])
        all_maxs = np.array([r["bbox"]["max"] for r in rooms])
        scene_bbox = {"min": all_mins.min(axis=0).tolist(), "max": all_maxs.max(axis=0).tolist()}
    else:
        scene_bbox = {"min": [0, 0, 0], "max": [1, 1, 1]}
    
    return {
        "scene_id": scene_id, "bbox": scene_bbox,
        "rooms": rooms, "doors": doors, "windows": windows,
        "furniture": furniture, "furniture_count": len(furniture),
        "room_count": len(rooms), "invalid_rooms": invalid_rooms, "excluded_rooms": excluded_rooms
    }


def compile_room_metadata(
    scene_data: Dict, scene_id: str, room: Dict, room_index: int,
    model_info_map: Dict, taxonomy: Dict,
    all_doors: List[Dict], all_windows: List[Dict]
) -> Optional[Dict]:
    """Compile room-level metadata."""
    room_type = room.get("type", "UnknownRoom")
    
    if is_excluded_room(room_type):
        return None
    
    arch_map = {m["uid"]: m for m in scene_data.get("mesh", [])}
    furniture_map = {f["uid"]: f for f in scene_data.get("furniture", [])}
    
    bbox = compute_room_bbox(room, arch_map, furniture_map)
    if bbox is None:
        return None
    
    centroid = compute_room_centroid(bbox)
    bbox_min = np.array(bbox["min"])
    bbox_max = np.array(bbox["max"])
    
    # Extract room furniture
    room_furniture = []
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if ref_id not in furniture_map:
            continue
        
        furn = furniture_map[ref_id]
        jid = furn.get("jid")
        if not jid:
            continue
        
        model_info = model_info_map.get(jid, {})
        category = model_info.get('category') or furn.get('title') or "UnknownCategory"
        
        room_furniture.append({
            "uid": ref_id, "jid": jid, "title": furn.get("title", ""), "category": category,
            "transform": {
                "pos": child.get("pos", [0, 0, 0]),
                "rot": child.get("rot", [0, 0, 0, 1]),
                "scale": child.get("scale", [1, 1, 1])
            }
        })
    
    # Find doors/windows in room
    margin = 0.5
    room_doors = [d for d in all_doors if 
                  bbox_min[0] - margin <= d["position"][0] <= bbox_max[0] + margin and
                  bbox_min[2] - margin <= d["position"][2] <= bbox_max[2] + margin]
    room_windows = [w for w in all_windows if
                    bbox_min[0] - margin <= w["position"][0] <= bbox_max[0] + margin and
                    bbox_min[2] - margin <= w["position"][2] <= bbox_max[2] + margin]
    
    if len(room_doors) == 0 and len(room_windows) == 0:
        return None
    
    room_id = f"{room_type}_{room_index + 1}" if room_index > 0 else room_type
    
    return {
        "scene_id": scene_id, "room_type": room_type, "room_index": room_index, "room_id": room_id,
        "bbox": bbox, "centroid": centroid,
        "doors": room_doors, "windows": room_windows,
        "furniture": room_furniture, "furniture_count": len(room_furniture),
        "is_empty": len(room_furniture) == 0,
        "pov_camera": compute_pov_camera(bbox, room_doors[0] if room_doors else None),
        "adjacent_rooms": []
    }


# ==============================================================================
# Scene Processing
# ==============================================================================

def process_one_scene(
    scene_path: Path, model_info_map: Dict, taxonomy: Dict, output_dir: Path
) -> Tuple[bool, Optional[str]]:
    """Process a single scene."""
    scene_id = scene_path.stem
    
    with open(scene_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)
    
    try:
        scene_meta = compile_scene_metadata(scene_data, scene_id, model_info_map, taxonomy)
        all_doors = scene_meta["doors"]
        all_windows = scene_meta["windows"]
        
        if len(all_doors) == 0 and len(all_windows) == 0:
            return False, "No doors or windows"
        
        # Save scene metadata
        scene_output = output_dir / "scenes" / f"{scene_id}.json"
        scene_output.parent.mkdir(parents=True, exist_ok=True)
        with open(scene_output, "w", encoding="utf-8") as f:
            json.dump(scene_meta, f, indent=2)
        
        # Process rooms
        rooms_output_dir = output_dir / "rooms"
        rooms_output_dir.mkdir(parents=True, exist_ok=True)
        
        room_type_counts = {}
        valid_rooms = 0
        
        for room_index, room in enumerate(scene_data.get("scene", {}).get("room", [])):
            room_type = room.get("type", "UnknownRoom")
            
            room_meta = compile_room_metadata(
                scene_data, scene_id, room, room_index,
                model_info_map, taxonomy, all_doors, all_windows
            )
            
            if room_meta is None:
                continue
            
            valid_rooms += 1
            room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            
            count = room_type_counts[room_type]
            room_filename = f"{scene_id}_{room_type}_{count}.json" if count > 1 else f"{scene_id}_{room_type}.json"
            
            with open(rooms_output_dir / room_filename, "w", encoding="utf-8") as f:
                json.dump(room_meta, f, indent=2)
        
        if valid_rooms == 0:
            if scene_output.exists():
                scene_output.unlink()
            return False, "No valid rooms"
        
        return True, None
    
    except Exception as e:
        logger.exception(f"Failed {scene_id}: {e}")
        return False, str(e)


# ==============================================================================
# Scene File Discovery
# ==============================================================================

def load_scene_list(scene_list_path: Path) -> List[str]:
    """Load scene IDs from a text file."""
    scenes = []
    with open(scene_list_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scenes.append(scene_id)
    return scenes


def find_scene_files(scene_ids: List[str], scenes_dir: Path) -> Tuple[List[Path], List[str]]:
    """Find scene JSON files with multiple search strategies."""
    found_files = []
    not_found_ids = []
    
    # Build cache of all JSON files
    logger.info(f"Scanning {scenes_dir} for JSON files...")
    all_json_files = {json_file.stem: json_file for json_file in scenes_dir.rglob("*.json")}
    logger.info(f"Found {len(all_json_files)} JSON files")
    
    for scene_id in scene_ids:
        # Direct match
        if scene_id in all_json_files:
            found_files.append(all_json_files[scene_id])
            continue
        
        # Direct path
        direct_path = scenes_dir / f"{scene_id}.json"
        if direct_path.exists():
            found_files.append(direct_path)
            continue
        
        # Partial match
        if len(scene_id) >= 8:
            matches = [p for stem, p in all_json_files.items() if stem.startswith(scene_id) or scene_id in stem]
            if len(matches) == 1:
                found_files.append(matches[0])
                continue
        
        not_found_ids.append(scene_id)
    
    return found_files, not_found_ids


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Compile metadata")
    parser.add_argument("--config", required=True, help="Path to paths.yaml configuration file")
    parser.add_argument("--scene-list", required=True, help="File containing scene IDs (one per line)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(Path(args.config))
    
    output_dataset_root = Path(config["output_dataset_root"])
    scenes_dir = Path(config["front3d_scenes_dir"])
    model_info_path = Path(config["front3d_model_info"])
    taxonomy_path = Path(config["taxonomy_path"])
    
    logger.info("=" * 60)
    logger.info("Stage 2: Compile Metadata")
    logger.info("=" * 60)
    logger.info(f"Output Dataset Root: {output_dataset_root}")
    logger.info(f"Scenes Dir: {scenes_dir}")
    logger.info(f"Model Info: {model_info_path}")
    logger.info(f"Taxonomy: {taxonomy_path}")
    logger.info("=" * 60)
    
    # Validate paths
    if not scenes_dir.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        return 1
    if not model_info_path.exists():
        logger.error(f"Model info not found: {model_info_path}")
        return 1
    if not taxonomy_path.exists():
        logger.error(f"Taxonomy not found: {taxonomy_path}")
        return 1
    
    # Load model info and taxonomy
    with open(model_info_path, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}
    taxonomy = load_taxonomy(taxonomy_path)
    
    # Load scene list
    scene_list_path = Path(args.scene_list)
    if not scene_list_path.exists():
        logger.error(f"Scene list not found: {scene_list_path}")
        return 1
    
    scene_ids = load_scene_list(scene_list_path)
    logger.info(f"Loaded {len(scene_ids)} scene IDs")
    
    # Find scene files
    scene_files, not_found = find_scene_files(scene_ids, scenes_dir)
    
    if not_found:
        logger.warning(f"Could not find {len(not_found)} scenes")
        for sid in not_found[:5]:
            logger.warning(f"  - {sid}")
        if len(not_found) > 5:
            logger.warning(f"  ... and {len(not_found) - 5} more")
    
    if not scene_files:
        logger.error("No scene files found")
        return 1
    
    if args.limit:
        scene_files = scene_files[:args.limit]
    
    # Process scenes
    output_dir = output_dataset_root / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(scene_files)} scenes...")
    
    success_count = 0
    for i, scene_path in enumerate(scene_files, 1):
        success, error = process_one_scene(scene_path, model_info_map, taxonomy, output_dir)
        
        if success:
            success_count += 1
            logger.info(f"[{i}/{len(scene_files)}] ✓ {scene_path.stem}")
        else:
            logger.warning(f"[{i}/{len(scene_files)}] ✗ {scene_path.stem}: {error}")
    
    logger.info(f"\nDone: {success_count}/{len(scene_files)}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    exit(main())