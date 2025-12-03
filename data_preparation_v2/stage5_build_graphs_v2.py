#!/usr/bin/env python3
"""
Stage 5 v2: POV-Normalized Graph Generation

Key changes from v1:
1. POV-relative spatial relations (relative to camera viewing direction)
2. Descriptive object naming instead of numerical suffixes
3. One unique graph per POV (no duplicates across POVs)
4. Spatial terms relative to viewer: "ahead", "to your left", "behind you"

Object naming strategy:
- Single objects: "the sofa", "the table"
- Multiple objects: Use positional descriptors based on POV-relative position
  - "the left chair", "the right chair"
  - "the near table", "the far table"
  - "the chair ahead", "the chair behind"
  - "the corner cabinet", "the central table"

POV-relative directions:
- Viewing direction = "ahead" / "in front"
- Opposite to viewing = "behind"
- Right of viewing = "to your right"
- Left of viewing = "to your left"
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# POV-Relative Coordinate System
# ============================================================================

def compute_pov_transform(eye: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute transformation for POV-relative coordinates.
    
    Returns:
        (pov_origin, rotation_angle)
        - pov_origin: The eye position (camera location)
        - rotation_angle: Angle to rotate world coords to POV coords
    """
    # Viewing direction in world space (XZ plane)
    view_dir = np.array([center[0] - eye[0], center[2] - eye[2]])
    
    if np.linalg.norm(view_dir) < 1e-6:
        return eye, 0.0
    
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    # Angle from +Z axis (forward in world) to viewing direction
    # We want viewing direction to become +Z in POV space (ahead)
    angle = math.atan2(view_dir[0], view_dir[1])
    
    return eye, angle


def world_to_pov_coords(pos: np.ndarray, pov_origin: np.ndarray, rotation_angle: float) -> np.ndarray:
    """
    Transform world position to POV-relative coordinates.
    
    In POV space:
    - +Z is "ahead" (viewing direction)
    - -Z is "behind"
    - +X is "to your right"
    - -X is "to your left"
    """
    # Translate to POV origin
    rel_pos = pos - pov_origin
    
    # Rotate around Y axis (vertical)
    cos_a = math.cos(-rotation_angle)
    sin_a = math.sin(-rotation_angle)
    
    new_x = rel_pos[0] * cos_a - rel_pos[2] * sin_a
    new_z = rel_pos[0] * sin_a + rel_pos[2] * cos_a
    
    return np.array([new_x, rel_pos[1], new_z])


# ============================================================================
# POV-Relative Spatial Relations
# ============================================================================

def get_pov_relative_direction(pov_pos: np.ndarray, threshold: float = 0.3) -> str:
    """
    Get direction description relative to POV.
    
    pov_pos: Position in POV-relative coordinates (from world_to_pov_coords)
    """
    x, y, z = pov_pos[0], pov_pos[1], pov_pos[2]
    
    # Primary direction based on larger component
    abs_x, abs_z = abs(x), abs(z)
    
    if abs_x < threshold and abs_z < threshold:
        return "nearby"
    
    if abs_z > abs_x:
        # Depth is dominant
        if z > threshold:
            return "ahead"
        elif z < -threshold:
            return "behind you"
        else:
            return "nearby"
    else:
        # Horizontal is dominant
        if x > threshold:
            return "to your right"
        elif x < -threshold:
            return "to your left"
        else:
            return "nearby"


def get_pov_relative_relation(from_pov_pos: np.ndarray, to_pov_pos: np.ndarray, threshold: float = 0.3) -> str:
    """
    Get spatial relation between two objects in POV-relative coordinates.
    Returns relation of 'to' relative to 'from'.
    """
    diff = to_pov_pos - from_pov_pos
    dx, dz = diff[0], diff[2]
    
    abs_dx, abs_dz = abs(dx), abs(dz)
    
    if abs_dx < threshold and abs_dz < threshold:
        return "next to"
    
    if abs_dz > abs_dx:
        # Depth is dominant
        if dz > threshold:
            return "ahead of"
        else:
            return "behind"
    else:
        # Horizontal is dominant
        if dx > threshold:
            return "to the right of"
        else:
            return "to the left of"


def get_inverse_pov_relation(relation: str) -> str:
    """Get the inverse of a POV-relative spatial relation."""
    inverses = {
        "to the right of": "to the left of",
        "to the left of": "to the right of",
        "ahead of": "behind",
        "behind": "ahead of",
        "next to": "next to",
        "nearby": "nearby",
    }
    return inverses.get(relation, relation)


def get_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Get horizontal distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2)


# ============================================================================
# Descriptive Object Naming
# ============================================================================

def get_position_descriptor(pov_pos: np.ndarray, all_pov_positions: List[np.ndarray], 
                           category: str, same_category_positions: List[Tuple[int, np.ndarray]]) -> str:
    """
    Generate a descriptive name for an object based on its POV-relative position.
    
    Args:
        pov_pos: This object's position in POV coordinates
        all_pov_positions: All object positions in POV coordinates
        category: Object category (e.g., "chair", "table")
        same_category_positions: List of (index, pov_pos) for objects of same category
    
    Returns:
        Descriptive name like "the left chair", "the far table", etc.
    """
    if len(same_category_positions) == 1:
        # Only one of this category - just use "the <category>"
        return f"the {category}"
    
    # Multiple objects of same category - need position-based descriptor
    x, y, z = pov_pos[0], pov_pos[1], pov_pos[2]
    
    # Compute relative position among same-category objects
    all_x = [p[1][0] for p in same_category_positions]
    all_z = [p[1][2] for p in same_category_positions]
    
    min_x, max_x = min(all_x), max(all_x)
    min_z, max_z = min(all_z), max(all_z)
    
    x_range = max_x - min_x
    z_range = max_z - min_z
    
    descriptors = []
    
    # Determine position along each axis
    if x_range > 0.5:  # Significant spread in X
        if x < min_x + x_range * 0.33:
            descriptors.append("left")
        elif x > max_x - x_range * 0.33:
            descriptors.append("right")
        else:
            descriptors.append("middle")
    
    if z_range > 0.5:  # Significant spread in Z
        if z < min_z + z_range * 0.33:
            descriptors.append("near")
        elif z > max_z - z_range * 0.33:
            descriptors.append("far")
    
    # If no clear position, use distance from POV
    if not descriptors:
        dist = np.sqrt(x**2 + z**2)
        all_dists = [np.sqrt(p[1][0]**2 + p[1][2]**2) for p in same_category_positions]
        min_dist, max_dist = min(all_dists), max(all_dists)
        
        if max_dist - min_dist > 0.5:
            if dist < min_dist + (max_dist - min_dist) * 0.33:
                descriptors.append("near")
            elif dist > max_dist - (max_dist - min_dist) * 0.33:
                descriptors.append("far")
    
    if descriptors:
        descriptor = " ".join(descriptors)
        return f"the {descriptor} {category}"
    else:
        # Fallback: use general direction
        direction = get_pov_relative_direction(pov_pos, threshold=0.3)
        if direction == "ahead":
            return f"the {category} ahead"
        elif direction == "behind you":
            return f"the {category} behind"
        elif direction == "to your left":
            return f"the left {category}"
        elif direction == "to your right":
            return f"the right {category}"
        else:
            # Ultimate fallback - use index based on distance
            all_dists = [(i, np.sqrt(p[1][0]**2 + p[1][2]**2)) for i, p in enumerate(same_category_positions)]
            all_dists.sort(key=lambda x: x[1])
            
            my_dist = np.sqrt(x**2 + z**2)
            for rank, (idx, d) in enumerate(all_dists):
                if abs(d - my_dist) < 0.1:
                    if rank == 0:
                        return f"the nearest {category}"
                    elif rank == len(all_dists) - 1:
                        return f"the farthest {category}"
                    else:
                        ordinals = ["first", "second", "third", "fourth", "fifth"]
                        if rank < len(ordinals):
                            return f"the {ordinals[rank]} nearest {category}"
            
            return f"the {category}"


def generate_object_names(objects: List[Dict], pov_origin: np.ndarray, rotation_angle: float) -> Dict[str, str]:
    """
    Generate descriptive names for all objects based on POV-relative positions.
    
    Returns:
        Dict mapping uid -> descriptive name
    """
    # Convert all positions to POV coordinates
    pov_positions = {}
    for obj in objects:
        pos = np.array(obj["position"])
        pov_pos = world_to_pov_coords(pos, pov_origin, rotation_angle)
        pov_positions[obj["uid"]] = pov_pos
    
    # Group objects by category
    category_objects = defaultdict(list)
    for obj in objects:
        category_objects[obj["category"]].append((obj["uid"], pov_positions[obj["uid"]]))
    
    # Generate names
    names = {}
    for obj in objects:
        uid = obj["uid"]
        category = obj["category"]
        pov_pos = pov_positions[uid]
        same_category = category_objects[category]
        
        name = get_position_descriptor(pov_pos, list(pov_positions.values()), category, same_category)
        names[uid] = name
    
    # Handle duplicates by adding more specific descriptors
    name_counts = defaultdict(list)
    for uid, name in names.items():
        name_counts[name].append(uid)
    
    for name, uids in name_counts.items():
        if len(uids) > 1:
            # Need to differentiate
            for i, uid in enumerate(uids):
                pov_pos = pov_positions[uid]
                dist = np.sqrt(pov_pos[0]**2 + pov_pos[2]**2)
                
                # Add distance qualifier
                sorted_by_dist = sorted(uids, key=lambda u: np.sqrt(pov_positions[u][0]**2 + pov_positions[u][2]**2))
                rank = sorted_by_dist.index(uid)
                
                if rank == 0:
                    names[uid] = name.replace("the ", "the closer ")
                elif rank == len(uids) - 1:
                    names[uid] = name.replace("the ", "the farther ")
                else:
                    # Use angle-based descriptor
                    angle = math.atan2(pov_pos[0], pov_pos[2])
                    if angle > 0:
                        names[uid] = name.replace("the ", "the rightward ")
                    else:
                        names[uid] = name.replace("the ", "the leftward ")
    
    return names


# ============================================================================
# POV-Normalized Room Graph
# ============================================================================

def build_pov_room_graph(
    room_meta: Dict, 
    scene_meta: Dict, 
    pov_id: str,
    camera: Dict
) -> Dict:
    """
    Build POV-normalized room graph.
    
    Args:
        room_meta: Room metadata
        scene_meta: Scene metadata
        pov_id: POV identifier (e.g., "door0", "window1")
        camera: POV camera with "eye" and "center"
    
    Returns:
        Graph with POV-relative spatial relations and descriptive object names
    """
    room_type = room_meta.get("room_type", "Unknown")
    room_id = room_meta.get("room_id", room_type)
    
    # Get POV transform
    eye = np.array(camera["eye"])
    center = np.array(camera["center"])
    pov_origin, rotation_angle = compute_pov_transform(eye, center)
    
    # Collect objects (furniture)
    objects = []
    furniture = room_meta.get("furniture", [])
    
    for item in furniture:
        category = item.get("category", "object")
        
        # Skip walls and floors
        if any(skip in category.lower() for skip in ["wall", "floor", "ceiling"]):
            continue
        
        # Clean up category name
        category = category.replace("_", " ").lower()
        
        transform = item.get("transform", {})
        pos = transform.get("pos", [0, 0, 0])
        
        objects.append({
            "uid": item.get("uid", f"obj_{len(objects)}"),
            "category": category,
            "title": item.get("title", category),
            "position": pos,
        })
    
    # Add doors
    for i, door in enumerate(room_meta.get("doors", [])):
        door_center = None
        bbox = door.get("bbox")
        if bbox and "min" in bbox and "max" in bbox:
            bbox_min = np.array(bbox["min"])
            bbox_max = np.array(bbox["max"])
            door_center = ((bbox_min + bbox_max) / 2.0).tolist()
        
        if door_center is not None:
            objects.append({
                "uid": f"door_{i}",
                "category": "door",
                "title": "door",
                "position": door_center,
            })
    
    # Add windows
    for i, window in enumerate(room_meta.get("windows", [])):
        window_center = None
        bbox = window.get("bbox")
        if bbox and "min" in bbox and "max" in bbox:
            bbox_min = np.array(bbox["min"])
            bbox_max = np.array(bbox["max"])
            window_center = ((bbox_min + bbox_max) / 2.0).tolist()
        
        if window_center is not None:
            objects.append({
                "uid": f"window_{i}",
                "category": "window",
                "title": "window",
                "position": window_center,
            })
    
    # Generate descriptive names for objects
    object_names = generate_object_names(objects, pov_origin, rotation_angle)
    
    # Compute POV-relative positions
    pov_positions = {}
    for obj in objects:
        pos = np.array(obj["position"])
        pov_pos = world_to_pov_coords(pos, pov_origin, rotation_angle)
        pov_positions[obj["uid"]] = pov_pos
    
    # Build relations between objects
    relations = []
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            
            pos1 = np.array(obj1["position"])
            pos2 = np.array(obj2["position"])
            dist = get_distance(pos1, pos2)
            
            if dist > 8.0:  # Skip very far objects
                continue
            
            pov_pos1 = pov_positions[obj1["uid"]]
            pov_pos2 = pov_positions[obj2["uid"]]
            
            relation = get_pov_relative_relation(pov_pos1, pov_pos2, threshold=0.3)
            inverse = get_inverse_pov_relation(relation)
            
            name1 = object_names[obj1["uid"]]
            name2 = object_names[obj2["uid"]]
            
            relations.append({
                "from": name1,
                "from_uid": obj1["uid"],
                "from_category": obj1["category"],
                "to": name2,
                "to_uid": obj2["uid"],
                "to_category": obj2["category"],
                "relation": relation,
                "distance": round(dist, 2),
            })
            relations.append({
                "from": name2,
                "from_uid": obj2["uid"],
                "from_category": obj2["category"],
                "to": name1,
                "to_uid": obj1["uid"],
                "to_category": obj1["category"],
                "relation": inverse,
                "distance": round(dist, 2),
            })
    
    # Count objects by category
    category_counts = defaultdict(int)
    for obj in objects:
        category_counts[obj["category"]] += 1
    
    # Compute object locations relative to POV
    object_locations = []
    for obj in objects:
        pov_pos = pov_positions[obj["uid"]]
        direction = get_pov_relative_direction(pov_pos)
        dist = np.sqrt(pov_pos[0]**2 + pov_pos[2]**2)
        
        object_locations.append({
            "name": object_names[obj["uid"]],
            "uid": obj["uid"],
            "category": obj["category"],
            "direction": direction,
            "distance": round(dist, 2),
            "pov_position": pov_pos.tolist(),
        })
    
    # Sort by distance from POV
    object_locations.sort(key=lambda x: x["distance"])
    
    # Build graph structure
    graph = {
        "room_id": room_id,
        "room_type": room_type,
        "pov_id": pov_id,
        "pov_type": "door" if pov_id.startswith("door") else "window",
        "camera": camera,
        "pov_transform": {
            "origin": pov_origin.tolist(),
            "rotation_angle_rad": rotation_angle,
            "rotation_angle_deg": math.degrees(rotation_angle),
        },
        "num_objects": len(objects),
        "is_empty": len(objects) == 0,
        "object_counts": dict(category_counts),
        "objects": object_locations,
        "relations": relations,
    }
    
    return graph


def pov_room_graph_to_text(graph: Dict) -> str:
    """
    Convert POV-normalized room graph to natural language description.
    
    Uses second-person perspective ("you see", "to your left").
    """
    lines = []
    
    room_type = graph["room_type"]
    pov_type = graph["pov_type"]
    num_objects = graph["num_objects"]
    counts = graph["object_counts"]
    is_empty = graph.get("is_empty", num_objects == 0)
    objects = graph["objects"]
    
    # Opening sentence - from POV perspective
    if pov_type == "door":
        lines.append(f"You are standing at the doorway, looking into the {room_type}.")
    else:
        lines.append(f"You are looking into the {room_type} from the window.")
    
    if is_empty or num_objects == 0:
        lines.append("The room appears to be empty.")
        return "\n".join(lines)
    
    # Object overview
    obj_list = []
    for cat, count in sorted(counts.items()):
        if cat in ["door", "window"]:
            continue  # Skip doors/windows in listing
        if count == 1:
            obj_list.append(f"a {cat}")
        else:
            obj_list.append(f"{count} {cat}s")
    
    if obj_list:
        lines.append(f"The room contains {', '.join(obj_list)}.")
    
    lines.append("")
    
    # Describe objects by their location relative to POV
    # Group by direction
    direction_objects = defaultdict(list)
    for obj in objects:
        if obj["category"] not in ["door", "window"]:
            direction_objects[obj["direction"]].append(obj)
    
    direction_order = ["ahead", "to your left", "to your right", "behind you", "nearby"]
    
    for direction in direction_order:
        objs = direction_objects.get(direction, [])
        if not objs:
            continue
        
        if direction == "ahead":
            prefix = "Directly ahead,"
        elif direction == "to your left":
            prefix = "To your left,"
        elif direction == "to your right":
            prefix = "To your right,"
        elif direction == "behind you":
            prefix = "Behind you,"
        else:
            prefix = "Nearby,"
        
        obj_names = [obj["name"] for obj in objs]
        if len(obj_names) == 1:
            lines.append(f"{prefix} you see {obj_names[0]}.")
        elif len(obj_names) == 2:
            lines.append(f"{prefix} you see {obj_names[0]} and {obj_names[1]}.")
        else:
            lines.append(f"{prefix} you see {', '.join(obj_names[:-1])}, and {obj_names[-1]}.")
    
    # Describe key spatial relationships (limited)
    if graph["relations"]:
        lines.append("")
        lines.append("Spatial arrangement:")
        
        seen = set()
        sorted_relations = sorted(graph["relations"], key=lambda x: x["distance"])
        
        count = 0
        for rel in sorted_relations:
            if count >= 10:  # Limit
                break
            
            # Skip door/window relations
            if rel["from_category"] in ["door", "window"] or rel["to_category"] in ["door", "window"]:
                continue
            
            key = tuple(sorted([rel["from_uid"], rel["to_uid"]]))
            if key in seen:
                continue
            seen.add(key)
            
            lines.append(f"  {rel['from'].capitalize()} is {rel['relation']} {rel['to']}.")
            count += 1
    
    return "\n".join(lines)


# ============================================================================
# Scene Graph (kept similar but with cleaner output)
# ============================================================================

def build_scene_graph(scene_meta: Dict, rooms_metadata: List[Dict]) -> Dict:
    """Build scene-level graph with rooms and their relations."""
    rooms = []
    for room_meta in rooms_metadata:
        room_type = room_meta.get("room_type", "Unknown")
        room_id = room_meta.get("room_id", room_type)
        bbox = room_meta.get("bbox", {})
        
        if not bbox or "min" not in bbox:
            continue
        
        bbox_min = np.array(bbox["min"])
        bbox_max = np.array(bbox["max"])
        center = (bbox_min + bbox_max) / 2.0
        
        rooms.append({
            "room_id": room_id,
            "room_type": room_type,
            "center": center,
        })
    
    # Build relations
    relations = []
    for i, room1 in enumerate(rooms):
        for j, room2 in enumerate(rooms):
            if i >= j:
                continue
            
            dx = room2["center"][0] - room1["center"][0]
            dz = room2["center"][2] - room1["center"][2]
            
            if abs(dx) > abs(dz):
                relation = "to the right of" if dx > 0 else "to the left of"
                inverse = "to the left of" if dx > 0 else "to the right of"
            else:
                relation = "in front of" if dz > 0 else "behind"
                inverse = "behind" if dz > 0 else "in front of"
            
            relations.append({
                "from": room1["room_type"],
                "from_id": room1["room_id"],
                "to": room2["room_type"],
                "to_id": room2["room_id"],
                "relation": relation,
            })
            relations.append({
                "from": room2["room_type"],
                "from_id": room2["room_id"],
                "to": room1["room_type"],
                "to_id": room1["room_id"],
                "relation": inverse,
            })
    
    return {
        "scene_id": scene_meta.get("scene_id", "unknown"),
        "num_rooms": len(rooms),
        "rooms": [{"room_id": r["room_id"], "room_type": r["room_type"]} for r in rooms],
        "relations": relations,
    }


def scene_graph_to_text(graph: Dict) -> str:
    """Convert scene graph to text."""
    lines = []
    
    num_rooms = graph["num_rooms"]
    room_types = [r["room_type"] for r in graph["rooms"]]
    room_counts = defaultdict(int)
    for rt in room_types:
        room_counts[rt] += 1
    
    lines.append(f"This home contains {num_rooms} rooms.")
    
    room_list = []
    for rt, count in sorted(room_counts.items()):
        if count == 1:
            room_list.append(f"a {rt}")
        else:
            room_list.append(f"{count} {rt}s")
    
    if room_list:
        lines.append(f"The rooms include: {', '.join(room_list)}.")
    
    if graph["relations"]:
        lines.append("")
        seen = set()
        for rel in graph["relations"]:
            key = tuple(sorted([rel["from_id"], rel["to_id"]]))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"The {rel['from']} is {rel['relation']} the {rel['to']}.")
    
    return "\n".join(lines)


# ============================================================================
# Main Processing
# ============================================================================

def process_one_scene(
    scene_id: str,
    scene_meta: Dict,
    rooms_metadata: List[Dict],
    output_dir: Path,
    generate_pov_graphs: bool = True
) -> Tuple[bool, Optional[str], List[Dict]]:
    """
    Process one scene: generate graphs and text descriptions.
    
    Returns:
        (success, error_message, pov_graph_info_list)
    """
    try:
        jsons_dir = output_dir / "jsons"
        texts_dir = output_dir / "texts"
        jsons_dir.mkdir(parents=True, exist_ok=True)
        texts_dir.mkdir(parents=True, exist_ok=True)
        
        # Build scene graph
        scene_graph = build_scene_graph(scene_meta, rooms_metadata)
        scene_text = scene_graph_to_text(scene_graph)
        
        with open(jsons_dir / f"{scene_id}_scene_graph.json", "w") as f:
            json.dump(scene_graph, f, indent=2)
        with open(texts_dir / f"{scene_id}_scene_description.txt", "w") as f:
            f.write(scene_text)
        
        pov_graph_info_list = []
        
        # Build POV-specific room graphs
        for room_meta in rooms_metadata:
            room_id = room_meta.get("room_id", room_meta.get("room_type", "Unknown"))
            room_bbox = room_meta.get("bbox", {})
            
            if not room_bbox or "min" not in room_bbox:
                continue
            
            bbox_min = np.array(room_bbox["min"])
            bbox_max = np.array(room_bbox["max"])
            room_center = (bbox_min + bbox_max) / 2.0
            
            # Get all POVs for this room
            povs = []
            
            # From doors
            for i, door in enumerate(room_meta.get("doors", [])):
                bbox = door.get("bbox")
                if bbox and "min" in bbox:
                    door_center = (np.array(bbox["min"]) + np.array(bbox["max"])) / 2.0
                    camera = {
                        "eye": [door_center[0], 1.5, door_center[2]],
                        "center": [room_center[0], 1.0, room_center[2]],
                        "up": [0.0, 1.0, 0.0]
                    }
                    povs.append((f"door{i}", camera))
            
            # From windows
            for i, window in enumerate(room_meta.get("windows", [])):
                bbox = window.get("bbox")
                if bbox and "min" in bbox:
                    window_center = (np.array(bbox["min"]) + np.array(bbox["max"])) / 2.0
                    camera = {
                        "eye": [window_center[0], 1.5, window_center[2]],
                        "center": [room_center[0], 1.0, room_center[2]],
                        "up": [0.0, 1.0, 0.0]
                    }
                    povs.append((f"window{i}", camera))
            
            # Also use pov_layouts if available (from stage 3 v2)
            for pov_layout in room_meta.get("pov_layouts", []):
                pov_id = pov_layout["pov_id"]
                camera = pov_layout.get("camera")
                if camera and not any(p[0] == pov_id for p in povs):
                    povs.append((pov_id, camera))
            
            if not povs:
                # Fallback: create a default POV
                camera = {
                    "eye": [bbox_min[0], 1.5, (bbox_min[2] + bbox_max[2]) / 2],
                    "center": [room_center[0], 1.0, room_center[2]],
                    "up": [0.0, 1.0, 0.0]
                }
                povs.append(("default", camera))
            
            if not generate_pov_graphs:
                # Generate only one room graph (legacy mode)
                pov_id, camera = povs[0]
                room_graph = build_pov_room_graph(room_meta, scene_meta, pov_id, camera)
                room_text = pov_room_graph_to_text(room_graph)
                
                filename_base = f"{scene_id}_{room_id}"
                with open(jsons_dir / f"{filename_base}_room_graph.json", "w") as f:
                    json.dump(room_graph, f, indent=2)
                with open(texts_dir / f"{filename_base}_room_description.txt", "w") as f:
                    f.write(room_text)
                
                continue
            
            # Generate POV-specific graphs
            for pov_id, camera in povs:
                room_graph = build_pov_room_graph(room_meta, scene_meta, pov_id, camera)
                room_text = pov_room_graph_to_text(room_graph)
                
                # Save with POV ID in filename
                filename_base = f"{scene_id}_{room_id}_{pov_id}"
                
                graph_path = jsons_dir / f"{filename_base}_room_graph.json"
                text_path = texts_dir / f"{filename_base}_room_description.txt"
                
                with open(graph_path, "w") as f:
                    json.dump(room_graph, f, indent=2)
                with open(text_path, "w") as f:
                    f.write(room_text)
                
                pov_graph_info_list.append({
                    "scene_id": scene_id,
                    "room_id": room_id,
                    "pov_id": pov_id,
                    "pov_type": "door" if pov_id.startswith("door") else "window",
                    "graph_path": str(graph_path.relative_to(output_dir.parent)),
                    "text_path": str(text_path.relative_to(output_dir.parent)),
                    "camera": camera,
                })
                
                logger.debug(f"    Generated graph for {room_id}/{pov_id}")
        
        return True, None, pov_graph_info_list
    
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return False, str(e), []


def load_scene_list(scene_list_path: Path) -> List[str]:
    """Load scene IDs from a text file."""
    scenes = []
    with open(scene_list_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scenes.append(scene_id)
    return scenes


def main():
    parser = argparse.ArgumentParser(description="Stage 5 v2: POV-normalized graph generation")
    parser.add_argument("--dataset-root", default=None, help="Root directory of dataset")
    parser.add_argument("--metadata-dir", default=None, help="Directory with metadata")
    parser.add_argument("--output-dir", default=None, help="Output directory for graphs")
    parser.add_argument("--scene-list", default=None, help="File containing scene IDs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes with existing outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-pov-graphs", action="store_true", 
                        help="Generate only one graph per room (legacy mode)")
    args = parser.parse_args()
    
    # Resolve paths
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        metadata_dir = Path(args.metadata_dir) if args.metadata_dir else dataset_root / "metadata"
        output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "graphs"
    else:
        if not args.metadata_dir or not args.output_dir:
            parser.error("--dataset-root or both --metadata-dir and --output-dir required")
        metadata_dir = Path(args.metadata_dir)
        output_dir = Path(args.output_dir)
    
    # Get scene list
    if args.scene_list:
        scene_ids = load_scene_list(Path(args.scene_list))
        logger.info(f"Loaded {len(scene_ids)} scene IDs from list")
    else:
        scene_meta_files = list((metadata_dir / "scenes").glob("*.json"))
        scene_ids = [f.stem for f in scene_meta_files]
        logger.info(f"Discovered {len(scene_ids)} scenes from metadata")
    
    if args.limit:
        scene_ids = scene_ids[:args.limit]
    
    # Skip existing check
    if args.skip_existing:
        jsons_dir = output_dir / "jsons"
        scenes_to_process = []
        for scene_id in scene_ids:
            scene_graph = jsons_dir / f"{scene_id}_scene_graph.json"
            if not scene_graph.exists():
                scenes_to_process.append(scene_id)
        logger.info(f"Skipping {len(scene_ids) - len(scenes_to_process)} scenes with existing outputs")
        scene_ids = scenes_to_process
    
    logger.info(f"Processing {len(scene_ids)} scenes...")
    
    all_pov_graph_info = []
    success_count = 0
    error_count = 0
    
    for i, scene_id in enumerate(scene_ids, 1):
        scene_meta_path = metadata_dir / "scenes" / f"{scene_id}.json"
        if not scene_meta_path.exists():
            logger.warning(f"Scene metadata not found: {scene_meta_path}")
            error_count += 1
            continue
        
        with open(scene_meta_path, "r") as f:
            scene_meta = json.load(f)
        
        # Load room metadata
        rooms_metadata = []
        for room_meta_path in (metadata_dir / "rooms").glob(f"{scene_id}_*.json"):
            with open(room_meta_path, "r") as f:
                room_data = json.load(f)
                if room_data.get("scene_id") == scene_id:
                    rooms_metadata.append(room_data)
        
        if not rooms_metadata:
            logger.warning(f"No room metadata for {scene_id}")
            error_count += 1
            continue
        
        success, error, pov_graph_info = process_one_scene(
            scene_id, scene_meta, rooms_metadata, output_dir,
            generate_pov_graphs=not args.no_pov_graphs
        )
        
        if success:
            success_count += 1
            all_pov_graph_info.extend(pov_graph_info)
            logger.info(f"[{i}/{len(scene_ids)}] ✓ {scene_id} ({len(pov_graph_info)} POV graphs)")
        else:
            error_count += 1
            logger.warning(f"[{i}/{len(scene_ids)}] ✗ {scene_id}: {error}")
    
    # Save POV graph info summary
    if all_pov_graph_info:
        info_path = output_dir / "pov_graphs_info.json"
        with open(info_path, "w") as f:
            json.dump(all_pov_graph_info, f, indent=2)
        logger.info(f"Saved POV graph info to {info_path}")
    
    logger.info(f"\nDone: {success_count} succeeded, {error_count} errors")
    logger.info(f"Total POV graphs generated: {len(all_pov_graph_info)}")


if __name__ == "__main__":
    main()
