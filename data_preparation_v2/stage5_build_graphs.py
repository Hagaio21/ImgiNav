#!/usr/bin/env python3
"""
Stage 5: Scene and Room Graph Generation

Generates graph structures that can be converted to natural language descriptions.

Scene Graph:
- Lists all rooms
- Spatial relations between rooms (kitchen is left of bedroom, etc.)
- Bidirectional relations

Room Graph:
- Room type with list of objects
- Spatial relations between objects
- Includes doors and windows, excludes walls/floors
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Spatial Relations
# ============================================================================

def get_horizontal_relation(pos1: np.ndarray, pos2: np.ndarray, threshold: float = 0.5) -> Optional[str]:
    """
    Get horizontal relation: left/right based on X axis.
    Returns relation of pos2 relative to pos1.
    """
    dx = pos2[0] - pos1[0]
    if abs(dx) < threshold:
        return None
    return "right of" if dx > 0 else "left of"


def get_depth_relation(pos1: np.ndarray, pos2: np.ndarray, threshold: float = 0.5) -> Optional[str]:
    """
    Get depth relation: in front of / behind based on Z axis.
    Returns relation of pos2 relative to pos1.
    """
    dz = pos2[2] - pos1[2]
    if abs(dz) < threshold:
        return None
    return "in front of" if dz > 0 else "behind"


def get_spatial_relation(pos1: np.ndarray, pos2: np.ndarray, threshold: float = 0.5) -> str:
    """
    Get primary spatial relation between two positions.
    Returns relation of pos2 relative to pos1.
    """
    dx = pos2[0] - pos1[0]
    dz = pos2[2] - pos1[2]
    
    # Determine primary direction
    if abs(dx) > abs(dz):
        # Horizontal is dominant
        if abs(dx) < threshold:
            return "near"
        return "to the right of" if dx > 0 else "to the left of"
    else:
        # Depth is dominant
        if abs(dz) < threshold:
            return "near"
        return "in front of" if dz > 0 else "behind"


def get_inverse_relation(relation: str) -> str:
    """Get the inverse of a spatial relation."""
    inverses = {
        "to the right of": "to the left of",
        "to the left of": "to the right of",
        "in front of": "behind",
        "behind": "in front of",
        "near": "near",
        "next to": "next to",
    }
    return inverses.get(relation, relation)


def get_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Get horizontal distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2)


# ============================================================================
# Scene Graph
# ============================================================================

def build_scene_graph(scene_meta: Dict, rooms_metadata: List[Dict]) -> Dict:
    """
    Build scene-level graph with rooms and their relations.
    """
    # Collect room info
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
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        })
    
    # Build relations between rooms
    relations = []
    for i, room1 in enumerate(rooms):
        for j, room2 in enumerate(rooms):
            if i >= j:
                continue
            
            relation = get_spatial_relation(room1["center"], room2["center"], threshold=1.0)
            inverse = get_inverse_relation(relation)
            
            relations.append({
                "from": room1["room_id"],
                "to": room2["room_id"],
                "relation": relation,
            })
            relations.append({
                "from": room2["room_id"],
                "to": room1["room_id"],
                "relation": inverse,
            })
    
    # Build graph structure
    graph = {
        "scene_id": scene_meta.get("scene_id", "unknown"),
        "num_rooms": len(rooms),
        "rooms": [
            {
                "room_id": r["room_id"],
                "room_type": r["room_type"],
                "center": r["center"].tolist(),
            }
            for r in rooms
        ],
        "relations": relations,
    }
    
    return graph


def scene_graph_to_text(graph: Dict) -> str:
    """
    Convert scene graph to natural language description.
    """
    lines = []
    
    # Scene overview
    num_rooms = graph["num_rooms"]
    room_types = [r["room_type"] for r in graph["rooms"]]
    room_counts = {}
    for rt in room_types:
        room_counts[rt] = room_counts.get(rt, 0) + 1
    
    # Opening sentence
    lines.append(f"This scene contains {num_rooms} rooms.")
    
    # List rooms
    room_list = []
    for rt, count in room_counts.items():
        if count == 1:
            room_list.append(f"a {rt}")
        else:
            room_list.append(f"{count} {rt}s")
    
    if room_list:
        lines.append(f"The rooms include: {', '.join(room_list)}.")
    
    # Describe relations
    if graph["relations"]:
        lines.append("")
        lines.append("Spatial layout:")
        
        # Group relations by source room
        seen = set()
        for rel in graph["relations"]:
            key = (rel["from"], rel["to"])
            if key in seen:
                continue
            seen.add(key)
            seen.add((rel["to"], rel["from"]))
            
            lines.append(f"  The {rel['from']} is {rel['relation']} the {rel['to']}.")
    
    return "\n".join(lines)


# ============================================================================
# Room Graph
# ============================================================================

def build_room_graph(room_meta: Dict, scene_meta: Dict) -> Dict:
    """
    Build room-level graph with objects and their relations.
    """
    room_type = room_meta.get("room_type", "Unknown")
    room_id = room_meta.get("room_id", room_type)
    
    # Collect objects (furniture)
    objects = []
    furniture = room_meta.get("furniture", [])
    
    for item in furniture:
        category = item.get("category", "object")
        
        # Skip walls and floors
        if any(skip in category.lower() for skip in ["wall", "floor", "ceiling"]):
            continue
        
        transform = item.get("transform", {})
        pos = transform.get("pos", [0, 0, 0])
        
        objects.append({
            "uid": item.get("uid", "unknown"),
            "category": category,
            "title": item.get("title", category),
            "position": np.array(pos),
        })
    
    # Add doors from room metadata
    for i, door in enumerate(room_meta.get("doors", [])):
        door_center = None
        bbox = door.get("bbox")
        if bbox and "min" in bbox and "max" in bbox:
            bbox_min = np.array(bbox["min"])
            bbox_max = np.array(bbox["max"])
            door_center = (bbox_min + bbox_max) / 2.0
        
        if door_center is not None:
            objects.append({
                "uid": f"door_{i}",
                "category": "door",
                "title": "door",
                "position": door_center,
            })
    
    # Add windows from room metadata
    for i, window in enumerate(room_meta.get("windows", [])):
        window_center = None
        bbox = window.get("bbox")
        if bbox and "min" in bbox and "max" in bbox:
            bbox_min = np.array(bbox["min"])
            bbox_max = np.array(bbox["max"])
            window_center = (bbox_min + bbox_max) / 2.0
        
        if window_center is not None:
            objects.append({
                "uid": f"window_{i}",
                "category": "window",
                "title": "window",
                "position": window_center,
            })
    
    # If room has no doors/windows from metadata, try to get from scene
    if not room_meta.get("doors") and not room_meta.get("windows"):
        room_bbox = room_meta.get("bbox", {})
        if room_bbox and "min" in room_bbox:
            bbox_min = np.array(room_bbox["min"])
            bbox_max = np.array(room_bbox["max"])
            
            # Check scene doors
            for i, door in enumerate(scene_meta.get("doors", [])):
                door_bbox = door.get("bbox")
                if door_bbox and "min" in door_bbox:
                    door_center = (np.array(door_bbox["min"]) + np.array(door_bbox["max"])) / 2.0
                    # Check if door is near room
                    margin = 1.0
                    if (bbox_min[0] - margin <= door_center[0] <= bbox_max[0] + margin and
                        bbox_min[2] - margin <= door_center[2] <= bbox_max[2] + margin):
                        objects.append({
                            "uid": f"door_{i}",
                            "category": "door",
                            "title": "door",
                            "position": door_center,
                        })
            
            # Check scene windows
            for i, window in enumerate(scene_meta.get("windows", [])):
                window_bbox = window.get("bbox")
                if window_bbox and "min" in window_bbox:
                    window_center = (np.array(window_bbox["min"]) + np.array(window_bbox["max"])) / 2.0
                    margin = 1.0
                    if (bbox_min[0] - margin <= window_center[0] <= bbox_max[0] + margin and
                        bbox_min[2] - margin <= window_center[2] <= bbox_max[2] + margin):
                        objects.append({
                            "uid": f"window_{i}",
                            "category": "window",
                            "title": "window",
                            "position": window_center,
                        })
    
    # Build relations between objects
    relations = []
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            
            dist = get_distance(obj1["position"], obj2["position"])
            if dist > 10.0:  # Skip very far objects
                continue
            
            relation = get_spatial_relation(obj1["position"], obj2["position"], threshold=0.3)
            inverse = get_inverse_relation(relation)
            
            relations.append({
                "from": obj1["category"],
                "from_uid": obj1["uid"],
                "to": obj2["category"],
                "to_uid": obj2["uid"],
                "relation": relation,
                "distance": round(dist, 2),
            })
            relations.append({
                "from": obj2["category"],
                "from_uid": obj2["uid"],
                "to": obj1["category"],
                "to_uid": obj1["uid"],
                "relation": inverse,
                "distance": round(dist, 2),
            })
    
    # Count objects by category
    category_counts = {}
    for obj in objects:
        cat = obj["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Build graph structure
    graph = {
        "room_id": room_id,
        "room_type": room_type,
        "num_objects": len(objects),
        "is_empty": len(objects) == 0,
        "object_counts": category_counts,
        "objects": [
            {
                "uid": o["uid"],
                "category": o["category"],
                "title": o["title"],
                "position": o["position"].tolist(),
            }
            for o in objects
        ],
        "relations": relations,
    }
    
    return graph


def room_graph_to_text(graph: Dict) -> str:
    """
    Convert room graph to natural language description.
    """
    lines = []
    
    room_type = graph["room_type"]
    num_objects = graph["num_objects"]
    counts = graph["object_counts"]
    is_empty = graph.get("is_empty", num_objects == 0)
    
    # Opening sentence
    if is_empty or num_objects == 0:
        lines.append(f"This {room_type} is empty.")
        return "\n".join(lines)
    
    lines.append(f"This {room_type} contains {num_objects} objects.")
    
    # List objects by category
    obj_list = []
    for cat, count in sorted(counts.items()):
        if count == 1:
            obj_list.append(f"a {cat}")
        else:
            obj_list.append(f"{count} {cat}s")
    
    if obj_list:
        lines.append(f"Objects: {', '.join(obj_list)}.")
    
    # Describe spatial relations (limited to avoid too much text)
    if graph["relations"]:
        lines.append("")
        lines.append("Spatial layout:")
        
        # Only include each pair once, prioritize nearby objects
        seen = set()
        sorted_relations = sorted(graph["relations"], key=lambda x: x["distance"])
        
        count = 0
        for rel in sorted_relations:
            if count >= 15:  # Limit number of relations
                break
            
            key = tuple(sorted([rel["from_uid"], rel["to_uid"]]))
            if key in seen:
                continue
            seen.add(key)
            
            lines.append(f"  The {rel['from']} is {rel['relation']} the {rel['to']}.")
            count += 1
    
    return "\n".join(lines)


# ============================================================================
# Main Processing
# ============================================================================

def process_one_scene(
    scene_id: str,
    scene_meta: Dict,
    rooms_metadata: List[Dict],
    output_dir: Path
) -> Tuple[bool, Optional[str]]:
    """Process one scene: generate graphs and text descriptions."""
    try:
        # Build scene graph
        scene_graph = build_scene_graph(scene_meta, rooms_metadata)
        scene_text = scene_graph_to_text(scene_graph)
        
        # Save scene graph and text
        # Output structure: graphs/jsons/<scene_id>_scene_graph.json
        jsons_output_dir = output_dir / "jsons"
        jsons_output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(jsons_output_dir / f"{scene_id}_scene_graph.json", "w") as f:
            json.dump(scene_graph, f, indent=2)
        
        # Also save text descriptions (optional, for debugging)
        texts_output_dir = output_dir / "texts"
        texts_output_dir.mkdir(parents=True, exist_ok=True)
        with open(texts_output_dir / f"{scene_id}_scene_description.txt", "w") as f:
            f.write(scene_text)
        
        # Build room graphs
        for room_meta in rooms_metadata:
            room_id = room_meta.get("room_id", room_meta.get("room_type", "Unknown"))
            
            room_graph = build_room_graph(room_meta, scene_meta)
            room_text = room_graph_to_text(room_graph)
            
            # Use scene_id + room_id for unique filename
            # Output structure: graphs/jsons/<scene_id>_<room_id>_room_graph.json
            filename_base = f"{scene_id}_{room_id}"
            
            with open(jsons_output_dir / f"{filename_base}_room_graph.json", "w") as f:
                json.dump(room_graph, f, indent=2)
            
            with open(texts_output_dir / f"{filename_base}_room_description.txt", "w") as f:
                f.write(room_text)
        
        return True, None
    
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return False, str(e)


def load_scene_list(scene_list_path: Path) -> List[str]:
    """Load scene IDs from a text file (one per line)."""
    scenes = []
    with open(scene_list_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scenes.append(scene_id)
    return scenes


def check_scene_graphs_exist(scene_id: str, output_dir: Path) -> bool:
    """Check if graph outputs already exist for a scene."""
    scene_graph = output_dir / "jsons" / f"{scene_id}_scene_graph.json"
    return scene_graph.exists()


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Generate scene and room graphs")
    parser.add_argument("--dataset-root", default=None, help="Root directory of dataset (derives paths from this)")
    parser.add_argument("--metadata-dir", default=None, help="Directory with metadata (required if --dataset-root not provided)")
    parser.add_argument("--output-dir", default=None, help="Output directory for graphs (required if --dataset-root not provided)")
    parser.add_argument("--scene-list", default=None, help="File containing scene IDs (one per line)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes that already have output files")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    # If dataset-root is provided, derive paths from it
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        metadata_dir = Path(args.metadata_dir) if args.metadata_dir else dataset_root / "metadata"
        output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "graphs"
    else:
        # Backward compatibility: require individual paths
        if not args.metadata_dir:
            parser.error("--metadata-dir is required when --dataset-root is not provided")
        if not args.output_dir:
            parser.error("--output-dir is required when --dataset-root is not provided")
        metadata_dir = Path(args.metadata_dir)
        output_dir = Path(args.output_dir)
    
    # Determine which scenes to process
    if args.scene_list:
        scene_list_path = Path(args.scene_list)
        if not scene_list_path.exists():
            logger.error(f"Scene list file not found: {scene_list_path}")
            return
        
        scene_ids = load_scene_list(scene_list_path)
        logger.info(f"Loaded {len(scene_ids)} scene IDs from {scene_list_path}")
        
        # Filter scene metadata files to only those in the list
        scene_meta_files = []
        for scene_id in scene_ids:
            scene_meta_path = metadata_dir / "scenes" / f"{scene_id}.json"
            if scene_meta_path.exists():
                scene_meta_files.append(scene_meta_path)
            else:
                logger.warning(f"Scene metadata not found: {scene_meta_path}")
    else:
        # Discover scenes from metadata directory
        scene_meta_files = list((metadata_dir / "scenes").glob("*.json"))
        if not scene_meta_files:
            logger.error("No scene metadata found")
            return
        logger.info(f"Discovered {len(scene_meta_files)} scenes from metadata directory")
    
    if args.limit:
        scene_meta_files = scene_meta_files[:args.limit]
    
    logger.info(f"Processing {len(scene_meta_files)} scenes...")
    
    success_count = 0
    skip_count = 0
    
    for i, scene_meta_path in enumerate(scene_meta_files, 1):
        scene_id = scene_meta_path.stem
        
        # Skip if outputs exist and --skip-existing is set
        if args.skip_existing:
            if check_scene_graphs_exist(scene_id, output_dir):
                logger.info(f"[{i}/{len(scene_meta_files)}] ⏭ {scene_id} (exists)")
                skip_count += 1
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
            continue
        
        success, _ = process_one_scene(scene_id, scene_meta, rooms_metadata, output_dir)
        
        if success:
            success_count += 1
            logger.info(f"[{i}/{len(scene_meta_files)}] ✓ {scene_id}")
        else:
            logger.warning(f"[{i}/{len(scene_meta_files)}] ✗ {scene_id}")
    
    logger.info(f"\nDone: {success_count}/{len(scene_meta_files)}, {skip_count} skipped")


if __name__ == "__main__":
    main()