#!/usr/bin/env python3

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cdist

from common.taxonomy import Taxonomy
from common.utils import safe_mkdir, write_json, create_progress_tracker
from data_preparation.utils.file_discovery import gather_paths_from_sources, infer_ids_from_path
from data_preparation.utils.geometry_utils import angle_from_center, compute_directional_relations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

TAXONOMY: Taxonomy = None

# ---------------------------------------------------------------------
# Graph Building Functions
# ---------------------------------------------------------------------

def check_room_adjacency(room_a_bbox: Dict, room_b_bbox: Dict, threshold: float = 0.3) -> bool:
    """Check if two rooms are adjacent based on bounding boxes."""
    a_min = np.array(room_a_bbox['min'])
    a_max = np.array(room_a_bbox['max'])
    b_min = np.array(room_b_bbox['min'])
    b_max = np.array(room_b_bbox['max'])
    
    # Compute minimum distance between bounding boxes
    # For each axis, compute distance
    dists = []
    for i in range(3):
        if a_max[i] < b_min[i]:
            dists.append(b_min[i] - a_max[i])
        elif b_max[i] < a_min[i]:
            dists.append(a_min[i] - b_max[i])
        else:
            dists.append(0)  # Overlapping
    
    min_distance = np.sqrt(sum(d**2 for d in dists))
    return min_distance < threshold

def check_object_proximity(obj_a_bbox: Dict, obj_b_bbox: Dict, threshold: float = 0.5) -> Optional[str]:
    """Check proximity between two objects. Returns 'near' if close, None otherwise."""
    a_min = np.array(obj_a_bbox['min'])
    a_max = np.array(obj_a_bbox['max'])
    b_min = np.array(obj_b_bbox['min'])
    b_max = np.array(obj_b_bbox['max'])
    
    # Compute minimum distance
    dists = []
    for i in range(3):
        if a_max[i] < b_min[i]:
            dists.append(b_min[i] - a_max[i])
        elif b_max[i] < a_min[i]:
            dists.append(a_min[i] - b_max[i])
        else:
            dists.append(0)
    
    min_distance = np.sqrt(sum(d**2 for d in dists))
    
    if min_distance < threshold:
        return "near"
    return None

def build_scene_graph(metadata: Dict, adjacency_threshold: float = 0.3) -> Dict:
    """Build scene graph with rooms as nodes and spatial relations as edges."""
    scene_id = metadata["scene_id"]
    rooms = metadata.get("rooms", [])
    
    if len(rooms) == 0:
        return None
    
    # Compute scene center from room locations
    room_centers = [np.array(r["room_location"]) for r in rooms]
    scene_center = np.array(room_centers).mean(axis=0)
    
    # Build nodes
    nodes = []
    for room in rooms:
        nodes.append({
            "id": f"room_{room['room_id']}",
            "room_id": room["room_id"],
            "room_type": room["room_type"],
            "location": room["room_location"],
            "bbox": room["room_bbox"]
        })
    
    # Build edges
    edges = []
    for i, room_a in enumerate(rooms):
        for j, room_b in enumerate(rooms):
            if j <= i:
                continue
            
            # Check adjacency
            is_adjacent = check_room_adjacency(room_a["room_bbox"], room_b["room_bbox"], adjacency_threshold)
            dist_rel = "adjacent" if is_adjacent else None
            
            # Compute directional relations
            center_a = np.array(room_a["room_location"][:2])  # XY only
            center_b = np.array(room_b["room_location"][:2])
            scene_center_2d = scene_center[:2]
            
            ang_a = angle_from_center(scene_center_2d, center_a)
            ang_b = angle_from_center(scene_center_2d, center_b)
            dir_a_to_b, dir_b_to_a = compute_directional_relations(ang_a, ang_b)
            
            edges.append({
                "room_a": f"room_{room_a['room_id']}",
                "room_b": f"room_{room_b['room_id']}",
                "distance_relation": dist_rel,
                "direction_relation": dir_a_to_b
            })
            edges.append({
                "room_a": f"room_{room_b['room_id']}",
                "room_b": f"room_{room_a['room_id']}",
                "distance_relation": dist_rel,
                "direction_relation": dir_b_to_a
            })
    
    scene_graph = {
        "scene_id": scene_id,
        "scene_center": scene_center.tolist(),
        "nodes": nodes,
        "edges": edges
    }
    
    return scene_graph

def build_room_graph(room_metadata: Dict, proximity_threshold: float = 0.5) -> Dict:
    """Build room graph with objects as nodes and spatial relations as edges."""
    scene_id = room_metadata["scene_id"]
    room_id = room_metadata["room_id"]
    objects = room_metadata.get("objects", [])
    
    if len(objects) == 0:
        return None
    
    # Compute room center from object locations
    obj_centers = [np.array(obj["location"]) for obj in objects]
    room_center = np.array(obj_centers).mean(axis=0)
    
    # Build nodes
    nodes = []
    for obj in objects:
        nodes.append({
            "id": f"obj_{obj['object_id']}",
            "object_id": obj["object_id"],
            "label": obj["label"],
            "label_id": obj["label_id"],
            "location": obj["location"],
            "bbox": obj["bbox"]
        })
    
    # Build edges
    edges = []
    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if j <= i:
                continue
            
            # Check proximity
            prox_rel = check_object_proximity(obj_a["bbox"], obj_b["bbox"], proximity_threshold)
            
            # Compute directional relations
            center_a = np.array(obj_a["location"][:2])  # XY only
            center_b = np.array(obj_b["location"][:2])
            room_center_2d = room_center[:2]
            
            ang_a = angle_from_center(room_center_2d, center_a)
            ang_b = angle_from_center(room_center_2d, center_b)
            dir_a_to_b, dir_b_to_a = compute_directional_relations(ang_a, ang_b)
            
            edges.append({
                "obj_a": f"obj_{obj_a['object_id']}",
                "obj_b": f"obj_{obj_b['object_id']}",
                "distance_relation": prox_rel,
                "direction_relation": dir_a_to_b
            })
            edges.append({
                "obj_a": f"obj_{obj_b['object_id']}",
                "obj_b": f"obj_{obj_a['object_id']}",
                "distance_relation": prox_rel,
                "direction_relation": dir_b_to_a
            })
    
    room_graph = {
        "scene_id": scene_id,
        "room_id": room_id,
        "room_center": room_center.tolist(),
        "nodes": nodes,
        "edges": edges
    }
    
    return room_graph

# ---------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------

def process_one_scene(
    metadata_dir: Path, out_dir: Path, scene_id: str,
    adjacency_threshold: float = 0.3, proximity_threshold: float = 0.5
) -> Tuple[bool, Optional[str]]:
    """Process one scene to create scene and room graphs."""
    scene_metadata_path = metadata_dir / scene_id / f"{scene_id}_metadata.json"
    
    if not scene_metadata_path.exists():
        return False, f"Metadata not found for scene {scene_id}"
    
    try:
        # Load scene metadata
        with open(scene_metadata_path, "r", encoding="utf-8") as f:
            scene_metadata = json.load(f)
        
        # Build scene graph
        scene_graph = build_scene_graph(scene_metadata, adjacency_threshold)
        if scene_graph:
            scene_graph_path = out_dir / scene_id / f"{scene_id}_scene_graph.json"
            safe_mkdir(scene_graph_path.parent)
            write_json(scene_graph, scene_graph_path)
            logger.info(f"Created scene graph for {scene_id}")
        
        # Build room graphs
        room_count = 0
        for room in scene_metadata.get("rooms", []):
            room_id = room["room_id"]
            
            # Load room metadata
            room_metadata_path = metadata_dir / scene_id / f"{scene_id}_room_{room_id}_metadata.json"
            if not room_metadata_path.exists():
                logger.warning(f"Room metadata not found for room {room_id} in scene {scene_id}")
                continue
            
            with open(room_metadata_path, "r", encoding="utf-8") as f:
                room_metadata = json.load(f)
            
            # Build room graph
            room_graph = build_room_graph(room_metadata, proximity_threshold)
            if room_graph:
                room_graph_path = out_dir / scene_id / "rooms" / str(room_id) / f"{scene_id}_room_{room_id}_graph.json"
                safe_mkdir(room_graph_path.parent)
                write_json(room_graph, room_graph_path)
                room_count += 1
                logger.info(f"Created room graph for room {room_id} in scene {scene_id}")
        
        return True, None
    except Exception as e:
        error_msg = f"Failed to create graphs for scene {scene_id}: {e}"
        logger.exception(error_msg)
        return False, error_msg

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global TAXONOMY
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_dir", required=True, help="Directory with metadata from stage 2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--adjacency_threshold", type=float, default=0.3,
                    help="Threshold for room adjacency (meters)")
    ap.add_argument("--proximity_threshold", type=float, default=0.5,
                    help="Threshold for object proximity (meters)")
    ap.add_argument("--scene_ids", nargs="+", help="Specific scene IDs to process")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenes to process (default: all)"
    )
    args = ap.parse_args()
    
    TAXONOMY = Taxonomy(Path(args.taxonomy))
    
    metadata_dir = Path(args.metadata_dir)
    out_root = Path(args.out_dir)
    safe_mkdir(out_root)
    
    # Find all scene metadata files
    if args.scene_ids:
        scene_ids = args.scene_ids
    else:
        scene_metadata_files = list(metadata_dir.glob("*/metadata.json"))
        scene_ids = [f.parent.name for f in scene_metadata_files]
    
    if args.limit is not None:
        scene_ids = scene_ids[:args.limit]
    
    if not scene_ids:
        print("No scenes found")
        return
    
    progress = create_progress_tracker(len(scene_ids), "scenes")
    success_count = 0
    failed_scenes = []
    
    for i, scene_id in enumerate(scene_ids, 1):
        try:
            success, error_msg = process_one_scene(
                metadata_dir, out_root, scene_id,
                adjacency_threshold=args.adjacency_threshold,
                proximity_threshold=args.proximity_threshold
            )
            
            if success:
                success_count += 1
                progress(i, scene_id, True)
            else:
                failed_scenes.append({
                    "scene_id": scene_id,
                    "error": error_msg or "Unknown error"
                })
                progress(i, f"failed {scene_id}: {error_msg or 'Unknown error'}", False)
        except Exception as e:
            logger.exception(f"Exception processing scene {scene_id}: {e}")
            failed_scenes.append({
                "scene_id": scene_id,
                "error": str(e)
            })
            progress(i, f"failed {scene_id}: {e}", False)
    
    if failed_scenes:
        failed_manifest_path = out_root / "failed_scenes.csv"
        try:
            import csv
            with open(failed_manifest_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["scene_id", "error"])
                writer.writeheader()
                writer.writerows(failed_scenes)
            logger.info(f"Wrote failed scenes manifest to {failed_manifest_path} ({len(failed_scenes)} failures)")
        except Exception as e:
            logger.error(f"Failed to write failed scenes manifest: {e}")
    
    print(f"\nSuccessfully processed {success_count}/{len(scene_ids)} scenes")


if __name__ == "__main__":
    main()

