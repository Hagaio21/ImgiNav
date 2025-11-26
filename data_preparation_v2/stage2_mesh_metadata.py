#!/usr/bin/env python3

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from common.taxonomy import Taxonomy
from common.utils import safe_mkdir, write_json, create_progress_tracker
from data_preparation.utils.file_discovery import gather_paths_from_sources, infer_ids_from_path
from data_preparation.utils.geometry_utils import pca_plane_fit, build_orthonormal_frame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

TAXONOMY: Taxonomy = None

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def load_mesh(model_dir: Path, jid: str):
    obj_path = model_dir / jid / "raw_model.obj"
    glb_path = model_dir / jid / "raw_model.glb"

    if obj_path.exists():
        resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
        return trimesh.load(str(obj_path), force="mesh", process=False,
                            maintain_order=True, resolver=resolver)
    elif glb_path.exists():
        return trimesh.load(str(glb_path), force="mesh", process=False,
                            maintain_order=True)
    else:
        raise FileNotFoundError(f"Model not found: {obj_path} or {glb_path}")

def create_arch_mesh(arch: Dict):
    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
    faces = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)
    return mesh

def build_transform(child: Dict):
    pos = np.array(child.get("pos", [0, 0, 0]), dtype=np.float64)
    rot = np.array(child.get("rot", [0, 0, 0, 1]), dtype=np.float64)
    scl = np.array(child.get("scale", [1, 1, 1]), dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    Rm = Rotation.from_quat(rot).as_matrix()
    Sm = np.diag(scl)
    T[:3, :3] = Rm @ Sm
    return T

def compute_mesh_bbox(mesh: trimesh.Trimesh) -> Dict:
    """Compute bounding box of a mesh."""
    if mesh.is_empty or len(mesh.vertices) == 0:
        return {"min": [0, 0, 0], "max": [0, 0, 0]}
    vertices = mesh.vertices
    return {
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist()
    }

def compute_mesh_centroid(mesh: trimesh.Trimesh) -> List[float]:
    """Compute centroid of a mesh."""
    if mesh.is_empty or len(mesh.vertices) == 0:
        return [0, 0, 0]
    return mesh.vertices.mean(axis=0).tolist()

def compute_room_floor_plane(room_meshes: List[trimesh.Trimesh], floor_label_ids: List[int], taxonomy: Taxonomy) -> Tuple[np.ndarray, np.ndarray]:
    """Compute floor plane from room meshes using floor-labeled geometry."""
    floor_points = []
    
    for mesh_info in room_meshes:
        label = mesh_info.get('label', '')
        label_id = taxonomy.translate(label, output="id")
        
        if label_id in floor_label_ids or 'floor' in label.lower():
            vertices = mesh_info['mesh'].vertices
            floor_points.append(vertices)
    
    if len(floor_points) == 0:
        # Fallback: use lowest Z points
        all_vertices = np.vstack([m['mesh'].vertices for m in room_meshes])
        if len(all_vertices) == 0:
            return np.array([0, 0, 0]), np.array([0, 0, 1])
        z_min = all_vertices[:, 2].min()
        z_threshold = z_min + 0.1  # 10cm above minimum
        floor_points = [all_vertices[all_vertices[:, 2] <= z_threshold]]
    
    if len(floor_points) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 1])
    
    floor_vertices = np.vstack(floor_points)
    if len(floor_vertices) < 3:
        return np.array([0, 0, 0]), np.array([0, 0, 1])
    
    origin, normal = pca_plane_fit(floor_vertices)
    # Ensure normal points upward
    if normal[2] < 0:
        normal = -normal
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    
    return origin, normal

# ---------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------

def extract_scene_metadata(scene_data: Dict, model_dir: Path, model_info_map: Dict, 
                          taxonomy: Taxonomy, scene_id: str) -> Dict:
    """Extract comprehensive metadata from scene."""
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}
    
    # Get floor label IDs from taxonomy
    floor_label_ids = taxonomy.get_floor_ids()
    
    # Process all objects and group by room
    room_data = {}
    all_objects = []
    
    for room_idx, room in enumerate(scene_data.get("scene", {}).get("room", [])):
        room_type = room.get("type", "UnknownRoom")
        room_id = room_idx
        
        room_objects = []
        room_meshes = []
        
        for child_index, child in enumerate(room.get("children", [])):
            ref_id = child.get("ref")
            if not ref_id:
                continue
            
            try:
                mesh, label, model_item = None, "unknown", None
                
                if ref_id in furniture_map:
                    item_info = furniture_map[ref_id]
                    jid = item_info.get('jid')
                    if not jid:
                        continue
                    
                    mesh = load_mesh(model_dir, jid)
                    label = (model_info_map.get(jid, {}).get('category')
                             or item_info.get('title') or "unknown")
                    model_item = item_info
                
                elif ref_id in arch_map:
                    arch = arch_map[ref_id]
                    if "Ceiling" in arch.get("type", ""):
                        continue  # Skip ceilings
                    mesh = create_arch_mesh(arch)
                    label = 'floor' if 'Floor' in arch.get("type", "") else 'wall'
                else:
                    continue
                
                if mesh is None or mesh.is_empty:
                    continue
                
                transform = build_transform(child)
                mesh_world = mesh.copy()
                mesh_world.apply_transform(transform)
                
                # Get taxonomy IDs
                label_id = taxonomy.translate(label, output="id") or 0
                category_id = taxonomy.translate(label, output="id") or 0
                super_id = taxonomy.get_sup(label, output="id") or 0
                room_id_tax = taxonomy.translate(room_type, output="id") or 0
                
                # Object metadata
                obj_bbox = compute_mesh_bbox(mesh_world)
                obj_centroid = compute_mesh_centroid(mesh_world)
                
                obj_metadata = {
                    "object_id": ref_id,
                    "label": label,
                    "label_id": int(label_id),
                    "category_id": int(category_id),
                    "super_id": int(super_id),
                    "bbox": obj_bbox,
                    "location": obj_centroid,
                    "transform": transform.tolist(),
                    "room_id": room_id
                }
                
                room_objects.append(obj_metadata)
                all_objects.append(obj_metadata)
                room_meshes.append({
                    'mesh': mesh_world,
                    'label': label
                })
                
            except Exception as e:
                logger.warning(f"Failed to process object {ref_id}: {e}")
                continue
        
        if not room_objects:
            continue
        
        # Compute room bounding box
        room_vertices = np.vstack([m['mesh'].vertices for m in room_meshes])
        if len(room_vertices) == 0:
            continue
        
        room_bbox = {
            "min": room_vertices.min(axis=0).tolist(),
            "max": room_vertices.max(axis=0).tolist()
        }
        room_centroid = room_vertices.mean(axis=0).tolist()
        
        # Compute room floor plane and up direction
        floor_origin, up_normal = compute_room_floor_plane(room_meshes, floor_label_ids, taxonomy)
        
        room_data[room_id] = {
            "room_id": room_id,
            "room_type": room_type,
            "room_id_taxonomy": int(taxonomy.translate(room_type, output="id") or 0),
            "room_bbox": room_bbox,
            "room_location": room_centroid,
            "floor_origin": floor_origin.tolist(),
            "up_direction": up_normal.tolist(),
            "objects": room_objects
        }
    
    # Compute scene-level bounding box
    if not all_objects:
        return None
    
    all_vertices = []
    for obj in all_objects:
        bbox = obj['bbox']
        all_vertices.append(bbox['min'])
        all_vertices.append(bbox['max'])
    
    all_vertices = np.array(all_vertices)
    scene_bbox = {
        "min": all_vertices.min(axis=0).tolist(),
        "max": all_vertices.max(axis=0).tolist()
    }
    
    # Compute scene up direction (use first room's up direction or default)
    scene_up = [0, 0, 1]
    if room_data:
        first_room = list(room_data.values())[0]
        scene_up = first_room['up_direction']
    
    # Build comprehensive metadata
    metadata = {
        "scene_id": scene_id,
        "up_direction": scene_up,
        "scene_bbox": scene_bbox,
        "scene_size": (np.array(scene_bbox["max"]) - np.array(scene_bbox["min"])).tolist(),
        "num_rooms": len(room_data),
        "num_objects": len(all_objects),
        "rooms": list(room_data.values())
    }
    
    return metadata

def process_one_scene(
    scene_path: Path, model_dir: Path, model_info_file: Path, 
    out_root: Path, taxonomy: Taxonomy
) -> Tuple[bool, Optional[str]]:
    scene_id = infer_ids_from_path(scene_path)
    if isinstance(scene_id, tuple):
        scene_id = scene_id[0]
    scene_id = str(scene_id)
    
    out_dir = out_root / scene_id
    safe_mkdir(out_dir)
    
    # Load scene JSON
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    
    # Load model_info.json
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}
    
    # Extract metadata
    try:
        metadata = extract_scene_metadata(scene, model_dir, model_info_map, taxonomy, scene_id)
        
        if metadata is None:
            return False, "No objects found in scene"
        
        # Save scene-level metadata
        write_json(metadata, out_dir / f"{scene_id}_metadata.json")
        
        # Save per-room metadata
        for room in metadata["rooms"]:
            room_id = room["room_id"]
            room_metadata = {
                "scene_id": scene_id,
                "room_id": room_id,
                "room_type": room["room_type"],
                "room_id_taxonomy": room["room_id_taxonomy"],
                "room_bbox": room["room_bbox"],
                "room_location": room["room_location"],
                "floor_origin": room["floor_origin"],
                "up_direction": room["up_direction"],
                "objects": room["objects"]
            }
            write_json(room_metadata, out_dir / f"{scene_id}_room_{room_id}_metadata.json")
        
        return True, None
    except Exception as e:
        error_msg = f"Failed to extract metadata for scene {scene_id}: {e}"
        logger.exception(error_msg)
        return False, error_msg

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global TAXONOMY
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--scene_list")
    ap.add_argument("--scene_file")
    ap.add_argument("--in_dir", required=True, help="Input directory with scene JSON files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_info", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenes to process (default: all)"
    )
    args = ap.parse_args()
    
    TAXONOMY = Taxonomy(Path(args.taxonomy))
    
    scene_paths = gather_paths_from_sources(args.scene_file, args.scenes, args.scene_list)
    if not scene_paths:
        # Try to find scenes in input directory
        in_dir = Path(args.in_dir)
        scene_paths = list(in_dir.glob("*.json"))
        if not scene_paths:
            print("No scenes found")
            return
    
    if args.limit is not None:
        scene_paths = scene_paths[:args.limit]
    
    out_root = Path(args.out_dir)
    safe_mkdir(out_root)
    
    progress = create_progress_tracker(len(scene_paths), "scenes")
    success_count = 0
    failed_scenes = []
    
    for i, scene_path in enumerate(scene_paths, 1):
        try:
            success, error_msg = process_one_scene(
                scene_path, Path(args.model_dir), Path(args.model_info), 
                out_root, TAXONOMY
            )
            if success:
                success_count += 1
                progress(i, scene_path.name, True)
            else:
                failed_scenes.append({
                    "scene_id": scene_path.stem,
                    "scene_path": str(scene_path),
                    "error": error_msg or "Unknown error"
                })
                progress(i, f"failed {scene_path.name}: {error_msg or 'Unknown error'}", False)
        except Exception as e:
            error_msg = f"Exception processing scene {scene_path.name}: {e}"
            logger.exception(error_msg)
            failed_scenes.append({
                "scene_id": scene_path.stem,
                "scene_path": str(scene_path),
                "error": str(e)
            })
            progress(i, f"failed {scene_path.name}: {e}", False)
    
    if failed_scenes:
        failed_manifest_path = out_root / "failed_scenes.csv"
        try:
            import csv
            with open(failed_manifest_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["scene_id", "scene_path", "error"])
                writer.writeheader()
                writer.writerows(failed_scenes)
            logger.info(f"Wrote failed scenes manifest to {failed_manifest_path} ({len(failed_scenes)} failures)")
        except Exception as e:
            logger.error(f"Failed to write failed scenes manifest: {e}")
    
    print(f"\nSuccessfully processed {success_count}/{len(scene_paths)} scenes")


if __name__ == "__main__":
    main()

