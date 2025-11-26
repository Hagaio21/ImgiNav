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

# --- imports ---
from common.taxonomy import Taxonomy
from common.utils import (
    load_config_with_profile, create_progress_tracker,
    safe_mkdir, write_json
)
from data_preparation.utils.file_discovery import gather_paths_from_sources, infer_ids_from_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global Taxonomy Object ---
TAXONOMY: Taxonomy = None
ARGS = None

# ---------------------------------------------------------------------
# Utility Functions (reused from stage1_build_scenes.py)
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

def apply_segmented_colors(mesh: trimesh.Trimesh, label: str, taxonomy: Taxonomy) -> trimesh.Trimesh:
    """Apply taxonomy category colors to mesh vertices."""
    # Get color from taxonomy (using category mode)
    color = taxonomy.get_color(label, mode="category")
    if color is None:
        color = (128, 128, 128)  # Default gray
    
    # Ensure color is RGB tuple
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        rgb = tuple(int(c) for c in color[:3])
    else:
        rgb = (128, 128, 128)
    
    # Apply color to all vertices
    num_vertices = len(mesh.vertices)
    vertex_colors = np.full((num_vertices, 4), 255, dtype=np.uint8)
    vertex_colors[:, :3] = rgb
    
    mesh_seg = mesh.copy()
    mesh_seg.visual.vertex_colors = vertex_colors
    
    # Clear any existing textures/materials for segmented version
    if hasattr(mesh_seg.visual, 'material'):
        mesh_seg.visual.material = None
    
    return mesh_seg

# ---------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------

def process_scene_meshes(scene_data, model_dir, model_info_map, taxonomy: Taxonomy):
    """Process scene and create both segmented and textured meshes."""
    failed_models = {}
    scene_objects = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}

    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "UnknownRoom")

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
                        raise ValueError("Missing 'jid'")

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
                    failed_models[ref_id] = "Reference not found"
                    continue

                if mesh is None or mesh.is_empty:
                    raise ValueError("Empty mesh")

                transform = build_transform(child)

                scene_objects.append({
                    "mesh": mesh,
                    "transform": transform,
                    "label": label,
                    "room_type": room_type,
                    "node_name": f"{ref_id}_{child_index}",
                    "model_item": model_item,
                    "ref_id": ref_id
                })

            except Exception as e:
                logger.exception(f"Failed to process model {ref_id} in scene: {e}")
                failed_models[ref_id] = str(e)

    if not scene_objects:
        return None, None, failed_models

    # Build segmented and textured scenes
    segmented_scene = trimesh.Scene()
    textured_scene = trimesh.Scene()
    
    all_vertices = []
    
    for obj in scene_objects:
        # Create segmented version with taxonomy colors
        mesh_seg = apply_segmented_colors(obj['mesh'], obj['label'], taxonomy)
        segmented_scene.add_geometry(mesh_seg, node_name=obj['node_name'],
                                     transform=obj['transform'])
        
        # Keep original textured version
        textured_scene.add_geometry(obj['mesh'], node_name=obj['node_name'],
                                  transform=obj['transform'])
        
        # Collect vertices for bbox computation
        mesh_world = obj['mesh'].copy()
        mesh_world.apply_transform(obj['transform'])
        all_vertices.append(mesh_world.vertices)
    
    if not all_vertices:
        return None, None, failed_models
    
    # Compute scene bounding box
    all_vertices = np.vstack(all_vertices)
    scene_bbox = {
        "min": all_vertices.min(axis=0).tolist(),
        "max": all_vertices.max(axis=0).tolist()
    }
    
    return segmented_scene, textured_scene, failed_models, scene_bbox

def process_one_scene(
    scene_path: Path, model_dir: Path, model_info_file: Path, out_root: Path,
    args: argparse.Namespace, taxonomy: Taxonomy
) -> Tuple[bool, Optional[str]]:
    scene_id = infer_ids_from_path(scene_path)
    if isinstance(scene_id, tuple):
        scene_id = scene_id[0]
    scene_id = str(scene_id)

    out_dir = out_root / scene_id if args.per_scene_subdir else out_root
    if args.per_scene_subdir:
        safe_mkdir(out_dir)

    # Load scene JSON
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    # Load model_info.json
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}

    # Process scene: returns (segmented_scene, textured_scene, failed_models, scene_bbox)
    segmented_scene, textured_scene, failed, scene_bbox = process_scene_meshes(
        scene, model_dir, model_info_map, taxonomy)

    if segmented_scene is None or textured_scene is None:
        error_msg = f"No meshes created for scene {scene_id}"
        logger.warning(error_msg)
        return False, error_msg

    # Export meshes
    try:
        # Determine export format
        export_format = args.format.lower()
        
        # Export segmented mesh
        if export_format == "glb":
            segmented_scene.export(out_dir / f"{scene_id}_segmented.glb", file_type="glb")
        elif export_format == "obj":
            segmented_scene.export(out_dir / f"{scene_id}_segmented.obj", file_type="obj")
        elif export_format == "ply":
            # For PLY, we need to merge all meshes first
            meshes_list = list(segmented_scene.geometry.values())
            if meshes_list:
                merged = trimesh.util.concatenate(meshes_list)
                merged.export(out_dir / f"{scene_id}_segmented.ply", file_type="ply")
        else:
            raise ValueError(f"Unsupported format: {export_format}")
        
        # Export textured mesh
        if export_format == "glb":
            textured_scene.export(out_dir / f"{scene_id}_textured.glb", file_type="glb")
        elif export_format == "obj":
            textured_scene.export(out_dir / f"{scene_id}_textured.obj", file_type="obj")
        elif export_format == "ply":
            meshes_list = list(textured_scene.geometry.values())
            if meshes_list:
                merged = trimesh.util.concatenate(meshes_list)
                merged.export(out_dir / f"{scene_id}_textured.ply", file_type="ply")
        
        # Save initial scene metadata
        scene_metadata = {
            "scene_id": scene_id,
            "bounds": scene_bbox,
            "size": (np.array(scene_bbox["max"]) - np.array(scene_bbox["min"])).tolist(),
            "up_normal": [0.0, 0.0, 1.0],  # Default, will be refined in stage 2
            "segmented_mesh": f"{scene_id}_segmented.{export_format}",
            "textured_mesh": f"{scene_id}_textured.{export_format}"
        }
        write_json(scene_metadata, out_dir / f"{scene_id}_scene_metadata.json")
        
        return True, None
    except Exception as e:
        error_msg = f"Failed to export meshes for scene {scene_id}: {e}"
        logger.exception(error_msg)
        return False, error_msg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global TAXONOMY
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--scene_list")
    ap.add_argument("--scene_file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_info", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--format", choices=["obj", "glb", "ply"], default="glb",
                    help="Mesh export format")
    ap.add_argument("--per_scene_subdir", action="store_true", default=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenes to process (default: all)"
    )
    args = ap.parse_args()
    ARGS = args
    TAXONOMY = Taxonomy(Path(args.taxonomy))

    scene_paths = gather_paths_from_sources(args.scene_file, args.scenes, args.scene_list)
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
            success, error_msg = process_one_scene(scene_path, Path(args.model_dir),
                                                   Path(args.model_info), out_root, args, TAXONOMY)
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
    
    # Write failed scenes manifest if any failures occurred
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

