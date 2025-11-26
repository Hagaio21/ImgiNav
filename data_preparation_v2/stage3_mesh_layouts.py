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
import cv2

from common.taxonomy import Taxonomy
from common.utils import safe_mkdir, write_json, create_progress_tracker
from data_preparation.utils.file_discovery import gather_paths_from_sources, infer_ids_from_path
from data_preparation_v2.utils.mesh_renderer import (
    render_orthographic_topdown, filter_ceiling_geometry, clip_walls_from_top
)

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

def apply_segmented_colors(mesh: trimesh.Trimesh, label: str, taxonomy: Taxonomy) -> trimesh.Trimesh:
    """Apply taxonomy category colors to mesh vertices."""
    color = taxonomy.get_color(label, mode="category")
    if color is None:
        color = (128, 128, 128)
    
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        rgb = tuple(int(c) for c in color[:3])
    else:
        rgb = (128, 128, 128)
    
    num_vertices = len(mesh.vertices)
    vertex_colors = np.full((num_vertices, 4), 255, dtype=np.uint8)
    vertex_colors[:, :3] = rgb
    
    mesh_seg = mesh.copy()
    mesh_seg.visual.vertex_colors = vertex_colors
    
    if hasattr(mesh_seg.visual, 'material'):
        mesh_seg.visual.material = None
    
    return mesh_seg

# ---------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------

def load_scene_meshes(scene_data: Dict, model_dir: Path, model_info_map: Dict, 
                     taxonomy: Taxonomy, segmented: bool = False) -> Tuple[List[trimesh.Trimesh], List[str], Dict]:
    """Load all meshes from scene, optionally with segmented colors."""
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}
    
    meshes = []
    labels = []
    all_vertices = []
    
    for room in scene_data.get("scene", {}).get("room", []):
        for child in room.get("children", []):
            ref_id = child.get("ref")
            if not ref_id:
                continue
            
            try:
                mesh, label = None, "unknown"
                
                if ref_id in furniture_map:
                    item_info = furniture_map[ref_id]
                    jid = item_info.get('jid')
                    if not jid:
                        continue
                    
                    mesh = load_mesh(model_dir, jid)
                    label = (model_info_map.get(jid, {}).get('category')
                             or item_info.get('title') or "unknown")
                
                elif ref_id in arch_map:
                    arch = arch_map[ref_id]
                    if "Ceiling" in arch.get("type", ""):
                        continue
                    mesh = create_arch_mesh(arch)
                    label = 'floor' if 'Floor' in arch.get("type", "") else 'wall'
                else:
                    continue
                
                if mesh is None or mesh.is_empty:
                    continue
                
                transform = build_transform(child)
                mesh_world = mesh.copy()
                mesh_world.apply_transform(transform)
                
                # Apply segmented colors if requested
                if segmented:
                    mesh_world = apply_segmented_colors(mesh_world, label, taxonomy)
                
                meshes.append(mesh_world)
                labels.append(label)
                all_vertices.append(mesh_world.vertices)
                
            except Exception as e:
                logger.warning(f"Failed to load mesh {ref_id}: {e}")
                continue
    
    if not all_vertices:
        return [], [], {}
    
    all_vertices = np.vstack(all_vertices)
    bbox = {
        "min": all_vertices.min(axis=0).tolist(),
        "max": all_vertices.max(axis=0).tolist()
    }
    
    return meshes, labels, bbox

def load_room_meshes(scene_data: Dict, model_dir: Path, model_info_map: Dict,
                    taxonomy: Taxonomy, room_id: int, segmented: bool = False) -> Tuple[List[trimesh.Trimesh], List[str], Dict]:
    """Load meshes for a specific room."""
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}
    
    rooms = scene_data.get("scene", {}).get("room", [])
    if room_id >= len(rooms):
        return [], [], {}
    
    room = rooms[room_id]
    meshes = []
    labels = []
    all_vertices = []
    
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if not ref_id:
            continue
        
        try:
            mesh, label = None, "unknown"
            
            if ref_id in furniture_map:
                item_info = furniture_map[ref_id]
                jid = item_info.get('jid')
                if not jid:
                    continue
                
                mesh = load_mesh(model_dir, jid)
                label = (model_info_map.get(jid, {}).get('category')
                         or item_info.get('title') or "unknown")
            
            elif ref_id in arch_map:
                arch = arch_map[ref_id]
                if "Ceiling" in arch.get("type", ""):
                    continue
                mesh = create_arch_mesh(arch)
                label = 'floor' if 'Floor' in arch.get("type", "") else 'wall'
            else:
                continue
            
            if mesh is None or mesh.is_empty:
                continue
            
            transform = build_transform(child)
            mesh_world = mesh.copy()
            mesh_world.apply_transform(transform)
            
            if segmented:
                mesh_world = apply_segmented_colors(mesh_world, label, taxonomy)
            
            meshes.append(mesh_world)
            labels.append(label)
            all_vertices.append(mesh_world.vertices)
            
        except Exception as e:
            logger.warning(f"Failed to load mesh {ref_id}: {e}")
            continue
    
    if not all_vertices:
        return [], [], {}
    
    all_vertices = np.vstack(all_vertices)
    bbox = {
        "min": all_vertices.min(axis=0).tolist(),
        "max": all_vertices.max(axis=0).tolist()
    }
    
    return meshes, labels, bbox

def create_scene_layouts(scene_path: Path, model_dir: Path, model_info_file: Path,
                        out_dir: Path, taxonomy: Taxonomy, resolution: int = 512,
                        margin: float = 0.05, clip_walls_amount: float = 0.2):
    """Create scene-level layouts (segmented and textured)."""
    scene_id = infer_ids_from_path(scene_path)
    if isinstance(scene_id, tuple):
        scene_id = scene_id[0]
    scene_id = str(scene_id)
    
    # Load scene JSON
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    
    # Load model_info.json
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}
    
    # Create layouts directory
    layouts_dir = out_dir / scene_id / "layouts"
    safe_mkdir(layouts_dir)
    
    # Load segmented meshes
    seg_meshes, seg_labels, seg_bbox = load_scene_meshes(
        scene, model_dir, model_info_map, taxonomy, segmented=True
    )
    
    # Load textured meshes
    tex_meshes, tex_labels, tex_bbox = load_scene_meshes(
        scene, model_dir, model_info_map, taxonomy, segmented=False
    )
    
    if not seg_meshes or not tex_meshes:
        logger.warning(f"No meshes found for scene {scene_id}")
        return False
    
    # Use scene bbox (from metadata if available, otherwise computed)
    bbox = seg_bbox
    
    # Render segmented layout
    seg_image = render_orthographic_topdown(
        seg_meshes, bbox, resolution=resolution, margin=margin,
        exclude_ceiling=True, clip_walls=True, clip_amount=clip_walls_amount,
        labels=seg_labels
    )
    
    # Render textured layout
    tex_image = render_orthographic_topdown(
        tex_meshes, bbox, resolution=resolution, margin=margin,
        exclude_ceiling=True, clip_walls=True, clip_amount=clip_walls_amount,
        labels=tex_labels
    )
    
    # Save images
    seg_path = layouts_dir / f"{scene_id}_scene_segmented.png"
    tex_path = layouts_dir / f"{scene_id}_scene_textured.png"
    
    cv2.imwrite(str(seg_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tex_path), cv2.cvtColor(tex_image, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Created scene layouts for {scene_id}")
    return True

def create_room_layouts(scene_path: Path, model_dir: Path, model_info_file: Path,
                       metadata_dir: Path, out_dir: Path, taxonomy: Taxonomy,
                       resolution: int = 512, margin: float = 0.02, clip_walls_amount: float = 0.2):
    """Create room-level layouts (segmented and textured) with tight margins."""
    scene_id = infer_ids_from_path(scene_path)
    if isinstance(scene_id, tuple):
        scene_id = scene_id[0]
    scene_id = str(scene_id)
    
    # Load scene JSON
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    
    # Load model_info.json
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}
    
    # Load scene metadata to get room information
    metadata_path = metadata_dir / scene_id / f"{scene_id}_metadata.json"
    if not metadata_path.exists():
        logger.warning(f"Metadata not found for scene {scene_id}")
        return False
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    success_count = 0
    
    for room in metadata.get("rooms", []):
        room_id = room["room_id"]
        
        # Create room layouts directory
        room_layouts_dir = out_dir / scene_id / "rooms" / str(room_id) / "layouts"
        safe_mkdir(room_layouts_dir)
        
        # Load room meshes
        seg_meshes, seg_labels, seg_bbox = load_room_meshes(
            scene, model_dir, model_info_map, taxonomy, room_id, segmented=True
        )
        
        tex_meshes, tex_labels, tex_bbox = load_room_meshes(
            scene, model_dir, model_info_map, taxonomy, room_id, segmented=False
        )
        
        if not seg_meshes or not tex_meshes:
            logger.warning(f"No meshes found for room {room_id} in scene {scene_id}")
            continue
        
        # Use room bbox from metadata (more accurate)
        room_bbox = room.get("room_bbox", seg_bbox)
        
        # Render with tight margin
        seg_image = render_orthographic_topdown(
            seg_meshes, room_bbox, resolution=resolution, margin=margin,
            exclude_ceiling=True, clip_walls=True, clip_amount=clip_walls_amount,
            labels=seg_labels
        )
        
        tex_image = render_orthographic_topdown(
            tex_meshes, room_bbox, resolution=resolution, margin=margin,
            exclude_ceiling=True, clip_walls=True, clip_amount=clip_walls_amount,
            labels=tex_labels
        )
        
        # Save images
        seg_path = room_layouts_dir / f"{scene_id}_room_{room_id}_segmented.png"
        tex_path = room_layouts_dir / f"{scene_id}_room_{room_id}_textured.png"
        
        cv2.imwrite(str(seg_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tex_path), cv2.cvtColor(tex_image, cv2.COLOR_RGB2BGR))
        
        success_count += 1
        logger.info(f"Created layouts for room {room_id} in scene {scene_id}")
    
    return success_count > 0

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
    ap.add_argument("--metadata_dir", required=True, help="Directory with metadata from stage 2")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_info", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--resolution", type=int, default=512, help="Layout image resolution")
    ap.add_argument("--scene_margin", type=float, default=0.05, help="Scene layout margin (fraction)")
    ap.add_argument("--room_margin", type=float, default=0.02, help="Room layout margin (fraction)")
    ap.add_argument("--clip_walls", type=float, default=0.2, help="Amount to clip walls from top (meters)")
    ap.add_argument("--scene_only", action="store_true", help="Only create scene layouts")
    ap.add_argument("--room_only", action="store_true", help="Only create room layouts")
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
        in_dir = Path(args.in_dir)
        scene_paths = list(in_dir.glob("*.json"))
        if not scene_paths:
            print("No scenes found")
            return
    
    if args.limit is not None:
        scene_paths = scene_paths[:args.limit]
    
    out_root = Path(args.out_dir)
    metadata_dir = Path(args.metadata_dir)
    safe_mkdir(out_root)
    
    progress = create_progress_tracker(len(scene_paths), "scenes")
    success_count = 0
    
    for i, scene_path in enumerate(scene_paths, 1):
        try:
            success = True
            
            if not args.room_only:
                success_scene = create_scene_layouts(
                    scene_path, Path(args.model_dir), Path(args.model_info),
                    out_root, TAXONOMY, resolution=args.resolution,
                    margin=args.scene_margin, clip_walls_amount=args.clip_walls
                )
                success = success and success_scene
            
            if not args.scene_only:
                success_room = create_room_layouts(
                    scene_path, Path(args.model_dir), Path(args.model_info),
                    metadata_dir, out_root, TAXONOMY, resolution=args.resolution,
                    margin=args.room_margin, clip_walls_amount=args.clip_walls
                )
                success = success and success_room
            
            if success:
                success_count += 1
                progress(i, scene_path.name, True)
            else:
                progress(i, f"failed {scene_path.name}", False)
        except Exception as e:
            logger.exception(f"Exception processing scene {scene_path.name}: {e}")
            progress(i, f"failed {scene_path.name}: {e}", False)
    
    print(f"\nSuccessfully processed {success_count}/{len(scene_paths)} scenes")


if __name__ == "__main__":
    main()

