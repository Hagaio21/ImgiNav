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
from data_preparation_v2.utils.mesh_renderer import render_perspective
from data_preparation_v2.stage3_mesh_layouts import (
    load_room_meshes, load_mesh, create_arch_mesh, build_transform, apply_segmented_colors
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

def compute_room_corners(room_bbox: Dict, floor_origin: List[float], up_direction: List[float]) -> List[np.ndarray]:
    """
    Compute 4 corner positions for room camera placement.
    Corners are at the floor level, offset slightly inward from bbox corners.
    """
    bbox_min = np.array(room_bbox['min'])
    bbox_max = np.array(room_bbox['max'])
    
    # Compute floor plane (XY plane at floor level)
    floor_z = floor_origin[2] if len(floor_origin) >= 3 else bbox_min[2]
    
    # Get XY bounds
    x_min, y_min = bbox_min[0], bbox_min[1]
    x_max, y_max = bbox_max[0], bbox_max[1]
    
    # Offset inward by small amount (5% of room size)
    x_size = x_max - x_min
    y_size = y_max - y_min
    offset_x = x_size * 0.05
    offset_y = y_size * 0.05
    
    x_min_offset = x_min + offset_x
    x_max_offset = x_max - offset_x
    y_min_offset = y_min + offset_y
    y_max_offset = y_max - offset_y
    
    # Four corners
    corners = [
        np.array([x_min_offset, y_min_offset, floor_z]),  # Bottom-left
        np.array([x_max_offset, y_min_offset, floor_z]),  # Bottom-right
        np.array([x_max_offset, y_max_offset, floor_z]),  # Top-right
        np.array([x_min_offset, y_max_offset, floor_z]),  # Top-left
    ]
    
    return corners

def create_room_povs(scene_path: Path, model_dir: Path, model_info_file: Path,
                     metadata_dir: Path, out_dir: Path, taxonomy: Taxonomy,
                     resolution: Tuple[int, int] = (1280, 800), fov_deg: float = 70.0,
                     eye_height: float = 1.6, clip_walls_amount: float = 0.2):
    """Create POV images for all rooms in a scene."""
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
    
    # Load scene metadata
    metadata_path = metadata_dir / scene_id / f"{scene_id}_metadata.json"
    if not metadata_path.exists():
        logger.warning(f"Metadata not found for scene {scene_id}")
        return False
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    success_count = 0
    
    for room in metadata.get("rooms", []):
        room_id = room["room_id"]
        room_bbox = room["room_bbox"]
        room_location = np.array(room["room_location"])
        floor_origin = room.get("floor_origin", room_location.tolist())
        up_direction = room.get("up_direction", [0, 0, 1])
        
        # Create POV directory
        pov_dir = out_dir / scene_id / "rooms" / str(room_id) / "povs"
        safe_mkdir(pov_dir)
        seg_dir = pov_dir / "seg"
        tex_dir = pov_dir / "tex"
        safe_mkdir(seg_dir)
        safe_mkdir(tex_dir)
        
        # Load room meshes
        seg_meshes, seg_labels, _ = load_room_meshes(
            scene, model_dir, model_info_map, taxonomy, room_id, segmented=True
        )
        
        tex_meshes, tex_labels, _ = load_room_meshes(
            scene, model_dir, model_info_map, taxonomy, room_id, segmented=False
        )
        
        if not seg_meshes or not tex_meshes:
            logger.warning(f"No meshes found for room {room_id} in scene {scene_id}")
            continue
        
        # Compute camera positions (corners)
        corners = compute_room_corners(room_bbox, floor_origin, up_direction)
        
        if len(corners) != 4:
            logger.warning(f"Could not compute corners for room {room_id}")
            continue
        
        # Camera target is room center
        camera_target = room_location
        
        # Render from each corner
        for view_idx, corner_pos in enumerate(corners, start=1):
            view_id = f"v{view_idx:02d}"
            
            # Render segmented version
            seg_image = render_perspective(
                seg_meshes, corner_pos, camera_target, fov_deg=fov_deg,
                resolution=resolution, eye_height=eye_height,
                exclude_ceiling=True, clip_walls=True, clip_amount=clip_walls_amount,
                labels=seg_labels, bg_color=(0, 0, 0)
            )
            
            # Render textured version
            tex_image = render_perspective(
                tex_meshes, corner_pos, camera_target, fov_deg=fov_deg,
                resolution=resolution, eye_height=eye_height,
                exclude_ceiling=True, clip_walls=True, clip_amount=clip_walls_amount,
                labels=tex_labels, bg_color=(0, 0, 0)
            )
            
            # Save images
            seg_path = seg_dir / f"{scene_id}_room_{room_id}_pov_{view_id}_seg.png"
            tex_path = tex_dir / f"{scene_id}_room_{room_id}_pov_{view_id}_tex.png"
            
            cv2.imwrite(str(seg_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(tex_path), cv2.cvtColor(tex_image, cv2.COLOR_RGB2BGR))
        
        success_count += 1
        logger.info(f"Created POVs for room {room_id} in scene {scene_id} (8 images)")
    
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
    ap.add_argument("--width", type=int, default=1280, help="Image width")
    ap.add_argument("--height", type=int, default=800, help="Image height")
    ap.add_argument("--fov", type=float, default=70.0, help="Field of view in degrees")
    ap.add_argument("--eye_height", type=float, default=1.6, help="Eye height in meters")
    ap.add_argument("--clip_walls", type=float, default=0.2, help="Amount to clip walls from top (meters)")
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
    
    resolution = (args.width, args.height)
    
    progress = create_progress_tracker(len(scene_paths), "scenes")
    success_count = 0
    
    for i, scene_path in enumerate(scene_paths, 1):
        try:
            success = create_room_povs(
                scene_path, Path(args.model_dir), Path(args.model_info),
                metadata_dir, out_root, TAXONOMY, resolution=resolution,
                fov_deg=args.fov, eye_height=args.eye_height, clip_walls_amount=args.clip_walls
            )
            
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

