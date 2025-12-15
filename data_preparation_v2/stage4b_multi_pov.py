#!/usr/bin/env python3
"""
Stage 4b: Multi-POV Generation for Iterative Refinement

Extends stage4_render_povs_v2.py to generate multiple POVs per door/window:
- Base POV: At the door/window (existing)
- Step 1 + sweep: 1 step into room, looking left/center/right
- Step 2: 2 steps into room, looking center

This provides POV sequences for iterative refinement experiments.

Usage:
    # From existing pov_info.json
    python stage4b_multi_pov.py \
        --dataset-root /path/to/dataset \
        --existing-pov-info /path/to/pov_info.json \
        --step-distance 0.5 \
        --sweep-angles -30 0 30

    # From evaluation manifest CSV
    python stage4b_multi_pov.py \
        --dataset-root /path/to/dataset \
        --manifest /path/to/manifest_val.csv \
        --hpc --backend xvfb
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_xvfb_display = None


# ============================================================================
# HPC Headless Rendering Setup
# ============================================================================

def setup_hpc_rendering(backend: str = "auto") -> bool:
    """Set up rendering backend for HPC headless rendering."""
    global _xvfb_display
    
    if backend == "auto":
        for try_backend in ["xvfb", "osmesa", "egl"]:
            if setup_hpc_rendering(try_backend):
                return True
        return False
    
    elif backend == "xvfb":
        try:
            from xvfbwrapper import Xvfb
            _xvfb_display = Xvfb(width=1920, height=1080, colordepth=24)
            _xvfb_display.start()
            os.environ['DISPLAY'] = f':{_xvfb_display.new_display}'
            logger.info(f"Xvfb started on display :{_xvfb_display.new_display}")
            return True
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
            logger.info("Using EGL backend")
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
            logger.info("Using OSMesa backend")
            return True
        except Exception as e:
            logger.debug(f"OSMesa backend failed: {e}")
            if "PYOPENGL_PLATFORM" in os.environ:
                del os.environ["PYOPENGL_PLATFORM"]
            return False
    
    return False


def cleanup_hpc_rendering():
    """Clean up HPC rendering resources."""
    global _xvfb_display
    if _xvfb_display is not None:
        try:
            _xvfb_display.stop()
            logger.info("Xvfb stopped")
        except Exception:
            pass
        _xvfb_display = None


# ============================================================================
# GLB Loading and Rendering (lazy imports)
# ============================================================================

def load_glb_with_transforms(glb_path: Path) -> List:
    """Load GLB and return meshes with transforms applied."""
    import trimesh
    
    if not glb_path.exists():
        logger.warning(f"GLB not found: {glb_path}")
        return []
    
    try:
        scene = trimesh.load(str(glb_path), process=False, force="scene")
        
        if not isinstance(scene, trimesh.Scene):
            return [scene] if isinstance(scene, trimesh.Trimesh) else []
        
        meshes = scene.dump(concatenate=False)
        if meshes is None:
            meshes = []
        return [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
    except Exception as e:
        logger.error(f"Failed to load GLB {glb_path}: {e}")
        return []


def try_render_pov_pyrender(
    meshes: List,
    camera_info: Dict,
    width: int,
    height: int,
    fov: float = 80.0
) -> Optional[Image.Image]:
    """Render first-person POV using pyrender."""
    try:
        import pyrender
        import trimesh
    except ImportError:
        logger.warning("pyrender not available for POV rendering")
        return None
    
    try:
        eye = np.array(camera_info["eye"])
        look_at = np.array(camera_info["look_at"])
        up = np.array([0.0, 1.0, 0.0])
        
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
        
        forward = look_at - eye
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
        logger.warning(f"pyrender POV failed: {e}")
        return None


# ============================================================================
# Camera Position Computation
# ============================================================================

def compute_stepped_camera_positions(
    base_camera_info: Dict,
    step_distances: List[float],
    sweep_angles: List[float],
    eye_height: float = 1.6,
) -> List[Dict]:
    """
    Compute camera positions for stepped exploration with angular sweep.
    
    Args:
        base_camera_info: Original camera info from door/window POV
        step_distances: List of step distances into the room (meters)
        sweep_angles: List of yaw angles for sweep (degrees, 0=forward)
        eye_height: Camera height (meters)
    
    Returns:
        List of camera_info dicts for each POV
    """
    cameras = []
    
    # Base position (at door/window)
    base_pos = np.array(base_camera_info["opening_center"])
    base_rotation = base_camera_info["rotation_deg"]
    
    # Forward direction (into the room) in world coords
    forward_rad = math.radians(base_rotation)
    forward_dir = np.array([
        math.sin(forward_rad),  # X
        0,                       # Y (no vertical movement)
        math.cos(forward_rad)   # Z
    ])
    
    # Eye position
    eye_pos = base_pos.copy()
    eye_pos[1] = eye_height
    
    # POV 0: Base position (at door), looking straight
    look_at = eye_pos + forward_dir * 3.0
    cameras.append({
        "position": base_pos.copy(),
        "eye": eye_pos.tolist(),
        "look_at": look_at.tolist(),
        "rotation_deg": base_rotation,
        "step_idx": 0,
        "sweep_angle": 0,
        "pov_name": "step0_center",
        "opening_center": base_camera_info["opening_center"],
    })
    
    # Generate stepped positions
    for step_idx, step_dist in enumerate(step_distances, 1):
        # Position: move forward into room
        stepped_pos = base_pos + forward_dir * step_dist
        stepped_eye = stepped_pos.copy()
        stepped_eye[1] = eye_height
        
        for sweep_angle in sweep_angles:
            # Rotation: base rotation + sweep angle
            total_rotation = base_rotation + sweep_angle
            total_rad = math.radians(total_rotation)
            
            # Look direction with sweep
            look_dir = np.array([
                math.sin(total_rad),
                0,
                math.cos(total_rad)
            ])
            look_at = stepped_eye + look_dir * 3.0
            
            sweep_name = "left" if sweep_angle < -10 else "right" if sweep_angle > 10 else "center"
            
            cameras.append({
                "position": stepped_pos.copy(),
                "eye": stepped_eye.tolist(),
                "look_at": look_at.tolist(),
                "rotation_deg": total_rotation,
                "step_idx": step_idx,
                "sweep_angle": sweep_angle,
                "pov_name": f"step{step_idx}_{sweep_name}",
                "opening_center": base_camera_info["opening_center"],
            })
    
    return cameras


def render_multi_povs(
    meshes: List,
    camera_positions: List[Dict],
    width: int = 1280,
    height: int = 720,
    fov: float = 80.0,
) -> List[Image.Image]:
    """Render POV images for multiple camera positions."""
    images = []
    
    for cam_info in camera_positions:
        img = try_render_pov_pyrender(meshes, cam_info, width, height, fov)
        if img is None:
            img = Image.new("RGB", (width, height), (135, 206, 235))
        images.append(img)
    
    return images


# ============================================================================
# Manifest Processing
# ============================================================================

def compute_opening_center(opening: Dict) -> Optional[np.ndarray]:
    """Get center position of door/window from bbox."""
    bbox = opening.get("bbox")
    if bbox and "min" in bbox and "max" in bbox:
        bbox_min = np.array(bbox["min"])
        bbox_max = np.array(bbox["max"])
        return (bbox_min + bbox_max) / 2.0
    
    pos = opening.get("position")
    if pos and len(pos) >= 3:
        return np.array(pos, dtype=np.float64)
    
    return None


def determine_opening_wall(
    opening: Dict,
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray
) -> str:
    """Determine which wall the opening is on."""
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return "min_z"
    
    dist_to_min_x = abs(opening_center[0] - room_bbox_min[0])
    dist_to_max_x = abs(opening_center[0] - room_bbox_max[0])
    dist_to_min_z = abs(opening_center[2] - room_bbox_min[2])
    dist_to_max_z = abs(opening_center[2] - room_bbox_max[2])
    
    min_dist = min(dist_to_min_x, dist_to_max_x, dist_to_min_z, dist_to_max_z)
    
    if min_dist == dist_to_min_x:
        return "min_x"
    elif min_dist == dist_to_max_x:
        return "max_x"
    elif min_dist == dist_to_min_z:
        return "min_z"
    else:
        return "max_z"


def get_wall_rotation(wall: str) -> float:
    """Get rotation angle (degrees) to look into room from wall."""
    rotation_map = {
        "min_z": 0.0,    # Looking +Z
        "max_z": 180.0,  # Looking -Z
        "min_x": 90.0,   # Looking +X
        "max_x": 270.0,  # Looking -X
    }
    return rotation_map.get(wall, 0.0)


def get_inside_direction(wall: str) -> np.ndarray:
    """Get unit vector pointing from wall into room."""
    directions = {
        "min_x": np.array([1.0, 0.0, 0.0]),
        "max_x": np.array([-1.0, 0.0, 0.0]),
        "min_z": np.array([0.0, 0.0, 1.0]),
        "max_z": np.array([0.0, 0.0, -1.0]),
    }
    return directions.get(wall, np.array([0.0, 0.0, 1.0]))


def load_room_metadata(scene_id: str, room_id: str, dataset_root: Path) -> Optional[Dict]:
    """Load room metadata JSON."""
    # Try different path patterns
    possible_paths = [
        dataset_root / "metadata" / "rooms" / f"{scene_id}_{room_id}.json",
        dataset_root / "metadata" / "rooms" / f"{scene_id}_{room_id}_room.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load room metadata {path}: {e}")
    
    logger.warning(f"Room metadata not found for {scene_id}/{room_id}")
    return None


def get_opening_from_metadata(room_meta: Dict, pov_id: str, pov_type: str) -> Optional[Dict]:
    """Get specific door/window opening from room metadata."""
    # Parse pov_id to get index (e.g., "door0" -> 0, "window1" -> 1)
    try:
        if pov_type == "door":
            idx = int(pov_id.replace("door", ""))
            openings = room_meta.get("doors", [])
        else:  # window
            idx = int(pov_id.replace("window", ""))
            openings = room_meta.get("windows", [])
        
        if idx < len(openings):
            return openings[idx]
        else:
            logger.warning(f"Opening index {idx} out of range for {pov_type}s (have {len(openings)})")
            return None
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse pov_id {pov_id}: {e}")
        return None


def compute_camera_from_opening(
    opening: Dict,
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray,
    eye_height: float = 1.6,
) -> Optional[Dict]:
    """Compute camera info from opening and room bbox (like original script)."""
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return None
    
    wall = determine_opening_wall(opening, room_bbox_min, room_bbox_max)
    rotation_deg = get_wall_rotation(wall)
    inside_dir = get_inside_direction(wall)
    
    # Eye position at opening center, at eye height
    eye = opening_center.copy()
    eye[1] = eye_height
    
    # Look at point: 3 meters into room
    look_at = eye + inside_dir * 3.0
    
    return {
        "opening_center": opening_center.tolist(),
        "rotation_deg": rotation_deg,
        "wall": wall,
        "inside_direction": inside_dir.tolist(),
        "eye": eye.tolist(),
        "look_at": look_at.tolist(),
    }


def process_manifest(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    step_distances: List[float] = [0.5, 1.0],
    sweep_angles: List[float] = [-30, 0, 30],
    render_images: bool = True,
    pov_width: int = 1280,
    pov_height: int = 720,
    pov_fov: float = 80.0,
    limit: int = None,
) -> List[Dict]:
    """
    Process evaluation manifest and generate multi-step POVs.
    
    Args:
        manifest_path: Path to manifest CSV
        dataset_root: Dataset root directory
        output_dir: Output directory for new POVs
        step_distances: Distances to step into room (meters)
        sweep_angles: Yaw angles for sweep (degrees)
        render_images: Whether to render images (False = metadata only)
        
    Returns:
        Extended POV info list
    """
    # Load manifest
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(df)} entries")
    
    # Filter out rejected samples
    if 'rejected' in df.columns:
        df = df[df['rejected'] != True]
        logger.info(f"After filtering rejected: {len(df)} entries")
    
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {len(df)} entries")
    
    # Group by scene for efficient GLB loading
    grouped = df.groupby('scene_id')
    
    # Output directories
    multi_pov_dir = output_dir / "povs_multi" / "tex"
    multi_pov_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache for room metadata
    room_meta_cache = {}
    
    extended_povs = []
    total_scenes = len(grouped)
    skipped = 0
    
    for scene_idx, (scene_id, scene_df) in enumerate(grouped, 1):
        logger.info(f"[{scene_idx}/{total_scenes}] Processing scene {scene_id} ({len(scene_df)} POVs)")
        
        # Load GLB if rendering
        tex_meshes = None
        if render_images:
            tex_glb = dataset_root / "geometry" / "tex" / f"{scene_id}_tex.glb"
            if tex_glb.exists():
                tex_meshes = load_glb_with_transforms(tex_glb)
                logger.debug(f"  Loaded {len(tex_meshes)} meshes from {tex_glb}")
            else:
                logger.warning(f"  GLB not found: {tex_glb}")
        
        for _, row in scene_df.iterrows():
            room_id = row['room_id']
            pov_id = row['pov_id']
            pov_type = row.get('pov_type', 'door')
            
            # Load room metadata (with caching)
            cache_key = f"{scene_id}_{room_id}"
            if cache_key not in room_meta_cache:
                room_meta_cache[cache_key] = load_room_metadata(scene_id, room_id, dataset_root)
            
            room_meta = room_meta_cache[cache_key]
            if room_meta is None:
                logger.warning(f"  Skipping {scene_id}/{room_id}/{pov_id}: no room metadata")
                skipped += 1
                continue
            
            # Get room bbox
            room_bbox = room_meta.get("bbox", {})
            if "min" not in room_bbox or "max" not in room_bbox:
                logger.warning(f"  Skipping {scene_id}/{room_id}/{pov_id}: no room bbox")
                skipped += 1
                continue
            
            room_bbox_min = np.array(room_bbox["min"])
            room_bbox_max = np.array(room_bbox["max"])
            
            # Get the specific opening (door/window)
            opening = get_opening_from_metadata(room_meta, pov_id, pov_type)
            if opening is None:
                logger.warning(f"  Skipping {scene_id}/{room_id}/{pov_id}: opening not found")
                skipped += 1
                continue
            
            # Compute camera info from opening
            camera_info = compute_camera_from_opening(opening, room_bbox_min, room_bbox_max)
            if camera_info is None:
                logger.warning(f"  Skipping {scene_id}/{room_id}/{pov_id}: could not compute camera")
                skipped += 1
                continue
            
            # Build base camera info for multi-POV generation
            base_camera_info = {
                "opening_center": camera_info["opening_center"],
                "rotation_deg": camera_info["rotation_deg"],
            }
            
            # Generate multi-step camera positions
            cameras = compute_stepped_camera_positions(
                base_camera_info,
                step_distances,
                sweep_angles
            )
            
            # Render images
            if render_images and tex_meshes:
                images = render_multi_povs(
                    tex_meshes, cameras, pov_width, pov_height, pov_fov
                )
            else:
                images = [None] * len(cameras)
            
            # Save and create extended POV info
            for cam, img in zip(cameras, images):
                pov_name = cam["pov_name"]
                
                # Save image
                img_path = ""
                if img is not None:
                    img_filename = f"{scene_id}_{room_id}_{pov_id}_{pov_name}.png"
                    img_full_path = multi_pov_dir / img_filename
                    img.save(img_full_path)
                    img_path = str(img_full_path.relative_to(output_dir))
                
                # Create extended POV entry
                extended_pov = {
                    "scene_id": scene_id,
                    "room_id": room_id,
                    "room_type": row.get('room_type', 'Unknown'),
                    "pov_id": pov_id,
                    "pov_type": pov_type,
                    "base_pov_id": pov_id,
                    "multi_pov_name": pov_name,
                    "multi_pov_step": cam["step_idx"],
                    "multi_pov_sweep_angle": cam["sweep_angle"],
                    "multi_pov_position": cam["position"].tolist() if isinstance(cam["position"], np.ndarray) else cam["position"],
                    "multi_pov_rotation_deg": cam["rotation_deg"],
                    "multi_pov_eye": cam["eye"],
                    "multi_pov_look_at": cam["look_at"],
                    "multi_pov_image_path": img_path,
                    # Original manifest fields
                    "original_layout_path": row.get('layout_path', ''),
                    "original_pov_path": row.get('pov_path', ''),
                    "latent_embedding_path": row.get('latent_embedding_path', ''),
                    "graph_embedding_path": row.get('graph_embedding_path', ''),
                    "pov_embedding_path": row.get('pov_embedding_path', ''),
                    # Camera info
                    "camera_info": camera_info,
                }
                
                extended_povs.append(extended_pov)
        
        # Clear meshes to free memory
        tex_meshes = None
    
    logger.info(f"Processed {len(extended_povs)} multi-POVs, skipped {skipped} entries")
    return extended_povs


def process_existing_povs(
    pov_info_path: Path,
    dataset_root: Path,
    output_dir: Path,
    step_distances: List[float] = [0.5, 1.0],
    sweep_angles: List[float] = [-30, 0, 30],
    render_images: bool = True,
    pov_width: int = 1280,
    pov_height: int = 720,
    pov_fov: float = 80.0,
    limit: int = None,
) -> List[Dict]:
    """
    Process existing POV info and generate multi-step POVs.
    (Original functionality)
    """
    # Load existing POV info
    with open(pov_info_path, "r") as f:
        existing_povs = json.load(f)
    
    logger.info(f"Loaded {len(existing_povs)} existing POVs")
    
    if limit:
        existing_povs = existing_povs[:limit]
    
    # Group by scene for efficient GLB loading
    povs_by_scene = {}
    for pov in existing_povs:
        scene_id = pov.get("scene_id")
        if scene_id not in povs_by_scene:
            povs_by_scene[scene_id] = []
        povs_by_scene[scene_id].append(pov)
    
    # Output directories
    multi_pov_dir = output_dir / "povs_multi" / "tex"
    multi_pov_dir.mkdir(parents=True, exist_ok=True)
    
    extended_povs = []
    
    for scene_id, scene_povs in povs_by_scene.items():
        logger.info(f"Processing scene {scene_id} ({len(scene_povs)} POVs)")
        
        # Load GLB if rendering
        tex_meshes = None
        if render_images:
            tex_glb = dataset_root / "geometry" / "tex" / f"{scene_id}_tex.glb"
            if tex_glb.exists():
                tex_meshes = load_glb_with_transforms(tex_glb)
        
        for pov in scene_povs:
            # Build base camera info from existing POV
            camera = pov.get("camera", {})
            base_camera_info = {
                "opening_center": camera.get("opening_center", pov.get("camera_position", [0, 1.5, 0])),
                "rotation_deg": camera.get("rotation_deg", pov.get("rotation_deg", 0)),
            }
            
            # Generate multi-step camera positions
            cameras = compute_stepped_camera_positions(
                base_camera_info,
                step_distances,
                sweep_angles
            )
            
            # Render images
            if render_images and tex_meshes:
                images = render_multi_povs(
                    tex_meshes, cameras, pov_width, pov_height, pov_fov
                )
            else:
                images = [None] * len(cameras)
            
            # Save and create extended POV info
            room_id = pov.get("room_id", "unknown")
            pov_id = pov.get("pov_id", "pov0")
            
            for cam, img in zip(cameras, images):
                pov_name = cam["pov_name"]
                
                # Save image
                img_path = ""
                if img is not None:
                    img_filename = f"{scene_id}_{room_id}_{pov_id}_{pov_name}.png"
                    img_full_path = multi_pov_dir / img_filename
                    img.save(img_full_path)
                    img_path = str(img_full_path.relative_to(output_dir))
                
                # Create extended POV entry
                extended_pov = {
                    **pov,  # Copy all original fields
                    "multi_pov_name": pov_name,
                    "multi_pov_step": cam["step_idx"],
                    "multi_pov_sweep_angle": cam["sweep_angle"],
                    "multi_pov_position": cam["position"].tolist() if isinstance(cam["position"], np.ndarray) else cam["position"],
                    "multi_pov_rotation_deg": cam["rotation_deg"],
                    "multi_pov_eye": cam["eye"],
                    "multi_pov_look_at": cam["look_at"],
                    "multi_pov_image_path": img_path,
                    "base_pov_id": pov_id,
                }
                
                extended_povs.append(extended_pov)
    
    return extended_povs


def create_refinement_manifest_extension(
    extended_povs: List[Dict],
    output_path: Path,
):
    """
    Create a manifest extension that groups multi-POVs by room.
    """
    # Group by room
    by_room = {}
    for pov in extended_povs:
        scene_id = pov.get("scene_id")
        room_id = pov.get("room_id")
        base_pov_id = pov.get("base_pov_id", pov.get("pov_id"))
        key = f"{scene_id}_{room_id}_{base_pov_id}"
        
        if key not in by_room:
            by_room[key] = {
                "scene_id": scene_id,
                "room_id": room_id,
                "room_type": pov.get("room_type", "Unknown"),
                "base_pov_id": base_pov_id,
                "pov_type": pov.get("pov_type", "door"),
                "original_layout_path": pov.get("original_layout_path", pov.get("layout_tex", "")),
                "latent_embedding_path": pov.get("latent_embedding_path", ""),
                "graph_embedding_path": pov.get("graph_embedding_path", ""),
                "povs": []
            }
        
        by_room[key]["povs"].append({
            "pov_name": pov.get("multi_pov_name"),
            "step": pov.get("multi_pov_step"),
            "sweep_angle": pov.get("multi_pov_sweep_angle"),
            "image_path": pov.get("multi_pov_image_path"),
            "eye": pov.get("multi_pov_eye"),
            "look_at": pov.get("multi_pov_look_at"),
            "rotation_deg": pov.get("multi_pov_rotation_deg"),
        })
    
    # Sort POVs within each room by step then sweep angle
    for room_data in by_room.values():
        room_data["povs"].sort(key=lambda p: (p["step"], p["sweep_angle"]))
        room_data["num_povs"] = len(room_data["povs"])
    
    with open(output_path, "w") as f:
        json.dump(list(by_room.values()), f, indent=2)
    
    logger.info(f"Created refinement manifest with {len(by_room)} rooms")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-step POVs for refinement")
    
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Dataset root directory")
    
    # Input source (one of these required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path,
                             help="Path to evaluation manifest CSV")
    input_group.add_argument("--existing-pov-info", type=Path,
                             help="Path to existing pov_info.json")
    
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: dataset_root)")
    
    # Step configuration
    parser.add_argument("--step-distances", type=float, nargs="+", default=[0.5, 1.0],
                        help="Step distances into room (meters)")
    parser.add_argument("--sweep-angles", type=float, nargs="+", default=[-30, 0, 30],
                        help="Sweep angles (degrees, 0=forward)")
    
    # Rendering options
    parser.add_argument("--no-render", action="store_true",
                        help="Skip rendering, only generate metadata")
    parser.add_argument("--pov-width", type=int, default=1280)
    parser.add_argument("--pov-height", type=int, default=720)
    parser.add_argument("--pov-fov", type=float, default=80.0)
    
    # HPC options
    parser.add_argument("--hpc", action="store_true",
                        help="Enable HPC headless rendering")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "xvfb", "egl", "osmesa"],
                        help="Rendering backend for HPC")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process")
    
    args = parser.parse_args()
    
    # Defaults
    if args.output_dir is None:
        args.output_dir = args.dataset_root
    
    # Setup HPC rendering if needed
    if args.hpc:
        if not setup_hpc_rendering(args.backend):
            logger.error("Failed to setup HPC rendering")
            return 1
    
    try:
        # Process based on input source
        if args.manifest:
            logger.info(f"Processing from manifest: {args.manifest}")
            extended_povs = process_manifest(
                manifest_path=args.manifest,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                step_distances=args.step_distances,
                sweep_angles=args.sweep_angles,
                render_images=not args.no_render,
                pov_width=args.pov_width,
                pov_height=args.pov_height,
                pov_fov=args.pov_fov,
                limit=args.limit,
            )
        else:
            logger.info(f"Processing from POV info: {args.existing_pov_info}")
            extended_povs = process_existing_povs(
                pov_info_path=args.existing_pov_info,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                step_distances=args.step_distances,
                sweep_angles=args.sweep_angles,
                render_images=not args.no_render,
                pov_width=args.pov_width,
                pov_height=args.pov_height,
                pov_fov=args.pov_fov,
                limit=args.limit,
            )
        
        # Save extended POV info
        pov_info_dir = args.output_dir / "pov_info"
        pov_info_dir.mkdir(parents=True, exist_ok=True)
        
        extended_path = pov_info_dir / "pov_info_multi.json"
        with open(extended_path, "w") as f:
            json.dump(extended_povs, f, indent=2)
        logger.info(f"Saved extended POV info: {extended_path}")
        
        # Create refinement manifest
        refinement_manifest_path = pov_info_dir / "refinement_rooms.json"
        create_refinement_manifest_extension(extended_povs, refinement_manifest_path)
        
        # Summary
        n_per_base = 1 + len(args.step_distances) * len(args.sweep_angles)
        n_original = len(extended_povs) // n_per_base if n_per_base > 0 else 0
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Original POVs: {n_original}")
        logger.info(f"POVs per base: {n_per_base}")
        logger.info(f"Total POVs: {len(extended_povs)}")
        logger.info(f"Step distances: {args.step_distances}")
        logger.info(f"Sweep angles: {args.sweep_angles}")
        
        return 0
        
    finally:
        if args.hpc:
            cleanup_hpc_rendering()


if __name__ == "__main__":
    exit(main() or 0)