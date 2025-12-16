#!/usr/bin/env python3
"""
Prepare Refinement Evaluation Dataset

Complete pipeline to create a refinement evaluation dataset:
1. Load evaluation manifest and sample N rooms
2. Render multi-POV images for each room (step into room with sweep angles)
3. Embed POV images using ResNet18
4. Create refinement dataset manifest

Uses the EXACT same rendering code as stage4_render_povs_v2.py

Usage:
    python prepare_refinement_dataset.py \
        --manifest /path/to/manifest_val.csv \
        --dataset-root /path/to/dataset \
        --output-dir /path/to/refinement_dataset \
        --num-samples 50 \
        --seed 42 \
        --hpc --backend xvfb
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_xvfb_display = None


# ============================================================================
# HPC Setup (same as original)
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
            _xvfb_display = Xvfb(width=1280, height=720)
            _xvfb_display.start()
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
    global _xvfb_display
    if _xvfb_display is not None:
        try:
            _xvfb_display.stop()
        except Exception:
            pass
        _xvfb_display = None


# ============================================================================
# GLB Loading (same as original)
# ============================================================================

def load_glb_with_transforms(glb_path: Path) -> List[trimesh.Trimesh]:
    """Load GLB and return meshes with transforms applied."""
    try:
        scene = trimesh.load(str(glb_path), process=False, force="scene")
        
        if not isinstance(scene, trimesh.Scene):
            return [scene] if isinstance(scene, trimesh.Trimesh) else []
        
        try:
            meshes = scene.dump(concatenate=False)
            if meshes is None:
                meshes = []
            return [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        except Exception as e:
            logger.warning(f"Failed to dump meshes from scene: {e}")
            return []
    except Exception as e:
        logger.error(f"Failed to load GLB {glb_path}: {e}")
        return []


# ============================================================================
# Opening / Wall Detection (same as original)
# ============================================================================

def compute_opening_center(opening: Dict) -> Optional[np.ndarray]:
    """Get center position of door/window."""
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
    """Get rotation angle (degrees) to bring wall to bottom of image."""
    rotation_map = {
        "min_z": 0.0,
        "max_z": 180.0,
        "min_x": 90.0,
        "max_x": 270.0,
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


# ============================================================================
# POV Rendering (same as original)
# ============================================================================

def try_render_pov_pyrender(
    meshes: List[trimesh.Trimesh],
    camera_info: Dict,
    width: int,
    height: int,
    fov: float = 80.0
) -> Optional[Image.Image]:
    """Render first-person POV using pyrender. EXACT copy from original."""
    try:
        import pyrender
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


def render_pov_image(
    meshes: List[trimesh.Trimesh],
    camera_info: Dict,
    width: int = 1280,
    height: int = 720,
    fov: float = 80.0
) -> Image.Image:
    """Render first-person POV image."""
    img = try_render_pov_pyrender(meshes, camera_info, width, height, fov)
    if img is not None:
        return img
    
    # Fallback: return placeholder
    return Image.new("RGB", (width, height), (135, 206, 235))


# ============================================================================
# Camera Info Computation (based on original)
# ============================================================================

def compute_base_camera_info(
    opening: Dict,
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray,
) -> Optional[Dict]:
    """
    Compute camera info for an opening - EXACTLY like original script.
    
    From original (lines 451-458):
        camera_info = {
            "wall": wall,
            "rotation_deg": rotation_deg,
            "opening_center": opening_center.tolist(),
            "inside_direction": inside_dir.tolist(),
            "eye": [opening_center[0], 1.6, opening_center[2]],
            "look_at": (opening_center + inside_dir * 3.0).tolist(),
        }
    """
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return None
    
    wall = determine_opening_wall(opening, room_bbox_min, room_bbox_max)
    rotation_deg = get_wall_rotation(wall)
    inside_dir = get_inside_direction(wall)
    
    camera_info = {
        "wall": wall,
        "rotation_deg": rotation_deg,
        "opening_center": opening_center.tolist(),
        "inside_direction": inside_dir.tolist(),
        "eye": [opening_center[0], 1.6, opening_center[2]],
        "look_at": (opening_center + inside_dir * 3.0).tolist(),
    }
    
    return camera_info


def compute_stepped_camera_info(
    base_camera_info: Dict,
    step_distance: float,
    sweep_angle: float,
) -> Dict:
    """
    Compute camera info for a stepped position with sweep angle.
    
    Args:
        base_camera_info: Original camera info from compute_base_camera_info
        step_distance: Distance to step into room (0 = at opening)
        sweep_angle: Additional rotation in degrees (0 = straight, negative = left, positive = right)
    
    Returns:
        New camera_info dict with updated eye and look_at
    """
    opening_center = np.array(base_camera_info["opening_center"])
    inside_dir = np.array(base_camera_info["inside_direction"])
    base_rotation = base_camera_info["rotation_deg"]
    
    # Step into room
    stepped_pos = opening_center + inside_dir * step_distance
    eye = [stepped_pos[0], 1.6, stepped_pos[2]]
    
    # Apply sweep angle to look direction
    total_rotation = base_rotation + sweep_angle
    rotation_rad = math.radians(total_rotation)
    
    # Compute look direction from rotation
    # Based on get_wall_rotation mapping:
    # min_z (0°) -> look +Z
    # max_z (180°) -> look -Z  
    # min_x (90°) -> look +X
    # max_x (270°) -> look -X
    look_dir = np.array([
        math.sin(rotation_rad),
        0.0,
        math.cos(rotation_rad)
    ])
    
    look_at = np.array(eye) + look_dir * 3.0
    
    return {
        "wall": base_camera_info["wall"],
        "rotation_deg": total_rotation,
        "opening_center": base_camera_info["opening_center"],
        "inside_direction": base_camera_info["inside_direction"],
        "eye": eye,
        "look_at": look_at.tolist(),
        "step_distance": step_distance,
        "sweep_angle": sweep_angle,
    }


# ============================================================================
# POV Embedding (ResNet18)
# ============================================================================

class POVEncoder(nn.Module):
    """ResNet18-based POV image encoder (512-dim output)."""
    
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(1)


def get_pov_transform():
    """ImageNet-style transform for POV images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# Room Metadata Loading
# ============================================================================

def load_room_metadata(scene_id: str, room_id: str, dataset_root: Path) -> Optional[Dict]:
    """
    Load room metadata JSON.
    
    Manifest room_id format: "SecondBedroom_4", "Kitchen_2"
    File format: "{scene_id}_SecondBedroom.json", "{scene_id}_Kitchen.json"
    
    So we need to strip the trailing _N from room_id.
    """
    metadata_dir = dataset_root / "metadata" / "rooms"
    
    # Strip trailing _N from room_id (e.g., "SecondBedroom_4" -> "SecondBedroom")
    room_type = re.sub(r'_\d+$', '', room_id)
    
    # Try exact match with room_type
    path = metadata_dir / f"{scene_id}_{room_type}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    
    return None


def get_opening_from_metadata(room_meta: Dict, pov_id: str, pov_type: str) -> Optional[Dict]:
    """Get specific door/window opening from room metadata."""
    try:
        if pov_type == "door":
            idx = int(pov_id.replace("door", ""))
            openings = room_meta.get("doors", [])
        else:
            idx = int(pov_id.replace("window", ""))
            openings = room_meta.get("windows", [])
        
        if idx < len(openings):
            return openings[idx]
    except (ValueError, IndexError):
        pass
    return None


# ============================================================================
# Main Pipeline
# ============================================================================

def sample_from_manifest(
    df: pd.DataFrame,
    num_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Sample rooms from manifest."""
    # Filter out rejected
    if 'rejected' in df.columns:
        df = df[df['rejected'] != True].copy()
    
    random.seed(seed)
    np.random.seed(seed)
    
    if len(df) <= num_samples:
        logger.warning(f"Requested {num_samples} samples but only {len(df)} available")
        return df
    
    sampled_indices = random.sample(range(len(df)), num_samples)
    sampled_indices.sort()
    
    return df.iloc[sampled_indices].copy()


def process_sample(
    row: pd.Series,
    room_meta: Dict,
    seg_meshes: List[trimesh.Trimesh],
    output_dir: Path,
    step_distances: List[float],
    sweep_angles: List[float],
    pov_width: int,
    pov_height: int,
    pov_fov: float,
    encoder: nn.Module,
    transform,
    device: torch.device,
) -> Optional[Dict]:
    """Process a single sample: render multi-POVs and embed them."""
    scene_id = row['scene_id']
    room_id = row['room_id']
    pov_id = row['pov_id']
    pov_type = row.get('pov_type', 'door')
    
    # Get room bbox
    room_bbox = room_meta.get("bbox", {})
    if "min" not in room_bbox or "max" not in room_bbox:
        logger.warning(f"No bbox for {scene_id}/{room_id}")
        return None
    
    room_bbox_min = np.array(room_bbox["min"])
    room_bbox_max = np.array(room_bbox["max"])
    
    # Get opening
    opening = get_opening_from_metadata(room_meta, pov_id, pov_type)
    if opening is None:
        logger.warning(f"Opening not found: {scene_id}/{room_id}/{pov_id}")
        return None
    
    # Compute base camera info (exactly like original)
    base_camera_info = compute_base_camera_info(opening, room_bbox_min, room_bbox_max)
    if base_camera_info is None:
        logger.warning(f"Could not compute camera info: {scene_id}/{room_id}/{pov_id}")
        return None
    
    # Output directories
    pov_images_dir = output_dir / "povs" / "seg"
    pov_emb_dir = output_dir / "povs" / "embeddings"
    pov_images_dir.mkdir(parents=True, exist_ok=True)
    pov_emb_dir.mkdir(parents=True, exist_ok=True)
    
    pov_entries = []
    
    # Step 0: At opening (same as original)
    camera_info_step0 = base_camera_info.copy()
    camera_info_step0["step_distance"] = 0.0
    camera_info_step0["sweep_angle"] = 0.0
    
    pov_name = "step0_center"
    base_name = f"{scene_id}_{room_id}_{pov_id}_{pov_name}"
    
    # Render
    img = render_pov_image(seg_meshes, camera_info_step0, pov_width, pov_height, pov_fov)
    img_path = pov_images_dir / f"{base_name}_seg_pov.png"
    img.save(img_path)
    
    # Embed
    with torch.no_grad():
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
        emb = encoder(tensor).cpu().squeeze(0)
    emb_path = pov_emb_dir / f"{base_name}.pt"
    torch.save(emb, emb_path)
    
    pov_entries.append({
        "pov_name": pov_name,
        "step_idx": 0,
        "step_distance": 0.0,
        "sweep_angle": 0.0,
        "eye": camera_info_step0["eye"],
        "look_at": camera_info_step0["look_at"],
        "rotation_deg": camera_info_step0["rotation_deg"],
        "image_path": str(img_path.relative_to(output_dir)),
        "embedding_path": str(emb_path.relative_to(output_dir)),
    })
    
    # Steps into room with sweep angles
    for step_idx, step_dist in enumerate(step_distances, 1):
        for sweep_angle in sweep_angles:
            # Compute camera info for this step/sweep
            camera_info = compute_stepped_camera_info(base_camera_info, step_dist, sweep_angle)
            
            sweep_name = "left" if sweep_angle < -10 else "right" if sweep_angle > 10 else "center"
            pov_name = f"step{step_idx}_{sweep_name}"
            base_name = f"{scene_id}_{room_id}_{pov_id}_{pov_name}"
            
            # Render
            img = render_pov_image(seg_meshes, camera_info, pov_width, pov_height, pov_fov)
            img_path = pov_images_dir / f"{base_name}_seg_pov.png"
            img.save(img_path)
            
            # Embed
            with torch.no_grad():
                tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
                emb = encoder(tensor).cpu().squeeze(0)
            emb_path = pov_emb_dir / f"{base_name}.pt"
            torch.save(emb, emb_path)
            
            pov_entries.append({
                "pov_name": pov_name,
                "step_idx": step_idx,
                "step_distance": step_dist,
                "sweep_angle": sweep_angle,
                "eye": camera_info["eye"],
                "look_at": camera_info["look_at"],
                "rotation_deg": camera_info["rotation_deg"],
                "image_path": str(img_path.relative_to(output_dir)),
                "embedding_path": str(emb_path.relative_to(output_dir)),
            })
    
    return {
        "scene_id": scene_id,
        "room_id": room_id,
        "room_type": row.get('room_type', room_meta.get('room_type', 'Unknown')),
        "pov_id": pov_id,
        "pov_type": pov_type,
        "base_camera_info": base_camera_info,
        "original_layout_path": row.get('layout_path', ''),
        "original_pov_path": row.get('pov_path', ''),
        "latent_embedding_path": row.get('latent_embedding_path', ''),
        "graph_embedding_path": row.get('graph_embedding_path', ''),
        "original_pov_embedding_path": row.get('pov_embedding_path', ''),
        "num_multi_povs": len(pov_entries),
        "multi_povs": pov_entries,
    }


def create_refinement_manifest(
    samples: List[Dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Create a manifest CSV for refinement evaluation."""
    rows = []
    
    for sample in samples:
        row = {
            "scene_id": sample["scene_id"],
            "room_id": sample["room_id"],
            "room_type": sample["room_type"],
            "pov_id": sample["pov_id"],
            "pov_type": sample["pov_type"],
            "layout_path": sample["original_layout_path"],
            "pov_path": sample["original_pov_path"],
            "latent_embedding_path": sample["latent_embedding_path"],
            "graph_embedding_path": sample["graph_embedding_path"],
            "original_pov_embedding_path": sample["original_pov_embedding_path"],
            "num_multi_povs": sample["num_multi_povs"],
        }
        
        for pov in sample["multi_povs"]:
            pov_name = pov["pov_name"]
            row[f"pov_image_{pov_name}"] = pov["image_path"]
            row[f"pov_emb_{pov_name}"] = pov["embedding_path"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare refinement evaluation dataset")
    
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--step-distances", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--sweep-angles", type=float, nargs="+", default=[-30, 0, 30])
    parser.add_argument("--pov-width", type=int, default=1280)
    parser.add_argument("--pov-height", type=int, default=720)
    parser.add_argument("--pov-fov", type=float, default=80.0)
    
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "egl", "osmesa", "xvfb"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-render", action="store_true")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.hpc:
        if not setup_hpc_rendering(args.backend):
            logger.error("Failed to setup HPC rendering")
            return 1
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load manifest and sample
        logger.info(f"Loading manifest: {args.manifest}")
        df = pd.read_csv(args.manifest)
        logger.info(f"Manifest has {len(df)} entries")
        
        sampled_df = sample_from_manifest(df, args.num_samples, args.seed)
        logger.info(f"Sampled {len(sampled_df)} entries")
        
        # Save sample info
        sample_info = {
            "source_manifest": str(args.manifest),
            "num_samples": len(sampled_df),
            "seed": args.seed,
            "step_distances": args.step_distances,
            "sweep_angles": args.sweep_angles,
            "pov_size": [args.pov_width, args.pov_height],
            "pov_fov": args.pov_fov,
        }
        with open(args.output_dir / "sample_info.json", "w") as f:
            json.dump(sample_info, f, indent=2)
        
        # Load encoder
        logger.info("Loading POV encoder (ResNet18)...")
        encoder = POVEncoder().to(device).eval()
        transform = get_pov_transform()
        
        # Group by scene for efficient GLB loading
        grouped = sampled_df.groupby('scene_id')
        
        all_samples = []
        total_scenes = len(grouped)
        
        for scene_idx, (scene_id, scene_df) in enumerate(grouped, 1):
            logger.info(f"[{scene_idx}/{total_scenes}] Processing scene {scene_id} ({len(scene_df)} samples)")
            
            # First check if any room metadata exists for this scene
            room_meta_cache = {}
            valid_rows = []
            
            for _, row in scene_df.iterrows():
                room_id = row['room_id']
                if room_id not in room_meta_cache:
                    room_meta_cache[room_id] = load_room_metadata(scene_id, room_id, args.dataset_root)
                
                if room_meta_cache[room_id] is not None:
                    valid_rows.append((row, room_meta_cache[room_id]))
                else:
                    logger.warning(f"  No room metadata for {scene_id}/{room_id}")
            
            if not valid_rows:
                logger.warning(f"  Skipping scene {scene_id}: no valid room metadata")
                continue
            
            # Load seg GLB only if we have valid rooms
            seg_meshes = []
            if not args.skip_render:
                seg_glb = args.dataset_root / "geometry" / "seg" / f"{scene_id}_seg.glb"
                if seg_glb.exists():
                    seg_meshes = load_glb_with_transforms(seg_glb)
                    logger.info(f"  Loaded {len(seg_meshes)} meshes from seg GLB")
                else:
                    logger.warning(f"  Seg GLB not found: {seg_glb}")
                    continue
            
            # Process valid rows
            for row, room_meta in valid_rows:
                result = process_sample(
                    row=row,
                    room_meta=room_meta,
                    seg_meshes=seg_meshes,
                    output_dir=args.output_dir,
                    step_distances=args.step_distances,
                    sweep_angles=args.sweep_angles,
                    pov_width=args.pov_width,
                    pov_height=args.pov_height,
                    pov_fov=args.pov_fov,
                    encoder=encoder,
                    transform=transform,
                    device=device,
                )
                
                if result is not None:
                    all_samples.append(result)
            
            # Clear meshes and force garbage collection
            seg_meshes = None
            gc.collect()
        
        logger.info(f"Processed {len(all_samples)} samples successfully")
        
        # Save detailed POV info
        with open(args.output_dir / "pov_info_multi.json", "w") as f:
            json.dump(all_samples, f, indent=2)
        
        # Create and save manifest
        manifest_df = create_refinement_manifest(all_samples, args.output_dir)
        manifest_path = args.output_dir / "manifest_refinement.csv"
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"Saved manifest: {manifest_path}")
        
        # Summary
        n_per_sample = 1 + len(args.step_distances) * len(args.sweep_angles)
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Samples processed: {len(all_samples)}")
        logger.info(f"POVs per sample: {n_per_sample}")
        logger.info(f"Total POV images: {len(all_samples) * n_per_sample}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"{'='*60}")
        
        return 0
        
    finally:
        if args.hpc:
            cleanup_hpc_rendering()


if __name__ == "__main__":
    exit(main() or 0)