#!/usr/bin/env python3
"""
Prepare Refinement Evaluation Dataset

Complete pipeline to create a refinement evaluation dataset:
1. Load evaluation manifest and sample N rooms
2. Render multi-POV images for each room (step into room with sweep angles)
3. Embed POV images using ResNet18
4. Create refinement dataset manifest

Usage:
    python prepare_refinement_dataset.py \
        --manifest /path/to/manifest_val.csv \
        --dataset-root /path/to/dataset \
        --output-dir /path/to/refinement_dataset \
        --num-samples 50 \
        --seed 42 \
        --hpc --backend xvfb

Output:
    refinement_dataset/
    ├── manifest_refinement.csv           # Main manifest for refinement eval
    ├── pov_info_multi.json               # Detailed POV info
    ├── povs/
    │   ├── images/                       # Rendered POV images
    │   │   └── {scene}_{room}_{pov}_{step}.png
    │   └── embeddings/                   # POV embeddings
    │       └── {scene}_{room}_{pov}_{step}.pt
    └── sample_info.json                  # Sampling metadata
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# GLB Loading and Rendering
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


def render_pov_pyrender(
    meshes: List,
    eye: np.ndarray,
    look_at: np.ndarray,
    width: int,
    height: int,
    fov: float = 80.0
) -> Optional[Image.Image]:
    """Render first-person POV using pyrender."""
    try:
        import pyrender
    except ImportError:
        logger.warning("pyrender not available")
        return None
    
    try:
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
        logger.warning(f"pyrender failed: {e}")
        return None


# ============================================================================
# Camera Position Computation
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


def compute_multi_pov_cameras(
    opening: Dict,
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray,
    step_distances: List[float] = [0.5, 1.0],
    sweep_angles: List[float] = [-30, 0, 30],
    eye_height: float = 1.6,
) -> List[Dict]:
    """Compute camera positions for multi-POV generation."""
    opening_center = compute_opening_center(opening)
    if opening_center is None:
        return []
    
    wall = determine_opening_wall(opening, room_bbox_min, room_bbox_max)
    base_rotation = get_wall_rotation(wall)
    inside_dir = get_inside_direction(wall)
    
    # Forward direction
    forward_rad = math.radians(base_rotation)
    forward_dir = np.array([
        math.sin(forward_rad),
        0,
        math.cos(forward_rad)
    ])
    
    cameras = []
    
    # Step 0: At opening, looking straight in
    eye = opening_center.copy()
    eye[1] = eye_height
    look_at = eye + forward_dir * 3.0
    
    cameras.append({
        "pov_name": "step0_center",
        "step_idx": 0,
        "sweep_angle": 0,
        "eye": eye.tolist(),
        "look_at": look_at.tolist(),
        "rotation_deg": base_rotation,
    })
    
    # Stepped positions with sweep
    for step_idx, step_dist in enumerate(step_distances, 1):
        stepped_pos = opening_center + forward_dir * step_dist
        stepped_eye = stepped_pos.copy()
        stepped_eye[1] = eye_height
        
        for sweep_angle in sweep_angles:
            total_rotation = base_rotation + sweep_angle
            total_rad = math.radians(total_rotation)
            
            look_dir = np.array([
                math.sin(total_rad),
                0,
                math.cos(total_rad)
            ])
            look_at = stepped_eye + look_dir * 3.0
            
            sweep_name = "left" if sweep_angle < -10 else "right" if sweep_angle > 10 else "center"
            
            cameras.append({
                "pov_name": f"step{step_idx}_{sweep_name}",
                "step_idx": step_idx,
                "sweep_angle": sweep_angle,
                "eye": stepped_eye.tolist(),
                "look_at": look_at.tolist(),
                "rotation_deg": total_rotation,
            })
    
    return cameras


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


def embed_pov_images(
    images: List[Image.Image],
    encoder: nn.Module,
    transform,
    device: torch.device,
) -> List[torch.Tensor]:
    """Embed a list of POV images."""
    embeddings = []
    
    with torch.no_grad():
        for img in images:
            if img is None:
                embeddings.append(torch.zeros(512))
                continue
            
            tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
            emb = encoder(tensor).cpu().squeeze(0)
            embeddings.append(emb)
    
    return embeddings


# ============================================================================
# Room Metadata Loading
# ============================================================================

def load_room_metadata(scene_id: str, room_id: str, dataset_root: Path) -> Optional[Dict]:
    """Load room metadata JSON."""
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
                logger.warning(f"Failed to load {path}: {e}")
    
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
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Sample
    if len(df) <= num_samples:
        logger.warning(f"Requested {num_samples} samples but only {len(df)} available")
        return df
    
    sampled_indices = random.sample(range(len(df)), num_samples)
    sampled_indices.sort()
    
    return df.iloc[sampled_indices].copy()


def process_sample(
    row: pd.Series,
    dataset_root: Path,
    output_dir: Path,
    tex_meshes: List,
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
    
    # Load room metadata
    room_meta = load_room_metadata(scene_id, room_id, dataset_root)
    if room_meta is None:
        logger.warning(f"No room metadata for {scene_id}/{room_id}")
        return None
    
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
    
    # Compute camera positions
    cameras = compute_multi_pov_cameras(
        opening, room_bbox_min, room_bbox_max,
        step_distances, sweep_angles
    )
    
    if not cameras:
        logger.warning(f"No cameras for {scene_id}/{room_id}/{pov_id}")
        return None
    
    # Render POVs
    images = []
    for cam in cameras:
        eye = np.array(cam["eye"])
        look_at = np.array(cam["look_at"])
        
        if tex_meshes:
            img = render_pov_pyrender(tex_meshes, eye, look_at, pov_width, pov_height, pov_fov)
        else:
            img = Image.new("RGB", (pov_width, pov_height), (135, 206, 235))
        
        images.append(img)
    
    # Embed POVs
    embeddings = embed_pov_images(images, encoder, transform, device)
    
    # Save images and embeddings
    pov_images_dir = output_dir / "povs" / "images"
    pov_emb_dir = output_dir / "povs" / "embeddings"
    pov_images_dir.mkdir(parents=True, exist_ok=True)
    pov_emb_dir.mkdir(parents=True, exist_ok=True)
    
    pov_entries = []
    for cam, img, emb in zip(cameras, images, embeddings):
        pov_name = cam["pov_name"]
        base_name = f"{scene_id}_{room_id}_{pov_id}_{pov_name}"
        
        # Save image
        img_path = pov_images_dir / f"{base_name}.png"
        if img is not None:
            img.save(img_path)
        
        # Save embedding
        emb_path = pov_emb_dir / f"{base_name}.pt"
        torch.save(emb, emb_path)
        
        pov_entries.append({
            "pov_name": pov_name,
            "step_idx": cam["step_idx"],
            "sweep_angle": cam["sweep_angle"],
            "eye": cam["eye"],
            "look_at": cam["look_at"],
            "rotation_deg": cam["rotation_deg"],
            "image_path": str(img_path.relative_to(output_dir)),
            "embedding_path": str(emb_path.relative_to(output_dir)),
        })
    
    return {
        "scene_id": scene_id,
        "room_id": room_id,
        "room_type": row.get('room_type', 'Unknown'),
        "pov_id": pov_id,
        "pov_type": pov_type,
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
        # Create one row per sample with multi-POV info
        row = {
            "scene_id": sample["scene_id"],
            "room_id": sample["room_id"],
            "room_type": sample["room_type"],
            "pov_id": sample["pov_id"],
            "pov_type": sample["pov_type"],
            "latent_embedding_path": sample["latent_embedding_path"],
            "graph_embedding_path": sample["graph_embedding_path"],
            "original_pov_embedding_path": sample["original_pov_embedding_path"],
            "num_multi_povs": sample["num_multi_povs"],
        }
        
        # Add paths for each multi-POV
        for pov in sample["multi_povs"]:
            pov_name = pov["pov_name"]
            row[f"pov_image_{pov_name}"] = pov["image_path"]
            row[f"pov_emb_{pov_name}"] = pov["embedding_path"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare refinement evaluation dataset")
    
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Input evaluation manifest CSV")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Dataset root directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for refinement dataset")
    
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to include")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    
    # POV generation options
    parser.add_argument("--step-distances", type=float, nargs="+", default=[0.5, 1.0],
                        help="Step distances into room (meters)")
    parser.add_argument("--sweep-angles", type=float, nargs="+", default=[-30, 0, 30],
                        help="Sweep angles (degrees)")
    parser.add_argument("--pov-width", type=int, default=1280)
    parser.add_argument("--pov-height", type=int, default=720)
    parser.add_argument("--pov-fov", type=float, default=80.0)
    
    # HPC options
    parser.add_argument("--hpc", action="store_true",
                        help="Enable HPC headless rendering")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "xvfb", "egl", "osmesa"])
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip rendering (for testing)")
    
    args = parser.parse_args()
    
    # Setup
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
            "sampled_indices": sampled_df.index.tolist(),
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
            
            # Load GLB
            tex_meshes = []
            if not args.skip_render:
                tex_glb = args.dataset_root / "geometry" / "tex" / f"{scene_id}_tex.glb"
                if tex_glb.exists():
                    tex_meshes = load_glb_with_transforms(tex_glb)
                    logger.info(f"  Loaded {len(tex_meshes)} meshes")
                else:
                    logger.warning(f"  GLB not found: {tex_glb}")
            
            # Process each sample in this scene
            for _, row in scene_df.iterrows():
                result = process_sample(
                    row=row,
                    dataset_root=args.dataset_root,
                    output_dir=args.output_dir,
                    tex_meshes=tex_meshes,
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
            
            # Clear meshes
            tex_meshes = None
        
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
        logger.info(f"Step distances: {args.step_distances}")
        logger.info(f"Sweep angles: {args.sweep_angles}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"{'='*60}")
        
        return 0
        
    finally:
        if args.hpc:
            cleanup_hpc_rendering()


if __name__ == "__main__":
    exit(main() or 0)