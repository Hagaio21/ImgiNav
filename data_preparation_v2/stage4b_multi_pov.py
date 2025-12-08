#!/usr/bin/env python3
"""
Stage 4b: Multi-POV Generation for Iterative Refinement

Extends stage4_render_povs_v2.py to generate multiple POVs per door/window:
- Base POV: At the door/window (existing)
- Step 1 + sweep: 1 step into room, looking left/center/right
- Step 2: 2 steps into room, looking center

This provides POV sequences for iterative refinement experiments.

Usage:
    python stage4b_multi_pov.py \
        --dataset-root /path/to/dataset \
        --existing-pov-info /path/to/pov_info.json \
        --step-distance 0.5 \
        --sweep-angles -30 0 30
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from existing script (or copy the functions)
try:
    from stage4_render_povs_v2 import (
        setup_hpc_rendering,
        cleanup_hpc_rendering,
        load_glb_with_transforms,
        try_render_pov_pyrender,
        load_taxonomy,
        get_door_window_colors,
    )
    HAS_RENDER = True
except ImportError:
    logger.warning("Could not import from stage4_render_povs_v2, will only update metadata")
    HAS_RENDER = False


def compute_stepped_camera_positions(
    base_camera_info: Dict,
    step_distances: List[float],
    sweep_angles: List[float],
) -> List[Dict]:
    """
    Compute camera positions for stepped exploration with angular sweep.
    
    Args:
        base_camera_info: Original camera info from door/window POV
        step_distances: List of step distances into the room (meters)
        sweep_angles: List of yaw angles for sweep (degrees, 0=forward)
    
    Returns:
        List of camera_info dicts for each POV
    """
    cameras = []
    
    # Base position (at door/window)
    base_pos = np.array(base_camera_info["opening_center"])
    base_rotation = base_camera_info["rotation_deg"]  # Rotation to look into room
    
    # Forward direction (into the room) in world coords
    forward_rad = math.radians(base_rotation)
    forward_dir = np.array([
        math.sin(forward_rad),  # X
        0,                       # Y (no vertical movement)
        math.cos(forward_rad)   # Z
    ])
    
    # POV 0: Base position (at door), looking straight
    cameras.append({
        "position": base_pos.copy(),
        "rotation_deg": base_rotation,
        "step_idx": 0,
        "sweep_angle": 0,
        "pov_name": "step0_center",
        **{k: v for k, v in base_camera_info.items() 
           if k not in ["opening_center", "rotation_deg"]}
    })
    
    # Generate stepped positions
    for step_idx, step_dist in enumerate(step_distances, 1):
        # Position: move forward into room
        stepped_pos = base_pos + forward_dir * step_dist
        
        for sweep_angle in sweep_angles:
            # Rotation: base rotation + sweep angle
            total_rotation = base_rotation + sweep_angle
            
            sweep_name = "left" if sweep_angle < -10 else "right" if sweep_angle > 10 else "center"
            
            cameras.append({
                "position": stepped_pos.copy(),
                "rotation_deg": total_rotation,
                "step_idx": step_idx,
                "sweep_angle": sweep_angle,
                "pov_name": f"step{step_idx}_{sweep_name}",
                "opening_center": base_camera_info["opening_center"],  # Keep reference
                **{k: v for k, v in base_camera_info.items() 
                   if k not in ["opening_center", "rotation_deg"]}
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
        # Adapt camera_info for render function
        render_cam = {
            "opening_center": cam_info["position"].tolist() if isinstance(cam_info["position"], np.ndarray) else cam_info["position"],
            "rotation_deg": cam_info["rotation_deg"],
        }
        
        if HAS_RENDER:
            img = try_render_pov_pyrender(meshes, render_cam, width, height, fov)
            if img is None:
                img = Image.new("RGB", (width, height), (135, 206, 235))
        else:
            img = Image.new("RGB", (width, height), (200, 200, 200))
        
        images.append(img)
    
    return images


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
    
    Args:
        pov_info_path: Path to existing pov_info.json
        dataset_root: Dataset root directory
        output_dir: Output directory for new POVs
        step_distances: Distances to step into room (meters)
        sweep_angles: Yaw angles for sweep (degrees)
        render_images: Whether to render images (False = metadata only)
        
    Returns:
        Extended POV info list
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
        if render_images and HAS_RENDER:
            tex_glb = dataset_root / "geometry" / "tex" / f"{scene_id}_tex.glb"
            if tex_glb.exists():
                tex_meshes = load_glb_with_transforms(tex_glb)
        
        for pov in scene_povs:
            # Build base camera info from existing POV
            base_camera_info = {
                "opening_center": pov.get("camera_position", pov.get("opening_center", [0, 1.5, 0])),
                "rotation_deg": pov.get("rotation_deg", 0),
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
                    "multi_pov_image_path": img_path,
                    "base_pov_id": pov_id,  # Reference to original
                }
                
                extended_povs.append(extended_pov)
    
    return extended_povs


def create_refinement_manifest_extension(
    extended_povs: List[Dict],
    output_path: Path,
):
    """
    Create a manifest extension that groups multi-POVs by room.
    
    This can be merged with the main manifest or used separately.
    """
    # Group by room
    by_room = {}
    for pov in extended_povs:
        scene_id = pov.get("scene_id")
        room_id = pov.get("room_id")
        key = f"{scene_id}_{room_id}"
        
        if key not in by_room:
            by_room[key] = {
                "scene_id": scene_id,
                "room_id": room_id,
                "povs": []
            }
        
        by_room[key]["povs"].append({
            "pov_name": pov.get("multi_pov_name"),
            "step": pov.get("multi_pov_step"),
            "sweep_angle": pov.get("multi_pov_sweep_angle"),
            "image_path": pov.get("multi_pov_image_path"),
            "base_pov_id": pov.get("base_pov_id"),
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
    parser.add_argument("--existing-pov-info", type=Path, default=None,
                        help="Path to existing pov_info.json (default: dataset_root/pov_info/pov_info.json)")
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
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--backend", default="auto")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of POVs to process")
    
    args = parser.parse_args()
    
    # Defaults
    if args.existing_pov_info is None:
        args.existing_pov_info = args.dataset_root / "pov_info" / "pov_info.json"
    
    if args.output_dir is None:
        args.output_dir = args.dataset_root
    
    # Validate
    if not args.existing_pov_info.exists():
        logger.error(f"POV info not found: {args.existing_pov_info}")
        return
    
    # Setup HPC if needed
    if args.hpc and HAS_RENDER:
        if not setup_hpc_rendering(args.backend):
            logger.error("Failed to setup HPC rendering")
            return
    
    try:
        # Process
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
        n_original = len(extended_povs) // (1 + len(args.step_distances) * len(args.sweep_angles))
        n_per_base = 1 + len(args.step_distances) * len(args.sweep_angles)
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Original POVs: {n_original}")
        logger.info(f"POVs per base: {n_per_base}")
        logger.info(f"Total POVs: {len(extended_povs)}")
        logger.info(f"Step distances: {args.step_distances}")
        logger.info(f"Sweep angles: {args.sweep_angles}")
        
    finally:
        if args.hpc and HAS_RENDER:
            cleanup_hpc_rendering()


if __name__ == "__main__":
    main()
