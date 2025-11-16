#!/usr/bin/env python3
"""
Create new layout images with modified coloring:
- Height band: [-1, 1.8] (more permissive for floor points)
- Floor points are colored separately
- All images saved to a single 'layout_new' folder
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from utils.file_discovery import find_room_files, discover_scenes
from utils.geometry_utils import load_room_meta, extract_frame_from_meta
from common.utils import create_progress_tracker, safe_mkdir
from common.taxonomy import Taxonomy
from utils.geometry_utils import (
    world_to_local_coords, points_to_image_coords
)

TAXONOMY = None


def draw_point(canvas: np.ndarray, x: int, y: int, color: np.ndarray, size: int = 1):
    """Draw a point on the canvas."""
    half = size // 2
    x0, x1 = max(x - half, 0), min(x + half, canvas.shape[1] - 1)
    y0, y1 = max(y - half, 0), min(y + half, canvas.shape[0] - 1)
    
    # Ensure color is a proper numpy array
    if isinstance(color, (tuple, list)):
        color = np.array(color, dtype=np.uint8)
    elif color.ndim == 1 and color.shape[0] == 3:
        # Color is (3,) - explicitly assign to each channel
        canvas[y0:y1 + 1, x0:x1 + 1, 0] = color[0]
        canvas[y0:y1 + 1, x0:x1 + 1, 1] = color[1]
        canvas[y0:y1 + 1, x0:x1 + 1, 2] = color[2]
        return
    
    # For multi-dimensional color arrays, assign directly
    canvas[y0:y1 + 1, x0:x1 + 1] = color


def create_room_layout_new(
    parquet_path: Path,
    output_path: Path,
    taxonomy: Taxonomy,
    color_mode: str = "category",
    resolution: int = 512,
    margin: int = 10,
    height_min: float = -1.0,
    height_max: float = 1.8,
    point_size: int = 1,
    floor_point_size: int = None,
    floor_bbox_buffer: float = 0.1,
):
    """
    Create room layout with new coloring scheme:
    - Uses height band [height_min, height_max] for all points (more permissive)
    - Floor points are colored separately (rendered first, then regular points on top)
    - Floor points use larger point size to fill gaps
    - Image is cropped to floor point bounding box with buffer
    """
    if floor_point_size is None:
        floor_point_size = max(point_size * 2, 3)  # Default: 2x regular size, minimum 3
    # Load room metadata
    meta = load_room_meta(parquet_path.parent)
    if meta is None:
        raise RuntimeError(f"No metadata found for {parquet_path}")

    origin, u, v, n, uv_bounds, _, map_band = extract_frame_from_meta(meta)

    # Get floor IDs from taxonomy
    floor_ids = taxonomy.get_floor_ids()
    if not floor_ids:
        print(f"[warn] No floor IDs found in taxonomy for {parquet_path.parent}", flush=True)
        floor_ids = []

    # Load and process point cloud
    df = pd.read_parquet(parquet_path)
    required_cols = {"x", "y", "z", "label_id"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Missing required columns {required_cols} in {parquet_path}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    labels = df["label_id"].to_numpy(dtype=np.int32)

    # Transform to local coordinates
    uvh = world_to_local_coords(xyz, origin, u, v, n)

    # Identify floor points
    is_floor = np.isin(labels, np.array(floor_ids, dtype=labels.dtype)) if floor_ids else np.zeros(len(labels), dtype=bool)

    # Filter all points (both floor and regular) in height band [-1, 1.8]
    height_mask = (uvh[:, 2] >= height_min) & (uvh[:, 2] <= height_max)
    
    # Apply height filter
    total_mask = height_mask
    
    if total_mask.sum() == 0:
        print(f"[warn] no points in height band [{height_min},{height_max}] or floor points in {parquet_path}", flush=True)
        return

    # Get filtered data
    u_vals = uvh[total_mask, 0]
    v_vals = uvh[total_mask, 1]
    heights = uvh[total_mask, 2]  # z-coordinates (heights)
    filtered_labels = labels[total_mask]
    filtered_is_floor = is_floor[total_mask]

    # Convert to image coordinates
    x_img, y_img = points_to_image_coords(u_vals, v_vals, uv_bounds, resolution, margin)

    # Sort points by height (low to high) so floor appears under other points
    sort_indices = np.argsort(heights)
    sorted_heights = heights[sort_indices]
    sorted_labels = filtered_labels[sort_indices]
    sorted_is_floor = filtered_is_floor[sort_indices]
    sorted_x_img = x_img[sort_indices]
    sorted_y_img = y_img[sort_indices]

    # Render to canvas - points are already sorted from low to high
    canvas = np.full((resolution, resolution, 3), 240, dtype=np.uint8)
    
    # First pass: render all points and collect floor point coordinates
    floor_x_coords = []
    floor_y_coords = []
    
    for is_floor_pt, lbl, x, y in zip(sorted_is_floor, sorted_labels, sorted_x_img, sorted_y_img):
        lbl_int = int(lbl)
        if color_mode == "category":
            color_tuple = taxonomy.get_color(lbl_int, mode=color_mode)
        else:
            color_tuple = taxonomy.get_color(lbl_int)
        color = np.array(color_tuple, dtype=np.uint8)
        # Use larger point size for floor points to fill gaps
        size = floor_point_size if is_floor_pt else point_size
        draw_point(canvas, x, y, color, size=size)
        
        # Collect floor point coordinates for bbox calculation
        if is_floor_pt:
            floor_x_coords.append(x)
            floor_y_coords.append(y)
    
    # Crop to floor point bounding box with buffer
    if floor_x_coords and floor_y_coords:
        floor_x_coords = np.array(floor_x_coords)
        floor_y_coords = np.array(floor_y_coords)
        
        # Calculate floor bbox
        floor_x_min = int(np.min(floor_x_coords))
        floor_x_max = int(np.max(floor_x_coords))
        floor_y_min = int(np.min(floor_y_coords))
        floor_y_max = int(np.max(floor_y_coords))
        
        # Add buffer (as percentage of bbox size)
        bbox_width = floor_x_max - floor_x_min
        bbox_height = floor_y_max - floor_y_min
        buffer_x = int(bbox_width * floor_bbox_buffer)
        buffer_y = int(bbox_height * floor_bbox_buffer)
        
        # Expand bbox with buffer
        crop_x_min = max(0, floor_x_min - buffer_x)
        crop_x_max = min(resolution - 1, floor_x_max + buffer_x)
        crop_y_min = max(0, floor_y_min - buffer_y)
        crop_y_max = min(resolution - 1, floor_y_max + buffer_y)
        
        # Crop the canvas
        canvas = canvas[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
        
        # Resize back to original resolution to maintain consistency
        canvas_pil = Image.fromarray(canvas)
        canvas_pil = canvas_pil.resize((resolution, resolution), Image.Resampling.NEAREST)
        canvas = np.array(canvas_pil)
    
    # Save image
    safe_mkdir(output_path.parent)
    Image.fromarray(canvas).save(output_path)


def create_scene_layout_new(
    scene_dir: Path,
    output_path: Path,
    taxonomy: Taxonomy,
    color_mode: str = "category",
    resolution: int = 512,
    margin: int = 10,
    height_min: float = -1.0,
    height_max: float = 1.8,
    point_size: int = 1,
    floor_point_size: int = None,
    floor_bbox_buffer: float = 0.1,
):
    """
    Create scene layout with new coloring scheme.
    Image is cropped to floor point bounding box with buffer.
    """
    if floor_point_size is None:
        floor_point_size = max(point_size * 2, 3)  # Default: 2x regular size, minimum 3
    scene_id = scene_dir.name
    room_parquets = sorted(scene_dir.rglob("rooms/*/*.parquet"))
    if not room_parquets:
        print(f"[warn] no room parquets found in {scene_dir}", flush=True)
        return

    # Get reference frame from first room
    first_meta = load_room_meta(room_parquets[0].parent)
    if first_meta is None:
        print(f"[warn] missing metadata for {room_parquets[0]}", flush=True)
        return
    origin, u, v, n, _, _, _ = extract_frame_from_meta(first_meta)

    # Get floor IDs from taxonomy
    floor_ids = taxonomy.get_floor_ids()
    if not floor_ids:
        print(f"[warn] No floor IDs found in taxonomy", flush=True)
        floor_ids = []

    # Collect global bounds
    all_u_bounds, all_v_bounds = [], []
    for parquet_path in room_parquets:
        try:
            df = pd.read_parquet(parquet_path, columns=["x", "y", "z"])
            xyz = df.to_numpy(dtype=np.float32)
            uvh = world_to_local_coords(xyz, origin, u, v, n)
            all_u_bounds.extend([uvh[:, 0].min(), uvh[:, 0].max()])
            all_v_bounds.extend([uvh[:, 1].min(), uvh[:, 1].max()])
        except Exception as e:
            print(f"[warn] failed bounds for {parquet_path}: {e}", flush=True)

    if not all_u_bounds:
        print(f"[warn] no usable points in {scene_dir}", flush=True)
        return
    global_uv_bounds = (min(all_u_bounds), max(all_u_bounds), min(all_v_bounds), max(all_v_bounds))

    # Render all rooms to single canvas
    canvas = np.full((resolution, resolution, 3), 240, dtype=np.uint8)
    
    # Collect all points first (floor and regular)
    all_points = []
    for parquet_path in room_parquets:
        try:
            df = pd.read_parquet(parquet_path)
            required_cols = {"x", "y", "z", "label_id"}
            if not required_cols.issubset(df.columns):
                continue

            xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            labels = df["label_id"].to_numpy(dtype=np.int32)

            uvh = world_to_local_coords(xyz, origin, u, v, n)
            
            # Identify floor points
            is_floor = np.isin(labels, np.array(floor_ids, dtype=labels.dtype)) if floor_ids else np.zeros(len(labels), dtype=bool)
            
            # Filter all points in height band [-1, 1.8]
            height_mask = (uvh[:, 2] >= height_min) & (uvh[:, 2] <= height_max)
            
            mask = height_mask
            if mask.sum() == 0:
                continue

            u_vals, v_vals = uvh[mask, 0], uvh[mask, 1]
            heights = uvh[mask, 2]  # z-coordinates (heights)
            f_labels = labels[mask]
            f_is_floor = is_floor[mask]

            x_img, y_img = points_to_image_coords(u_vals, v_vals, global_uv_bounds, resolution, margin)
            
            # Store points with heights for rendering (will sort by height later)
            for height, is_floor_pt, lbl, x, y in zip(heights, f_is_floor, f_labels, x_img, y_img):
                all_points.append((height, is_floor_pt, lbl, x, y))
        except Exception as e:
            print(f"[warn] skipping {parquet_path}: {e}", flush=True)

    # Sort all points by height (low to high) so floor appears under other points
    all_points.sort(key=lambda p: p[0])  # Sort by height (first element)
    
    # Render points in order from low to high and collect floor coordinates
    floor_x_coords = []
    floor_y_coords = []
    
    for height, is_floor_pt, lbl, x, y in all_points:
        lbl_int = int(lbl)
        if color_mode == "category":
            color_tuple = taxonomy.get_color(lbl_int, mode=color_mode)
        else:
            color_tuple = taxonomy.get_color(lbl_int)
        color = np.array(color_tuple, dtype=np.uint8)
        # Use larger point size for floor points to fill gaps
        size = floor_point_size if is_floor_pt else point_size
        draw_point(canvas, x, y, color, size=size)
        
        # Collect floor point coordinates for bbox calculation
        if is_floor_pt:
            floor_x_coords.append(x)
            floor_y_coords.append(y)
    
    # Crop to floor point bounding box with buffer
    if floor_x_coords and floor_y_coords:
        floor_x_coords = np.array(floor_x_coords)
        floor_y_coords = np.array(floor_y_coords)
        
        # Calculate floor bbox
        floor_x_min = int(np.min(floor_x_coords))
        floor_x_max = int(np.max(floor_x_coords))
        floor_y_min = int(np.min(floor_y_coords))
        floor_y_max = int(np.max(floor_y_coords))
        
        # Add buffer (as percentage of bbox size)
        bbox_width = floor_x_max - floor_x_min
        bbox_height = floor_y_max - floor_y_min
        buffer_x = int(bbox_width * floor_bbox_buffer)
        buffer_y = int(bbox_height * floor_bbox_buffer)
        
        # Expand bbox with buffer
        crop_x_min = max(0, floor_x_min - buffer_x)
        crop_x_max = min(resolution - 1, floor_x_max + buffer_x)
        crop_y_min = max(0, floor_y_min - buffer_y)
        crop_y_max = min(resolution - 1, floor_y_max + buffer_y)
        
        # Crop the canvas
        canvas = canvas[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
        
        # Resize back to original resolution to maintain consistency
        canvas_pil = Image.fromarray(canvas)
        canvas_pil = canvas_pil.resize((resolution, resolution), Image.Resampling.NEAREST)
        canvas = np.array(canvas_pil)

    safe_mkdir(output_path.parent)
    Image.fromarray(canvas).save(output_path)


def main():
    ap = argparse.ArgumentParser(
        description="Create new layout images with modified coloring (height band [-1, 1.8] and floor coloring)"
    )
    ap.add_argument("--in_root", required=True, help="Root folder with scenes or room dataset")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON file")
    ap.add_argument("--output_dir", required=True, help="Output directory (will create 'layout_new' subfolder)")
    ap.add_argument("--pattern", help="Glob pattern for parquet files")
    ap.add_argument("--res", type=int, default=512, help="Output image resolution")
    ap.add_argument("--hmin", type=float, default=-1.0, help="Minimum height filter (default: -1.0)")
    ap.add_argument("--hmax", type=float, default=1.8, help="Maximum height filter (default: 1.8)")
    ap.add_argument("--point-size", type=int, default=5, help="Point rendering size for regular points")
    ap.add_argument("--floor-point-size", type=int, default=None, help="Point rendering size for floor points (default: 2x point-size, minimum 3)")
    ap.add_argument("--floor-bbox-buffer", type=float, default=0.1, help="Buffer around floor bbox as fraction of bbox size (default: 0.1 = 10%%)")
    ap.add_argument("--manifest", help="Optional manifest CSV")
    ap.add_argument("--mode", choices=["room", "scene", "both"], default="both")
    ap.add_argument("--color-mode", choices=["category", "super"], default="category",
                    help="Color mode for rendering")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    taxonomy_path = Path(args.taxonomy)
    manifest_path = Path(args.manifest) if args.manifest else None
    output_base = Path(args.output_dir)
    output_dir = output_base / "layout_new"
    
    # Load taxonomy once
    global TAXONOMY
    TAXONOMY = Taxonomy(taxonomy_path)

    # Rooms
    if args.mode in ("room", "both"):
        room_files = find_room_files(in_root, manifest_path, args.pattern)
        progress = create_progress_tracker(len(room_files), "room layouts")
        for i, parquet_path in enumerate(room_files, 1):
            scene_id, room_id = parquet_path.stem.split("_")[:2]
            # Save to single layout_new folder
            output_path = output_dir / f"{scene_id}_{room_id}_room_seg_layout.png"
            try:
                create_room_layout_new(
                    parquet_path, output_path, TAXONOMY,
                    args.color_mode,
                    resolution=args.res, 
                    height_min=args.hmin,
                    height_max=args.hmax,
                    point_size=args.point_size,
                    floor_point_size=args.floor_point_size,
                    floor_bbox_buffer=args.floor_bbox_buffer
                )
                progress(i, f"{parquet_path.name} -> {output_path}", True)
            except Exception as e:
                progress(i, f"failed {parquet_path.name}: {e}", False)

    # Scenes
    if args.mode in ("scene", "both"):
        if manifest_path:
            # Use scene IDs from manifest
            scene_ids = discover_scenes(manifest=manifest_path)
        else:
            # Default discovery
            scene_info_files = list(in_root.rglob("*_scene_info.json"))
            scene_ids = [p.stem.replace("_scene_info", "") for p in scene_info_files]

        progress = create_progress_tracker(len(scene_ids), "scene layouts")
        for i, scene_id in enumerate(scene_ids, 1):
            scene_dir = in_root / scene_id
            # Save to single layout_new folder
            output_path = output_dir / f"{scene_id}_scene_layout.png"
            try:
                create_scene_layout_new(
                    scene_dir, output_path, TAXONOMY,
                    args.color_mode,
                    resolution=args.res, 
                    height_min=args.hmin,
                    height_max=args.hmax,
                    point_size=args.point_size,
                    floor_point_size=args.floor_point_size,
                    floor_bbox_buffer=args.floor_bbox_buffer
                )
                progress(i, f"{scene_id} -> {output_path}", True)
            except Exception as e:
                progress(i, f"failed {scene_id}: {e}", False)

    print(f"\n[INFO] All layout images saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

