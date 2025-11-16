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
from utils.layout_analysis import count_distinct_colors
# We'll use our own numpy-based function instead

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


def compute_whiteness_ratio(canvas: np.ndarray, white_threshold: int = 230) -> float:
    """
    Compute the ratio of white pixels in the canvas.
    
    Args:
        canvas: Image array (H, W, 3) with values in [0, 255]
        white_threshold: Pixel value threshold for "white" (default: 230)
    
    Returns:
        float: Ratio of white pixels (0.0 to 1.0)
    """
    # Check if all channels are above threshold
    white_mask = (canvas > white_threshold).all(axis=2)
    total_pixels = white_mask.size
    white_pixels = white_mask.sum()
    return white_pixels / total_pixels if total_pixels > 0 else 0.0


def count_distinct_colors_numpy(canvas: np.ndarray, min_pixel_threshold: int = 0) -> int:
    """
    Count distinct colors in a numpy canvas array.
    Includes ALL colors (background, floor, walls, objects).
    Empty rooms typically have only 3 colors: background + floor + wall.
    
    Args:
        canvas: Image array (H, W, 3) with values in [0, 255]
        min_pixel_threshold: Minimum number of pixels for a color to be counted
    
    Returns:
        Number of distinct colors
    """
    # Reshape to (N, 3) where N = H * W
    pixels = canvas.reshape(-1, 3)
    
    # Get unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    distinct_colors = 0
    for color, count in zip(unique_colors, counts):
        # Check minimum pixel threshold
        if count < min_pixel_threshold:
            continue
        
        distinct_colors += 1
    
    return distinct_colors


def create_room_layout_new(
    parquet_path: Path,
    output_path: Path,
    taxonomy: Taxonomy,
    color_mode: str = "category",
    resolution: int = 512,
    margin: int = 10,
    height_min: float = 0.1,
    height_max: float = 1.8,
    point_size: int = 1,
    floor_point_size: int = None,
    floor_bbox_buffer: float = 0.1,
    min_colors: int = 4,
    max_whiteness: float = 0.95,
    object_point_size_multiplier: float = 1.5,
):
    """
    Create room layout with new coloring scheme:
    - Uses height band [height_min, height_max] for all points (more permissive)
    - Floor points are colored separately (rendered first, then regular points on top)
    - Floor points use larger point size to fill gaps
    - Objects use larger point size to reduce sparsity
    - Image is cropped to floor point bounding box with buffer
    - Filters out empty/sparse rooms (min_colors, max_whiteness)
    
    Returns:
        bool: True if image was created and passed filters, False if filtered out
    """
    if floor_point_size is None:
        floor_point_size = max(point_size * 2, 3)  # Default: 2x regular size, minimum 3
    
    # Calculate object point size to reduce sparsity
    object_point_size = max(int(point_size * object_point_size_multiplier), point_size + 1)
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

    # Filter points:
    # - Regular (non-floor) points: height between 0.1 and 1.8 (original layout band)
    # - Floor points: include all floor points (regardless of height)
    regular_height_mask = (uvh[:, 2] >= 0.1) & (uvh[:, 2] <= 1.8)
    floor_mask = is_floor  # Include all floor points
    
    # Combine: regular points in height band OR floor points
    total_mask = (regular_height_mask & ~is_floor) | floor_mask
    
    if total_mask.sum() == 0:
        print(f"[warn] no points (regular in [0.1, 1.8] or floor) in {parquet_path}", flush=True)
        return False

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
        # Use larger point size for floor points and objects to reduce sparsity
        if is_floor_pt:
            size = floor_point_size
        else:
            size = object_point_size  # Larger size for objects too
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
    
    # Filter empty/sparse rooms before saving
    # Count ALL colors (including background, floor, walls, objects)
    # Empty rooms typically have only 3 colors: background + floor + wall
    # But walls might be same color as background, so we need to check non-background colors too
    color_count_all = count_distinct_colors_numpy(canvas, min_pixel_threshold=10)
    
    # Also count non-background colors to catch cases where walls = background
    background_color = np.array([240, 240, 240])
    pixels = canvas.reshape(-1, 3)
    non_bg_mask = ~np.all(pixels == background_color, axis=1)
    non_bg_pixels = pixels[non_bg_mask]
    if len(non_bg_pixels) > 0:
        unique_non_bg, counts_non_bg = np.unique(non_bg_pixels, axis=0, return_counts=True)
        non_bg_color_count = np.sum(counts_non_bg >= 10)
    else:
        non_bg_color_count = 0
    
    # Filter if: total colors < 4 OR non-background colors < 2 (background + floor + wall = 3, but if wall=background then only 2)
    if color_count_all < min_colors or non_bg_color_count < 2:
        print(f"[filter] Skipping {parquet_path.name}: {color_count_all} total colors, {non_bg_color_count} non-bg colors (min: {min_colors} total, 2 non-bg) - likely empty room", flush=True)
        return False
    
    # Check whiteness ratio
    whiteness_ratio = compute_whiteness_ratio(canvas, white_threshold=230)
    if whiteness_ratio > max_whiteness:
        print(f"[filter] Skipping {parquet_path.name}: whiteness ratio {whiteness_ratio:.3f} > {max_whiteness}", flush=True)
        return False
    
    # Image passed filters, save it
    safe_mkdir(output_path.parent)
    Image.fromarray(canvas).save(output_path)
    return True


def create_scene_layout_new(
    scene_dir: Path,
    output_path: Path,
    taxonomy: Taxonomy,
    color_mode: str = "category",
    resolution: int = 512,
    margin: int = 10,
    height_min: float = 0.1,
    height_max: float = 1.8,
    point_size: int = 1,
    floor_point_size: int = None,
    floor_bbox_buffer: float = 0.1,
    object_point_size_multiplier: float = 1.0,
):
    """
    Create scene layout with new coloring scheme.
    Image is cropped to floor point bounding box with buffer.
    Scenes typically use smaller point sizes than rooms.
    """
    if floor_point_size is None:
        floor_point_size = max(point_size * 2, 3)  # Default: 2x regular size, minimum 3
    
    # Calculate object point size (scenes typically use smaller multiplier)
    object_point_size = max(int(point_size * object_point_size_multiplier), point_size)
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
            
            # Filter points:
            # - Regular (non-floor) points: height between 0.1 and 1.8 (original layout band)
            # - Floor points: include all floor points (regardless of height)
            regular_height_mask = (uvh[:, 2] >= 0.1) & (uvh[:, 2] <= 1.8)
            floor_mask = is_floor  # Include all floor points
            
            # Combine: regular points in height band OR floor points
            mask = (regular_height_mask & ~is_floor) | floor_mask
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
        # Use larger point size for floor points, and object size for regular points
        if is_floor_pt:
            size = floor_point_size
        else:
            size = object_point_size
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
    ap.add_argument("--hmin", type=float, default=0.1, help="Minimum height filter for regular points (default: 0.1, original layout band). Floor points are always included regardless of height.")
    ap.add_argument("--hmax", type=float, default=1.8, help="Maximum height filter (default: 1.8)")
    ap.add_argument("--point-size", type=int, default=5, help="Point rendering size for regular points (rooms)")
    ap.add_argument("--scene-point-size", type=int, default=None, help="Point rendering size for scenes (default: same as point-size, but typically smaller)")
    ap.add_argument("--floor-point-size", type=int, default=None, help="Point rendering size for floor points (default: 2x point-size, minimum 3)")
    ap.add_argument("--scene-floor-point-size", type=int, default=None, help="Point rendering size for floor points in scenes (default: 2x scene-point-size)")
    ap.add_argument("--floor-bbox-buffer", type=float, default=0.1, help="Buffer around floor bbox as fraction of bbox size (default: 0.1 = 10%%)")
    ap.add_argument("--object-point-size-multiplier", type=float, default=1.5, help="Multiply point size for objects in rooms to reduce sparsity (default: 1.5)")
    ap.add_argument("--scene-object-point-size-multiplier", type=float, default=1.0, help="Multiply point size for objects in scenes (default: 1.0, smaller than rooms)")
    ap.add_argument("--min-colors", type=int, default=4, help="Minimum number of distinct colors (including background) to keep image. Empty rooms have 3 (background+floor+wall), so default 4 filters them out (default: 4)")
    ap.add_argument("--max-whiteness", type=float, default=0.95, help="Maximum whiteness ratio to keep image (default: 0.95)")
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
                success = create_room_layout_new(
                    parquet_path, output_path, TAXONOMY,
                    args.color_mode,
                    resolution=args.res, 
                    height_min=args.hmin,
                    height_max=args.hmax,
                    point_size=args.point_size,
                    floor_point_size=args.floor_point_size,
                    floor_bbox_buffer=args.floor_bbox_buffer,
                    min_colors=args.min_colors,
                    max_whiteness=args.max_whiteness,
                    object_point_size_multiplier=args.object_point_size_multiplier
                )
                if success:
                    progress(i, f"{parquet_path.name} -> {output_path}", True)
                else:
                    progress(i, f"{parquet_path.name} -> filtered out", False)
            except Exception as e:
                progress(i, f"failed {parquet_path.name}: {e}", False)

    # Scenes
    if args.mode in ("scene", "both"):
        # Priority: 1) manifest_path (if provided), 2) scene_list.csv in datasets/, 3) file discovery
        scene_list_manifest = None
        
        # If manifest_path is provided and mode is scene, use it (for job arrays with sharded manifests)
        if manifest_path and manifest_path.exists():
            scene_list_manifest = manifest_path
        else:
            # Try to find scene_list.csv in datasets/ directory
            if in_root.parent:
                # Try datasets/scene_list.csv (if in_root is datasets/scenes)
                scene_list_manifest = in_root.parent / "scene_list.csv"
                if not scene_list_manifest.exists():
                    # Try datasets/scene_list.csv directly
                    datasets_dir = Path("/work3/s233249/ImgiNav/datasets")
                    scene_list_manifest = datasets_dir / "scene_list.csv"
                    if not scene_list_manifest.exists():
                        scene_list_manifest = None
        
        if scene_list_manifest and scene_list_manifest.exists():
            # Use find_scene_pointclouds to get scene_id and verify paths
            # Manifest format: scene_id,parquet_file_path,meta_file_path
            from utils.file_discovery import find_scene_pointclouds
            scene_tuples = find_scene_pointclouds(in_root, manifest=scene_list_manifest)
            # Extract scene_ids and verify scene directories exist
            scene_ids = []
            for scene_id, pc_path in scene_tuples:
                # Get scene directory from parquet file path (parent directory)
                # parquet_file_path format: /work3/.../scenes/{scene_id}/{scene_id}_sem_pointcloud.parquet
                # So pc_path.parent gives us the scene directory
                scene_dir_from_path = pc_path.parent
                # Verify scene directory exists and has rooms subdirectory
                if scene_dir_from_path.exists() and (scene_dir_from_path / "rooms").exists():
                    scene_ids.append(scene_id)
                else:
                    # Fallback: try constructing from scene_id
                    scene_dir = in_root / scene_id
                    if scene_dir.exists() and (scene_dir / "rooms").exists():
                        scene_ids.append(scene_id)
                    else:
                        print(f"[warn] Scene directory not found for {scene_id}: {scene_dir_from_path} or {scene_dir}", flush=True)
        else:
            # Default discovery
            scene_info_files = list(in_root.rglob("*_scene_info.json"))
            scene_ids = [p.stem.replace("_scene_info", "") for p in scene_info_files]

        progress = create_progress_tracker(len(scene_ids), "scene layouts")
        # Use scene-specific point sizes (default to smaller values than rooms)
        if args.scene_point_size is not None:
            scene_point_size = args.scene_point_size
        else:
            # Default to smaller than room point size (e.g., 3 instead of 5)
            scene_point_size = max(1, args.point_size - 2)
        
        scene_floor_point_size = args.scene_floor_point_size if args.scene_floor_point_size is not None else None
        
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
                    point_size=scene_point_size,
                    floor_point_size=scene_floor_point_size,
                    floor_bbox_buffer=args.floor_bbox_buffer,
                    object_point_size_multiplier=args.scene_object_point_size_multiplier
                )
                progress(i, f"{scene_id} -> {output_path}", True)
            except Exception as e:
                progress(i, f"failed {scene_id}: {e}", False)

    print(f"\n[INFO] All layout images saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

