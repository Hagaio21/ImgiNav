#!/usr/bin/env python3
"""
Create category-based zmaps from scene parquet files.

Creates two scales of zmaps:
1. scene_zmap: Global statistics aggregated across all scenes
2. room_zmap: Per-room statistics for each room

For each room:
1. Identifies floor points using label_id
2. Performs PCA on floor points to find the up direction (normal to floor plane)
3. For each category/color, computes height distribution (min, max, mean, std)
4. Aggregates statistics at scene and room scales

The zmap maps colors/categories to height distributions for reconstruction.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy
from common.utils import write_json, safe_mkdir, create_progress_tracker
from data_preparation.utils.geometry_utils import pca_plane_fit, world_to_local_coords, build_orthonormal_frame
from data_preparation.utils.file_discovery import find_room_files, infer_ids_from_path


def orient_normal_upward(normal: np.ndarray, all_xyz: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Ensure normal points upward (more points above than below the plane)."""
    heights = (all_xyz - origin) @ normal
    if (heights > 0).sum() < (heights < 0).sum():
        normal = -normal
    return normal / (np.linalg.norm(normal) + 1e-12)


def compute_room_up_direction(parquet_path: Path, taxonomy: Taxonomy) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the up direction for a room using PCA on floor points.
    
    Returns:
        origin: Origin point on the floor plane
        up_direction: Normalized vector pointing upward (perpendicular to floor)
    """
    df = pd.read_parquet(parquet_path)
    required_cols = {"x", "y", "z"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Missing columns {required_cols} in {parquet_path}")
    
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)
    
    # Get floor IDs from taxonomy
    floor_ids = taxonomy.get_floor_ids()
    
    # Extract floor points
    floor_pts = np.empty((0, 3), dtype=np.float64)
    if ("label_id" in df.columns) and floor_ids:
        mask = np.isin(df["label_id"].to_numpy(), np.array(floor_ids, dtype=df["label_id"].dtype))
        floor_pts = xyz[mask]
    
    # Fallback to low-Z points if insufficient floor labels
    if floor_pts.shape[0] < 50:
        z = xyz[:, 2]
        z_cutoff = np.quantile(z, 0.02)
        candidates = xyz[z <= z_cutoff + 1e-6]
        
        if floor_pts.shape[0] == 0 and "label_id" in df.columns and floor_ids:
            pass  # Keep empty to make error obvious
        else:
            floor_pts = candidates
    
    if floor_pts.shape[0] < 3:
        raise RuntimeError(f"Too few floor points to compute plane: {floor_pts.shape[0]}")
    
    # Compute floor plane using PCA
    origin, normal = pca_plane_fit(floor_pts)
    
    # Ensure normal points upward
    normal = orient_normal_upward(normal, xyz, origin)
    
    return origin, normal


def compute_category_heights(
    parquet_path: Path,
    origin: np.ndarray,
    up_direction: np.ndarray,
    taxonomy: Taxonomy
) -> Dict[str, List[float]]:
    """
    Compute height values (along up direction) for each category/color.
    
    Args:
        parquet_path: Path to room parquet file
        origin: Origin point on floor plane
        up_direction: Normalized vector pointing upward
        taxonomy: Taxonomy object for category/color mapping
    
    Returns:
        Dictionary mapping color strings (e.g., "rgb(228,26,28)") to list of height values
    """
    df = pd.read_parquet(parquet_path)
    
    required_cols = {"x", "y", "z"}
    if not required_cols.issubset(df.columns):
        return {}
    
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)
    
    # Compute height along up direction (distance from floor plane)
    # Height = (point - origin) @ up_direction
    heights = (xyz - origin) @ up_direction
    
    # Get category/color for each point
    category_heights = defaultdict(list)
    
    # Prioritize category_id (all parquet files from stage1 should have this)
    if "category_id" in df.columns:
        category_ids = df["category_id"].to_numpy()
        
        for i, category_id in enumerate(category_ids):
            category_id_int = int(category_id)
            if category_id_int == 0:
                continue  # Skip unknown categories
            
            # Get color for this category
            color = taxonomy.get_color(category_id_int, mode="category")
            
            # Convert color tuple to string format
            color_str = f"rgb({color[0]},{color[1]},{color[2]})"
            category_heights[color_str].append(float(heights[i]))
    
    # Fallback to label_id if category_id is not available
    elif "label_id" in df.columns:
        label_ids = df["label_id"].to_numpy()
        
        for i, label_id in enumerate(label_ids):
            label_id_int = int(label_id)
            if label_id_int == 0:
                continue  # Skip unknown labels
            
            # Resolve to category ID
            category_id = taxonomy._resolve_category(label_id_int)
            
            # Get color for this category
            if category_id is not None:
                color = taxonomy.get_color(category_id, mode="category")
            else:
                # Try super category
                super_id = taxonomy.resolve_super(label_id_int)
                if super_id is not None:
                    color = taxonomy.get_color(super_id, mode="none")
                else:
                    continue
            
            # Convert color tuple to string format
            color_str = f"rgb({color[0]},{color[1]},{color[2]})"
            category_heights[color_str].append(float(heights[i]))
    else:
        print(f"[warn] No category_id or label_id column found in {parquet_path}", flush=True)
    
    return category_heights


def process_room_parquet(
    parquet_path: Path,
    taxonomy: Taxonomy
) -> Optional[Dict[str, List[float]]]:
    """
    Process a single room parquet file and return category heights.
    
    Returns:
        Dictionary mapping color strings to lists of height values, or None if processing fails
    """
    try:
        # Compute up direction from floor points
        origin, up_direction = compute_room_up_direction(parquet_path, taxonomy)
        
        # Compute heights for each category
        category_heights = compute_category_heights(parquet_path, origin, up_direction, taxonomy)
        
        return category_heights
    
    except Exception as e:
        print(f"[warn] Failed to process {parquet_path}: {e}", flush=True)
        return None


def aggregate_zmap_statistics(
    all_category_heights: Dict[str, List[float]],
    taxonomy: Taxonomy
) -> Dict[str, Dict]:
    """
    Aggregate height statistics for each category/color.
    
    Args:
        all_category_heights: Dictionary mapping color strings to lists of height values
        taxonomy: Taxonomy object for semantic ID lookup
    
    Returns:
        Dictionary in zmap format: { "rgb(r,g,b)": { "min": ..., "max": ..., "mean": ..., "std": ..., "semantic_id": ..., "samples": ... } }
    """
    zmap = {}
    
    for color_str, heights in all_category_heights.items():
        if len(heights) == 0:
            continue
        
        heights_array = np.array(heights, dtype=np.float32)
        
        # Extract RGB from color string
        rgb_str = color_str.replace("rgb(", "").replace(")", "")
        r, g, b = map(int, rgb_str.split(","))
        rgb_tuple = (r, g, b)
        
        # Find semantic ID from color
        # Try super ID first (for super categories and wall)
        super_id = taxonomy.get_super_id_from_color(rgb_tuple)
        semantic_id = super_id if super_id is not None else None
        
        # If not found, try to find category ID by searching id2color
        if semantic_id is None:
            id2color = taxonomy.data.get("id2color", {})
            for id_str, color_list in id2color.items():
                if isinstance(color_list, list) and len(color_list) == 3:
                    color_tuple = tuple(color_list)
                    if color_tuple == rgb_tuple:
                        semantic_id = int(id_str)
                        break
        
        # If still not found, use 0 (Unknown)
        if semantic_id is None:
            semantic_id = 0
        
        zmap[color_str] = {
            "min": float(np.min(heights_array)),
            "max": float(np.max(heights_array)),
            "mean": float(np.mean(heights_array)),
            "std": float(np.std(heights_array)),
            "semantic_id": int(semantic_id),
            "samples": int(len(heights))
        }
    
    return zmap


def main():
    parser = argparse.ArgumentParser(
        description="Create category-based zmaps (scene and room scales) from parquet files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        required=True,
        help="Root directory containing scene parquet files"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("config/taxonomy.json"),
        help="Path to taxonomy.json file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("config"),
        help="Output directory for zmap files"
    )
    parser.add_argument(
        "--room-list",
        type=Path,
        default=None,
        help="Room list manifest CSV file (scene_id, room_id, room_parquet_file_path, ...)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="rooms/*/*.parquet",
        help="Pattern to match parquet files (e.g., 'rooms/*/*.parquet')"
    )
    
    args = parser.parse_args()
    
    # Load taxonomy
    if not args.taxonomy.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {args.taxonomy}")
    taxonomy = Taxonomy(args.taxonomy)
    
    # Find all room parquet files
    print(f"[INFO] Discovering room parquet files...")
    
    room_files = []
    room_info = []  # List of (scene_id, room_id, parquet_path) tuples
    
    # If room_list is provided, use it to find room files
    if args.room_list and args.room_list.exists():
        print(f"[INFO] Reading room_list from {args.room_list}...")
        with open(args.room_list, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene_id = row.get("scene_id", "").strip()
                room_id = row.get("room_id", "").strip()
                
                # Try to get parquet path from manifest
                parquet_path = None
                for col_name in ['room_parquet_file_path', 'parquet_file_path', 'parquet_path', 'file_path']:
                    if col_name in row and row[col_name]:
                        parquet_path = Path(row[col_name])
                        if parquet_path.exists():
                            break
                
                # If not in manifest, try to construct path
                if parquet_path is None or not parquet_path.exists():
                    scene_dir = args.parquet_root / scene_id
                    if scene_dir.exists():
                        # Look for rooms/{room_id}/*.parquet
                        room_dir = scene_dir / "rooms" / room_id
                        if room_dir.exists():
                            parquet_files = list(room_dir.glob("*.parquet"))
                            if parquet_files:
                                parquet_path = parquet_files[0]
                
                if parquet_path and parquet_path.exists():
                    room_files.append(parquet_path)
                    room_info.append((scene_id, room_id, parquet_path))
                else:
                    print(f"[warn] Room parquet not found for scene_id={scene_id}, room_id={room_id}", flush=True)
        
        print(f"[INFO] Found {len(room_files)} room parquet files from room_list")
    else:
        # Use standard discovery
        room_files = find_room_files(args.parquet_root, manifest=None, pattern=args.pattern)
        # Extract scene_id and room_id from paths
        for parquet_path in room_files:
            scene_id, room_id = infer_ids_from_path(parquet_path)
            room_info.append((str(scene_id), str(room_id), parquet_path))
        print(f"[INFO] Found {len(room_files)} room parquet files from filesystem")
    
    if not room_files:
        print(f"[ERROR] No room parquet files found")
        return 1
    
    print(f"[INFO] Processing {len(room_files)} room parquet files")
    
    # Process all rooms - collect both scene-level and room-level statistics
    scene_category_heights = defaultdict(list)  # Global aggregation for scene_zmap
    room_zmaps = {}  # Per-room zmaps: {scene_id: {room_id: zmap}}
    
    progress = create_progress_tracker(len(room_files), "processing rooms")
    
    for i, (scene_id, room_id, parquet_path) in enumerate(room_info, 1):
        category_heights = process_room_parquet(parquet_path, taxonomy)
        
        if category_heights is not None:
            # Aggregate for scene-level zmap (global)
            for color_str, heights in category_heights.items():
                scene_category_heights[color_str].extend(heights)
            
            # Create room-level zmap
            room_zmap = aggregate_zmap_statistics(category_heights, taxonomy)
            
            # Store in nested structure: scene_id -> room_id -> zmap
            if scene_id not in room_zmaps:
                room_zmaps[scene_id] = {}
            room_zmaps[scene_id][room_id] = room_zmap
        
        progress(i, f"processed {scene_id}/{room_id}", True)
    
    # Create scene-level zmap (global aggregation)
    print(f"\n[INFO] Aggregating scene-level statistics for {len(scene_category_heights)} categories...")
    scene_zmap = aggregate_zmap_statistics(scene_category_heights, taxonomy)
    
    # Save zmaps
    safe_mkdir(args.output_dir)
    
    scene_zmap_path = args.output_dir / "scene_zmap.json"
    room_zmap_path = args.output_dir / "room_zmap.json"
    
    write_json(scene_zmap, scene_zmap_path)
    write_json(room_zmaps, room_zmap_path)
    
    print(f"\n[INFO] Scene zmap saved to {scene_zmap_path}")
    print(f"[INFO] Room zmap saved to {room_zmap_path}")
    print(f"[INFO] Scene zmap: {len(scene_zmap)} categories with height statistics")
    
    # Count total rooms in room_zmap
    total_rooms = sum(len(rooms) for rooms in room_zmaps.values())
    print(f"[INFO] Room zmap: {len(room_zmaps)} scenes, {total_rooms} rooms with height statistics")
    
    # Print scene zmap summary
    print(f"\n[INFO] Scene zmap summary:")
    for color_str, stats in sorted(scene_zmap.items()):
        print(f"  {color_str}: {stats['samples']} samples, height range [{stats['min']:.3f}, {stats['max']:.3f}], mean={stats['mean']:.3f}±{stats['std']:.3f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
