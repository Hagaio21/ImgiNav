#!/usr/bin/env python3

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import json

from utils.utils import (
    discover_files, load_room_meta, extract_frame_from_meta, 
    find_semantic_maps_json, load_global_palette, create_progress_tracker, safe_mkdir
)
from utils.semantic_utils import Taxonomy
from utils.geometry_utils import (
    world_to_local_coords

)

TAXONOMY = None
# ---------- Rendering Helpers ----------

def draw_point(canvas: np.ndarray, x: int, y: int, color: np.ndarray, size: int = 1):
    half = size // 2
    x0, x1 = max(x - half, 0), min(x + half, canvas.shape[1] - 1)
    y0, y1 = max(y - half, 0), min(y + half, canvas.shape[0] - 1)
    canvas[y0:y1 + 1, x0:x1 + 1] = color

def points_to_image_coords(u_vals: np.ndarray, v_vals: np.ndarray, 
                          uv_bounds: Tuple[float, float, float, float],
                          resolution: int, margin: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    umin, umax, vmin, vmax = uv_bounds
    span = max(umax - umin, vmax - vmin, 1e-6)
    scale = (resolution - 2 * margin) / span
    
    u_pix = (u_vals - umin) * scale + margin
    v_pix = (v_vals - vmin) * scale + margin
    
    # Flip V for image coordinates (origin at top-left)
    x_img = np.clip(np.round(u_pix).astype(np.int32), 0, resolution - 1)
    y_img = np.clip(np.round((resolution - 1) - v_pix).astype(np.int32), 0, resolution - 1)
    
    return x_img, y_img

def load_taxonomy(taxonomy_path):
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    category2color = {}
    super2color = {}

    # Structural categories (direct mapping)
    if "structural" in taxonomy:
        for cat, info in taxonomy["structural"].items():
            if "color" in info:
                category2color[cat.lower()] = tuple(info["color"])

    # Furniture categories (nested under super)
    if "furniture" in taxonomy:
        for super_name, super_info in taxonomy["furniture"].items():
            if "color" in super_info:
                super2color[super_name.lower()] = tuple(super_info["color"])
            if "categories" in super_info:
                for cat, info in super_info["categories"].items():
                    if "color" in info:
                        category2color[cat.lower()] = tuple(info["color"])

    return category2color, super2color


# ---------- Room Layout Generation ----------


def create_room_layout(
    parquet_path: Path,
    output_path: Path,
    color_mode: str = "category",
    resolution: int = 512,
    margin: int = 10,
    height_min: float = None,
    height_max: float = None,
    point_size: int = 1,
):
    # Load room metadata
    meta = load_room_meta(parquet_path.parent)
    if meta is None:
        raise RuntimeError(f"No metadata found for {parquet_path}")

    origin, u, v, n, uv_bounds, _, map_band = extract_frame_from_meta(meta)

    # Resolve height filtering bounds
    if height_min is None or height_max is None:
        if map_band and len(map_band) == 2:
            if height_min is None:
                height_min = float(map_band[0])
            if height_max is None:
                height_max = float(map_band[1])

    if height_min is None:
        height_min = 0.0
    if height_max is None:
        height_max = 2.5
    if height_max <= height_min:
        raise ValueError(f"height_max ({height_max}) must be > height_min ({height_min})")

    # Load and process point cloud
    df = pd.read_parquet(parquet_path)
    required_cols = {"x", "y", "z", "label_id"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Missing required columns {required_cols} in {parquet_path}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    labels = df["label_id"].to_numpy(dtype=np.int32)

    # Transform to local coordinates
    uvh = world_to_local_coords(xyz, origin, u, v, n)

    # Apply height filtering
    #print("DEBUG layout:", height_min, height_max)
    height_mask = (uvh[:, 2] >= height_min) & (uvh[:, 2] <= height_max)
    if height_mask.sum() == 0:
        print(f"[warn] no points in height band [{height_min},{height_max}] m in {parquet_path}",flush=True)
        return

    u_vals, v_vals = uvh[height_mask, 0], uvh[height_mask, 1]
    filtered_labels = labels[height_mask]

    # Convert to image coordinates
    x_img, y_img = points_to_image_coords(u_vals, v_vals, uv_bounds, resolution, margin)

    # Render to canvas
    canvas = np.full((resolution, resolution, 3), 240, dtype=np.uint8)
    for lbl, x, y in zip(filtered_labels, x_img, y_img):
        if lbl is None:
            raise ValueError(f"Unexpected None label in {parquet_path}")
        color = TAXONOMY.get_color(lbl, mode=color_mode)
        draw_point(canvas, x, y, np.array(color, dtype=np.uint8), size=point_size)

    # Save image
    safe_mkdir(output_path.parent)
    Image.fromarray(canvas).save(output_path)

# ---------- Scene Layout Generation ----------

# def create_scene_layout(
#     scene_dir: Path,
#     output_path: Path,
#     color_mode: str = "category",
#     resolution: int = 512,
#     margin: int = 10,
#     height_min: float = None,
#     height_max: float = None,
#     point_size: int = 1,
# ):
#     """Generate combined layout image for entire scene using taxonomy palette."""
#     scene_id = scene_dir.name
#     room_parquets = sorted(scene_dir.rglob("rooms/*/*.parquet"))
#     if not room_parquets:
#         print(f"[warn] no room parquets found in {scene_dir}",flush=True)
#         return

#     # Get reference frame from first room
#     first_meta = load_room_meta(room_parquets[0].parent)
#     if first_meta is None:
#         print(f"[warn] missing metadata for {room_parquets[0]}",flush=True)
#         return
#     origin, u, v, n, _, _, _ = extract_frame_from_meta(first_meta)

#     # Collect global bounds
#     all_u_bounds, all_v_bounds = [], []
#     for parquet_path in room_parquets:
#         try:
#             df = pd.read_parquet(parquet_path, columns=["x", "y", "z"])
#             xyz = df.to_numpy(dtype=np.float32)
#             uvh = world_to_local_coords(xyz, origin, u, v, n)
#             all_u_bounds.extend([uvh[:, 0].min(), uvh[:, 0].max()])
#             all_v_bounds.extend([uvh[:, 1].min(), uvh[:, 1].max()])
#         except Exception as e:
#             print(f"[warn] failed bounds for {parquet_path}: {e}",flush=True)

#     if not all_u_bounds:
#         print(f"[warn] no usable points in {scene_dir}",flush=True)
#         return
#     global_uv_bounds = (min(all_u_bounds), max(all_u_bounds), min(all_v_bounds), max(all_v_bounds))

#     # Render all rooms to single canvas
#     canvas = np.full((resolution, resolution, 3), 240, dtype=np.uint8)
#     for parquet_path in room_parquets:
#         try:
#             df = pd.read_parquet(parquet_path)
#             required_cols = {"x", "y", "z", "label_id"}
#             if not required_cols.issubset(df.columns):
#                 continue

#             xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
#             labels = df["label_id"].to_numpy(dtype=np.int32)

#             uvh = world_to_local_coords(xyz, origin, u, v, n)
#             mask = np.ones(len(xyz), dtype=bool)
#             if height_min is not None:
#                 mask &= uvh[:, 2] >= height_min
#             if height_max is not None:
#                 mask &= uvh[:, 2] <= height_max

#             u_vals, v_vals = uvh[mask, 0], uvh[mask, 1]
#             f_labels = labels[mask]

#             x_img, y_img = points_to_image_coords(u_vals, v_vals, global_uv_bounds, resolution, margin)
#             for lbl, x, y in zip(f_labels, x_img, y_img):
#                 color = TAXONOMY.get_color(lbl, mode=color_mode)
#                 draw_point(canvas, x, y, np.array(color, dtype=np.uint8), size=point_size)
#         except Exception as e:
#             print(f"[warn] skipping {parquet_path}: {e}",flush=True)

#     safe_mkdir(output_path.parent)
#     Image.fromarray(canvas).save(output_path)



def create_scene_layout(
    scene_dir: Path,
    output_path: Path,
    color_mode: str = "category",
    resolution: int = 512,
    margin: int = 10,
    height_min: float = None,
    height_max: float = None,
    point_size: int = 1,
):
    scene_id = scene_dir.name
    room_parquets = sorted(scene_dir.rglob("rooms/*/*.parquet"))
    if not room_parquets:
        print(f"[warn] no room parquets found in {scene_dir}",flush=True)
        return

    # Get reference frame from first room
    first_meta = load_room_meta(room_parquets[0].parent)
    if first_meta is None:
        print(f"[warn] missing metadata for {room_parquets[0]}",flush=True)
        return
    origin, u, v, n, _, _, _ = extract_frame_from_meta(first_meta)

    # SAVE THE SCENE COORDINATE FRAME TO scene_info.json
    scene_info_path = scene_dir / f"{scene_id}_scene_info.json"
    if scene_info_path.exists():
        # Load existing scene_info
        scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    else:
        scene_info = {}
    
    # Add coordinate frame (same frame used to generate layout)
    scene_info["origin_world"] = origin.tolist()
    scene_info["u_world"] = u.tolist()
    scene_info["v_world"] = v.tolist()
    scene_info["n_world"] = n.tolist()
    
    # Save updated scene_info
    scene_info_path.write_text(json.dumps(scene_info, indent=2), encoding="utf-8")

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
            print(f"[warn] failed bounds for {parquet_path}: {e}",flush=True)

    if not all_u_bounds:
        print(f"[warn] no usable points in {scene_dir}",flush=True)
        return
    global_uv_bounds = (min(all_u_bounds), max(all_u_bounds), min(all_v_bounds), max(all_v_bounds))

    # Render all rooms to single canvas
    canvas = np.full((resolution, resolution, 3), 240, dtype=np.uint8)
    for parquet_path in room_parquets:
        try:
            df = pd.read_parquet(parquet_path)
            required_cols = {"x", "y", "z", "label_id"}
            if not required_cols.issubset(df.columns):
                continue

            xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            labels = df["label_id"].to_numpy(dtype=np.int32)

            uvh = world_to_local_coords(xyz, origin, u, v, n)
            mask = np.ones(len(xyz), dtype=bool)
            if height_min is not None:
                mask &= uvh[:, 2] >= height_min
            if height_max is not None:
                mask &= uvh[:, 2] <= height_max

            u_vals, v_vals = uvh[mask, 0], uvh[mask, 1]
            f_labels = labels[mask]

            x_img, y_img = points_to_image_coords(u_vals, v_vals, global_uv_bounds, resolution, margin)
            for lbl, x, y in zip(f_labels, x_img, y_img):
                color = TAXONOMY.get_color(lbl, mode=color_mode)
                draw_point(canvas, x, y, np.array(color, dtype=np.uint8), size=point_size)
        except Exception as e:
            print(f"[warn] skipping {parquet_path}: {e}",flush=True)

    safe_mkdir(output_path.parent)
    Image.fromarray(canvas).save(output_path)
# ---------- Discovery Helpers ----------
from utils.file_discovery import discover_rooms, discover_scenes_from_rooms, discover_scenes_from_manifest

# ---------- Main Processing ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root folder with scenes or room dataset")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON file")
    ap.add_argument("--pattern", help="Glob pattern for parquet files")
    ap.add_argument("--res", type=int, default=512, help="Output image resolution")
    ap.add_argument("--hmin", type=float, default=0.1, help="Minimum height filter")
    ap.add_argument("--hmax", type=float, default=1.8, help="Maximum height filter")
    ap.add_argument("--point-size", type=int, default=5, help="Point rendering size")
    ap.add_argument("--manifest", help="Optional manifest CSV")
    ap.add_argument("--mode", choices=["room", "scene", "both"], default="both")
    ap.add_argument("--color-mode", choices=["category", "super"], default="category",
                    help="Color mode for rendering")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    taxonomy_path = Path(args.taxonomy)
    manifest_path = Path(args.manifest) if args.manifest else None
    
    # Load taxonomy once
    global TAXONOMY
    TAXONOMY = Taxonomy(taxonomy_path)


    # Rooms
    if args.mode in ("room", "both"):
        room_files = discover_rooms(in_root, args.pattern, manifest_path)
        progress = create_progress_tracker(len(room_files), "room layouts")
        for i, parquet_path in enumerate(room_files, 1):
            scene_id, room_id = parquet_path.stem.split("_")[:2]
            output_path = parquet_path.parent / "layouts" / f"{scene_id}_{room_id}_room_seg_layout.png"
            try:
                create_room_layout(parquet_path, output_path,
                                   args.color_mode,
                                   resolution=args.res, height_min=args.hmin,
                                   height_max=args.hmax, point_size=args.point_size)
                progress(i, f"{parquet_path.name} -> {output_path}", True)
            except Exception as e:
                progress(i, f"failed {parquet_path.name}: {e}", False)

    # Scenes
    if args.mode in ("scene", "both"):
        if manifest_path:
            # Use scene IDs from manifest
            scene_ids = discover_scenes_from_manifest(manifest_path)
            scene_info_files = [in_root / sid / f"{sid}_scene_info.json" for sid in scene_ids]
        else:
            # Default discovery
            scene_info_files = list(in_root.rglob("*_scene_info.json"))
            scene_ids = [p.stem.replace("_scene_info", "") for p in scene_info_files]

        progress = create_progress_tracker(len(scene_ids), "scene layouts")
        for i, scene_id in enumerate(scene_ids, 1):
            scene_dir = in_root / scene_id
            output_path = scene_dir / "layouts" / f"{scene_id}_scene_layout.png"
            try:
                create_scene_layout(scene_dir, output_path,
                                    args.color_mode,
                                    resolution=args.res, height_min=args.hmin,
                                    height_max=args.hmax, point_size=args.point_size)
                progress(i, f"{scene_id} -> {output_path}", True)
            except Exception as e:
                progress(i, f"failed {scene_id}: {e}", False)




if __name__ == "__main__":
    main()