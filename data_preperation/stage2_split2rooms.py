#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Tuple, List
from utils.file_discovery import discover_files
from utils.common import safe_mkdir, write_json, create_progress_tracker
from utils.file_discovery import infer_scene_id
from utils.semantic_utils import Taxonomy
from utils.geometry_utils import pca_plane_fit
TAXONOMY: Taxonomy = None


# ---------- Frame Computation ----------

def orient_normal_upward(normal: np.ndarray, all_xyz: np.ndarray, origin: np.ndarray) -> np.ndarray:
    heights = (all_xyz - origin) @ normal
    if (heights > 0).sum() < (heights < 0).sum():
        normal = -normal
    return normal / (np.linalg.norm(normal) + 1e-12)

def build_orthonormal_frame(origin: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = np.array([0, 1, 0], dtype=np.float64)
    X = np.array([1, 0, 0], dtype=np.float64)
    
    # Project Y onto plane perpendicular to normal
    v = Y - (Y @ normal) * normal
    if np.linalg.norm(v) < 1e-9:
        # Y is parallel to normal, use X instead
        v = X - (X @ normal) * normal
    v = v / (np.linalg.norm(v) + 1e-12)
    
    # Complete right-handed frame
    u = np.cross(normal, v)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(u, normal)
    v = v / (np.linalg.norm(v) + 1e-12)
    
    return u, v, normal

def world_to_local(xyz: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    R = np.stack([u, v, n], axis=1)  # world -> local transformation
    return (xyz - origin) @ R

def compute_room_frame(parquet_path: Path, floor_label_ids=None, height_band=(0.05, 0.50)) -> dict:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required for room frame computation")
    
    # Read point cloud
    df = pd.read_parquet(parquet_path)
    required_cols = {"x", "y", "z"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Missing columns {required_cols} in {parquet_path}")
    
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)
    
    # Extract floor points if label info available
    floor_pts = np.empty((0, 3), dtype=np.float64)
    if ("label_id" in df.columns) and (floor_label_ids is not None):
        mask = np.isin(df["label_id"].to_numpy(), np.array(floor_label_ids, dtype=df["label_id"].dtype))
        floor_pts = xyz[mask]
    
    # Fallback to low-Z points if insufficient floor labels
    if floor_pts.shape[0] < 50:
        z = xyz[:, 2]
        z_cutoff = np.quantile(z, 0.02)
        candidates = xyz[z <= z_cutoff + 1e-6]
        
        if floor_pts.shape[0] == 0 and "label_id" in df.columns and floor_label_ids is not None:
            pass  # Keep empty to make error obvious
        else:
            floor_pts = candidates
    
    if floor_pts.shape[0] < 3:
        raise RuntimeError(f"Too few floor points to compute plane (check floor_label_ids) point num = {floor_pts.shape[0]}")
    
    # Compute floor plane
    origin, normal = pca_plane_fit(floor_pts)
    normal = orient_normal_upward(normal, xyz, origin)
    u, v, n = build_orthonormal_frame(origin, normal)
    
    # Transform all points to local coordinates
    uvh = world_to_local(xyz, origin, u, v, n)
    umin, umax = float(uvh[:, 0].min()), float(uvh[:, 0].max())
    vmin, vmax = float(uvh[:, 1].min()), float(uvh[:, 1].max())
    
    # Auto-orient: longer axis becomes forward (+v)
    yaw_auto = 0.0 if (vmax - vmin) >= (umax - umin) else 90.0
    
    return {
        "origin_world": origin.tolist(),
        "u_world": u.tolist(),
        "v_world": v.tolist(),
        "n_world": n.tolist(),
        "uv_bounds": [umin, umax, vmin, vmax],
        "yaw_auto": float(yaw_auto),
        "map_band_m": [float(height_band[0]), float(height_band[1])]
    }

# ---------- Meta Writing ----------

def write_room_meta_files(root: Path, layout: str, floor_label_ids=None, height_band=(0.05, 0.50)):
    if layout == "inplace":
        pattern = "*/rooms/*/*.parquet"
    else:
        pattern = "part-*.parquet"
    
    parquet_files = list(root.rglob(pattern))
    progress = create_progress_tracker(len(parquet_files), "room frames")
    
    for i, parquet_path in enumerate(parquet_files, 1):
        try:
            meta = compute_room_frame(parquet_path, floor_label_ids, height_band)
            
            if layout == "inplace":
                # New layout: <scene>_<room>_meta.json
                stem = parquet_path.stem
                meta_path = parquet_path.parent / f"{stem}_meta.json"
            else:
                # Old layout: meta.json
                meta_path = parquet_path.parent / "meta.json"
            
            write_json(meta, meta_path)
            progress(i, f"frame: {meta_path}", True)
            
        except Exception as e:
            progress(i, f"[warn] frame failed for {parquet_path}: {e}", False)

# ---------- Main Processing ----------

def split_scene_to_rooms(input_files: List[Path], output_config: dict, columns: List[str] = None) -> int:
    processed_count = 0
    progress = create_progress_tracker(len(input_files), "scene splits")
    
    # Determine processing engine
    engine = output_config.get("engine", "auto")
    if engine in ("auto", "dataset"):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            engine = "dataset"
        except ImportError:
            if output_config.get("engine") == "dataset":
                raise RuntimeError("pyarrow not available; use --engine pandas or install pyarrow")
            engine = "pandas"

    for i, input_path in enumerate(input_files, 1):
        try:
            scene_id = infer_scene_id(input_path)
            scene_dir = input_path.parent
            
            if engine == "dataset":
                import pyarrow.parquet as pq
                table = pq.read_table(input_path)
                
                if "room_id" not in table.column_names:
                    progress(i, f"skip {input_path.name}: missing room_id", False)
                    continue
                
                # Apply column filtering
                if columns:
                    keep_cols = [c for c in columns if c in table.column_names]
                    for required in ("scene_id", "room_id"):
                        if required not in keep_cols and required in table.column_names:
                            keep_cols.append(required)
                    table = table.select(keep_cols)
                
                df = table.to_pandas()
            else:
                # Pandas fallback
                df = pd.read_parquet(input_path)
                if "room_id" not in df.columns:
                    progress(i, f"skip {input_path.name}: missing room_id", False)
                    continue
                
                # Apply column filtering
                if columns:
                    keep_cols = [c for c in columns if c in df.columns]
                    for required in ("scene_id", "room_id"):
                        if required not in keep_cols and required in df.columns:
                            keep_cols.append(required)
                    df = df[keep_cols]
            
            # Ensure scene_id column exists
            if "scene_id" not in df.columns:
                df["scene_id"] = scene_id
            
            # Split by room and save
            for (sc_id, room_id), group in df.groupby(["scene_id", "room_id"]):
                if output_config["inplace"]:
                    # New layout: scenes/<scene>/rooms/<rid>/<scene>_<rid>.parquet
                    room_dir = scene_dir / "rooms" / str(int(room_id))
                    safe_mkdir(room_dir)
                    output_path = room_dir / f"{scene_id}_{room_id}.parquet"
                else:
                    # Old layout: dataset/scene_id=<>/rooms/room_id=<>/part-*.parquet
                    room_dir = (output_config["output_dir"] / f"scene_id={sc_id}" / 
                               "rooms" / f"room_id={int(room_id)}")
                    safe_mkdir(room_dir)
                    output_path = room_dir / "part-00000.parquet"
                
                group.to_parquet(output_path, index=False)
            
            processed_count += 1
            progress(i, f"split {scene_id}", True)
            
        except Exception as e:
            progress(i, f"failed {input_path.name}: {e}", False)
    
    return processed_count



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root folder with scene parquet files")
    ap.add_argument("--glob", default="*_sem_pointcloud.parquet", help="Glob for input files")
    ap.add_argument("--out_root", help="(Default mode) Root folder for partitioned dataset")
    ap.add_argument("--dataset_name", default="room_dataset", help="(Default mode) Dataset folder name")
    ap.add_argument("--columns", nargs="*", default=[], help="Optional column restriction")
    ap.add_argument("--engine", choices=["auto", "dataset", "pandas"], default="auto")
    ap.add_argument("--inplace", action="store_true", help="Write inside each scene directory instead of separate dataset")

    # Frame computation
    ap.add_argument("--compute-frames", action="store_true", help="Compute per-room floor frames")
    ap.add_argument("--floor-label-ids", type=int, nargs="*", help="Override floor label IDs manually")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    ap.add_argument("--map-band", type=float, nargs=2, default=[0.05, 0.50],
                    help="Height band [min max] above floor")
    ap.add_argument("--manifest", help="Optional manifest CSV listing files to process")

    args = ap.parse_args()
    global TAXONOMY
    TAXONOMY = Taxonomy(Path(args.taxonomy))

    in_dir = Path(args.in_dir)

    # Validate output configuration
    if not args.inplace and not args.out_root:
        print("ERROR: --out_root required unless using --inplace", file=sys.stderr)
        sys.exit(2)

    # Set up output configuration
    output_config = {
        "inplace": args.inplace,
        "engine": args.engine
    }

    if not args.inplace:
        output_config["output_dir"] = Path(args.out_root) / args.dataset_name
        safe_mkdir(output_config["output_dir"])

    # Discover input files
    manifest_path = Path(args.manifest) if args.manifest else None
    input_files = discover_files("pointcloud", in_dir, manifest_path, args.glob, "parquet_file_path")

    if not input_files:
        print("No input files found", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(input_files)} input files")

    # Process splits
    processed = split_scene_to_rooms(input_files, output_config, args.columns)
    print(f"Processed {processed} scene files")

    # Compute frames if requested
    if args.compute_frames:
        if args.floor_label_ids:
            floor_ids = tuple(args.floor_label_ids)
            print(f"Using provided floor label IDs: {list(floor_ids)}")
        else:
            floor_ids = TAXONOMY.get_floor_ids()
            print(f"Using floor label IDs from taxonomy {args.taxonomy}: {list(floor_ids)}")

        # --- Write per-room frames ---
        print("Computing per-room floor frames...")
        layout = "inplace" if args.inplace else "default"
        search_root = in_dir if args.inplace else output_config["output_dir"]

        write_room_meta_files(
            search_root, layout,
            floor_label_ids=floor_ids,
            height_band=tuple(args.map_band)
        )
        print("Done computing frames")


if __name__ == "__main__":
    main()