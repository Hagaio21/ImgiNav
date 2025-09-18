#!/usr/bin/env python3
"""
room_split.py  (enhanced)

Create a partitioned Parquet dataset by room from scene-level semantic point clouds,
and (optionally) compute a per-room floor-aligned frame.

Inputs (from main_data_processing.py):
  <in_dir>/<scene_id>/<scene_id>_sem_pointcloud.parquet
  cols: x,y,z,r,g,b,label_id,room_id[, cat_id, super_id, merged_id...]

Modes:
  (A) Default (original behavior): write to out_root/dataset_name/scene_id=.../(rooms/)?room_id=.../part-*.parquet + meta.json
  (B) --inplace: write inside each scene directory as:
       <in_dir>/<scene_id>/rooms/<room_id>/<scene_id>_<room_id>.parquet
       <in_dir>/<scene_id>/rooms/<room_id>/<scene_id>_<room_id>_meta.json
"""

import argparse, sys, re, json
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

# ---------- split helpers ----------

def find_inputs(in_dir: Path, pattern: str):
    return sorted(in_dir.rglob(pattern))

def infer_scene_id(p: Path):
    scene = p.parent.name
    m = re.match(r"(.+)_sem_pointcloud\.parquet$", p.name)
    if m: scene = m.group(1)
    return scene

# ---------- semantic maps (auto floor ids) ----------

def find_semantic_maps_json(start: Path) -> Optional[Path]:
    """Walk up from 'start' to locate semantic_maps.json once per dataset root."""
    for p in [start, *start.parents]:
        cand = p / "semantic_maps.json"
        if cand.exists():
            return cand
    return None

def floor_label_ids_from_maps(maps_path: Path) -> Tuple[int, ...]:
    j = json.loads(maps_path.read_text(encoding="utf-8"))
    ids = set()
    if isinstance(j, dict) and "label2id" in j:
        for name, lid in j["label2id"].items():
            if str(name).strip().lower() == "floor":
                ids.add(int(lid))
    if isinstance(j, dict) and "id2label" in j:
        for lid, name in j["id2label"].items():
            if str(name).strip().lower() == "floor":
                try: ids.add(int(lid))
                except Exception: pass
    if not ids:
        raise RuntimeError(f"'floor' not found in {maps_path} (label2id/id2label).")
    return tuple(sorted(ids))

# ---------- frame helpers (fast PCA) ----------

def pca_plane(points: np.ndarray):
    """Return (origin, unit normal) using PCA on candidate floor points."""
    origin = points.mean(axis=0)
    X = points - origin
    if X.shape[0] < 3:
        n = np.array([0,0,1.0], dtype=np.float64)
        return origin.astype(np.float64), n
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)  # ascending
    n = V[:,0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return origin.astype(np.float64), n

def orient_n_up(n: np.ndarray, all_xyz: np.ndarray, origin: np.ndarray):
    """Choose normal sign so most points have positive height along +n."""
    h = (all_xyz - origin) @ n
    if (h > 0).sum() < (h < 0).sum():
        n = -n
    return n / (np.linalg.norm(n)+1e-12)

def build_uvn(origin, n):
    """Coherent in-plane axes: v = proj(+Y), fallback +X; u = n x v; re-orthonormalize."""
    Y = np.array([0,1,0], dtype=np.float64)
    X = np.array([1,0,0], dtype=np.float64)
    v = Y - (Y @ n) * n
    if np.linalg.norm(v) < 1e-9:
        v = X - (X @ n) * n
    v = v / (np.linalg.norm(v)+1e-12)
    u = np.cross(n, v); u /= (np.linalg.norm(u)+1e-12)
    v = np.cross(u, n); v /= (np.linalg.norm(v)+1e-12)
    return u, v, n

def world_to_uvh(xyz, origin, u, v, n):
    R = np.stack([u, v, n], axis=1)  # world 2 local (columns are local axes in world)
    return (xyz - origin) @ R        # row-wise dot with each column  [u,v,h]

def compute_room_frame(parquet_path: Path, floor_label_ids=None, band=(0.05, 0.50)):
    import pandas as pd
    # Read minimally once (robust to engines): then subset
    df = pd.read_parquet(parquet_path)
    if not {"x","y","z"}.issubset(df.columns):
        raise RuntimeError(f"Missing x/y/z in {parquet_path}")
    xyz = df[["x","y","z"]].to_numpy(dtype=np.float64, copy=False)

    # floor candidates strictly from labels if available
    if ("label_id" in df.columns) and (floor_label_ids is not None):
        m = np.isin(df["label_id"].to_numpy(), np.array(floor_label_ids, dtype=df["label_id"].dtype))
        floor_pts = xyz[m]
    else:
        m = None
        floor_pts = np.empty((0,3), dtype=np.float64)

    # cautious fallback only if we have too few labeled floor points
    if floor_pts.shape[0] < 50:
        z = xyz[:,2]
        zcut = np.quantile(z, 0.02)
        cand = xyz[z <= zcut + 1e-6]
        # if we *do* have a label column but wrong ids were passed, prefer labeled-over-heuristic when possible
        if floor_pts.shape[0] == 0 and "label_id" in df.columns and floor_label_ids is not None:
            pass  # keep empty; this will raise later and make the error obvious
        else:
            floor_pts = cand

    if floor_pts.shape[0] < 3:
        raise RuntimeError("Too few floor points to compute a plane (check floor_label_ids).")

    origin, n = pca_plane(floor_pts)
    n = orient_n_up(n, xyz, origin)
    u, v, n = build_uvn(origin, n)

    uvh = world_to_uvh(xyz, origin, u, v, n)
    umin, umax = float(uvh[:,0].min()), float(uvh[:,0].max())
    vmin, vmax = float(uvh[:,1].min()), float(uvh[:,1].max())

    # auto yaw: longer in-plane axis → forward (+v) vs right (+u)
    yaw_auto = 0.0 if (vmax - vmin) >= (umax - umin) else 90.0

    meta = {
        "origin_world": origin.tolist(),
        "u_world": u.tolist(),
        "v_world": v.tolist(),
        "n_world": n.tolist(),
        "uv_bounds": [umin, umax, vmin, vmax],
        "yaw_auto": float(yaw_auto),
        "map_band_m": [float(band[0]), float(band[1])]
    }
    return meta

# ---------- meta writing helpers (support both layouts) ----------

def write_meta_for_all_rooms_default_layout(dataset_root: Path, floor_label_ids=None, band=(0.05,0.50)):
    """Original layout: .../scene_id=*/(rooms/)?room_id=*/part-*.parquet → meta.json next to it."""
    parts = sorted(dataset_root.rglob("part-*.parquet"))
    for p in parts:
        try:
            meta = compute_room_frame(p, floor_label_ids=floor_label_ids, band=band)
            outp = p.parent / "meta.json"
            outp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"  ↳ frame: {outp}")
        except Exception as e:
            print(f"  [warn] frame failed for {p}: {e}")

def write_meta_for_all_rooms_inplace(in_root: Path, floor_label_ids=None, band=(0.05,0.50)):
    """In-place layout: scenes/<scene>/rooms/<rid>/<scene>_<rid>.parquet → <scene>_<rid>_meta.json."""
    parts = sorted(in_root.rglob("rooms/*/*.parquet"))
    for p in parts:
        try:
            meta = compute_room_frame(p, floor_label_ids=floor_label_ids, band=band)
            stem = p.stem  # "<scene>_<rid>"
            outp = p.parent / f"{stem}_meta.json"
            outp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"  ↳ frame: {outp}")
        except Exception as e:
            print(f"  [warn] frame failed for {p}: {e}")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root folder containing <scene_id>/*_sem_pointcloud.parquet")
    ap.add_argument("--glob", default="*_sem_pointcloud.parquet", help="Glob for input Parquet files")
    ap.add_argument("--out_root", required=False, help="(Default mode only) Root folder to write the partitioned dataset under")
    ap.add_argument("--dataset_name", default="room_dataset", help="(Default mode only) Dataset folder name under out_root")
    ap.add_argument("--columns", nargs="*", default=[],
                    help="Optional: restrict saved columns (default: keep all + ensure scene_id,room_id exist)")
    ap.add_argument("--engine", choices=["auto","dataset","pandas"], default="auto",
                    help="'dataset' uses pyarrow.dataset; 'pandas' writes per-group files; 'auto' tries dataset then falls back")
    ap.add_argument("--inplace", action="store_true",
                    help="Write inside each scene directory: scenes/<scene>/rooms/<rid>/<scene>_<rid>.parquet + _meta.json")

    # frame computation
    ap.add_argument("--compute-frames", action="store_true",
                    help="After writing partitions, compute per-room floor frame and save meta jsons")
    ap.add_argument("--floor-label-ids", type=int, nargs="*", default=None,
                    help="Optional override for 'floor' label id(s). If omitted, read from semantic_maps.json.")
    ap.add_argument("--map-band", type=float, nargs=2, default=[0.05, 0.50],
                    help="Local height band [min max] above floor, stored in meta json")
    
    ap.add_argument("--manifest", type=str,
    help="Optional manifest CSV listing files to process (overrides --in_root / auto discovery)")


    args = ap.parse_args()

    in_dir = Path(args.in_dir)

    # Default mode requires out_root; inplace mode ignores out_root/dataset_name
    if not args.inplace:
        if not args.out_root:
            print("ERROR: --out_root is required unless you pass --inplace.", file=sys.stderr)
            sys.exit(2)
        out_root = Path(args.out_root)
        out_dir = out_root / args.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_root = None
        out_dir = None  # unused in inplace mode

    inputs = []
    if args.manifest:
        import csv
        with open(args.manifest, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = None
                # prefer explicit columns
                if "scene_parquet" in row and row["scene_parquet"]:
                    path = row["scene_parquet"]
                elif "room_parquet" in row and row["room_parquet"]:
                    path = row["room_parquet"]
                else:
                    # fallback: first column with ".parquet" in the value
                    for val in row.values():
                        if val and val.endswith(".parquet"):
                            path = val
                            break
                if path:
                    p = Path(path).expanduser().resolve()
                    if p.exists():
                        inputs.append(p)
                    else:
                        print(f"[warn] listed file not found: {p}", file=sys.stderr)


    else:
        inputs = find_inputs(in_dir, args.glob)

    if not inputs:
        print(f"No inputs found (manifest or scan).", file=sys.stderr)
        sys.exit(2)


    # Decide engine
    engine = args.engine
    if engine in ("auto","dataset"):
        try:
            import pyarrow as pa  # noqa
            import pyarrow.dataset as ds  # noqa
            import pyarrow.parquet as pq  # noqa: F401
            engine = "dataset"
        except Exception:
            if args.engine == "dataset":
                print("pyarrow.dataset not available; use --engine pandas or install pyarrow.", file=sys.stderr)
                sys.exit(3)
            engine = "pandas"

    kept = 0
    if engine == "dataset":
        import pyarrow as pa
        import pyarrow.parquet as pq
        # We still read per scene, then group by room (so we can write to either layout)
        for p in inputs:
            scene_id = infer_scene_id(p)
            scene_dir = p.parent  # original scene folder
            t = pq.read_table(p)
            if "room_id" not in t.column_names:
                print(f"Skipping {p.name}: missing 'room_id' column.", file=sys.stderr)
                continue

            if args.columns:
                keep = [c for c in args.columns if c in t.column_names]
                for c in ("scene_id","room_id"):
                    if c not in keep and c in t.column_names:
                        keep.append(c)
                t = t.select(keep)

            df = t.to_pandas()  # group by room in pandas for simplicity/compatibility
            if "scene_id" not in df.columns:
                df["scene_id"] = scene_id

            for (sc, rid), g in df.groupby(["scene_id","room_id"]):
                if args.inplace:
                    sub = scene_dir / "rooms" / str(int(rid))
                    sub.mkdir(parents=True, exist_ok=True)
                    outp = sub / f"{scene_id}_{rid}.parquet"
                else:
                    # original dataset layout (kept intact)
                    sub = (out_dir / f"scene_id={sc}" / "rooms" / f"room_id={int(rid)}")
                    sub.mkdir(parents=True, exist_ok=True)
                    outp = sub / "part-00000.parquet"
                g.to_parquet(outp, index=False)
            kept += 1

        print(f"✔ Wrote partitioned dataset for {kept} scenes  (engine=pyarrow.dataset)")

    else:
        # pandas fallback
        import pandas as pd
        for p in inputs:
            scene_id = infer_scene_id(p)
            scene_dir = p.parent
            df = pd.read_parquet(p)
            if "room_id" not in df.columns:
                print(f"Skipping {p.name}: missing 'room_id' column.", file=sys.stderr)
                continue
            df["scene_id"] = df.get("scene_id", scene_id)

            if args.columns:
                keep = [c for c in args.columns if c in df.columns]
                for c in ("scene_id","room_id"):
                    if c not in keep and c in df.columns:
                        keep.append(c)
                df = df[keep]

            for (sc, rid), g in df.groupby(["scene_id","room_id"]):
                if args.inplace:
                    sub = scene_dir / "rooms" / str(int(rid))
                    sub.mkdir(parents=True, exist_ok=True)
                    outp = sub / f"{scene_id}_{rid}.parquet"
                else:
                    sub = (out_dir / f"scene_id={sc}" / "rooms" / f"room_id={int(rid)}")
                    sub.mkdir(parents=True, exist_ok=True)
                    outp = sub / "part-00000.parquet"
                g.to_parquet(outp, index=False)
            kept += 1

        print(f"✔ Wrote partitioned dataset for {kept} scenes  (engine=pandas)")

    # ---- post-pass: compute per-room frames once and cache ----
    if args.compute_frames:
        # resolve floor ids: CLI override > semantic_maps.json > error
        floor_ids = tuple(args.floor_label_ids) if args.floor_label_ids else None
        if floor_ids is None:
            maps = find_semantic_maps_json(in_dir if args.inplace else (out_dir or in_dir))
            if maps is None:
                # also try from input scenes directory (useful for inplace)
                maps = find_semantic_maps_json(in_dir)
            if maps is None:
                raise RuntimeError("semantic_maps.json not found (needed to read 'floor' label id).")
            floor_ids = floor_label_ids_from_maps(maps)
            print(f"Using floor label id(s) from {maps}: {list(floor_ids)}")
        else:
            print(f"Using provided floor label id(s): {list(floor_ids)}")

        print("⧉ Computing per-room floor frames ...")
        if args.inplace:
            write_meta_for_all_rooms_inplace(in_dir, floor_label_ids=floor_ids, band=tuple(args.map_band))
        else:
            write_meta_for_all_rooms_default_layout(out_dir, floor_label_ids=floor_ids, band=tuple(args.map_band))
        print("⧉ Done computing frames.")

if __name__ == "__main__":
    main()
