#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
room_layout.py — segmented top-down layout images per room

Changes:
- Support BOTH layouts:
    Old:   .../scene_id=*/room_id=*/*.parquet + meta.json
    New:   scenes/<scene>/rooms/<rid>/<scene>_<rid>.parquet + <scene>_<rid>_meta.json
- Height band filtering: keep points with local height h in [hmin, hmax]
  (h is the 3rd coord in local [u,v,h], i.e., meters above the floor plane)
  Defaults: if not provided, uses map_band_m from meta (if present), else [0.0, 2.5].
- Keeps --point-size and palette coloring via semantic_maps.json (id2color).
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional

# ---------------- IO helpers ----------------
def load_meta(room_dir: Path):
    """Load meta from room dir. Try new '<scene>_<rid>_meta.json' then legacy 'meta.json'/'room_meta.json'.
       Returns (origin,u,v,n,uv_bounds,map_band or None)."""
    cand = list(room_dir.glob("*_meta.json"))
    meta_path = None
    if cand:
        meta_path = cand[0]
    else:
        for name in ("meta.json", "room_meta.json"):
            p = room_dir / name
            if p.exists():
                meta_path = p
                break
    if meta_path is None:
        raise RuntimeError(f"No meta json found in {room_dir}")

    j = json.loads(meta_path.read_text(encoding="utf-8"))
    origin = np.array(j["origin_world"], dtype=np.float32)
    u = np.array(j["u_world"], dtype=np.float32)
    v = np.array(j["v_world"], dtype=np.float32)
    n = np.array(j["n_world"], dtype=np.float32)
    uv_bounds = tuple(j["uv_bounds"])  # (umin, umax, vmin, vmax)
    band = tuple(j.get("map_band_m", [])) if "map_band_m" in j else None
    return origin, u, v, n, uv_bounds, band

def load_global_palette(start: Path) -> dict:
    """Find semantic_maps.json and return id2color mapping."""
    for p in [start, *start.parents]:
        maps_path = p / "semantic_maps.json"
        if maps_path.exists():
            j = json.loads(maps_path.read_text(encoding="utf-8"))
            if "id2color" not in j:
                raise RuntimeError("id2color missing in semantic_maps.json. Run generate_palette.py first.")
            # keys may be str; normalize to int
            return {int(k): tuple(v) for k, v in j["id2color"].items()}
    raise RuntimeError("semantic_maps.json not found.")

# ---------------- rasterization helper ----------------
def draw_point(canvas, x, y, color, size=1):
    half = size // 2
    x0, x1 = max(x-half, 0), min(x+half, canvas.shape[1]-1)
    y0, y1 = max(y-half, 0), min(y+half, canvas.shape[0]-1)
    canvas[y0:y1+1, x0:x1+1] = color

# ---------------- layout generation ----------------
def create_seg_layout(parquet_path: Path, out_path: Path, res=768, margin=10,
                      hmin=None, hmax=None, point_size=1):
    # meta + palette
    origin, u, v, n, uv_bounds, band = load_meta(parquet_path.parent)
    palette = load_global_palette(parquet_path.parent)

    # Resolve height band
    if hmin is None or hmax is None:
        if band is not None and len(band) == 2:
            if hmin is None: hmin = float(band[0])
            if hmax is None: hmax = float(band[1])
    # final defaults
    if hmin is None: hmin = 0.0
    if hmax is None: hmax = 2.5
    if hmax <= hmin:
        raise ValueError(f"hmax ({hmax}) must be > hmin ({hmin})")

    # read point cloud
    df = pd.read_parquet(parquet_path)
    if "label_id" not in df.columns:
        raise RuntimeError(f"'label_id' column missing in {parquet_path}")
    xyz = df[["x","y","z"]].to_numpy(dtype=np.float32)
    labels = df["label_id"].to_numpy(dtype=np.int32)

    # project into local uvh coords
    R = np.stack([u, v, n], axis=1)
    uvh = (xyz - origin) @ R  # columns: [u, v, h]

    # height band mask
    mask = (uvh[:,2] >= hmin) & (uvh[:,2] <= hmax)
    if mask.sum() == 0:
        print(f"[warn] no points in height band [{hmin},{hmax}] m in {parquet_path}")
        return
    uvals, vvals, labels = uvh[mask,0], uvh[mask,1], labels[mask]

    # scale to image coords
    umin, umax, vmin, vmax = uv_bounds
    L = max(umax-umin, vmax-vmin, 1e-6)
    scale = (res-2*margin) / L
    upix = (uvals - umin) * scale + margin
    vpix = (vvals - vmin) * scale + margin
    xi = np.clip(np.round(upix).astype(np.int32), 0, res-1)
    yi = np.clip(np.round((res-1)-vpix).astype(np.int32), 0, res-1)

    # rasterize labels with point size control
    canvas = np.full((res,res,3), 240, dtype=np.uint8)
    for uid in np.unique(labels):
        color = palette.get(int(uid), (128,128,128))
        mask_uid = labels == uid
        # write small squares (size×size)
        col = np.array(color, dtype=np.uint8)
        for x, y in zip(xi[mask_uid], yi[mask_uid]):
            draw_point(canvas, x, y, col, size=point_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGB").save(out_path)
    print(f"✔ wrote {out_path} (h∈[{hmin},{hmax}] m)")

# ---------------- discovery ----------------
import csv
from typing import Optional, List
from pathlib import Path

def find_parquets(in_root: Path, pattern: Optional[str], manifest: Optional[Path] = None) -> List[Path]:
    """
    Discover parquet files either from a manifest CSV or by scanning the dataset root.
    """
    if manifest is not None:
        rows = []
        with open(manifest, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "room_parquet" in row:   # stage2/3/4 manifests all have this
                    rows.append(Path(row["room_parquet"]))
        return rows

    if pattern:
        return sorted(in_root.rglob(pattern))

    # Default discovery modes
    parts = sorted(in_root.rglob("part-*.parquet"))            # old partitioned
    parts += sorted(in_root.rglob("*_*[0-9].parquet"))         # new <scene>_<rid>.parquet
    if not parts:
        parts = sorted(in_root.rglob("rooms/*/*.parquet"))
    return parts

def create_scene_layout(scene_dir: Path, out_path: Path, res=768, margin=10,
                        hmin=None, hmax=None, point_size=1):
    import pandas as pd
    import numpy as np
    from PIL import Image
    import json

    # --- helpers ---
    def find_semantic_maps_json(start: Path):
        for p in [start, *start.parents]:
            cand = p / "semantic_maps.json"
            if cand.exists():
                return cand
        return None

    def load_global_palette(scene_dir: Path):
        maps_path = find_semantic_maps_json(scene_dir)
        if not maps_path:
            return {}
        maps = json.loads(maps_path.read_text(encoding="utf-8"))
        id2color = maps.get("id2color", {})
        # ensure keys are ints
        return {int(k): tuple(v) for k, v in id2color.items()}

    def draw_point(img, x, y, color, size=1):
        h, w = img.shape[:2]
        r = size // 2
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                xi, yj = x+i, y+j
                if 0 <= xi < w and 0 <= yj < h:
                    img[yj, xi] = color

    scene_id = scene_dir.name
    parquets = sorted(scene_dir.rglob("rooms/*/*.parquet"))
    if not parquets:
        print(f"[warn] no room parquets in {scene_dir}")
        return

    # --- orientation from first room's meta ---
    first_meta = parquets[0].with_name(parquets[0].stem + "_meta.json")
    if not first_meta.exists():
        print(f"[warn] missing meta.json for {parquets[0]}")
        return
    ref_meta = json.loads(first_meta.read_text(encoding="utf-8"))
    origin = np.array(ref_meta["origin_world"], dtype=np.float32)
    u = np.array(ref_meta["u_world"], dtype=np.float32)
    v = np.array(ref_meta["v_world"], dtype=np.float32)
    n = np.array(ref_meta["n_world"], dtype=np.float32)
    R = np.stack([u, v, n], axis=1)

    # --- collect floor label ids ---
    maps_path = find_semantic_maps_json(scene_dir)
    if not maps_path:
        print(f"[warn] semantic_maps.json not found for {scene_dir}")
        return
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    floor_ids = {int(lid) for name, lid in maps.get("label2id", {}).items()
                 if str(name).strip().lower() == "floor"}

    # --- compute global UV bounds using floor (or all) points ---
    umins, umaxs, vmins, vmaxs = [], [], [], []
    for parquet_path in parquets:
        try:
            cols = ["x","y","z","label_id"]
            df = pd.read_parquet(parquet_path, columns=cols)
            mask = df["label_id"].isin(floor_ids)
            if not mask.any():  # fallback to all points
                xyz = df[["x","y","z"]].to_numpy(dtype=np.float32)
            else:
                xyz = df.loc[mask, ["x","y","z"]].to_numpy(dtype=np.float32)

            uvh = (xyz - origin) @ R
            umins.append(float(uvh[:,0].min())); umaxs.append(float(uvh[:,0].max()))
            vmins.append(float(uvh[:,1].min())); vmaxs.append(float(uvh[:,1].max()))
        except Exception as e:
            print(f"[warn] failed bounds for {parquet_path}: {e}")

    if not umins:
        print(f"[warn] no usable points found in {scene_dir}")
        return
    umin, umax = min(umins), max(umaxs)
    vmin, vmax = min(vmins), max(vmaxs)

    # --- prepare canvas ---
    palette = load_global_palette(scene_dir)
    canvas = np.full((res, res, 3), 240, dtype=np.uint8)

    # --- draw all rooms ---
    for parquet_path in parquets:
        try:
            df = pd.read_parquet(parquet_path)
            if "label_id" not in df.columns:
                continue
            xyz = df[["x","y","z"]].to_numpy(dtype=np.float32)
            labels = df["label_id"].to_numpy(dtype=np.int32)

            uvh = (xyz - origin) @ R
            mask = np.ones(len(xyz), dtype=bool)
            if hmin is not None:
                mask &= uvh[:,2] >= hmin
            if hmax is not None:
                mask &= uvh[:,2] <= hmax
            uvals, vvals, labels = uvh[mask,0], uvh[mask,1], labels[mask]

            L = max(umax - umin, vmax - vmin, 1e-6)
            scale = (res - 2*margin) / L
            upix = (uvals - umin) * scale + margin
            vpix = (vvals - vmin) * scale + margin
            xi = np.clip(np.round(upix).astype(np.int32), 0, res-1)
            yi = np.clip(np.round((res-1)-vpix).astype(np.int32), 0, res-1)

            for uid in np.unique(labels):
                color = palette.get(int(uid), (128,128,128))
                mask_uid = labels == uid
                col = np.array(color, dtype=np.uint8)
                for x, y in zip(xi[mask_uid], yi[mask_uid]):
                    draw_point(canvas, x, y, col, size=point_size)
        except Exception as e:
            print(f"[warn] skipping {parquet_path}: {e}")

    # --- save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Image.fromarray(canvas, "RGB").save(out_path)
    Image.fromarray(canvas).save(out_path)
    print(f"✔ wrote scene layout {out_path}")


# ---------------- main ----------------
def main():
    import time, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root folder with scenes or room_dataset")
    ap.add_argument("--pattern", default=None, help="Glob for parquet files")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--hmin", type=float, default=0.1)
    ap.add_argument("--hmax", type=float, default=1.8)
    ap.add_argument("--ceiling-thresh", type=float, default=None)
    ap.add_argument("--point-size", type=int, default=5)
    ap.add_argument("--manifest", type=str, help="Optional manifest CSV listing files to process")
    ap.add_argument("--mode", choices=["room","scene","both"], default="room")
    args = ap.parse_args()

    in_root = Path(args.in_root)

    # ------------------- ROOM LAYOUTS -------------------
    if args.mode in ("room", "both"):
        parts = find_parquets(in_root, args.pattern, args.manifest)
        if not parts:
            print("No room parquets found (manifest or scan).", flush=True)
        else:
            print(f"Preparing to generate {len(parts)} room layouts...", flush=True)

        for idx, p in enumerate(parts, 1):
            # --- extract scene_id and room_id ---
            parts_name = p.stem.split("_")
            if len(parts_name) >= 2:
                # new layout: <scene_id>_<room_id>.parquet
                scene_id, room_id = parts_name[0], parts_name[1]
            else:
                # old layout: scenes/.../scene_id=<uuid>/room_id=<num>/*.parquet
                scene_id = p.parents[2].name   # scene folder
                room_id = p.parent.name        # room folder

            out_name = f"{scene_id}_{room_id}_room_seg_layout.png"
            out_path = p.parent / "layouts" / out_name

            print(f"[{idx}/{len(parts)}] Starting room {p}...", flush=True)
            start = time.time()
            try:
                create_seg_layout(
                    p, out_path,
                    res=args.res,
                    hmin=args.hmin,
                    hmax=args.hmax,
                    point_size=args.point_size
                )
                elapsed = time.time() - start
                print(f"[{idx}/{len(parts)}] Finished {p.name} in {elapsed:.2f} sec → {out_path}", flush=True)
            except Exception as e:
                print(f"[{idx}/{len(parts)}] [warn] failed for {p}: {e}", flush=True)


    # ------------------- SCENE LAYOUTS -------------------
    if args.mode in ("scene", "both"):
        if args.manifest:
            scene_ids = []
            with open(args.manifest, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "scene_id" in row and row["scene_id"]:
                        scene_ids.append(row["scene_id"])
        else:
            scene_ids = [p.stem.replace("_scene_info","") for p in in_root.rglob("*_scene_info.json")]

        if not scene_ids:
            print("No scenes found (manifest or scan).", flush=True)
        else:
            print(f"Preparing to generate {len(scene_ids)} scene layouts...", flush=True)

        for idx, sid in enumerate(scene_ids, 1):
            scene_dir = in_root / sid
            out_path = scene_dir / "layouts" / f"{sid}_scene_layout.png"
            print(f"[{idx}/{len(scene_ids)}] Starting scene {sid}...", flush=True)
            start = time.time()
            try:
                create_scene_layout(
                    scene_dir, out_path,
                    res=args.res,
                    hmin=args.hmin,
                    hmax=args.hmax,
                    point_size=args.point_size
                )
                elapsed = time.time() - start
                print(f"[{idx}/{len(scene_ids)}] Finished {sid} in {elapsed:.2f} sec → {out_path}", flush=True)
            except Exception as e:
                print(f"[{idx}/{len(scene_ids)}] [warn] failed for scene {sid}: {e}", flush=True)


if __name__ == "__main__":
    main()

