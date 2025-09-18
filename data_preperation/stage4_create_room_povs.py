#!/usr/bin/env python3

import argparse, json, math, re, sys, hashlib
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

import csv
from pathlib import Path
from typing import Optional, List


# only for HPC
# from xvfbwrapper import Xvfb
# from xvfbwrapper import Xvfb

# ----------- constants -----------
TILT_DEG = 10.0  # look slightly downward for better floor visibility

# ----------- IO helpers -----------
def infer_scene_id(p: Path) -> str:
    # new filename format: <scene_id>_<room_id>.parquet
    m = re.match(r"([0-9a-fA-F-]+)_\d+\.parquet$", p.name)
    if m: return m.group(1)
    # old path format: .../scene_id=<ID>/room_id=...
    m = re.search(r"scene_id=([^/\\]+)", str(p))
    if m: return m.group(1)
    # fallback from data if present later
    return p.stem

def infer_room_id(p: Path) -> int:
    # new filename format: ..._<room_id>.parquet
    m = re.match(r".+_(\d+)\.parquet$", p.name)
    if m: return int(m.group(1))
    # old path format
    m = re.search(r"room_id=(\d+)", str(p))
    if m: return int(m.group(1))
    return -1



def find_room_files(root: Path, manifest: Optional[Path] = None) -> List[Path]:
    """
    Discover room parquet files either from a manifest CSV or by scanning the dataset root.
    """
    if manifest is not None:
        rows = []
        with open(manifest, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "room_parquet" in row:   # manifests for stage2/3/4 always have this
                    rows.append(Path(row["room_parquet"]))
        return rows

    files = sorted(root.rglob("scene_id=*/room_id=*/*.parquet"))
    return files or sorted(root.rglob("*.parquet"))

def load_meta(parquet_path: Path):
    """Load meta from room folder. Tries new '<scene>_<rid>_meta.json', then legacy 'meta.json'."""
    room_dir = parquet_path.parent
    cand = list(room_dir.glob("*_meta.json"))
    if cand:
        mpath = cand[0]
    else:
        mpath = room_dir / "meta.json"
        if not mpath.exists():
            return None
    j = json.loads(mpath.read_text(encoding="utf-8"))
    to_arr = lambda k: np.array(j[k], dtype=np.float32)
    origin = to_arr("origin_world")
    u = to_arr("u_world")
    v = to_arr("v_world")
    n = to_arr("n_world")
    uv_bounds = tuple(j["uv_bounds"])         # (umin, umax, vmin, vmax)
    yaw_auto = float(j.get("yaw_auto", 0.0))
    map_band = tuple(j.get("map_band_m", [0.05, 0.50]))
    return origin, u, v, n, uv_bounds, yaw_auto, map_band

def find_semantic_maps_json(start: Path) -> Optional[Path]:
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
        raise RuntimeError("'floor' not found in semantic_maps.json")
    return tuple(sorted(ids))


def render_offscreen(pcd, width, height, eye, center, up, fov_deg, bg_rgb, point_size, out_path) -> bool:
    import open3d as o3d
    import time as _t, math as _m
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array(bg_rgb, dtype=np.float32) / 255.0

    fx = (0.5 * width) / _m.tan(_m.radians(fov_deg) / 2.0)
    fy = (0.5 * height) / _m.tan(_m.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f)
        l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l
        M[1, :3] = u2
        M[2, :3] = f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye_
        return M @ T

    pin.extrinsic = look_at(
        eye.astype(np.float64), center.astype(np.float64), up.astype(np.float64)
    )

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()
    _t.sleep(0.12)  # give OpenGL time
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    return True

def render_legacy_capture(pcd, width, height, eye, center, up, fov_deg, bg_rgb, point_size, out_path) -> bool:
    import open3d as o3d, time as _t, math as _m
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array(bg_rgb, dtype=np.float32)/255.0
    fx = (0.5*width) / _m.tan(_m.radians(fov_deg)/2.0)
    fy = (0.5*height) / _m.tan(_m.radians(fov_deg)/2.0)
    cx, cy = width/2.0, height/2.0
    pin = o3d.camera.PinholeCameraParameters()
    pin.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    def look_at(eye_, center_, up_):
        f = center_ - eye_
        f = f / (np.linalg.norm(f) + 1e-12)
        upn = up_ / (np.linalg.norm(up_) + 1e-12)
        l = np.cross(upn, f); l = l / (np.linalg.norm(l) + 1e-12)
        u2 = np.cross(f, l)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = l; M[1, :3] = u2; M[2, :3] = f
        T = np.eye(4, dtype=np.float64); T[:3, 3] = -eye_
        return M @ T
    pin.extrinsic = look_at(eye.astype(np.float64), center.astype(np.float64), up.astype(np.float64))
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
    vis.poll_events(); vis.update_renderer(); _t.sleep(0.12)
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    return True

# ----------- minimap -----------
def minimap_floor_black(uv: np.ndarray, is_floor: np.ndarray, res=768, margin=10,
                        floor_rgb=(255,0,0), other_rgb=(0,0,0), bg=(240,240,240)):
    from PIL import Image, ImageDraw
    if uv.shape[0]==0:
        return Image.new("RGB",(res,res),bg), (0,1,0,1)
    umin,vmin = uv.min(axis=0); umax,vmax = uv.max(axis=0)
    L = max(umax-umin, vmax-vmin, 1e-9); scale = (res-2*margin)/L
    upix = (uv[:,0]-umin)*scale + margin
    vpix = (uv[:,1]-vmin)*scale + margin
    xi = np.clip(np.round(upix).astype(np.int32), 0, res-1)
    yi = np.clip(np.round((res-1)-vpix).astype(np.int32), 0, res-1)
    floor_count = np.zeros((res,res), dtype=np.int32)
    other_count = np.zeros((res,res), dtype=np.int32)
    np.add.at(floor_count, (yi[is_floor], xi[is_floor]), 1)
    np.add.at(other_count, (yi[~is_floor], xi[~is_floor]), 1)
    canvas = np.full((res,res,3), np.array(bg,dtype=np.uint8), dtype=np.uint8)
    canvas[other_count>0] = np.array(other_rgb, dtype=np.uint8)
    canvas[floor_count>0] = np.array(floor_rgb, dtype=np.uint8)
    img = Image.fromarray(canvas)
    # axes legend u→, v↑
    draw = ImageDraw.Draw(img); ax=max(24,res//7); ox=margin+6; oy=res-margin-6
    draw.line([ox,oy,ox+ax,oy], fill=(0,0,0), width=2)
    draw.polygon([(ox+ax,oy),(ox+ax-8,oy-4),(ox+ax-8,oy+4)], fill=(0,0,0))
    draw.line([ox,oy,ox,oy-ax], fill=(0,0,0), width=2)
    draw.polygon([(ox,oy-ax),(ox-4,oy-ax+8),(ox+4,oy-ax+8)], fill=(0,0,0))
    return img, (umin,umax,vmin,vmax)

def draw_cam_arrows_on_minimap_uv(img, uv_bounds, cams_uv: np.ndarray, angles_deg: List[float], res: int):
    from PIL import ImageDraw
    umin,umax,vmin,vmax = uv_bounds; margin=10
    L = max(umax-umin, vmax-vmin, 1e-6); scale = (res-2*margin)/L
    draw = ImageDraw.Draw(img)
    Lp = max(16, res//10); head = max(6, res//40); r = max(3, res//90)
    for (cu,cv), ang in zip(cams_uv, angles_deg):
        cx = (cu-umin)*scale + margin
        cy = (cv-vmin)*scale + margin; cy = (res-1)-cy
        draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(220,30,30), outline=(20,20,20), width=1)
        th = math.radians(float(ang))
        ex,ey = cx + Lp*math.sin(th), cy - Lp*math.cos(th)  # 0° = +v
        draw.line([cx,cy,ex,ey], fill=(220,30,30), width=2)
        left = th+math.radians(150); right = th-math.radians(150)
        p2=(ex+head*math.sin(left),  ey-head*math.cos(left))
        p3=(ex+head*math.sin(right), ey-head*math.cos(right))
        draw.polygon([(ex,ey),p2,p3], fill=(220,30,30))

# ----------- utils -----------
def load_global_palette(start: Path) -> dict:
    maps_path = find_semantic_maps_json(start)
    if maps_path is None:
        raise RuntimeError("semantic_maps.json not found.")
    j = json.loads(maps_path.read_text(encoding="utf-8"))
    if "id2color" not in j:
        raise RuntimeError("id2color missing in semantic_maps.json. Run generate_palette.py first.")
    return {int(k): tuple(v) for k, v in j["id2color"].items()}

def stable_room_seed(scene_id: str, room_id: int, user_seed: int) -> np.random.RandomState:
    if user_seed >= 0:
        return np.random.RandomState(user_seed)
    key = f"{scene_id}:{room_id}".encode("utf-8")
    seed = int(hashlib.sha1(key).hexdigest()[:8], 16)
    return np.random.RandomState(seed)

def process_room(parquet_path: Path, root_out_unused: Path,
                 width=1280, height=800, fov_deg=70.0, eye_height=1.6,
                 point_size=2.0, bg_rgb=(0,0,0),
                 num_views: int = 6, seed: int = -1) -> bool:
    import open3d as o3d
    import pandas as pd

    meta = load_meta(parquet_path)
    if meta is None:
        print(f"[skip] no meta.json → {parquet_path.parent}")
        return False
    origin, u, v, n, uv_bounds_all, yaw_auto, _band = meta

    maps_path = find_semantic_maps_json(parquet_path.parent)
    if maps_path is None:
        raise RuntimeError(f"semantic_maps.json not found near {parquet_path.parent}")
    floor_ids = floor_label_ids_from_maps(maps_path)

    df = pd.read_parquet(parquet_path)
    xyz = df[["x","y","z"]].to_numpy(np.float32)
    raw = df[["r","g","b"]].to_numpy()
    rgb = (raw.astype(np.float32)/255.0) if not np.issubdtype(raw.dtype, np.floating) else raw.astype(np.float32)
    labels = df["label_id"].to_numpy() if "label_id" in df.columns else None
    scene_id = df["scene_id"].iloc[0] if "scene_id" in df.columns else infer_scene_id(parquet_path)
    room_id  = int(df["room_id"].iloc[0]) if "room_id" in df.columns else infer_room_id(parquet_path)

    if labels is None:
        raise RuntimeError(f"'label_id' column missing in {parquet_path}")

    # --- output dirs inside room folder ---
    out_dir = parquet_path.parent / "povs"
    tex_dir = out_dir / "tex"
    seg_dir = out_dir / "seg"
    tex_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # build Open3D clouds
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64, copy=False))

    palette = load_global_palette(parquet_path.parent)
    seg_cols = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for uid in np.unique(labels):
        color = palette.get(int(uid), (128, 128, 128))
        seg_cols[labels == uid] = np.array(color, dtype=np.float32) / 255.0
    seg = o3d.geometry.PointCloud()
    seg.points = pcd.points
    seg.colors = o3d.utility.Vector3dVector(seg_cols.astype(np.float64, copy=False))

    # local coords strictly from meta
    R = np.stack([u, v, n], axis=1)           # world <- local
    uvh = (xyz - origin) @ R
    is_floor = np.isin(labels, np.array(floor_ids, dtype=labels.dtype))

    # FLOOR AABB for placement
    if is_floor.any():
        fu = uvh[is_floor, 0]; fv = uvh[is_floor, 1]
        uminF, umaxF = float(fu.min()), float(fu.max())
        vminF, vmaxF = float(fv.min()), float(fv.max())
        center_u = float(np.median(fu))
        center_v = float(np.median(fv))
    else:
        uminF, umaxF, vminF, vmaxF = uv_bounds_all
        center_u = float(np.median(uvh[:,0]))
        center_v = float(np.median(uvh[:,1]))

    # -------- corner views only --------
    cams_uv = []
    used_f_world = []
    corner_names = []

    duF = umaxF - uminF
    dvF = vmaxF - vminF
    inset = max(0.05 * max(duF, dvF), 0.10)  # 5% span, min 10cm
    corners = [
        (uminF + inset, vminF + inset),
        (umaxF - inset, vminF + inset),
        (umaxF - inset, vmaxF - inset),
        (uminF + inset, vmaxF - inset),
    ]

    def horiz_from_yaw_inward(yaw_deg: float, cu: float, cv: float) -> Tuple[np.ndarray, float]:
        # candidate A: 0°=+v, 90°=+u
        fA = math.cos(math.radians(yaw_deg))*v + math.sin(math.radians(yaw_deg))*u
        # candidate B: 0°=+u, 90°=+v
        fB = math.cos(math.radians(yaw_deg))*u + math.sin(math.radians(yaw_deg))*v
        # choose the one pointing more toward the room center
        d = np.array([center_u - cu, center_v - cv], dtype=np.float32)
        if np.linalg.norm(d) < 1e-9:
            f = fA
        else:
            auA, avA = float(np.dot(fA, u)), float(np.dot(fA, v))
            auB, avB = float(np.dot(fB, u)), float(np.dot(fB, v))
            scoreA = auA*d[0] + avA*d[1]
            scoreB = auB*d[0] + avB*d[1]
            f = fA if scoreA >= scoreB else fB
        # if still outward (negative dot), flip 180°
        au, av = float(np.dot(f, u)), float(np.dot(f, v))
        if au*(center_u - cu) + av*(center_v - cv) < 0.0:
            f = -f
        return f, yaw_deg

    for j, (cu, cv) in enumerate(corners):
        cu = float(np.clip(cu, uminF, umaxF))
        cv = float(np.clip(cv, vminF, vmaxF))
        cams_uv.append((cu, cv))

        d = np.array([center_u - cu, center_v - cv], dtype=np.float32)
        if np.linalg.norm(d) < 1e-9:
            f_world, yaw_used = horiz_from_yaw_inward(float(yaw_auto), cu, cv)
        else:
            d = d / (np.linalg.norm(d) + 1e-12)
            f_world = float(d[0]) * u + float(d[1]) * v
            yaw_used = math.degrees(math.atan2(np.dot(f_world, u), np.dot(f_world, v)))
        used_f_world.append(f_world)
        corner_names.append(f"{j:02d}")

    all_names = corner_names

    # -------- render all POVs with a slight downward tilt --------
    tilt = math.tan(math.radians(TILT_DEG))  # magnitude for +n (down in image when up=-n)

    def render_pair(view_num: str, f_world: np.ndarray, cu: float, cv: float):
        try:
            aim = f_world - tilt * n
            eye = origin + cu*u + cv*v + eye_height*n
            center = eye + aim
            up = -n

            base_name = f"{scene_id}_{room_id}_v-{view_num}"
            tex_name = f"{base_name}_pov_tex.png"
            seg_name = f"{base_name}_pov_seg.png"

            tex_path = tex_dir / tex_name
            seg_path = seg_dir / seg_name

            render_offscreen(pcd, width, height, eye, center, up,
                             fov_deg, bg_rgb, point_size, tex_path)
            print(f"  ✔ {tex_path}", flush=True)

            render_offscreen(seg, width, height, eye, center, up,
                             fov_deg, bg_rgb, point_size, seg_path)
            print(f"  ✔ {seg_path}", flush=True)

            return aim
        except Exception as e:
            print(f"Failed to render pair for {scene_id}", flush=True)
            print(f"{e}")

    aims_used = []
    for (cu, cv), f_world, name in zip(cams_uv, used_f_world, all_names):
        aims_used.append(render_pair(name, f_world, cu, cv))

    # -------- minimap LAST --------
    uv = uvh[:, :2]
    mm_img, uvb = minimap_floor_black(uv, is_floor, res=768, margin=10)

    angles = []
    for aim in aims_used:
        horiz = aim - np.dot(aim, n) * n
        au, av = float(np.dot(horiz, u)), float(np.dot(horiz, v))
        angles.append(math.degrees(math.atan2(au, av)))

    mm_uv = np.array(cams_uv, dtype=np.float32)
    draw_cam_arrows_on_minimap_uv(mm_img, uvb, mm_uv, angles, 768)
    mm_name = f"{scene_id}_{room_id}_minimap.png"
    mm_path = out_dir / mm_name
    mm_img.save(str(mm_path))
    print(f"  ✔ {mm_path}", flush=True)

    return True

# ----------- CLI -----------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--path", type=str, help="Single room parquet (new or old layout)")
    g.add_argument("--dataset-root", type=str, help="Root with scenes or room_dataset")
    ap.add_argument("--out-dir", type=str, default="./pov_out",
                    help="(Kept for compatibility; in new layout outputs go beside each room)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fov-deg", type=float, default=70.0)
    ap.add_argument("--eye-height", type=float, default=1.6)
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--bg", type=int, nargs=3, default=[0,0,0])
    ap.add_argument("--num-views", type=int, default=6,
                    help="Total base views per room (>=1). 1 auto + (N-1) random. Corner views are added on top.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="Random seed; if <0 uses deterministic seed per room.")
    ap.add_argument("--manifest", type=str,
                    help="Optional manifest CSV listing files to process (overrides --dataset-root / auto discovery)")

    args = ap.parse_args()
    bg_rgb = tuple(args.bg)

    # --- Single room mode ---
    if args.path:
        ok = process_room(
            Path(args.path), Path(args.out_dir),
            args.width, args.height, args.fov_deg,
            args.eye_height, args.point_size, bg_rgb,
            num_views=args.num_views, seed=args.seed
        )
        sys.exit(0 if ok else 1)

    # --- Multi-room mode ---
    root = Path(args.dataset_root)
    files = find_room_files(root, args.manifest)
    if not files:
        print(f"No room parquets found (manifest or scan).", flush=True)
        sys.exit(2)

    vdisplay = None
    try:
        print("Starting virtual X display for Open3D...", flush=True)
        # only for HPC:
        # vdisplay = Xvfb(width=1920, height=1080, colordepth=24)
        # vdisplay.start()

        print(f"Found {len(files)} room files...", flush=True)
        print(f"Will generate {(args.num_views+4)*2} images per room, "
              f"{(args.num_views+4)*len(files)*2} images", flush=True)

        any_ok = False
        for f in files:
            try:
                print(f"- {f}", flush=True)
                ok = process_room(
                    f, Path(args.out_dir),
                    args.width, args.height, args.fov_deg,
                    args.eye_height, args.point_size, bg_rgb,
                    num_views=args.num_views, seed=args.seed
                )
                any_ok = any_ok or ok
            except Exception as e:
                print(f"  [error] {e}", flush=True)

        sys.exit(0 if any_ok else 1)

    finally:
        if vdisplay is not None:
            print("Stopping virtual X display.", flush=True)
            vdisplay.stop()

if __name__ == "__main__":
    main()
