#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage1_build_scenes.py (refactored)

Batch processor for 3D-FRONT scenes using shared utilities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

# Import shared utilities
from utils.utils import (
    gather_scene_paths, infer_scene_id, load_taxonomy, resolve_category,
    update_semantic_maps, resolve_config_with_profile, 
    create_progress_tracker, safe_mkdir, write_json
)

@dataclass
class SavePolicy:
    glb: bool = True
    obj: bool = True
    csv: bool = True
    npy: bool = False
    npz: bool = True
    parquet: bool = True

SAVE = SavePolicy()

def export_to_obj_manually(meshes_with_info: List[Dict], file_path: Path):
    """Export meshes to OBJ format (non-textured)."""
    vertex_offset = 1
    with file_path.open('w', encoding='utf-8') as f:
        f.write("# Manually constructed OBJ file (no textures)\n")
        for info in meshes_with_info:
            mesh = info['mesh']
            label_safe = info['label'].replace(' ', '_').replace('/', '-')
            f.write(f"\n# Object: {info['label']}\n")
            f.write(f"o {label_safe}\n")
            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in mesh.faces:
                f.write(f"f {face[0] + vertex_offset} {face[1] + vertex_offset} {face[2] + vertex_offset}\n")
            vertex_offset += len(mesh.vertices)

def get_point_colors(mesh: trimesh.Trimesh, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Sample colors from mesh (UV-aware or vertex colors)."""
    N = len(points)
    if N == 0 or mesh is None or mesh.is_empty:
        return np.zeros((0, 3), dtype=np.uint8)

    vis = getattr(mesh, "visual", None)

    # Try UV + material image first
    try:
        uv = getattr(vis, "uv", None) if vis is not None else None
        mat = getattr(vis, "material", None) if vis is not None else None
        img = getattr(mat, "image", None) if mat is not None else None
        if (uv is not None) and (img is not None):
            tris = mesh.triangles[face_indices]
            bary = trimesh.triangles.points_to_barycentric(tris, points)
            faces = mesh.faces[face_indices]
            tri_uv = uv[faces]
            uv_pts = (bary[:, :, None] * tri_uv).sum(axis=1)
            
            img_np = np.asarray(img.convert("RGB"))
            H, W = img_np.shape[:2]
            u = np.clip(uv_pts[:, 0], 0.0, 1.0) * (W - 1)
            v = (1.0 - np.clip(uv_pts[:, 1], 0.0, 1.0)) * (H - 1)
            ui = np.clip(np.round(u).astype(np.int64), 0, W - 1)
            vi = np.clip(np.round(v).astype(np.int64), 0, H - 1)
            cols = img_np[vi, ui, :].astype(np.uint8)
            return cols
    except Exception:
        pass

    # Try vertex colors
    try:
        vcols = getattr(vis, "vertex_colors", None) if vis is not None else None
        if vcols is not None and len(vcols) == len(mesh.vertices):
            tris = mesh.triangles[face_indices]
            bary = trimesh.triangles.points_to_barycentric(tris, points)
            faces = mesh.faces[face_indices]
            tri_vc = vcols[faces][:, :, :3]
            cols = (bary[:, :, None] * tri_vc.astype(np.float32)).sum(axis=1)
            return np.clip(np.round(cols), 0, 255).astype(np.uint8)
    except Exception:
        pass

    # Fallback: mid gray
    return np.full((N, 3), 128, dtype=np.uint8)

def save_scene_info(meshes_info: List[Dict], output_dir: Path, scene_id: str) -> Dict | None:
    """Calculate and save scene bounds/size/up_normal."""
    if not meshes_info:
        print("[WARNING] Cannot generate scene_info.json because no meshes were processed.")
        return None
        
    all_vertices_list = [info['mesh'].vertices for info in meshes_info]
    if not all_vertices_list:
        print("[WARNING] No vertices found to calculate scene info.")
        return None
        
    all_vertices = np.vstack(all_vertices_list)
    min_bounds = all_vertices.min(axis=0)
    max_bounds = all_vertices.max(axis=0)
    size = max_bounds - min_bounds

    # Simple Z-up heuristic
    floor_vertices_list = [info['mesh'].vertices for info in meshes_info if info['label'] == 'floor']
    furniture_vertices_list = [info['mesh'].vertices for info in meshes_info if info['label'] != 'floor']

    up_normal = [0.0, 0.0, 1.0]
    if floor_vertices_list and furniture_vertices_list:
        floor_vertices = np.vstack(floor_vertices_list)
        furniture_vertices = np.vstack(furniture_vertices_list)
        if floor_vertices[:, 2].mean() > furniture_vertices[:, 2].mean():
            up_normal = [0.0, 0.0, -1.0]

    scene_info = {
        "bounds": {"min": min_bounds.tolist(), "max": max_bounds.tolist()},
        "size": size.tolist(),
        "up_normal": up_normal
    }
    
    json_path = output_dir / f"{scene_id}_scene_info.json"
    write_json(scene_info, json_path)
    print(f"[INFO] Saved scene parameters: {json_path}")
    
    return scene_info

def process_scene_for_all_outputs(
    scene_data: dict,
    model_dir: Path,
    model_info_map: dict,
    total_points: int = 500_000,
    points_per_sq_meter: float = 0.0,
    min_pts_per_mesh: int = 100,
    max_pts_per_mesh: int = 0,
    taxonomy_path: str | None = None,
    alias_only_merge: bool = False,
) -> Tuple[trimesh.Scene, List[Dict], np.ndarray, Dict]:
    """Process scene and generate all outputs."""
    failed_models: Dict[str, str] = {}
    taxonomy = load_taxonomy(Path(taxonomy_path)) if taxonomy_path else None

    scene_objects: List[Dict] = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}

    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "UnknownRoom")
        for child_index, child in enumerate(room.get("children", [])):
            ref_id = child.get("ref")
            if not ref_id:
                continue
                
            node_name = f"{ref_id}_{child_index}"
            try:
                mesh, label = None, "unknown"

                if ref_id in furniture_map:
                    item_info = furniture_map[ref_id]
                    jid = item_info.get('jid')
                    if not jid:
                        raise ValueError("Missing 'jid' for furniture item")
                    
                    label = (model_info_map.get(jid, {}) or {}).get('category') \
                            or item_info.get('title') or "unknown"
                    
                    # Try OBJ first, fallback to GLB
                    obj_path = model_dir / jid / "raw_model.obj"
                    glb_path = model_dir / jid / "raw_model.glb"

                    if obj_path.exists():
                        resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
                        mesh = trimesh.load(str(obj_path), force='mesh', process=False, 
                                          maintain_order=True, resolver=resolver)
                    elif glb_path.exists():
                        mesh = trimesh.load(str(glb_path), force='mesh', process=False, 
                                          maintain_order=True)
                    else:
                        raise FileNotFoundError(f"Model not found: {obj_path} or {glb_path}")

                elif ref_id in arch_map:
                    arch = arch_map[ref_id]
                    atype = arch.get("type", "")
                    if "Ceiling" in atype:
                        continue  # skip ceilings
                    label = 'floor' if 'Floor' in atype else 'wall'
                    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
                    faces = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                    mesh.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)

                else:
                    failed_models[ref_id] = "Reference not found in furniture/mesh maps"
                    continue

                if (mesh is None) or mesh.is_empty or mesh.faces.shape[0] == 0:
                    raise ValueError("Mesh is empty or has no faces")

                # Apply node transform
                pos = np.array(child.get("pos", [0, 0, 0]), dtype=np.float64)
                rot = np.array(child.get("rot", [0, 0, 0, 1]), dtype=np.float64)  # x,y,z,w
                scl = np.array(child.get("scale", [1, 1, 1]), dtype=np.float64)

                T = np.eye(4, dtype=np.float64)
                T[:3, 3] = pos
                Rm = Rotation.from_quat(rot).as_matrix()
                Sm = np.diag(scl)
                T[:3, :3] = Rm @ Sm

                scene_objects.append({
                    "mesh": mesh,
                    "transform": T,
                    "label": label if isinstance(label, str) and label.strip() else "unknown",
                    "room_type": room_type if isinstance(room_type, str) and room_type.strip() else "UnknownRoom",
                    "node_name": node_name,
                    "model_item": furniture_map.get(ref_id) if ref_id in furniture_map else None
                })

            except Exception as e:
                failed_models[ref_id] = f"{type(e).__name__}: {e}"
                continue

    if not scene_objects:
        return None, None, None, failed_models

    # Build textured scene
    textured_scene = trimesh.Scene()
    for obj in scene_objects:
        textured_scene.add_geometry(obj['mesh'], node_name=obj['node_name'], transform=obj['transform'])

    # Create world-space copies
    meshes_info_world: List[Dict] = []
    for obj in scene_objects:
        mc = obj['mesh'].copy()
        mc.apply_transform(obj['transform'])
        mc.fix_normals()
        meshes_info_world.append({'mesh': mc, 'label': obj['label']})

    # Sample points
    areas = [float(max(0.0, o['mesh'].area)) for o in scene_objects]
    total_area = sum(a for a in areas if a > 0.0) or 1.0

    all_points, all_colors = [], []
    all_labels, all_room_types = [], []
    all_categories, all_supers, all_merged = [], [], []
    total_sampled = 0

    for obj, area in zip(scene_objects, areas):
        mesh_local: trimesh.Trimesh = obj['mesh']
        if area <= 0.0 or mesh_local.is_empty or mesh_local.faces.shape[0] == 0:
            continue

        # Determine sample count
        if points_per_sq_meter > 0.0:
            n_pts = int(round(area * points_per_sq_meter))
        else:
            n_pts = int(round((area / total_area) * total_points))
        
        n_pts = max(min_pts_per_mesh, n_pts)
        if max_pts_per_mesh > 0:
            n_pts = min(n_pts, max_pts_per_mesh)
        if n_pts <= 0:
            continue

        # Sample surface
        pts_local, face_indices = trimesh.sample.sample_surface(mesh_local, n_pts)
        cols = get_point_colors(mesh_local, pts_local, face_indices)
        pts_world = trimesh.transform_points(pts_local, obj['transform'])

        # Resolve categories
        if taxonomy:
            sem = resolve_category(obj['label'], obj.get("model_item"), taxonomy, alias_only_merge)
        else:
            sem = {"category": "", "super": obj['label'], "merged": obj['label']}

        total_sampled += n_pts
        all_points.append(pts_world)
        all_colors.append(cols)
        all_labels.extend([obj['label']] * n_pts)
        all_room_types.extend([obj['room_type']] * n_pts)
        all_categories.extend([sem["category"]] * n_pts)
        all_supers.extend([sem["super"]] * n_pts)
        all_merged.extend([sem["merged"]] * n_pts)

    print(f"ℹ️ sampled points (scene): {total_sampled}")

    # Build structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ('label', 'U100'), ('room_type', 'U50'),
        ('category', 'U80'), ('super', 'U80'), ('merged', 'U80')
    ]
    point_cloud_structured = np.array([], dtype=dtype)

    if all_points:
        P = np.vstack(all_points).astype(np.float32)
        C = np.vstack(all_colors).astype(np.uint8)
        N = P.shape[0]

        point_cloud_structured = np.empty(N, dtype=dtype)
        point_cloud_structured['x'], point_cloud_structured['y'], point_cloud_structured['z'] = P.T
        point_cloud_structured['r'], point_cloud_structured['g'], point_cloud_structured['b'] = C.T
        point_cloud_structured['label'] = np.array(all_labels, dtype='U100')
        point_cloud_structured['room_type'] = np.array(all_room_types, dtype='U50')
        point_cloud_structured['category'] = np.array(all_categories, dtype='U80')
        point_cloud_structured['super'] = np.array(all_supers, dtype='U80')
        point_cloud_structured['merged'] = np.array(all_merged, dtype='U80')

    return textured_scene, meshes_info_world, point_cloud_structured, failed_models

def export_pointcloud_compact(
    output_dir: Path,
    scene_id: str,
    point_cloud: np.ndarray,
    save_csv: bool = True,
    save_npz: bool = True,
    save_parquet: bool = True,
    csv_include_strings: bool = False,
    maps_root: Path | None = None,
    freeze_maps: bool = False,
):
    """Export point cloud in compact formats with integer IDs."""
    if point_cloud.size == 0:
        print("⚠️ Empty point cloud; nothing to export.")
        return

    maps_root = maps_root or output_dir

    # Extract geometry and color
    xyz = np.vstack([point_cloud["x"], point_cloud["y"], point_cloud["z"]]).T.astype(np.float32)
    rgb = np.vstack([point_cloud["r"], point_cloud["g"], point_cloud["b"]]).T.astype(np.uint8)

    # Extract string categories
    labels = point_cloud["label"].astype(str).tolist()
    rooms = point_cloud["room_type"].astype(str).tolist()
    cats = point_cloud["category"].astype(str).tolist() if "category" in point_cloud.dtype.names else []
    supers = point_cloud["super"].astype(str).tolist() if "super" in point_cloud.dtype.names else []
    merged = point_cloud["merged"].astype(str).tolist() if "merged" in point_cloud.dtype.names else []

    # Build value mapping
    values_by = {"label": labels, "room": rooms}
    if cats: values_by["category"] = cats
    if supers: values_by["super"] = supers
    if merged: values_by["merged"] = merged

    maps, _ = update_semantic_maps(maps_root, values_by, freeze=freeze_maps)

    # Convert to integer IDs
    label_id = np.array([maps["label"].get(s, 0) for s in labels], dtype=np.uint16)
    room_id = np.array([maps["room"].get(s, 0) for s in rooms], dtype=np.uint16)
    
    out_data = {"xyz": xyz, "rgb": rgb, "label_id": label_id, "room_id": room_id}
    if cats: out_data["cat_id"] = np.array([maps["category"].get(s, 0) for s in cats], dtype=np.uint16)
    if supers: out_data["super_id"] = np.array([maps["super"].get(s, 0) for s in supers], dtype=np.uint16)
    if merged: out_data["merged_id"] = np.array([maps["merged"].get(s, 0) for s in merged], dtype=np.uint16)

    # Save formats
    if save_npz:
        np.savez_compressed(output_dir / f"{scene_id}_sem_pointcloud.npz", **out_data)
        print(f"💾 NPZ → {output_dir / f'{scene_id}_sem_pointcloud.npz'}")

    if save_parquet:
        try:
            df_cols = {
                "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                "r": rgb[:, 0], "g": rgb[:, 1], "b": rgb[:, 2],
                "label_id": out_data["label_id"], "room_id": out_data["room_id"]
            }
            for k in ("cat_id", "super_id", "merged_id"):
                if k in out_data:
                    df_cols[k] = out_data[k]
            
            pd.DataFrame(df_cols).to_parquet(
                output_dir / f"{scene_id}_sem_pointcloud.parquet",
                compression="snappy", index=False
            )
            print(f"💾 Parquet → {output_dir / f'{scene_id}_sem_pointcloud.parquet'}")
        except Exception as e:
            print(f"ℹ️ Parquet skipped: {e}")

    if save_csv:
        data = {
            "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            "r": rgb[:, 0], "g": rgb[:, 1], "b": rgb[:, 2],
            "label_id": out_data["label_id"], "room_id": out_data["room_id"],
        }
        for k in ("cat_id", "super_id", "merged_id"):
            if k in out_data:
                data[k] = out_data[k]
        
        if csv_include_strings:
            data["label"] = labels
            data["room_type"] = rooms
            if cats: data["category"] = cats
            if supers: data["super"] = supers
            if merged: data["merged"] = merged
        
        pd.DataFrame(data).to_csv(output_dir / f"{scene_id}_sem_pointcloud.csv", index=False)
        print(f"💾 CSV → {output_dir / f'{scene_id}_sem_pointcloud.csv'}")

def process_one_scene(scene_path: Path, model_dir: Path, model_info_file: Path, 
                     out_root: Path, args) -> bool:
    """Process a single scene file."""
    try:
        with open(model_info_file, "r", encoding="utf-8") as f:
            model_info_map = {item["model_id"]: item for item in json.load(f)}
    except Exception as e:
        print(f"❌ model_info load failed: {e}")
        return False

    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
    except Exception as e:
        print(f"❌ scene load failed [{scene_path}]: {e}")
        return False

    scene_id = scene_path.stem
    out_dir = (out_root / scene_id) if args.per_scene_subdir else out_root
    safe_mkdir(out_dir)

    print(f"\n▶ Processing {scene_id} → {out_dir}")

    textured_scene, meshes_info, point_cloud, failed_models = process_scene_for_all_outputs(
        scene_data, model_dir, model_info_map,
        total_points=args.total_points,
        points_per_sq_meter=args.ppsm,
        min_pts_per_mesh=args.min_pts_per_mesh,
        max_pts_per_mesh=args.max_pts_per_mesh,
        taxonomy_path=args.taxonomy,
        alias_only_merge=args.alias_only_merge,
    )

    if failed_models:
        print(f"⚠️ Skipped {len(failed_models)} models:")
        for mid, why in failed_models.items():
            print(f"    - {mid}: {why}")

    if not meshes_info:
        print("❌ No meshes produced; skipping.")
        return False

    # Save outputs
    if args.save_glb:
        glb_path = out_dir / f"{scene_id}_textured.glb"
        textured_scene.export(file_obj=glb_path, file_type="glb")
        print(f"💾 GLB → {glb_path}")

    if args.save_obj:
        obj_path = out_dir / f"{scene_id}.obj"
        export_to_obj_manually(meshes_info, obj_path)
        print(f"💾 OBJ → {obj_path}")

    # Scene info
    scene_info = save_scene_info(meshes_info, out_dir, scene_id)

    # Point cloud
    if point_cloud.size:
        if args.save_npy:
            npy_path = out_dir / f"{scene_id}_sem_pointcloud.npy"
            np.save(npy_path, point_cloud)
            print(f"💾 NPY (legacy) → {npy_path}")

        export_pointcloud_compact(
            out_dir, scene_id, point_cloud,
            save_csv=args.save_csv,
            save_npz=args.save_npz,
            save_parquet=args.save_parquet,
            csv_include_strings=args.csv_with_strings,
            maps_root=Path(args.maps_root) if args.maps_root else out_root,
            freeze_maps=args.freeze_maps,
        )

    return True

def parse_args_with_config() -> argparse.Namespace:
    """Parse arguments with config file support."""
    Bool = argparse.BooleanOptionalAction

    # Pre-parse config and profile
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", help="Path to JSON/YAML config file")
    pre.add_argument("--profile", help="Named profile in config")
    cfg_ns, _ = pre.parse_known_args()
    
    # Load and resolve config
    config = resolve_config_with_profile(cfg_ns.config, cfg_ns.profile)

    # Full parser with config-driven defaults
    ap = argparse.ArgumentParser(parents=[pre])

    # Input sources
    ap.add_argument("--scene_file", help="Single scene JSON")
    ap.add_argument("--scenes", nargs="*", help="List/globs of scene JSONs")
    ap.add_argument("--scene_list", help="TXT or JSON array of scene paths")

    # Dependencies
    ap.add_argument("--model_dir", default=config.get("model_dir", "3D-FRONT_FUTURE/3D-FUTURE-model/3D-FUTURE-model"))
    ap.add_argument("--model_info", default=config.get("model_info", "3D-FRONT_FUTURE/3D-FUTURE-model/model_info.json"))
    ap.add_argument("--out_dir", default=config.get("out_dir", "test_"))

    # Processing options
    ap.add_argument("--total_points", type=int, default=int(config.get("total_points", 500_000)))
    ap.add_argument("--ppsm", type=float, default=float(config.get("ppsm", 0.0)))
    ap.add_argument("--min_pts_per_mesh", type=int, default=int(config.get("min_pts_per_mesh", 100)))
    ap.add_argument("--max_pts_per_mesh", type=int, default=int(config.get("max_pts_per_mesh", 0)))

    # Taxonomy
    ap.add_argument("--taxonomy", default=config.get("taxonomy", "canonical_categories.yaml"))
    ap.add_argument("--alias_only_merge", action=Bool, default=bool(config.get("alias_only_merge", False)))

    # Maps control
    ap.add_argument("--maps_root", default=config.get("maps_root"))
    ap.add_argument("--build_maps_only", action=Bool, default=bool(config.get("build_maps_only", False)))
    ap.add_argument("--freeze_maps", action=Bool, default=bool(config.get("freeze_maps", False)))

    # Save toggles
    ap.add_argument("--save_glb", action=Bool, default=bool(config.get("save_glb", SAVE.glb)))
    ap.add_argument("--save_obj", action=Bool, default=bool(config.get("save_obj", SAVE.obj)))
    ap.add_argument("--save_csv", action=Bool, default=bool(config.get("save_csv", SAVE.csv)))
    ap.add_argument("--csv_with_strings", action=Bool, default=bool(config.get("csv_with_strings", False)))
    ap.add_argument("--save_npy", action=Bool, default=bool(config.get("save_npy", SAVE.npy)))
    ap.add_argument("--save_npz", action=Bool, default=bool(config.get("save_npz", SAVE.npz)))
    ap.add_argument("--save_parquet", action=Bool, default=bool(config.get("save_parquet", SAVE.parquet)))

    # Batching options
    ap.add_argument("--per_scene_subdir", action=Bool, default=bool(config.get("per_scene_subdir", False)))
    ap.add_argument("--stop_on_error", action=Bool, default=bool(config.get("stop_on_error", False)))

    args = ap.parse_args()

    # Merge CLI args with config for input sources
    def pick(cli_val, cfg_key):
        return cli_val if (cli_val is not None and cli_val != []) else config.get(cfg_key)

    args.scene_file = pick(args.scene_file, "scene_file")
    args.scenes = pick(args.scenes, "scenes")
    args.scene_list = pick(args.scene_list, "scene_list")

    if args.maps_root is None:
        args.maps_root = args.out_dir

    return args

def prescan_build_maps(scene_paths: List[Path], model_dir: Path, model_info_file: Path,
                      maps_root: Path, taxonomy_path: Path = None):
    """Pre-scan scenes to build semantic maps."""
    taxonomy = load_taxonomy(taxonomy_path) if taxonomy_path else None

    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {item["model_id"]: item for item in json.load(f)}

    seen_labels, seen_rooms = set(), set()
    seen_cats, seen_supers, seen_merged = set(), set(), set()

    for scene_path in scene_paths:
        try:
            data = json.loads(scene_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ prescan skip [{scene_path.name}]: {e}")
            continue

        furniture_map = {f['uid']: f for f in data.get('furniture', [])}
        arch_map = {m['uid']: m for m in data.get('mesh', [])}

        for room in data.get("scene", {}).get("room", []):
            room_type = str(room.get("type", "UnknownRoom"))
            seen_rooms.add(room_type)

            for child in room.get("children", []):
                ref_id = child.get("ref")
                if not ref_id:
                    continue

                if ref_id in furniture_map:
                    item = furniture_map[ref_id]
                    jid = item.get("jid")
                    raw = (model_info_map.get(jid, {}) or {}).get("category") or item.get("title") or "unknown"
                    seen_labels.add(str(raw))
                    
                    if taxonomy:
                        sem = resolve_category(raw, item, taxonomy, alias_only_merge=False)
                        if sem["category"]:
                            seen_cats.add(sem["category"])
                        seen_supers.add(sem["super"])
                        seen_merged.add(sem["merged"])
                        
                elif ref_id in arch_map:
                    atype = arch_map[ref_id].get("type", "")
                    if "Ceiling" in atype:
                        continue
                    lbl = 'floor' if 'Floor' in atype else 'wall'
                    seen_labels.add(lbl)
                    
                    if taxonomy:
                        sem = resolve_category(lbl, None, taxonomy, alias_only_merge=False)
                        if sem["category"]:
                            seen_cats.add(sem["category"])
                        seen_supers.add(sem["super"])
                        seen_merged.add(sem["merged"])

    # Build semantic maps
    values_by = {
        "label": sorted(seen_labels),
        "room": sorted(seen_rooms),
    }
    
    if taxonomy:
        if seen_cats: values_by["category"] = sorted(seen_cats)
        if seen_supers: values_by["super"] = sorted(seen_supers)
        if seen_merged: values_by["merged"] = sorted(seen_merged)

    update_semantic_maps(maps_root, values_by, freeze=False)
    print(f"✅ Prescan complete. Maps at: {maps_root / 'semantic_maps.json'}")

def main():
    """Main entry point."""
    args = parse_args_with_config()

    out_root = Path(args.out_dir)
    safe_mkdir(out_root)

    model_dir = Path(args.model_dir)
    model_info_file = Path(args.model_info)
    
    if not model_dir.is_dir() or not model_info_file.exists():
        print("❌ model_dir/model_info paths invalid.")
        return

    scene_paths = gather_scene_paths(args.scene_file, args.scenes, args.scene_list)
    if not scene_paths:
        print("❌ No scenes provided. Use --scene_file, --scenes, or --scene_list.")
        return

    maps_root = Path(args.maps_root) if args.maps_root else out_root
    safe_mkdir(maps_root)

    # Prescan mode
    if args.build_maps_only:
        prescan_build_maps(
            scene_paths, model_dir, model_info_file, maps_root,
            Path(args.taxonomy) if args.taxonomy else None
        )
        return

    # Process scenes
    total = len(scene_paths)
    progress = create_progress_tracker(total, "scenes")
    success_count = 0
    
    for i, scene_path in enumerate(scene_paths, 1):
        print(f"\n====== [{i}/{total}] {scene_path} ======")
        if not scene_path.exists():
            print("❌ Missing scene file; skipping.")
            progress(i, scene_path.name, False)
            if args.stop_on_error:
                break
            continue
            
        try:
            success = process_one_scene(scene_path, model_dir, model_info_file, out_root, args)
            if success:
                success_count += 1
            progress(i, scene_path.name, success)
        except Exception as e:
            print(f"❌ Exception while processing {scene_path.name}: {e}")
            progress(i, scene_path.name, False)
            if args.stop_on_error:
                break

    print(f"\n✅ Done. {success_count}/{total} scenes completed.")

if __name__ == "__main__":
    main()