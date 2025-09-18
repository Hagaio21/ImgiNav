#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_processing_script.py

Batch processor for 3D-FRONT scenes:
- Loads scene JSON + 3D-FUTURE meshes
- Builds textured GLB (optional) and OBJ (optional)
- Samples semantic colored point clouds (area- or density-based)
- Exports compact NPZ/Parquet/CSV with integer IDs
- Supports canonical taxonomy (category/super) and alias merging
- Canonical semantic maps (freeze/extend) across runs
- Config via JSON/YAML (+ optional profiles), with CLI overrides

Dependencies:
  numpy, pandas, trimesh, scipy (Rotation)
  optional: pyarrow or fastparquet for Parquet
  optional: pyyaml (only if you use YAML config/taxonomy)

"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

# ----------------------------
# Defaults & Save policy
# ----------------------------

@dataclass
class SavePolicy:
    glb: bool = True
    obj: bool = True
    csv: bool = True
    npy: bool = False         # legacy OFF
    npz: bool = True
    parquet: bool = True
    normalized: bool = False  # write normalized variants

SAVE = SavePolicy()

# Normalization: "none" | "center_only" | "unit_cube"
DEFAULT_NORM_MODE = "center_only"


def _norm(s: str) -> str:
    """Lowercase, strip punctuation, unify separators; for robust matching."""
    import re
    s = (s or "").lower().strip()
    s = re.sub(r"[/_]+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def load_taxonomy_categories(path: Path) -> dict:
    """
    Loads a YAML/JSON taxonomy with:
      super_categories_3d: [{id, category}]
      categories_3d:      [{id, super, category}]
      aliases:            {normalized_label: "Canonical Fine OR Super"}
    Returns lookup dicts for fast resolution.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML taxonomy requested but 'pyyaml' is not installed. "
                               "Install it or use JSON.") from e
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    supers = data.get("super_categories_3d", [])
    cats   = data.get("categories_3d", [])
    aliases= data.get("aliases", {})

    super_by_name = {c["category"]: int(c["id"]) for c in supers}
    cat_by_name   = {c["category"]: int(c["id"]) for c in cats}
    cat_to_super  = {c["category"]: c.get("super", "Other") for c in cats}

    idx_super_norm = {_norm(k): k for k in super_by_name.keys()}
    idx_cat_norm   = {_norm(k): k for k in cat_by_name.keys()}
    idx_alias_norm = {_norm(k): v for k, v in aliases.items()}

    return {
        "super_by_name": super_by_name,
        "cat_by_name": cat_by_name,
        "cat_to_super": cat_to_super,
        "idx_super_norm": idx_super_norm,
        "idx_cat_norm": idx_cat_norm,
        "idx_alias_norm": idx_alias_norm,
    }

def resolve_category(raw_label: str,
                     model_info_item: dict | None,
                     taxo: dict,
                     alias_only_merge: bool = False) -> dict:
    """
    Map a raw label -> {'category': <fine or ''>, 'super': <always>, 'merged': <train label>}
    merged = super (default) or fine/alias-collapsed if alias_only_merge=True.
    """
    # prefer model_info category if present
    cand = (model_info_item or {}).get("category") or raw_label or "unknown"
    n = _norm(cand)

    # alias collapse (target can be fine or super canonical name)
    alias_target = taxo["idx_alias_norm"].get(n)
    if alias_target:
        cand = alias_target
        n = _norm(cand)

    # try fine category
    cat_name = taxo["idx_cat_norm"].get(n, "")
    if cat_name:
        super_name = taxo["cat_to_super"].get(cat_name, "Other")
    else:
        # maybe the (aliased) label already is a super
        super_name = taxo["idx_super_norm"].get(n, "Other")

    # choose merged
    if alias_only_merge:
        merged = cat_name or super_name   # keep fine if known; else super
    else:
        merged = super_name               # always super

    return {"category": cat_name, "super": super_name, "merged": merged}


def _load_or_create_maps_multi(root: Path,
                               values_by: dict[str, list[str]],
                               freeze: bool = False):
    """
    Maintain multiple name->id maps in one JSON: root/semantic_maps.json
    values_by: {"label":[...], "room":[...], "category":[...], "super":[...], "merged":[...]}
    If freeze=True, raise on any unseen value.
    """
    root.mkdir(parents=True, exist_ok=True)
    maps_path = root / "semantic_maps.json"
    maps = json.loads(maps_path.read_text(encoding="utf-8")) if maps_path.exists() else {}

    changed = False
    out = {}
    for key, vals in values_by.items():
        cur = {str(k): int(v) for k, v in maps.get(f"{key}2id", {}).items()}
        uniq = sorted({v for v in vals if v}, key=lambda s: s.lower())
        if freeze:
            unknown = [v for v in uniq if v not in cur]
            if unknown:
                raise RuntimeError(f"freeze_maps ON; unseen in '{key}': {unknown[:20]}")
        else:
            nxt = (max(cur.values()) + 1) if cur else 1
            for v in uniq:
                if v not in cur:
                    cur[v] = nxt; nxt += 1; changed = True
        out[key] = cur
        maps[f"{key}2id"] = cur

    if changed or not maps_path.exists():
        maps_path.write_text(json.dumps(maps, indent=2), encoding="utf-8")
    return out, maps_path


def _read_scene_list_file(path: Path) -> List[Path]:
    paths: List[Path] = []
    txt = path.read_text(encoding="utf-8").strip()
    # Try JSON array first
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            for p in arr:
                paths.append(Path(p))
            return paths
    except Exception:
        pass
    # Fallback: one path per line (ignore blanks/comments)
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    return paths

def _gather_scene_paths(scene_file: str | None,
                        scenes: List[str] | None,
                        scene_list: str | None) -> List[Path]:
    all_paths: List[Path] = []
    if scene_file:
        all_paths.append(Path(scene_file))
    if scenes:
        for pat in scenes:
            expanded = [Path(p) for p in glob.glob(pat)]
            if expanded:
                all_paths.extend(expanded)
            else:
                all_paths.append(Path(pat))
    if scene_list:
        all_paths.extend(_read_scene_list_file(Path(scene_list)))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in all_paths:
        q = p.resolve()
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq

def export_to_obj_manually(meshes_with_info: List[Dict], file_path: Path):
    """Non-textured OBJ (multi-object)."""
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
    """
    UV-aware color sampling:
      - If mesh has UVs + material image: barycentric UV -> sample image
      - Else if mesh has vertex colors: barycentric interpolate vcolors
      - Else fallback to mid-gray
    Returns uint8 RGB (N,3).
    """
    N = len(points)
    if N == 0 or mesh is None or mesh.is_empty:
        return np.zeros((0, 3), dtype=np.uint8)

    vis = getattr(mesh, "visual", None)

    # ---- Path 1: UV + single material image
    try:
        uv = getattr(vis, "uv", None) if vis is not None else None
        mat = getattr(vis, "material", None) if vis is not None else None
        img = getattr(mat, "image", None) if mat is not None else None
        if (uv is not None) and (img is not None):
            tris = mesh.triangles[face_indices]                      # (N,3,3)
            bary = trimesh.triangles.points_to_barycentric(tris, points)  # (N,3)

            faces = mesh.faces[face_indices]                         # (N,3)
            tri_uv = uv[faces]                                       # (N,3,2)

            # interpolate per-point UV in [0,1]
            uv_pts = (bary[:, :, None] * tri_uv).sum(axis=1)         # (N,2)

            # sample image (flip V because image origin is top-left)
            img_np = np.asarray(img.convert("RGB"))
            H, W = img_np.shape[:2]
            u = np.clip(uv_pts[:, 0], 0.0, 1.0) * (W - 1)
            v = (1.0 - np.clip(uv_pts[:, 1], 0.0, 1.0)) * (H - 1)
            ui = np.clip(np.round(u).astype(np.int64), 0, W - 1)
            vi = np.clip(np.round(v).astype(np.int64), 0, H - 1)
            cols = img_np[vi, ui, :].astype(np.uint8)
            return cols
    except Exception:
        # fall through to other paths
        pass

    # ---- Path 2: vertex colors (barycentric)
    try:
        vcols = getattr(vis, "vertex_colors", None) if vis is not None else None
        if vcols is not None and len(vcols) == len(mesh.vertices):
            tris = mesh.triangles[face_indices]                      # (N,3,3)
            bary = trimesh.triangles.points_to_barycentric(tris, points)  # (N,3)
            faces = mesh.faces[face_indices]                         # (N,3)
            tri_vc = vcols[faces][:, :, :3]                          # (N,3,3) RGB from RGBA
            cols = (bary[:, :, None] * tri_vc.astype(np.float32)).sum(axis=1)
            return np.clip(np.round(cols), 0, 255).astype(np.uint8)
    except Exception:
        pass

    # ---- Fallback: mid gray (architectural/walls/floors often have no textures)
    return np.full((N, 3), 128, dtype=np.uint8)


def save_scene_info(meshes_info: List[Dict], output_dir: Path, scene_id: str) -> Dict | None:
    """Bounds/size/up_normal (simple Z-up heuristic)."""
    if not meshes_info:
        print("[WARNING] Cannot generate scene_info.json because no meshes were processed.")
        return None
    all_vertices_list = [info['mesh'].vertices for info in meshes_info]
    if not all_vertices_list:
        print("[WARNING]  No vertices found to calculate scene info.")
        return None
    all_vertices = np.vstack(all_vertices_list)
    min_bounds = all_vertices.min(axis=0)
    max_bounds = all_vertices.max(axis=0)
    size = max_bounds - min_bounds

    floor_vertices_list = [info['mesh'].vertices for info in meshes_info if info['label'] == 'floor']
    furniture_vertices_list = [info['mesh'].vertices for info in meshes_info if info['label'] != 'floor']

    up_normal = [0.0, 0.0, 1.0]  # default Z-up
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
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scene_info, f, indent=4)
    print(f"[INFO] Saved scene parameters : {json_path}")
    return scene_info

def get_normalization_transform(scene_info: Dict, mode: str = DEFAULT_NORM_MODE) -> np.ndarray:
    """Build 4x4 transform per chosen normalization mode."""
    from trimesh.transformations import translation_matrix, scale_matrix
    if not scene_info or "bounds" not in scene_info:
        return np.eye(4, dtype=np.float32)
    bmin = np.array(scene_info["bounds"]["min"], dtype=np.float32)
    bmax = np.array(scene_info["bounds"]["max"], dtype=np.float32)
    center = (bmin + bmax) * 0.5
    size = (bmax - bmin)
    longest = float(max(size.max(), 1e-9))
    T = translation_matrix((-center).tolist())
    if mode == "none":
        return np.eye(4, dtype=np.float32)
    if mode == "center_only":
        return T.astype(np.float32)
    if mode == "unit_cube":
        s = 1.0 / longest
        S = scale_matrix(s)
        return (S @ T).astype(np.float32)
    return T.astype(np.float32)

def normalize_point_cloud(point_cloud: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply transform to structured point cloud (x,y,z only)."""
    if point_cloud.size == 0:
        return point_cloud
    points_xyz = np.vstack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).T
    normalized_xyz = trimesh.transform_points(points_xyz, transform_matrix)
    normalized_point_cloud = point_cloud.copy()
    normalized_point_cloud['x'], normalized_point_cloud['y'], normalized_point_cloud['z'] = normalized_xyz.T
    return normalized_point_cloud

def process_scene_for_all_outputs(
    scene_data: dict,
    model_dir: Path,
    model_info_map: dict,
    total_points: int = 500_000,
    points_per_sq_meter: float = 0.0,   # >0 → density-based sampling
    min_pts_per_mesh: int = 100,
    max_pts_per_mesh: int = 0,          # 0 = no cap
    taxonomy_path: str | None = None,
    alias_only_merge: bool = False,
) -> Tuple[trimesh.Scene, List[Dict], np.ndarray, Dict]:
    """
    Builds:
      - textured_scene (trimesh.Scene)
      - meshes_info_world: [{'mesh': Trimesh world-space, 'label': str}]
      - structured semantic point cloud (x,y,z,r,g,b,label,room_type,category,super,merged)
      - failed_models: {ref_id: reason}
    """
    failed_models: Dict[str, str] = {}
    taxo = load_taxonomy_categories(Path(taxonomy_path)) if taxonomy_path else None

    scene_objects: List[Dict] = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map      = {m['uid']: m for m in scene_data.get('mesh',      [])}

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
                    # label from model_info fallback
                    label = (model_info_map.get(jid, {}) or {}).get('category') \
                            or item_info.get('title') or "unknown"
                    # Prefer OBJ (with textures via resolver); fallback to GLB if needed
                    obj_path = model_dir / jid / "raw_model.obj"
                    glb_path = model_dir / jid / "raw_model.glb"

                    mesh = None
                    if obj_path.exists():
                        # keep visuals/uvs/materials intact
                        resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
                        mesh = trimesh.load(str(obj_path), force='mesh', process=False, maintain_order=True, resolver=resolver)
                    elif glb_path.exists():
                        mesh = trimesh.load(str(glb_path), force='mesh', process=False, maintain_order=True)
                    else:
                        raise FileNotFoundError(f"Model not found: {obj_path} or {glb_path}")


                elif ref_id in arch_map:
                    arch = arch_map[ref_id]
                    atype = arch.get("type", "")
                    if "Ceiling" in atype:
                        continue  # skip ceilings
                    label = 'floor' if 'Floor' in atype else 'wall'
                    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
                    faces    = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                    mesh.visual.vertex_colors = np.array([200,200,200,255], dtype=np.uint8)

                else:
                    failed_models[ref_id] = "Reference not found in furniture/mesh maps"
                    continue

                if (mesh is None) or mesh.is_empty or mesh.faces.shape[0] == 0:
                    raise ValueError("Mesh is empty or has no faces")

                # Node transform (pos + quat x,y,z,w + scale)
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

    # Textured scene
    textured_scene = trimesh.Scene()
    for obj in scene_objects:
        textured_scene.add_geometry(obj['mesh'], node_name=obj['node_name'], transform=obj['transform'])

    # World-space copies for OBJ / scene info
    meshes_info_world: List[Dict] = []
    for obj in scene_objects:
        mc = obj['mesh'].copy()
        mc.apply_transform(obj['transform'])
        mc.fix_normals()
        meshes_info_world.append({'mesh': mc, 'label': obj['label']})

    # Sampling plan
    areas = [float(max(0.0, o['mesh'].area)) for o in scene_objects]
    total_area = sum(a for a in areas if a > 0.0) or 1.0

    all_points, all_colors = [], []
    all_labels, all_room_types = [], []
    all_categories, all_supers, all_merged = [], [], []
    total_sampled = 0

    for obj, a in zip(scene_objects, areas):
        mesh_local: trimesh.Trimesh = obj['mesh']
        if a <= 0.0 or mesh_local.is_empty or mesh_local.faces.shape[0] == 0:
            continue

        if points_per_sq_meter > 0.0:
            n_pts = int(round(a * points_per_sq_meter))
        else:
            n_pts = int(round((a / total_area) * total_points))
        n_pts = max(min_pts_per_mesh, n_pts)
        if max_pts_per_mesh > 0:
            n_pts = min(n_pts, max_pts_per_mesh)
        if n_pts <= 0:
            continue

        pts_local, face_indices = trimesh.sample.sample_surface(mesh_local, n_pts)
        cols = get_point_colors(mesh_local, pts_local, face_indices)  # uint8 (N,3)
        pts_world = trimesh.transform_points(pts_local, obj['transform'])

        # resolve categories once per object
        if taxo:
            sem = resolve_category(obj['label'], obj.get("model_item"), taxo, alias_only_merge=alias_only_merge)
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

    # build structured array
    dtype = [
        ('x','f4'),('y','f4'),('z','f4'),
        ('r','u1'),('g','u1'),('b','u1'),
        ('label','U100'),('room_type','U50'),
        ('category','U80'),('super','U80'),('merged','U80')
    ]
    point_cloud_structured = np.array([], dtype=dtype)

    if all_points:
        P = np.vstack(all_points).astype(np.float32)
        C = np.vstack(all_colors).astype(np.uint8)
        N = P.shape[0]

        point_cloud_structured = np.empty(N, dtype=dtype)
        point_cloud_structured['x'], point_cloud_structured['y'], point_cloud_structured['z'] = P.T
        point_cloud_structured['r'], point_cloud_structured['g'], point_cloud_structured['b'] = C.T
        point_cloud_structured['label']     = np.array(all_labels, dtype='U100')
        point_cloud_structured['room_type'] = np.array(all_room_types, dtype='U50')
        point_cloud_structured['category']  = np.array(all_categories, dtype='U80')
        point_cloud_structured['super']     = np.array(all_supers, dtype='U80')
        point_cloud_structured['merged']    = np.array(all_merged, dtype='U80')

    return textured_scene, meshes_info_world, point_cloud_structured, failed_models

def _export_pointcloud_compact(
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
    """
    Writes compact arrays:
      xyz: float32 (N,3), rgb: uint8 (N,3),
      ids: label_id, room_id, cat_id, super_id, merged_id (uint16)
    semantic_maps.json is kept at maps_root (defaults to output_dir).
    """
    if point_cloud.size == 0:
        print("⚠️  Empty point cloud; nothing to export.")
        return

    maps_root = maps_root or output_dir

    # geometry/color
    xyz = np.vstack([point_cloud["x"], point_cloud["y"], point_cloud["z"]]).T.astype(np.float32)
    rgb = np.vstack([point_cloud["r"], point_cloud["g"], point_cloud["b"]]).T.astype(np.uint8)

    # strings
    labels = point_cloud["label"].astype(str).tolist()
    rooms  = point_cloud["room_type"].astype(str).tolist()
    cats   = point_cloud["category"].astype(str).tolist() if "category" in point_cloud.dtype.names else []
    supers = point_cloud["super"].astype(str).tolist()    if "super"    in point_cloud.dtype.names else []
    merged = point_cloud["merged"].astype(str).tolist()   if "merged"   in point_cloud.dtype.names else []

    values_by = {"label": labels, "room": rooms}
    if cats:   values_by["category"] = cats
    if supers: values_by["super"]    = supers
    if merged: values_by["merged"]   = merged

    maps, _ = _load_or_create_maps_multi(maps_root, values_by, freeze=freeze_maps)

    # ids
    label_id = np.array([maps["label"].get(s, 0) for s in labels], dtype=np.uint16)
    room_id  = np.array([maps["room"].get(s, 0)  for s in rooms ], dtype=np.uint16)
    out = {"xyz": xyz, "rgb": rgb, "label_id": label_id, "room_id": room_id}

    if cats:   out["cat_id"]    = np.array([maps["category"].get(s, 0) for s in cats],   dtype=np.uint16)
    if supers: out["super_id"]  = np.array([maps["super"].get(s, 0)    for s in supers], dtype=np.uint16)
    if merged: out["merged_id"] = np.array([maps["merged"].get(s, 0)   for s in merged], dtype=np.uint16)

    # save
    if save_npz:
        np.savez_compressed(output_dir / f"{scene_id}_sem_pointcloud.npz", **out)
        print(f"💾 NPZ  → {output_dir / f'{scene_id}_sem_pointcloud.npz'}")

    if save_parquet:
        try:
            df_cols = {
                "x": xyz[:,0], "y": xyz[:,1], "z": xyz[:,2],
                "r": rgb[:,0], "g": rgb[:,1], "b": rgb[:,2],
                "label_id": out["label_id"], "room_id": out["room_id"]
            }
            for k in ("cat_id","super_id","merged_id"):
                if k in out: df_cols[k] = out[k]
            pd.DataFrame(df_cols).to_parquet(
                output_dir / f"{scene_id}_sem_pointcloud.parquet",
                compression="snappy", index=False
            )
            print(f"💾 Parquet → {output_dir / f'{scene_id}_sem_pointcloud.parquet'}")
        except Exception as e:
            print(f"ℹ️ Parquet skipped: {e}")

    if save_csv:
        data = {
            "x": xyz[:,0], "y": xyz[:,1], "z": xyz[:,2],
            "r": rgb[:,0], "g": rgb[:,1], "b": rgb[:,2],
            "label_id": out["label_id"], "room_id": out["room_id"],
        }
        # include IDs for categories if present
        for k in ("cat_id","super_id","merged_id"):
            if k in out: data[k] = out[k]
        if csv_include_strings:
            data["label"]     = labels
            data["room_type"] = rooms
            if cats:   data["category"] = cats
            if supers: data["super"]    = supers
            if merged: data["merged"]   = merged
        pd.DataFrame(data).to_csv(output_dir / f"{scene_id}_sem_pointcloud.csv", index=False)
        print(f"💾 CSV  → {output_dir / f'{scene_id}_sem_pointcloud.csv'}")

def _prescan_build_maps(scene_paths: List[Path],
                        model_dir: Path,
                        model_info_file: Path,
                        maps_root: Path,
                        taxonomy_path: Path | None = None):
    """
    Fast pass: read scene JSONs, collect labels/rooms (+ resolved categories if taxonomy provided),
    then write semantic_maps.json once. Use before exporting with --freeze_maps.
    """
    taxo = load_taxonomy_categories(taxonomy_path) if taxonomy_path else None

    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {item["model_id"]: item for item in json.load(f)}

    seen_labels: set[str] = set()
    seen_rooms:  set[str] = set()
    seen_cats:   set[str] = set()
    seen_supers: set[str] = set()
    seen_merged: set[str] = set()

    for sp in scene_paths:
        try:
            data = json.loads(Path(sp).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ prescan skip [{sp.name}]: {e}")
            continue

        furniture_map = {f['uid']: f for f in data.get('furniture', [])}
        arch_map      = {m['uid']: m for m in data.get('mesh',      [])}

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
                    if taxo:
                        sem = resolve_category(raw, item, taxo, alias_only_merge=False)
                        if sem["category"]: seen_cats.add(sem["category"])
                        seen_supers.add(sem["super"])
                        seen_merged.add(sem["merged"])
                elif ref_id in arch_map:
                    atype = arch_map[ref_id].get("type", "")
                    if "Ceiling" in atype:  # skip
                        continue
                    lbl = 'floor' if 'Floor' in atype else 'wall'
                    seen_labels.add(lbl)
                    if taxo:
                        sem = resolve_category(lbl, None, taxo, alias_only_merge=False)
                        if sem["category"]: seen_cats.add(sem["category"])
                        seen_supers.add(sem["super"])
                        seen_merged.add(sem["merged"])

    # write canonical map
    values_by = {
        "label": sorted(seen_labels),
        "room":  sorted(seen_rooms),
    }
    if taxo:
        if seen_cats:   values_by["category"] = sorted(seen_cats)
        if seen_supers: values_by["super"]    = sorted(seen_supers)
        if seen_merged: values_by["merged"]   = sorted(seen_merged)

    _load_or_create_maps_multi(maps_root, values_by, freeze=False)
    print(f"✅ Prescan complete. Maps at: {maps_root / 'semantic_maps.json'}")

def _process_one_scene(scene_path: Path,
                       model_dir: Path,
                       model_info_file: Path,
                       out_root: Path,
                       args) -> bool:
    """Returns True on success, False on failure."""
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
    out_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"⚠️  Skipped {len(failed_models)} models:")
        for mid, why in failed_models.items():
            print(f"    - {mid}: {why}")

    if not meshes_info:
        print("❌ No meshes produced; skipping.")
        return False

    # GLB / OBJ
    if args.save_glb:
        glb_path = out_dir / f"{scene_id}_textured.glb"
        textured_scene.export(file_obj=glb_path, file_type="glb")
        print(f"💾 GLB  → {glb_path}")

    if args.save_obj:
        obj_path = out_dir / f"{scene_id}.obj"
        export_to_obj_manually(meshes_info, obj_path)
        print(f"💾 OBJ  → {obj_path}")

    # Scene info
    scene_info = save_scene_info(meshes_info, out_dir, scene_id)

    # Point cloud (compact)
    if point_cloud.size:
        if args.save_npy:
            npy_path = out_dir / f"{scene_id}_sem_pointcloud.npy"
            np.save(npy_path, point_cloud)
            print(f"💾 NPY (legacy) → {npy_path}")

        _export_pointcloud_compact(
            out_dir, scene_id, point_cloud,
            save_csv=args.save_csv,
            save_npz=args.save_npz,
            save_parquet=args.save_parquet,
            csv_include_strings=args.csv_with_strings,
            maps_root=Path(args.maps_root) if args.maps_root else out_root,
            freeze_maps=args.freeze_maps,
        )

    # Normalized variants (optional)
    if args.save_normalized and scene_info:
        print("… generating normalized variants")
        T_norm = get_normalization_transform(scene_info, mode=args.norm_mode)

        if args.save_glb:
            normalized_textured_scene = textured_scene.copy()
            normalized_textured_scene.apply_transform(T_norm)
            n_glb = out_dir / f"{scene_id}_textured_normalized.glb"
            normalized_textured_scene.export(file_obj=n_glb, file_type="glb")
            print(f"💾 GLB (norm) → {n_glb}")

        if args.save_obj:
            n_meshes = []
            for info in meshes_info:
                m = info["mesh"].copy()
                m.apply_transform(T_norm)
                n_meshes.append({"mesh": m, "label": info["label"]})
            n_obj = out_dir / f"{scene_id}_normalized.obj"
            export_to_obj_manually(n_meshes, n_obj)
            print(f"💾 OBJ (norm) → {n_obj}")

        normalized_point_cloud = normalize_point_cloud(point_cloud, T_norm)
        if args.save_npy:
            n_npy = out_dir / f"{scene_id}_sem_pointcloud_normalized.npy"
            np.save(n_npy, normalized_point_cloud)
            print(f"💾 NPY (norm, legacy) → {n_npy}")

        _export_pointcloud_compact(
            out_dir, f"{scene_id}_normalized", normalized_point_cloud,
            save_csv=args.save_csv,
            save_npz=args.save_npz,
            save_parquet=args.save_parquet,
            csv_include_strings=args.csv_with_strings,
            maps_root=Path(args.maps_root) if args.maps_root else out_root,
            freeze_maps=args.freeze_maps,
        )

    return True

def _load_config_dict(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if p.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML config requested but 'pyyaml' is not installed. "
                               "Install it or use JSON.") from e
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be an object/dict.")
    return data

def _parse_args_with_config() -> argparse.Namespace:
    """
    CLI > selected profile > base config > code defaults.
    Booleans use BooleanOptionalAction so you can turn them on/off from CLI
    even if the config set a different default (e.g., --no-save_obj).
    """
    Bool = argparse.BooleanOptionalAction

    # 1) pre-parse --config / --profile
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="Path to JSON/YAML config.")
    pre.add_argument("--profile", default=None, help="Named profile in config (under 'profiles').")
    cfg_ns, _ = pre.parse_known_args()
    cfg = _load_config_dict(cfg_ns.config) if cfg_ns.config else {}

    # Optional profile merge
    prof_name = getattr(cfg_ns, "profile", None) or cfg.get("profile")
    if isinstance(cfg.get("profiles"), dict) and prof_name:
        prof = cfg["profiles"].get(prof_name)
        if not isinstance(prof, dict):
            raise ValueError(f"Profile '{prof_name}' not found in config.")
        base = dict(cfg)
        base.pop("profiles", None); base.pop("profile", None)
        base.update(prof)   # profile overrides base
        cfg = base

    # 2) full parser with config-driven defaults
    ap = argparse.ArgumentParser(parents=[pre])

    # inputs
    ap.add_argument("--scene_file", default=None, help="Single scene JSON")
    ap.add_argument("--scenes", nargs="*", default=None, help="List/globs of scene JSONs")
    ap.add_argument("--scene_list", default=None,
                    help="TXT (one path per line) or JSON array of scene paths")

    # dependencies
    ap.add_argument("--model_dir",  default=cfg.get("model_dir", "3D-FRONT_FUTURE/3D-FUTURE-model/3D-FUTURE-model"))
    ap.add_argument("--model_info", default=cfg.get("model_info", "3D-FRONT_FUTURE/3D-FUTURE-model/model_info.json"))
    ap.add_argument("--out_dir",    default=cfg.get("out_dir", "test_"))

    # processing knobs
    ap.add_argument("--total_points", type=int, default=int(cfg.get("total_points", 500_000)))
    ap.add_argument("--ppsm", type=float, default=float(cfg.get("ppsm", 0.0)))
    ap.add_argument("--min_pts_per_mesh", type=int, default=int(cfg.get("min_pts_per_mesh", 100)))
    ap.add_argument("--max_pts_per_mesh", type=int, default=int(cfg.get("max_pts_per_mesh", 0)))
    ap.add_argument("--norm_mode", choices=["none","center_only","unit_cube"],
                    default=cfg.get("norm_mode", DEFAULT_NORM_MODE))

    # taxonomy
    ap.add_argument("--taxonomy", default=cfg.get("taxonomy", "canonical_categories.yaml"),
                    help="YAML/JSON taxonomy with super_categories_3d, categories_3d, aliases.")
    ap.add_argument("--alias_only_merge", action=Bool, default=bool(cfg.get("alias_only_merge", False)),
                    help="If on, merged=category when known; else merged=super.")

    # maps control
    ap.add_argument("--maps_root", default=cfg.get("maps_root"),
                    help="Folder to store/read semantic_maps.json (defaults to out_dir).")
    ap.add_argument("--build_maps_only", action=Bool, default=bool(cfg.get("build_maps_only", False)),
                    help="Pre-scan scenes, build maps, and exit (no exports).")
    ap.add_argument("--freeze_maps", action=Bool, default=bool(cfg.get("freeze_maps", False)),
                    help="Error if unseen labels/categories appear.")

    # save toggles
    ap.add_argument("--save_glb",       action=Bool, default=bool(cfg.get("save_glb",       SAVE.glb)))
    ap.add_argument("--save_obj",       action=Bool, default=bool(cfg.get("save_obj",       SAVE.obj)))
    ap.add_argument("--save_csv",       action=Bool, default=bool(cfg.get("save_csv",       SAVE.csv)))
    ap.add_argument("--csv_with_strings", action=Bool, default=bool(cfg.get("csv_with_strings", False)))
    ap.add_argument("--save_npy",       action=Bool, default=bool(cfg.get("save_npy",       SAVE.npy)))
    ap.add_argument("--save_npz",       action=Bool, default=bool(cfg.get("save_npz",       SAVE.npz)))
    # ap.add_argument("--save_parquet",   action=Bool, default=bool(cfg.get("save_parquet",   SAVE.parquet)))
    ap.add_argument("--save_parquet",   action=Bool, default=True)
    ap.add_argument("--save_normalized",action=Bool, default=bool(cfg.get("save_normalized",SAVE.normalized)))

    # batching niceties
    ap.add_argument("--per_scene_subdir", action=Bool, default=bool(cfg.get("per_scene_subdir", False)))
    ap.add_argument("--stop_on_error", action=Bool, default=bool(cfg.get("stop_on_error", False)))

    args = ap.parse_args()

    # 3) merge inputs: CLI wins; else config
    def pick(cli_val, cfg_key):
        return cli_val if (cli_val is not None and cli_val != []) else cfg.get(cfg_key)

    args.scene_file  = pick(args.scene_file,  "scene_file")
    args.scenes      = pick(args.scenes,      "scenes")
    args.scene_list  = pick(args.scene_list,  "scene_list")

    if args.maps_root is None:
        args.maps_root = args.out_dir

    return args

def main():
    args = _parse_args_with_config()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    model_info_file = Path(args.model_info)
    if not model_dir.is_dir() or not model_info_file.exists():
        print("❌ model_dir/model_info paths invalid.")
        return

    scene_paths = _gather_scene_paths(args.scene_file, args.scenes, args.scene_list)
    if not scene_paths:
        print("❌ No scenes provided. Use --scene_file, --scenes, or --scene_list.")
        return

    maps_root = Path(args.maps_root) if args.maps_root else out_root
    maps_root.mkdir(parents=True, exist_ok=True)

    # Prescan mode (two-pass workflow)
    if args.build_maps_only:
        _prescan_build_maps(
            scene_paths, model_dir, model_info_file, maps_root,
            Path(args.taxonomy) if args.taxonomy else None
        )
        return

    total = len(scene_paths)
    ok = 0
    for i, sp in enumerate(scene_paths, 1):
        print(f"\n====== [{i}/{total}] {sp} ======")
        if not sp.exists():
            print("❌ Missing scene file; skipping.")
            if args.stop_on_error:
                break
            continue
        try:
            success = _process_one_scene(sp, model_dir, model_info_file, out_root, args)
            ok += int(bool(success))
        except Exception as e:
            print(f"❌ Exception while processing {sp.name}: {e}")
            if args.stop_on_error:
                break

    print(f"\n✅ Done. {ok}/{total} scenes completed.")

if __name__ == "__main__":
    main()
