#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

# --- imports ---
from utils.semantic_utils import Taxonomy
from utils.utils import (
    load_config_with_profile, create_progress_tracker,
    safe_mkdir, write_json
)
from utils.file_discovery import gather_paths_from_sources, infer_ids_from_path

# --- Global Taxonomy Object ---
TAXONOMY: Taxonomy = None
ARGS = None

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def export_outputs(scene_id, output_dir, textured_scene, meshes_info, point_cloud, args, taxonomy):

    # ----- Save GLB -----
    if args.save_glb and textured_scene:
        textured_scene.export(output_dir / f"{scene_id}_textured.glb", file_type="glb")

    # ----- Save OBJ -----
    if args.save_obj and meshes_info:
        with open(output_dir / f"{scene_id}.obj", 'w') as f:
            f.write("# Scene OBJ\n")
            vertex_offset = 1
            for info in meshes_info:
                mesh = info['mesh']
                for v in mesh.vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in mesh.faces:
                    f.write(f"f {face[0]+vertex_offset} {face[1]+vertex_offset} {face[2]+vertex_offset}\n")
                vertex_offset += len(mesh.vertices)

    # ----- Scene Metadata -----
    if meshes_info:
        all_vertices = np.vstack([info['mesh'].vertices for info in meshes_info])
        scene_info = {
            "bounds": {
                "min": all_vertices.min(axis=0).tolist(),
                "max": all_vertices.max(axis=0).tolist()
            },
            "size": (all_vertices.max(axis=0) - all_vertices.min(axis=0)).tolist(),
            "up_normal": [0.0, 0.0, 1.0]
        }
        write_json(scene_info, output_dir / f"{scene_id}_scene_info.json")

    # ----- Point Cloud Exports -----
    if point_cloud is not None and point_cloud.size > 0:
        xyz = np.column_stack([point_cloud["x"], point_cloud["y"], point_cloud["z"]])
        rgb = np.column_stack([point_cloud["r"], point_cloud["g"], point_cloud["b"]])

        titles = point_cloud["title"].tolist()
        labels = point_cloud["label"].tolist()
        categories = point_cloud["category"].tolist()
        supers = point_cloud["super"].tolist()
        rooms = point_cloud["room_type"].tolist()

        df_data = {
            "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            "r": rgb[:, 0], "g": rgb[:, 1], "b": rgb[:, 2],
            "title": titles,
            "label": labels,
            "category": categories,
            "super": supers,
            "room_type": rooms,
            "title_id": point_cloud["title_id"].tolist(),
            "label_id": point_cloud["label_id"].tolist(),
            "category_id": point_cloud["category_id"].tolist(),
            "super_id": point_cloud["super_id"].tolist(),
            "room_id": point_cloud["room_id"].tolist(),
        }


        if args.save_parquet:
            pd.DataFrame(df_data).to_parquet(
                output_dir / f"{scene_id}_sem_pointcloud.parquet", index=False
            )

        if args.save_csv:
            pd.DataFrame(df_data).to_csv(
                output_dir / f"{scene_id}_sem_pointcloud.csv", index=False
            )


def get_point_colors(mesh, points, face_indices):
    N = len(points)
    if N == 0 or mesh is None or mesh.is_empty:
        return np.zeros((0, 3), dtype=np.uint8)

    vis = getattr(mesh, "visual", None)

    # Try UV texture sampling
    try:
        uv = getattr(vis, "uv", None) if vis else None
        mat = getattr(vis, "material", None) if vis else None
        img = getattr(mat, "image", None) if mat else None
        if uv is not None and img is not None:
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
            return img_np[vi, ui, :].astype(np.uint8)
    except Exception:
        pass

    # Try vertex colors
    try:
        vcols = getattr(vis, "vertex_colors", None) if vis else None
        if vcols is not None and len(vcols) == len(mesh.vertices):
            tris = mesh.triangles[face_indices]
            bary = trimesh.triangles.points_to_barycentric(tris, points)
            faces = mesh.faces[face_indices]
            tri_vc = vcols[faces][:, :, :3]
            cols = (bary[:, :, None] * tri_vc.astype(np.float32)).sum(axis=1)
            return np.clip(np.round(cols), 0, 255).astype(np.uint8)
    except Exception:
        pass

    # Fallback: flat gray
    return np.full((N, 3), 128, dtype=np.uint8)

def load_mesh(model_dir: Path, jid: str):
    obj_path = model_dir / jid / "raw_model.obj"
    glb_path = model_dir / jid / "raw_model.glb"

    if obj_path.exists():
        resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
        return trimesh.load(str(obj_path), force="mesh", process=False,
                            maintain_order=True, resolver=resolver)
    elif glb_path.exists():
        return trimesh.load(str(glb_path), force="mesh", process=False,
                            maintain_order=True)
    else:
        raise FileNotFoundError(f"Model not found: {obj_path} or {glb_path}")

def create_arch_mesh(arch: Dict):
    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
    faces = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)
    return mesh

def build_transform(child: Dict):
    pos = np.array(child.get("pos", [0, 0, 0]), dtype=np.float64)
    rot = np.array(child.get("rot", [0, 0, 0, 1]), dtype=np.float64)
    scl = np.array(child.get("scale", [1, 1, 1]), dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    Rm = Rotation.from_quat(rot).as_matrix()
    Sm = np.diag(scl)
    T[:3, :3] = Rm @ Sm
    return T

def sample_points(scene_objects):
    areas = [max(0.0, obj['mesh'].area) for obj in scene_objects]
    total_area = sum(a for a in areas if a > 0.0) or 1.0

    all_data = []
    for obj, area in zip(scene_objects, areas):
        if area <= 0.0:
            continue

        if ARGS.ppsm > 0.0:
            n_pts = int(round(area * ARGS.ppsm))
        else:
            n_pts = int(round((area / total_area) * ARGS.total_points))

        n_pts = max(ARGS.min_pts_per_mesh, n_pts)
        if ARGS.max_pts_per_mesh > 0:
            n_pts = min(n_pts, ARGS.max_pts_per_mesh)

        if n_pts <= 0:
            continue

        # ---- sample points ----
        pts_local, face_indices = trimesh.sample.sample_surface(obj['mesh'], n_pts)
        colors = get_point_colors(obj['mesh'], pts_local, face_indices)
        pts_world = trimesh.transform_points(pts_local, obj['transform'])

        title = obj['label']
        room_type = obj.get('room_type', 'UnknownRoom')

        # category
        category_id = TAXONOMY.translate(title, output="id")
        category = TAXONOMY.translate(category_id, output="name") if category_id else "UnknownCategory"

        # super (explicit name + id)
        super_cat = TAXONOMY.get_sup(title, output="name")
        super_id  = TAXONOMY.get_sup(title, output="id")


        # title id
        title_id = TAXONOMY.translate(title, output="id")

        # room
        room_id = TAXONOMY.translate(room_type, output="id")


        all_data.append({
            'xyz': pts_world.astype(np.float32),
            'rgb': colors.astype(np.uint8),
            'title': [title] * n_pts,
            'label': [title] * n_pts,
            'category': [category] * n_pts,
            'super': [super_cat] * n_pts,
            'room_type': [room_type] * n_pts,
            'title_id': [title_id] * n_pts,
            'label_id': [title_id] * n_pts,
            'category_id': [category_id or 0] * n_pts,
            'super_id': [super_id] * n_pts,
            'room_id': [room_id] * n_pts,
        })

    if not all_data:
        return np.array([])

    # ---- combine into structured array ----
    xyz = np.vstack([d['xyz'] for d in all_data])
    rgb = np.vstack([d['rgb'] for d in all_data])

    def flat(key): return sum([d[key] for d in all_data], [])

    dtype = [
        ('x','f4'), ('y','f4'), ('z','f4'),
        ('r','u1'), ('g','u1'), ('b','u1'),
        ('title','U100'), ('label','U100'),
        ('category','U80'), ('super','U80'),
        ('room_type','U50'),
        ('title_id','i4'), ('label_id','i4'),
        ('category_id','i4'), ('super_id','i4'),
        ('room_id','i4'),
    ]

    N = xyz.shape[0]
    structured = np.empty(N, dtype=dtype)
    structured['x'], structured['y'], structured['z'] = xyz.T
    structured['r'], structured['g'], structured['b'] = rgb.T
    structured['title'] = flat('title')
    structured['label'] = flat('label')
    structured['category'] = flat('category')
    structured['super'] = flat('super')
    structured['room_type'] = flat('room_type')
    structured['title_id'] = flat('title_id')
    structured['label_id'] = flat('label_id')
    structured['category_id'] = flat('category_id')
    structured['super_id'] = flat('super_id')
    structured['room_id'] = flat('room_id')

    return structured


# ---------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------
def process_scene(scene_data, model_dir, model_info_map):
    failed_models = {}
    scene_objects = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}

    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "UnknownRoom")  # <-- capture room type once

        for child_index, child in enumerate(room.get("children", [])):
            ref_id = child.get("ref")
            if not ref_id:
                continue

            try:
                mesh, label, model_item = None, "unknown", None

                if ref_id in furniture_map:
                    item_info = furniture_map[ref_id]
                    jid = item_info.get('jid')
                    if not jid:
                        raise ValueError("Missing 'jid'")

                    mesh = load_mesh(model_dir, jid)
                    label = (model_info_map.get(jid, {}).get('category')
                             or item_info.get('title') or "unknown")
                    model_item = item_info

                elif ref_id in arch_map:
                    arch = arch_map[ref_id]
                    if "Ceiling" in arch.get("type", ""):
                        continue
                    mesh = create_arch_mesh(arch)
                    label = 'floor' if 'Floor' in arch.get("type", "") else 'wall'
                else:
                    failed_models[ref_id] = "Reference not found"
                    continue

                if mesh is None or mesh.is_empty:
                    raise ValueError("Empty mesh")

                transform = build_transform(child)

                # ---- Save room type alongside the object ----
                scene_objects.append({
                    "mesh": mesh,
                    "transform": transform,
                    "label": label,
                    "room_type": room_type,            # << attach parent room type
                    "node_name": f"{ref_id}_{child_index}",
                    "model_item": model_item
                })

            except Exception as e:
                failed_models[ref_id] = str(e)

    if not scene_objects:
        return None, None, None, failed_models

    # Build scene and sample points
    textured_scene = trimesh.Scene()
    meshes_world = []
    for obj in scene_objects:
        textured_scene.add_geometry(obj['mesh'], node_name=obj['node_name'],
                                    transform=obj['transform'])
        mesh_copy = obj['mesh'].copy()
        mesh_copy.apply_transform(obj['transform'])
        meshes_world.append({'mesh': mesh_copy, 'label': obj['label']})

    point_cloud = sample_points(scene_objects)
    return textured_scene, meshes_world, point_cloud, failed_models

def process_one_scene(
    scene_path: Path, model_dir: Path, model_info_file: Path, out_root: Path,
    args: argparse.Namespace
) -> bool:
    scene_id = infer_ids_from_path(scene_path)
    if isinstance(scene_id, tuple):
        scene_id = scene_id[0]
    scene_id = str(scene_id)

    out_dir = out_root / scene_id if args.per_scene_subdir else out_root
    if args.per_scene_subdir:
        safe_mkdir(out_dir)

    # Load scene JSON
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    # Load model_info.json
    with open(model_info_file, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}

    # Build config dict
    config = {
        "ppsm": args.ppsm,
        "total_points": args.total_points,
        "min_pts": args.min_pts_per_mesh,
        "max_pts": args.max_pts_per_mesh,
    }

    # Process scene: returns (trimesh.Scene, meshes_world, structured_pointcloud, failed_models)
    textured_scene, meshes_info, point_cloud, failed = process_scene(
        scene, model_dir, model_info_map)

    if point_cloud is None or point_cloud.size == 0:
        print(f"[WARN] No points sampled for scene {scene_id}")
        return False

    # Delegate all saving to export_outputs
    export_outputs(scene_id, out_dir, textured_scene, meshes_info, point_cloud, args, TAXONOMY)

    return True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global TAXONOMY
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--scene_list")
    ap.add_argument("--scene_file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_info", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--total_points", type=int, default=500000)
    ap.add_argument("--ppsm", type=float, default=0.0)
    ap.add_argument("--min_pts_per_mesh", type=int, default=100)
    ap.add_argument("--max_pts_per_mesh", type=int, default=0)
    ap.add_argument("--save_glb", action="store_true")
    ap.add_argument("--save_obj", action="store_true")
    ap.add_argument("--save_parquet", action="store_true")
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--per_scene_subdir", action="store_true", default=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenes to process (default: all)"
    )
    args = ap.parse_args()
    ARGS = args
    TAXONOMY = Taxonomy(Path(args.taxonomy))

    scene_paths = gather_paths_from_sources(args.scene_file, args.scenes, args.scene_list)
    if not scene_paths:
        print("No scenes found")
        return

    if args.limit is not None:
        scene_paths = scene_paths[:args.limit]

    out_root = Path(args.out_dir)
    safe_mkdir(out_root)

    progress = create_progress_tracker(len(scene_paths), "scenes")
    success_count = 0
    for i, scene_path in enumerate(scene_paths, 1):
        try:
            success = process_one_scene(scene_path, Path(args.model_dir),
                                        Path(args.model_info), out_root, args)
            if success:
                success_count += 1
            progress(i, scene_path.name, success)
        except Exception as e:
            progress(i, f"failed {scene_path.name}: {e}", False)

    print(f"\nSuccessfully processed {success_count}/{len(scene_paths)} scenes")



if __name__ == "__main__":
    main()
