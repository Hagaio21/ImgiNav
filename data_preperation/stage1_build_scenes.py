#!/usr/bin/env python3
"""
stage1_build_scenes.py (simple refactor)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

from utils.utils import (
    gather_paths_from_sources, infer_ids_from_path, load_taxonomy_resolver,
    SemanticMaps, load_config_with_profile, create_progress_tracker, 
    safe_mkdir, write_json
)

def get_point_colors(mesh, points, face_indices):
    """Sample colors from mesh."""
    N = len(points)
    if N == 0 or mesh is None or mesh.is_empty:
        return np.zeros((0, 3), dtype=np.uint8)

    vis = getattr(mesh, "visual", None)

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

    return np.full((N, 3), 128, dtype=np.uint8)

def load_mesh(model_dir, jid):
    """Load mesh from OBJ or GLB."""
    obj_path = model_dir / jid / "raw_model.obj"
    glb_path = model_dir / jid / "raw_model.glb"

    if obj_path.exists():
        resolver = trimesh.visual.resolvers.FilePathResolver(obj_path.parent)
        return trimesh.load(str(obj_path), force='mesh', process=False, 
                          maintain_order=True, resolver=resolver)
    elif glb_path.exists():
        return trimesh.load(str(glb_path), force='mesh', process=False, 
                          maintain_order=True)
    else:
        raise FileNotFoundError(f"Model not found: {obj_path} or {glb_path}")

def create_arch_mesh(arch):
    """Create mesh from architectural data."""
    vertices = np.array(arch["xyz"], dtype=np.float64).reshape(-1, 3)
    faces = np.array(arch["faces"], dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)
    return mesh

def build_transform(child):
    """Build transform matrix from child data."""
    pos = np.array(child.get("pos", [0, 0, 0]), dtype=np.float64)
    rot = np.array(child.get("rot", [0, 0, 0, 1]), dtype=np.float64)
    scl = np.array(child.get("scale", [1, 1, 1]), dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    Rm = Rotation.from_quat(rot).as_matrix()
    Sm = np.diag(scl)
    T[:3, :3] = Rm @ Sm
    return T

def process_scene(scene_data, model_dir, model_info_map, config, taxonomy_resolver):
    """Process scene into objects and point cloud."""
    failed_models = {}
    scene_objects = []
    furniture_map = {f['uid']: f for f in scene_data.get('furniture', [])}
    arch_map = {m['uid']: m for m in scene_data.get('mesh', [])}

    for room in scene_data.get("scene", {}).get("room", []):
        room_type = room.get("type", "UnknownRoom")
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
                scene_objects.append({
                    "mesh": mesh, "transform": transform, "label": label,
                    "room_type": room_type, "node_name": f"{ref_id}_{child_index}",
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

    point_cloud = sample_points(scene_objects, config, taxonomy_resolver)
    return textured_scene, meshes_world, point_cloud, failed_models

def sample_points(scene_objects, config, taxonomy_resolver):
    """Sample points from scene objects."""
    areas = [max(0.0, obj['mesh'].area) for obj in scene_objects]
    total_area = sum(a for a in areas if a > 0.0) or 1.0

    all_data = []
    for obj, area in zip(scene_objects, areas):
        if area <= 0.0:
            continue

        if config['ppsm'] > 0.0:
            n_pts = int(round(area * config['ppsm']))
        else:
            n_pts = int(round((area / total_area) * config['total_points']))
        
        n_pts = max(config['min_pts'], n_pts)
        if config['max_pts'] > 0:
            n_pts = min(n_pts, config['max_pts'])
        
        if n_pts <= 0:
            continue

        pts_local, face_indices = trimesh.sample.sample_surface(obj['mesh'], n_pts)
        colors = get_point_colors(obj['mesh'], pts_local, face_indices)
        pts_world = trimesh.transform_points(pts_local, obj['transform'])

        if taxonomy_resolver:
            sem = taxonomy_resolver(obj['label'], obj.get("model_item"))
        else:
            sem = {"category": "", "super": obj['label'], "merged": obj['label']}

        all_data.append({
            'xyz': pts_world.astype(np.float32),
            'rgb': colors.astype(np.uint8),
            'labels': [obj['label']] * n_pts,
            'room_types': [obj['room_type']] * n_pts,
            'categories': [sem["category"]] * n_pts,
            'supers': [sem["super"]] * n_pts,
            'merged': [sem["merged"]] * n_pts
        })

    if not all_data:
        return np.array([])

    # Combine all data
    xyz = np.vstack([d['xyz'] for d in all_data])
    rgb = np.vstack([d['rgb'] for d in all_data])
    labels = sum([d['labels'] for d in all_data], [])
    room_types = sum([d['room_types'] for d in all_data], [])
    categories = sum([d['categories'] for d in all_data], [])
    supers = sum([d['supers'] for d in all_data], [])
    merged = sum([d['merged'] for d in all_data], [])

    dtype = [('x','f4'),('y','f4'),('z','f4'),('r','u1'),('g','u1'),('b','u1'),
             ('label','U100'),('room_type','U50'),('category','U80'),
             ('super','U80'),('merged','U80')]
    
    N = xyz.shape[0]
    structured = np.empty(N, dtype=dtype)
    structured['x'], structured['y'], structured['z'] = xyz.T
    structured['r'], structured['g'], structured['b'] = rgb.T
    structured['label'] = labels
    structured['room_type'] = room_types
    structured['category'] = categories
    structured['super'] = supers
    structured['merged'] = merged

    return structured

def export_outputs(scene_id, output_dir, textured_scene, meshes_info, point_cloud, args):
    """Export all requested formats."""
    if args.save_glb and textured_scene:
        textured_scene.export(output_dir / f"{scene_id}_textured.glb", file_type="glb")

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

    if meshes_info:
        all_vertices = np.vstack([info['mesh'].vertices for info in meshes_info])
        scene_info = {
            "bounds": {"min": all_vertices.min(axis=0).tolist(), 
                      "max": all_vertices.max(axis=0).tolist()},
            "size": (all_vertices.max(axis=0) - all_vertices.min(axis=0)).tolist(),
            "up_normal": [0.0, 0.0, 1.0]
        }
        write_json(scene_info, output_dir / f"{scene_id}_scene_info.json")

    if point_cloud.size > 0:
        xyz = np.column_stack([point_cloud["x"], point_cloud["y"], point_cloud["z"]])
        rgb = np.column_stack([point_cloud["r"], point_cloud["g"], point_cloud["b"]])
        
        labels = point_cloud["label"].tolist()
        rooms = point_cloud["room_type"].tolist()
        cats = point_cloud["category"].tolist()
        supers = point_cloud["super"].tolist()
        merged = point_cloud["merged"].tolist()

        values_by = {"label": labels, "room": rooms, "category": cats, 
                    "super": supers, "merged": merged}

        semantic_maps = SemanticMaps(output_dir)
        semantic_maps.update_with_values(values_by)
        maps_data = semantic_maps.data

        if args.save_parquet:
            df_data = {"x": xyz[:,0], "y": xyz[:,1], "z": xyz[:,2],
                      "r": rgb[:,0], "g": rgb[:,1], "b": rgb[:,2],
                      "label_id": [maps_data["label2id"].get(s,0) for s in labels],
                      "room_id": [maps_data["room2id"].get(s,0) for s in rooms]}
            pd.DataFrame(df_data).to_parquet(
                output_dir / f"{scene_id}_sem_pointcloud.parquet", index=False)

    if args.save_csv:
        csv_data = df_data.copy()  # Same data as parquet
        # Add string labels for debugging
        csv_data.update({
            "label": labels,
            "room_type": rooms,
            "category": cats,
            "super": supers,
            "merged": merged
        })
        pd.DataFrame(csv_data).to_csv(
            output_dir / f"{scene_id}_sem_pointcloud.csv", index=False)

def process_one_scene(scene_path, model_dir, model_info_file, out_root, args):
    """Process single scene."""
    with open(model_info_file) as f:
        model_info_map = {item["model_id"]: item for item in json.load(f)}
    
    with open(scene_path) as f:
        scene_data = json.load(f)

    scene_id, _ = infer_ids_from_path(scene_path)
    output_dir = (out_root / scene_id) if args.per_scene_subdir else out_root
    safe_mkdir(output_dir)

    taxonomy_resolver = None
    if args.taxonomy:
        taxonomy_resolver = load_taxonomy_resolver(Path(args.taxonomy))

    config = {
        'total_points': args.total_points, 'ppsm': args.ppsm,
        'min_pts': args.min_pts_per_mesh, 'max_pts': args.max_pts_per_mesh
    }

    textured_scene, meshes_info, point_cloud, failed = process_scene(
        scene_data, model_dir, model_info_map, config, taxonomy_resolver)

    if not meshes_info:
        return False

    export_outputs(scene_id, output_dir, textured_scene, meshes_info, point_cloud, args)
    return True

def main():
    """Main entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--scene_file")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--scene_list")
    ap.add_argument("--model_dir", default="3D-FRONT_FUTURE/3D-FUTURE-model")
    ap.add_argument("--model_info", default="3D-FRONT_FUTURE/3D-FUTURE-model/model_info.json")
    ap.add_argument("--out_dir", default="stage1_test_out")
    ap.add_argument("--taxonomy")
    ap.add_argument("--total_points", type=int, default=500000)
    ap.add_argument("--ppsm", type=float, default=0.0)
    ap.add_argument("--min_pts_per_mesh", type=int, default=100)
    ap.add_argument("--max_pts_per_mesh", type=int, default=0)
    ap.add_argument("--save_glb", action="store_true", default=False)
    ap.add_argument("--save_obj", action="store_true", default=False)
    ap.add_argument("--save_parquet", action="store_true", default=True)
    ap.add_argument("--per_scene_subdir", action="store_true",default=True)
    ap.add_argument("--save_csv", action="store_true")

    args = ap.parse_args()
    
    config = load_config_with_profile(args.config)
    for key, value in config.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)

    scene_paths = gather_paths_from_sources(args.scene_file, args.scenes, args.scene_list)
    if not scene_paths:
        print("No scenes found")
        return

    model_dir = Path(args.model_dir)
    model_info_file = Path(args.model_info)
    out_root = Path(args.out_dir)
    safe_mkdir(out_root)

    progress = create_progress_tracker(len(scene_paths), "scenes")
    success_count = 0
    
    for i, scene_path in enumerate(scene_paths, 1):
        try:
            success = process_one_scene(scene_path, model_dir, model_info_file, out_root, args)
            if success:
                success_count += 1
            progress(i, scene_path.name, success)
        except Exception as e:
            print(f"Error: {e}")
            progress(i, scene_path.name, False)

    print(f"Done. {success_count}/{len(scene_paths)} completed.")

if __name__ == "__main__":
    main()