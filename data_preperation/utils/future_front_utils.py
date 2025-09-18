"""
future_front_utils.py
---------------------
Dataset-specific utilities for 3D-FRONT + 3D-FUTURE.
This module assembles a scene and attaches object metadata.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import trimesh
from file_utils import load_json
import json

def _gather_scene_paths(scene_id: str, scenes_dir: str) -> Dict[str, str]:
    base = Path(scenes_dir)
    scene_file = base / f"{scene_id}.json"
    if not scene_file.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_file}")
    return {"scene_json": str(scene_file)}


def resolve_model_path(model_id: str, models_dir: Path) -> Path:
    """
    Resolve a 3D-FUTURE model_id to a local mesh path.

    Parameters
    ----------
    model_id : str
        The 3D-FUTURE model identifier.
    models_dir : Path
        Directory containing 3D-FUTURE models.

    Returns
    -------
    Path
        Path to the .obj or .glb file.

    Raises
    ------
    FileNotFoundError
        If neither .obj nor .glb is found.
    """
    obj_path = models_dir / model_id / "raw_model.obj"
    glb_path = models_dir / model_id / "raw_model.glb"

    if obj_path.exists():
        return obj_path
    elif glb_path.exists():
        return glb_path
    else:
        raise FileNotFoundError(f"No mesh found for {model_id} in {models_dir}")


def process_scene(scene_path: Path, model_info_path: Path, models_dir: Path):
    """
    Assemble a 3D-FRONT scene into a trimesh.Scene + object metadata.
    Handles furniture (3D-FUTURE), inline architectural meshes, and external arch refs.
    Always tags architectural objects with their type so they are segmentable.
    """
    # --- Load scene JSON
    scene_json = load_json(scene_path)

    # --- Load model info (3D-FUTURE metadata)
    with open(model_info_path, "r", encoding="utf-8") as f:
        model_info_map = {m["model_id"]: m for m in json.load(f)}

    # --- Lookup tables
    furniture_map = {f["uid"]: f for f in scene_json.get("furniture", [])}
    mesh_map = {m["uid"]: m for m in scene_json.get("mesh", [])}

    scene = trimesh.Scene()
    object_info = {}
    obj_index = 0
    furn_count, arch_count = 0, 0

    # --- Traverse rooms
    for room in scene_json.get("scene", {}).get("room", []):
        room_type = room.get("type", "UnknownRoom")

        for child in room.get("children", []):
            ref_id = child.get("ref")
            if ref_id is None:
                continue

            # ---------------- Furniture ----------------
            if ref_id in furniture_map:
                furn = furniture_map[ref_id]
                model_id = furn.get("jid")
                if not model_id:
                    continue

                meta = model_info_map.get(model_id, {})
                category = meta.get("category", "unknown")
                label = (
                    meta.get("label")
                    or meta.get("name")
                    or furn.get("title", "unknown")
                )

                # Transform
                pos = np.array(child.get("pos", [0, 0, 0]), dtype=float)
                rot = np.array(child.get("rot", [0, 0, 0, 1]), dtype=float)  # quaternion
                scale = np.array(child.get("scale", [1, 1, 1]), dtype=float)
                T = np.eye(4)
                T[:3, :3] = trimesh.transformations.quaternion_matrix(rot)[:3, :3]
                T[:3, 3] = pos
                T = T @ np.diag([*scale, 1.0])

                node_name = f"furn_{obj_index}_{model_id}"
                mesh = None
                try:
                    model_path = resolve_model_path(model_id, models_dir)
                    mesh = trimesh.load(model_path, force="mesh")
                    if not isinstance(mesh, trimesh.Trimesh):
                        mesh = mesh.dump().sum()
                    mesh.apply_transform(T)
                    scene.add_geometry(mesh, node_name=node_name)
                except Exception:
                    pass  # skip geometry if missing

                object_info[node_name] = {
                    "object_id": obj_index,
                    "model_id": model_id,
                    "category": category,
                    "label": label,
                    "room_type": room_type,
                    "is_architecture": False,
                }
                obj_index += 1
                furn_count += 1

            # ---------------- Inline architectural mesh ----------------
            elif ref_id in mesh_map:
                mitem = mesh_map[ref_id]
                arch_type = mitem.get("type", "mesh")
                verts = np.array(mitem.get("xyz", []), dtype=float).reshape(-1, 3)
                faces = np.array(mitem.get("faces", []), dtype=int).reshape(-1, 3)

                node_name = f"mesh_{obj_index}_{arch_type}"
                if verts.size > 0 and faces.size > 0:
                    try:
                        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                        scene.add_geometry(mesh, node_name=node_name)
                    except Exception:
                        pass  # skip geometry if corrupt

                object_info[node_name] = {
                    "object_id": obj_index,
                    "model_id": None,
                    "category": arch_type,
                    "label": arch_type,
                    "room_type": room_type,
                    "is_architecture": True,
                }
                obj_index += 1
                arch_count += 1

            # ---------------- External architecture refs ----------------
            elif ref_id.endswith("/model"):
                mesh_id = ref_id.split("/")[0]
                mesh_dir = scene_path.parent / "mesh" / mesh_id
                obj_path = mesh_dir / "model.obj"
                glb_path = mesh_dir / "model.glb"

                node_name = f"arch_{obj_index}_{mesh_id}"
                mesh = None
                mesh_file = None
                if obj_path.exists():
                    mesh_file = obj_path
                elif glb_path.exists():
                    mesh_file = glb_path

                if mesh_file is not None:
                    try:
                        mesh = trimesh.load(mesh_file, force="mesh")
                        if not isinstance(mesh, trimesh.Trimesh):
                            mesh = mesh.dump().sum()
                        scene.add_geometry(mesh, node_name=node_name)
                    except Exception:
                        pass  # skip geometry if load fails

                object_info[node_name] = {
                    "object_id": obj_index,
                    "model_id": mesh_id,
                    "category": "architecture",
                    "label": "architecture",
                    "room_type": room_type,
                    "is_architecture": True,
                }
                obj_index += 1
                arch_count += 1

            else:
                # silently ignore unknown refs
                continue

    # --- Report
    print(f"[INFO] Furniture objects: {furn_count}, Architectural objects: {arch_count}")

    return scene, object_info


def main():
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Assemble a 3D-FRONT scene with 3D-FUTURE models")
    parser.add_argument("--scene", required=True, help="Path to scene JSON")
    parser.add_argument("--model-info", required=True, help="Path to model_info.json")
    parser.add_argument("--models-dir", required=True, help="Path to 3D-FUTURE-model directory")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    model_info_path = Path(args.model_info)
    models_dir = Path(args.models_dir)

    scene, object_info = process_scene(scene_path, model_info_path, models_dir)

    print("[INFO] Scene loaded successfully")
    print(f"Scene ID: {scene_path.stem}")
    print(f"Objects loaded: {len(object_info)}")
    for k, v in list(object_info.items())[:5]:
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
