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

from utils.file_utils import load_json


def _gather_scene_paths(scene_id: str, scenes_dir: str) -> Dict[str, str]:
    base = Path(scenes_dir)
    scene_file = base / f"{scene_id}.json"
    if not scene_file.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_file}")
    return {"scene_json": str(scene_file)}


def resolve_model_path(model_id: str, models_dir: str) -> str:
    model_dir = Path(models_dir) / model_id
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    obj_file = model_dir / "raw_model.obj"
    glb_file = model_dir / "raw_model.glb"

    if obj_file.exists():
        return str(obj_file)
    elif glb_file.exists():
        return str(glb_file)
    else:
        raise FileNotFoundError(f"No mesh found for model {model_id} in {model_dir}")


def process_scene(scene_path: str, models_dir: str) -> Tuple[trimesh.Scene, Dict[str, Any]]:
    """
    Assemble a 3D-FRONT scene into a trimesh.Scene + metadata.

    Parameters
    ----------
    scene_path : str
        Path to the scene JSON file.
    models_dir : str
        Directory containing 3D-FUTURE models.

    Returns
    -------
    scene : trimesh.Scene
        Assembled geometry with transforms applied.
    object_info : dict
        Metadata for each object in the scene:
        {
            node_name: {
                "object_id": int,
                "model_id": str,
                "category": str,
                "room_label": str
            }
        }
    """
    scene_json = load_json(scene_path)
    scene = trimesh.Scene()
    object_info = {}

    obj_index = 0

    for room in scene_json.get("rooms", []):
        room_label = room.get("roomType", "unknown")

        for obj in room.get("children", []):
            model_id = obj.get("modelId")
            if model_id is None:
                continue

            transform = np.array(obj.get("transform", np.eye(4))).reshape(4, 4)
            category = obj.get("category", "unknown")

            try:
                model_path = resolve_model_path(model_id, models_dir)
            except FileNotFoundError as e:
                print(f"[WARN] Skipping object {model_id}: {e}")
                continue

            try:
                mesh = trimesh.load(model_path, force="mesh")
            except Exception as e:
                print(f"[WARN] Could not load mesh {model_id}: {e}")
                continue

            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()

            mesh.apply_transform(transform)

            # Use a unique node name (so we can link metadata)
            node_name = f"obj_{obj_index}_{model_id}"
            scene.add_geometry(mesh, node_name=node_name)

            # Store metadata
            object_info[node_name] = {
                "object_id": obj_index,
                "model_id": model_id,
                "category": category,
                "room_label": room_label
            }

            obj_index += 1

    return scene, object_info
