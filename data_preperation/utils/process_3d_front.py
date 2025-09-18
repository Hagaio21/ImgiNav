import json
from pathlib import Path
import os
import trimesh
from scipy.spatial.transform import Rotation
import numpy as np



def load_model_info(model_info_path):
    """Load model_info.json as a dictionary mapping model_id to info."""
    model_info_path = Path(model_info_path)
    with open(model_info_path, 'r') as f:
        info_list = json.load(f)
    # Map: model_id -> info dict
    return {entry["model_id"]: entry for entry in info_list}

def load_scene_json(scene_json_path):
    """Load a 3D-FRONT scene JSON file and return as dict."""
    scene_json_path = Path(scene_json_path)
    with open(scene_json_path, 'r') as f:
        return json.load(f)
    
def load_mesh_with_texture(obj_path):
    """
    Load OBJ mesh with textures, robust to relative texture paths.
    Changes working dir to OBJ's parent so textures resolve correctly.
    """
    obj_path = Path(obj_path)
    old_cwd = os.getcwd()
    try:
        os.chdir(obj_path.parent)
        mesh = trimesh.load(
            str(obj_path.name),   # load by filename
            force='mesh',
            process=False,
            maintain_order=True
        )
    finally:
        os.chdir(old_cwd)
    return mesh

def apply_transform(mesh, pos, rot, scale):
    """
    Apply translation, rotation (quaternion), and scale to a trimesh mesh.
    """
    pos = np.array(pos, dtype=np.float64)
    rot = np.array(rot, dtype=np.float64)
    scale = np.array(scale, dtype=np.float64)
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = Rotation.from_quat(rot).as_matrix() @ np.diag(scale)
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return mesh

def assemble_scene_meshes_with_room_ids(scene_dict):
    """
    Assigns each mesh object:
        - room_id: the room's 'instanceid' it belongs to
        - label: mesh 'type'
        - category_id: unique int for each mesh type
    Returns:
        meshes, metas
    """
    import trimesh
    import numpy as np
    from collections import defaultdict

    # Build mesh uid to room_id mapping
    mesh_uid_to_room = {}
    for room in scene_dict.get("scene", {}).get("room", []):
        room_id = room.get("instanceid")
        for child in room.get("children", []):
            # Children can be string (mesh uid) or dict (furniture)
            if isinstance(child, str):
                mesh_uid_to_room[child] = room_id
            elif isinstance(child, dict):
                # skip for now (furniture)
                continue

    # Build type-to-category_id mapping
    mesh_types = set()
    for mesh in scene_dict.get("mesh", []):
        mesh_types.add(mesh.get("type", "Unknown"))
    type_to_category = {t: i for i, t in enumerate(sorted(mesh_types), 1)}  # 1-based index

    meshes = []
    metas = []

    for mesh_obj in scene_dict.get("mesh", []):
        uid = mesh_obj["uid"]
        label = mesh_obj.get("type", "Unknown")
        category_id = type_to_category[label]
        room_id = mesh_uid_to_room.get(uid, None)
        vertices = np.array(mesh_obj["xyz"], dtype=np.float64).reshape(-1, 3)
        faces = np.array(mesh_obj["faces"], dtype=np.int64).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        mesh.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
        meta = {
            "object_id": uid,
            "category_id": category_id,
            "label": label,
            "room_id": room_id
        }
        meshes.append(mesh)
        metas.append(meta)

    return meshes, metas




def main():

    from pathlib import Path

    # Set your paths:
    scene_json_path = Path(r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\datasets\3D-FRONT_FUTURE\3D-FRONT\9b098112-9633-4328-b7f8-94054dd2d87e.json")
    model_info_path = Path(r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\model_info.json")
    models_root = Path(r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\datasets\3D-FRONT_FUTURE\3D-FUTURE-model\3D-FUTURE-model"
)

    # 1. Load model info and scene JSON
    model_info = load_model_info(model_info_path)
    scene_dict = load_scene_json(scene_json_path)

    # 2. Assemble meshes and meta
    meshes, metas = assemble_scene_meshes_with_room_ids(scene_dict)

    print(f"Loaded {len(meshes)} meshes from scene.")
    for i, (m, meta) in enumerate(zip(meshes, metas)):
        print(f"Mesh {i}: V={len(m.vertices)}, F={len(m.faces)}, meta={meta}")


if __name__ == "__main__":
    main()