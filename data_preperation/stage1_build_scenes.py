"""
stage-1: process the 3D-FUTURE-FRONT dataset -> saves a .npz/.npy/.csv/.parquet of a sampled pointclouds and a scene metadata.json file.

output:
    <scene_id>.npz/.npy/.csv/.parquet -> x,y,z,r,b,g,<object_id>,<category_id>,<room_label_id>
    <scene_id>scene_info.json:

    {
    "bounds": { "min": [...], "max": [...] },
    "size": [...],
    "up_normal": [0, 0, 1]  # always Z-up unless flipped
    }

"""

from utils.file_utils import SavePolicy, read_scene_list_file
from utils.future_front_utils import _gather_scene_paths, _process_one_scene, _prescan_build_maps
from utils.semantic_utils import load_taxonomy_categories
# (geometry utils are called inside future_front_utils)

def _export_pointcloud_compact(scene_id, output_dir, point_cloud, save_formats):
    pass


def _parse_args_with_config(config):
    pass



def main():
    args = _parse_args_with_config()
    scene_paths = _gather_scene_paths(args.scene_file, args.scenes, args.scene_list)
    for sp in scene_paths:
        _process_one_scene(sp, args.model_dir, args.model_info, args.out_dir, args)

