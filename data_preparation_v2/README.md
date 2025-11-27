# 3D-FRONT / 3D-FUTURE Data Preparation Pipeline v2

This pipeline prepares 3D-FRONT and 3D-FUTURE data into a clean, flat dataset structure.

## Overview

The v2 pipeline processes 3D scene data through 7 stages:

1. **Stage 0**: Taxonomy Generation - Creates deterministic category/supercategory ↔ color mappings
2. **Stage 1**: Geometry Reconstruction - Generates textured and segmented GLB files
3. **Stage 2**: Metadata Compilation - Extracts scene and room metadata
4. **Stage 3**: Layout Rendering - Generates top-down orthographic layout images
5. **Stage 4**: POV Rendering - Generates perspective point-of-view images
6. **Stage 5**: Graph Generation - Builds room and scene graphs
7. **Stage 6**: Manifest Generation - Creates global index files

## Output Structure

```
dataset/
├── geometry/
│   ├── tex/
│   │   └── <scene_id>_tex.glb
│   └── seg/
│       └── <scene_id>_seg.glb
├── layouts/
│   ├── tex/
│   │   ├── <scene_id>_tex_layout.png
│   │   └── <scene_id>_<room>_tex_layout.png
│   └── seg/
│       ├── <scene_id>_seg_layout.png
│       └── <scene_id>_<room>_seg_layout.png
├── povs/
│   ├── tex/
│   │   └── <scene_id>_<room>_tex_pov.png
│   └── seg/
│       └── <scene_id>_<room>_seg_pov.png
├── graphs/
│   └── jsons/
│       ├── <scene_id>_scene_graph.json
│       └── <scene_id>_<room>_room_graph.json
├── metadata/
│   ├── scenes/
│   │   └── <scene_id>.json
│   └── rooms/
│       └── <scene_id>_<room>.json
├── taxonomy/
│   └── taxonomy.json
└── manifests/
    ├── scenes.json
    └── rooms.json
```

## Usage

### Stage 0: Taxonomy Generation

```bash
python data_preparation_v2/stage0_build_taxonomy.py \
    --model-info <path_to_model_info.json> \
    --scenes-dir <path_to_3d_front_scenes> \
    --out dataset/taxonomy/taxonomy.json
```

### Stage 1: Geometry Reconstruction

```bash
python data_preparation_v2/stage1_reconstruct_geometry.py \
    --scenes-dir <path_to_3d_front_scenes> \
    --model-dir <path_to_3d_future_models> \
    --model-info <path_to_model_info.json> \
    --taxonomy dataset/taxonomy/taxonomy.json \
    --output-dir dataset/geometry \
    [--texture-dir <path_to_3d_front_texture>]
```

### Stage 2: Metadata Compilation

```bash
python data_preparation_v2/stage2_compile_metadata.py \
    --scenes-dir <path_to_3d_front_scenes> \
    --model-info <path_to_model_info.json> \
    --taxonomy dataset/taxonomy/taxonomy.json \
    --output-dir dataset/metadata
```

### Stage 3: Layout Rendering

```bash
python data_preparation_v2/stage3_render_layouts.py \
    --geometry-dir dataset/geometry \
    --metadata-dir dataset/metadata \
    --output-dir dataset/layouts \
    [--resolution 512]
```

### Stage 4: POV Rendering

```bash
python data_preparation_v2/stage4_render_povs.py \
    --geometry-dir dataset/geometry \
    --metadata-dir dataset/metadata \
    --output-dir dataset/povs \
    [--width 1280 --height 800 --fov-deg 70.0]
```

### Stage 5: Graph Generation

```bash
python data_preparation_v2/stage5_build_graphs.py \
    --layouts-dir dataset/layouts \
    --metadata-dir dataset/metadata \
    --taxonomy dataset/taxonomy/taxonomy.json \
    --output-dir dataset/graphs
```

### Stage 6: Manifest Generation

```bash
python data_preparation_v2/stage6_generate_manifests.py \
    --dataset-root dataset \
    --output-dir dataset/manifests
```

## Performance Optimizations

The pipeline is optimized for laptop use:

- **Stage 3 (Layouts)**: Loads each GLB file once and generates all layouts (scene + rooms) from it
- **Stage 4 (POVs)**: Loads each GLB file once and generates all POVs from it
- Memory is freed after processing each scene

## Testing

Run tests with a small subset of data:

```bash
# Run all stage tests
python data_preparation_v2/tests/test_runner.py

# Run individual stage tests
python -m pytest data_preparation_v2/tests/test_stage0.py
```

Test outputs are saved to `test_dataset/` directory.

## Dependencies

- trimesh
- open3d
- numpy
- pandas
- PIL/Pillow
- scipy
- tqdm
- cv2 (opencv-python)

## Notes

- All outputs use a flat structure with naming convention `<scene_id>_<room>_<type>.<ext>`
- Colors are deterministic (hash-based) for consistent segmentation
- The pipeline processes one scene at a time to manage memory usage

