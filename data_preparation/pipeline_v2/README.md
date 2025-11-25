# Pipeline v2: Direct 3D-FRONT Rendering

This pipeline assembles 3D-FRONT scenes in memory from JSON definitions and 3D-FUTURE meshes, then renders top-down layouts and perspective POVs.

## Features

- **In-memory scene assembly**: No pre-built GLB files required
- **Headless rendering**: Uses EGL for HPC environments
- **Layout rendering**: Top-down RGB and segmentation images (256x256)
- **POV rendering**: Perspective RGB and segmentation images (256x256)
- **Graph generation**: Automatic room graph creation from segmentation layouts
- **Architectural meshes**: Includes walls, floors, and ceilings from JSON

## Files

- `scene_loader.py`: Parses 3D-FRONT JSON and loads meshes into trimesh.Scene
- `render_worker.py`: Main rendering script for layouts and POVs
- `graph_builder.py`: Wrapper for graph building functionality
- `hpc_scripts/run_render.sh`: LSF job array script (max 20 jobs)

## Usage

### Single Scene

```bash
python data_preparation/pipeline_v2/render_worker.py \
    --scene_json /path/to/scene.json \
    --future_root /path/to/3d-future \
    --output_dir /path/to/dataset_v2 \
    --taxonomy /path/to/taxonomy.json \
    --num_povs 6 \
    --seed 42
```

### HPC Batch Processing

1. Edit `hpc_scripts/run_render.sh` and set:
   - `SCENES_DIR`: Directory containing 3D-FRONT JSON files
   - `FUTURE_DIR`: Directory containing 3D-FUTURE models
   - `OUTPUT_DIR`: Output dataset root
   - `TAXONOMY`: Path to taxonomy.json
   - `PROJECT_ROOT`: Path to ImgiNav repository

2. Submit job array:
```bash
bsub < data_preparation/pipeline_v2/hpc_scripts/run_render.sh
```

The script will:
- Generate a scene list from `SCENES_DIR`
- Distribute scenes across 20 jobs
- Each job processes its assigned scenes sequentially

## Output Structure

```
dataset_v2/
├── layouts/
│   ├── rgb/
│   │   └── {scene_id}.png
│   └── seg/
│       └── {scene_id}.png
├── povs/
│   ├── rgb/
│   │   └── {scene_id}_v{01-06}.png
│   └── seg/
│       └── {scene_id}_v{01-06}.png
└── graphs/
    ├── {scene_id}_0_graph.json
    ├── {scene_id}_0_graph.txt
    └── {scene_id}_0_graph_vis.png
```

## Specifications

- **Image resolution**: 256x256 for both layouts and POVs
- **POV sampling**: 20 attempts before failing to find valid camera position
- **Eye height**: 1.6m for POV cameras
- **Camera FOV**: 70 degrees for POVs
- **Ceiling filtering**: Ceilings are hidden in layouts, visible in POVs

## Dependencies

- trimesh
- pyrender
- numpy
- scipy
- PIL/Pillow
- opencv-python (for graph building)

## Environment Setup

For headless rendering, ensure:
```bash
export PYOPENGL_PLATFORM=egl
```

This is automatically set in `render_worker.py`.

