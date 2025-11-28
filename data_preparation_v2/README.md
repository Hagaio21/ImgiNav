# Data Preparation Pipeline

This pipeline processes 3D-FRONT scenes for the ImgiNav project. It supports sharding for parallel HPC processing and automatic stage chaining.

## Configuration: paths.yaml

All paths are configured in a single `paths.yaml` file:

```yaml
# OUTPUT - Where processed data goes
output_dataset_root: "/work3/s233249/ImgiNav/dataset_v2"

# 3D-FRONT/3D-FUTURE SOURCES (only needed for Stage 1 & 2)
front3d_scenes_dir: "/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
front3d_model_info: "/work3/s233249/ImgiNav/datasets/3D-FUTURE-model/model_info.json"
front3d_model_dir: "/work3/s233249/ImgiNav/datasets/3D-FUTURE-model"

# PIPELINE PATHS
base_dir: "/work3/s233249/ImgiNav/ImgiNav"
shards_dir: "data_preparation_v2/hpc_scripts/shards"  # relative to base_dir
log_dir: "data_preparation_v2/hpc_scripts/logs"       # relative to base_dir
```

**Key design:**
- `output_dataset_root` - Where ALL processed data goes (your new dataset)
- `front3d_*` paths - Only used by Stage 1 & 2 to read source data
- Stages 3-6 only need `output_dataset_root`

## Pipeline Overview

```
Stage 1: Reconstruct Geometry (needs 3D-FRONT sources)
   ↓
Stage 2: Compile Metadata (needs 3D-FRONT sources)
   ↓
Stage 3: Render Layouts (only needs OUTPUT_DATASET_ROOT)
   ↓
Stage 4: Render POVs (only needs OUTPUT_DATASET_ROOT)
   ↓
Stage 5: Build Graphs (only needs OUTPUT_DATASET_ROOT)
   ↓
Stage 6: Generate Manifests (only needs OUTPUT_DATASET_ROOT)
```

## Output Dataset Structure

```
$OUTPUT_DATASET_ROOT/
├── taxonomy/
│   └── taxonomy.json       # Category colors (create this first!)
├── geometry/
│   ├── tex/                # Textured GLB files
│   └── seg/                # Segmented GLB files
├── metadata/
│   ├── scenes/             # Scene-level JSON metadata
│   └── rooms/              # Room-level JSON metadata
├── layouts/
│   ├── tex/                # Textured layout images
│   └── seg/                # Segmented layout images
├── povs/
│   ├── tex/                # Textured POV images
│   └── seg/                # Segmented POV images
├── graphs/
│   ├── jsons/              # Scene and room graphs
│   └── texts/              # Text descriptions
└── manifests/
    ├── scenes.json         # Scene index
    └── rooms.json          # Room index
```

## Quick Start

### 1. Edit paths.yaml

Copy and edit `paths.yaml` for your environment:

```yaml
output_dataset_root: "/work3/s233249/ImgiNav/dataset_v2"
front3d_scenes_dir: "/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
front3d_model_info: "/work3/s233249/ImgiNav/datasets/3D-FUTURE-model/model_info.json"
front3d_model_dir: "/work3/s233249/ImgiNav/datasets/3D-FUTURE-model"
base_dir: "/work3/s233249/ImgiNav/ImgiNav"
shards_dir: "data_preparation_v2/hpc_scripts/shards"
log_dir: "data_preparation_v2/hpc_scripts/logs"
```

### 2. Create Taxonomy File

Create `$OUTPUT_DATASET_ROOT/taxonomy/taxonomy.json`:

```json
{
  "category_to_color": {
    "floor": [200, 200, 200],
    "wall": [50, 50, 50],
    "Door": [255, 100, 100],
    "Window": [100, 200, 255],
    "Bed": [255, 150, 150],
    "Chair": [150, 255, 150],
    "Table": [150, 150, 255]
  }
}
```

### 3. Create Shards

Split scenes into shards for parallel processing:

```bash
# Discover scenes from 3D-FRONT directory
python create_shards.py \
    --scenes-dir /dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT \
    --output-dir shards/ \
    --num-shards 10

# Or from a scene list file
python create_shards.py \
    --scene-list all_scenes.txt \
    --output-dir shards/ \
    --num-shards 10
```

### 4. Launch Pipeline

```bash
# Launch from stage 2
./launch_pipeline.sh --config paths.yaml --stage 2

# Dry run to see what would be submitted
./launch_pipeline.sh --config paths.yaml --stage 2 --dry-run

# Resume from a later stage (only needs output data, not 3D-FRONT sources)
./launch_pipeline.sh --config paths.yaml --stage 4
```

### 5. Monitor Jobs

```bash
bjobs -w
tail -f logs/stage2_metadata.*.out
```

## Running Individual Stages

### Stage 2: Compile Metadata

```bash
python stage2_compile_metadata.py \
    --config paths.yaml \
    --scene-list shard_1.txt
```

### Stage 3-6: Only need OUTPUT_DATASET_ROOT

```bash
# These stages use --dataset-root which is OUTPUT_DATASET_ROOT
python stage3_render_layouts.py \
    --dataset-root /work3/s233249/ImgiNav/dataset_v2 \
    --scene-list shard_1.txt \
    --hpc

python stage4_render_povs.py \
    --dataset-root /work3/s233249/ImgiNav/dataset_v2 \
    --scene-list shard_1.txt \
    --hpc

python stage5_build_graphs.py \
    --dataset-root /work3/s233249/ImgiNav/dataset_v2 \
    --scene-list shard_1.txt

python stage6_generate_manifests.py \
    --dataset-root /work3/s233249/ImgiNav/dataset_v2 \
    --scene-list shard_1.txt
```

## Troubleshooting

### Scene files not found (Stage 2)

The script searches recursively in `front3d_scenes_dir` for `<scene_id>.json`. Check:
- Is `front3d_scenes_dir` correct in paths.yaml?
- Do the scene IDs in your shard file match actual filenames?

### Config validation

```bash
python config_loader.py paths.yaml --validate --stage 2
```

### HPC rendering issues

- Ensure xvfbwrapper is installed: `pip install xvfbwrapper`
- The `--hpc` flag enables headless rendering
- Check logs for OpenGL errors

## Stage Chaining

Each stage automatically submits the next stage for the same shard:

```
Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
```

The `CONFIG_FILE` environment variable is passed to each stage.

## Files

| File | Description |
|------|-------------|
| `paths.yaml` | **Edit this!** All configuration in one place |
| `config_loader.py` | Python utility to load/validate config |
| `create_shards.py` | Create shard files from scene list |
| `launch_pipeline.sh` | Master launcher for HPC |
| `run_stage{2-6}_array.sh` | Individual stage scripts |
| `stage2_compile_metadata.py` | Stage 2 Python script (uses paths.yaml) |
