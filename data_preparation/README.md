# Data Preparation Module

This module contains scripts and utilities for preparing and processing the ImgiNav dataset.

## Overview

The data preparation pipeline processes raw 3D scene data into formats suitable for training:
- Scene building and processing
- Room splitting and organization
- Layout generation
- Point-of-view (POV) extraction
- Graph construction
- Embedding creation

## Pipeline Stages

### Stage 1: Build Scenes (`stage1_build_scenes.py`)

Processes raw 3D scene data and builds scene representations.

**Input:**
- Raw 3D scene files (GLB, OBJ, PLY)
- Scene metadata

**Output:**
- Processed scene files
- Scene manifests
- Textured meshes

**Usage:**
```bash
python data_preparation/stage1_build_scenes.py --input_dir /path/to/raw/scenes --output_dir /path/to/processed
```

### Stage 2: Split to Rooms (`stage2_split2rooms.py`)

Splits scenes into individual room components.

**Input:**
- Processed scenes from Stage 1

**Output:**
- Room-level data
- Room manifests
- Room metadata

**Usage:**
```bash
python data_preparation/stage2_split2rooms.py --input_dir /path/to/scenes --output_dir /path/to/rooms
```

### Stage 3: Create Room Scenes Layouts (`stage3_create_room_scenes_layouts.py`)

Generates layout images from room data.

**Input:**
- Room data from Stage 2

**Output:**
- Layout images (RGB)
- Segmentation maps
- Layout manifests

**Key Features:**
- Top-down view generation
- Semantic segmentation
- Room type classification

**Usage:**
```bash
python data_preparation/stage3_create_room_scenes_layouts.py --input_dir /path/to/rooms --output_dir /path/to/layouts
```

### Stage 4: Create Room POVs (`stage4_create_room_povs.py`)

Extracts point-of-view images from rooms.

**Input:**
- Room data from Stage 2

**Output:**
- POV images
- Camera parameters
- POV manifests

**Usage:**
```bash
python data_preparation/stage4_create_room_povs.py --input_dir /path/to/rooms --output_dir /path/to/povs
```

## Utility Scripts

### `collect.py`
Collects and organizes data from various sources.

**Features:**
- File discovery
- Manifest creation
- Data validation

### `build_graphs.py`
Builds graph representations of scenes.

**Features:**
- Spatial graph construction
- Relationship extraction
- Graph serialization

### `create_augmentations.py`
Creates data augmentations for training.

**Features:**
- Image augmentations
- Geometric transformations
- Augmented manifest creation

### `create_embeddings.py`
Pre-embeds images to latent representations.

**Features:**
- Batch embedding generation
- Latent storage
- Embedding manifest creation

**Usage:**
```bash
python data_preparation/create_embeddings.py \
    --model_checkpoint /path/to/autoencoder.pt \
    --manifest /path/to/layouts.csv \
    --output_dir /path/to/embeddings
```

## Utilities

### `utils/file_discovery.py`
Discovers and collects files from various sources.

**Functions:**
- `gather_paths_from_sources(sources)` - Collect files from multiple sources
- `infer_ids_from_path(path)` - Extract IDs from file paths

### `utils/geometry_utils.py`
Geometric utilities for 3D processing.

**Functions:**
- Coordinate transformations
- Mesh operations
- Camera calculations

### `utils/text_utils.py`
Text processing utilities.

**Functions:**
- Text normalization
- Label processing
- Taxonomy mapping

## HPC Scripts

The `hpc_scripts/` directory contains shell scripts for running data preparation on HPC clusters.

### Stage Scripts
- `run_stage1.sh` - Run Stage 1
- `run_stage2.sh` - Run Stage 2
- `run_stage3_*.sh` - Run Stage 3 variants
- `run_stage4_*.sh` - Run Stage 4 variants

### Utility Scripts
- `run_collect_*.sh` - Data collection scripts
- `create_augmentations.sh` - Create augmentations
- `embed_images.sh` - Pre-embed images
- `run_stage6_layout_embeddings.sh` - Create layout embeddings
- `run_stage7_create_pov_embeddings.sh` - Create POV embeddings
- `run_stage8_create_graph_embeddings.sh` - Create graph embeddings

## Data Formats

### Manifest Format
CSV files with columns:
- `scene_id`: Scene identifier
- `room_id`: Room identifier
- `layout_path`: Path to layout image
- `segmentation_path`: Path to segmentation map
- `type`: Room type classification
- `is_empty`: Boolean flag for empty rooms

### Layout Images
- **Format**: PNG
- **Size**: 512×512 pixels
- **Channels**: RGB (3 channels)
- **Segmentation**: 10 classes

### Embeddings
- **Format**: NumPy arrays (.npy)
- **Shape**: (H, W, C) matching autoencoder latent space
- **Storage**: Organized by scene/room ID

## Pipeline Workflow

1. **Stage 1**: Process raw scenes → Build scene representations
2. **Stage 2**: Split scenes → Extract rooms
3. **Stage 3**: Generate layouts → Create layout images and segmentation
4. **Stage 4**: Extract POVs → Create point-of-view images
5. **Optional**: Create augmentations → Augment dataset
6. **Optional**: Pre-embed → Create latent embeddings for faster training

## Running the Pipeline

### Full Pipeline
```bash
# Stage 1
bash data_preparation/hpc_scripts/run_stage1.sh

# Stage 2
bash data_preparation/hpc_scripts/run_stage2.sh

# Stage 3
bash data_preparation/hpc_scripts/run_stage3_rooms.sh

# Stage 4
bash data_preparation/hpc_scripts/run_stage4_povs.sh
```

### Individual Stages
Each stage can be run independently if previous stages are complete.

## Output Structure

```
datasets/
├── scenes/                          # Processed scene data
│   ├── {scene_id}/                  # Scene directory (new format, inplace)
│   │   ├── layouts/                 # Scene-level layouts
│   │   │   └── {scene_id}_scene_layout.png
│   │   └── rooms/
│   │       └── {room_id}/
│   │           ├── {scene_id}_{room_id}.parquet
│   │           ├── {scene_id}_{room_id}_meta.json
│   │           └── layouts/        # Room-level layouts
│   │               └── {scene_id}_{room_id}_room_seg_layout.png
│   └── scene_id={scene_id}/         # Scene directory (old format)
│       └── rooms/
│           └── room_id={room_id}/
│               └── part-*.parquet
├── layouts.csv                      # Layout manifest (main training data)
├── povs.csv                         # POV manifest
├── graphs.csv                       # Graph manifest
├── augmented/                       # Augmented dataset
│   ├── images/                      # Augmented images (original + variants)
│   │   └── {room_id}_aug_{variant}.png
│   ├── latents/                     # Pre-embedded latents
│   │   └── {room_id}_latent.npy
│   ├── manifest_images.csv          # Manifest for images only
│   └── manifest.csv                 # Full manifest with latents (for training)
└── manifests/                       # Additional manifest files (if created)
    └── ...
```

**Key Points:**
- **Scenes**: Processed scene data with room splits. Supports both old format (`scene_id={id}/rooms/room_id={id}/`) and new format (`{scene_id}/rooms/{room_id}/`)
- **Layouts**: Layout images are stored within scene/room directories:
  - Room layouts: `{scene_id}/rooms/{room_id}/layouts/{scene_id}_{room_id}_room_seg_layout.png`
  - Scene layouts: `{scene_id}/layouts/{scene_id}_scene_layout.png`
  - Manifest at `datasets/layouts.csv` tracks all layout paths
- **Augmented**: Contains both original and augmented images, plus pre-embedded latents for faster training
- **Manifests**: CSV files at `datasets/` root level track all data locations

## Notes

- All scripts support parallel processing where applicable
- Progress tracking is included for long-running operations
- Error handling and validation are built-in
- Manifest files are used for tracking data throughout the pipeline

