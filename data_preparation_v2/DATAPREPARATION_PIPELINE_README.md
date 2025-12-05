# Data Preparation Pipeline - Complete Guide

## Overview

This pipeline processes 3D-FRONT scenes into a structured dataset for training diffusion models. It converts 3D scene data into 2D layouts, point-of-view (POV) images, semantic graphs, and embeddings suitable for machine learning.

**Pipeline Flow:**
```
3D-FRONT Scenes → Geometry → Metadata → Layouts → POVs → Graphs → Manifests → Training Data
```

## Pipeline Stages

### Stage 0: Taxonomy Generation (`stage0_build_taxonomy.py`)

**Purpose**: Create a deterministic color mapping for object categories and supercategories.

**What it does:**
- Scans 3D-FUTURE `model_info.json` to collect all furniture categories
- Scans 3D-FRONT scene files to collect room types and object titles
- Creates deterministic color assignments for:
  - **Super-categories** (e.g., "Seating", "Storage", "Tables")
  - **Categories** (e.g., "Chair", "Sofa", "Coffee Table")
  - **Structural elements** (Floor, Wall, Ceiling)
  - **Architectural elements** (Doors, Windows)
- Assigns fixed colors to structural elements (floor=dark gray, wall=light gray)
- Uses hash-based color generation for furniture categories to ensure consistency

**Input:**
- `model_info.json` from 3D-FUTURE dataset
- 3D-FRONT scene JSON files

**Output:**
- `dataset/taxonomy/taxonomy.json` - Complete taxonomy with:
  - `id2color`: Maps category/supercategory IDs to RGB colors
  - `category2super`: Maps categories to their supercategories
  - `room2id`: Maps room types to IDs
  - `id2room`, `id2category`, `id2super`: Reverse mappings

**Key Features:**
- Deterministic: Same categories always get same colors
- Hierarchical: Categories inherit colors from supercategories with variations
- Structural elements have fixed, easily recognizable colors

---

### Stage 0b: Scene Validation (`stage0_validate_scenes.py`)

**Purpose**: Validate 3D-FRONT scenes and filter out invalid ones.

**What it does:**
- Checks scene files for required structure
- Validates room geometry and furniture placement
- Identifies scenes with missing or corrupted data
- Creates a list of valid scene IDs for downstream processing

**Output:**
- `valid_scenes.txt` - List of scene IDs that passed validation

---

### Stage 1: Geometry Reconstruction (`stage1_reconstruct_geometry.py`)

**Purpose**: Convert 3D-FRONT JSON scene descriptions into 3D mesh files (GLB format).

**What it does:**
- Loads 3D-FRONT scene JSON files
- For each scene:
  - Loads furniture meshes from 3D-FUTURE dataset
  - Applies transformations (position, rotation, scale) from scene data
  - Creates architectural meshes (walls, floors) from room geometry
  - Applies category colors from taxonomy for segmentation
- Generates two versions of each scene:
  - **Textured (`tex/`)**: Original furniture textures preserved
  - **Segmented (`seg/`)**: All objects colored by category (for training)

**Input:**
- 3D-FRONT scene JSON files
- 3D-FUTURE model meshes
- Taxonomy file (for colors)

**Output:**
- `dataset/geometry/tex/{scene_id}_tex.glb` - Textured 3D scenes
- `dataset/geometry/seg/{scene_id}_seg.glb` - Segmented 3D scenes

**Key Features:**
- Preserves original furniture textures in textured version
- Applies consistent category colors in segmented version
- Handles doors and windows as distinct categories
- Exports to GLB format for efficient loading

---

### Stage 2: Metadata Compilation (`stage2_compile_metadata.py`)

**Purpose**: Extract and organize all metadata about scenes and rooms.

**What it does:**
- Processes each scene to extract:
  - **Scene-level metadata**:
    - Scene ID, type, bounding box
    - List of rooms
    - Overall statistics
  - **Room-level metadata**:
    - Room type (bedroom, living room, etc.)
    - Room bounding box (min/max coordinates)
    - List of furniture with positions
    - Doors and windows with positions
    - Room adjacency information
    - Empty room flags
- Computes spatial relationships between objects
- Organizes metadata into structured JSON files

**Input:**
- 3D-FRONT scene JSON files
- Geometry files (for validation)

**Output:**
- `dataset/metadata/scenes/{scene_id}.json` - Scene metadata
- `dataset/metadata/rooms/{scene_id}_{room_id}.json` - Room metadata

**Key Features:**
- Comprehensive metadata extraction
- Spatial relationship computation
- Empty room detection
- Door/window position tracking

---

### Stage 3: Layout Rendering (`stage3_render_layouts.py`)

**Purpose**: Generate top-down 2D layout images from 3D scenes.

**What it does:**
- For each room:
  - Loads 3D geometry (textured or segmented)
  - Renders top-down orthographic view (bird's-eye view)
  - Clips geometry to room bounding box
  - Generates 2D layout image showing:
    - Floor (dark gray)
    - Walls (light gray)
    - Furniture (colored by category)
    - Doors and windows (colored by category)
- Creates two versions:
  - **Textured layouts**: Original colors/textures
  - **Segmented layouts**: Category colors only

**Input:**
- Geometry GLB files
- Room metadata (for bounding boxes)

**Output:**
- `dataset/layouts/tex/{scene_id}_{room_id}_tex_layout.png`
- `dataset/layouts/seg/{scene_id}_{room_id}_seg_layout.png`

**Key Features:**
- Top-down orthographic projection
- Room-specific layouts (not entire scene)
- Consistent resolution (typically 256x256)
- Preserves spatial relationships in 2D

---

### Stage 4: POV Rendering (`stage4_render_povs_v2.py`)

**Purpose**: Generate first-person point-of-view images and POV-normalized layouts.

**What it does:**
- For each room, identifies entry points (doors and windows)
- For each entry point:
  - **POV Image Rendering**:
    - Positions camera 0.8m outside door/window
    - Camera height: 1.6m (eye level)
    - Field of view: 80°
    - Looks into room (toward furniture centroid)
    - Renders first-person view
  - **Layout Rotation**:
    - Rotates room layout to match camera orientation
    - Door/window appears at bottom-center of layout
    - Layout is POV-normalized (viewer's perspective)
  - **POV Graph Generation**:
    - Creates spatial graph from viewer's perspective
    - Uses POV-relative directions ("ahead", "to your left", "behind you")
    - Descriptive object naming ("the left chair", "the table ahead")

**Input:**
- Room layouts (from Stage 3)
- Room geometry
- Room metadata (doors, windows, furniture positions)

**Output:**
- `dataset/povs/tex/{scene_id}_{room_id}_{pov_id}_tex_pov.png` - POV images
- `dataset/povs/seg/{scene_id}_{room_id}_{pov_id}_seg_pov.png` - Segmented POV images
- `dataset/layouts_pov/tex/{scene_id}_{room_id}_{pov_id}_tex_layout.png` - Rotated layouts
- `dataset/layouts_pov/seg/{scene_id}_{room_id}_{pov_id}_seg_layout.png` - Rotated segmented layouts
- `dataset/povs/pov_info.json` - Camera and POV metadata

**Key Features:**
- Natural camera positioning (standing at doorway)
- POV-normalized layouts (no re-rendering, just rotation)
- Multiple POVs per room (one per door/window)
- Improved camera parameters (better FOV, height, tilt)

---

### Stage 5: Graph Generation (`stage5_build_graphs_v2.py`)

**Purpose**: Create semantic graphs and text descriptions from room layouts.

**What it does:**
- For each POV:
  - **Object Detection**: Identifies all objects in the layout
  - **Spatial Relationships**: Computes POV-relative positions
    - "ahead", "behind", "to your left", "to your right"
    - Distance relationships ("near", "far")
  - **Object Naming**: Creates descriptive names
    - Single objects: "the sofa", "the table"
    - Multiple objects: "the left chair", "the right chair", "the table ahead"
  - **Graph Structure**: Creates JSON graph with:
    - Nodes: Objects with positions and categories
    - Edges: Spatial relationships between objects
  - **Text Description**: Generates natural language description
    - Example: "You are standing at the doorway, looking into the Living Room. There is a sofa ahead of you, a coffee table in front of the sofa, and a TV on the wall to your right."

**Input:**
- POV-normalized layouts
- Room metadata (object positions, categories)

**Output:**
- `dataset/graphs/jsons/{scene_id}_{room_id}_{pov_id}_room_graph.json` - Structured graphs
- `dataset/graphs/texts/{scene_id}_{room_id}_{pov_id}_room_description.txt` - Text descriptions

**Key Features:**
- POV-relative spatial descriptions
- Natural language generation
- Descriptive object naming (no numerical suffixes)
- Handles empty rooms with appropriate descriptions

---

### Stage 6: Manifest Generation (`stage6_generate_manifests.py`)

**Purpose**: Create global index files that reference all generated data.

**What it does:**
- Scans all generated files (layouts, POVs, graphs, metadata)
- Creates two manifest files:
  - **Scenes manifest**: Index of all scenes with paths to:
    - Geometry files
    - Metadata files
    - Scene-level information
  - **Rooms manifest**: Index of all rooms with paths to:
    - Layout images
    - POV images
    - Graph files
    - Room metadata
- Organizes data for easy dataset loading

**Input:**
- All files generated in previous stages

**Output:**
- `dataset/manifests/scenes.json` - Scene index
- `dataset/manifests/rooms.json` - Room index

**Key Features:**
- Complete file path references
- Easy dataset navigation
- Supports filtering and sharding

---

## Post-Processing Stages

### Embedding Generation

**Purpose**: Generate embeddings for training.

#### `embed_for_training.py`
- Generates CLIP embeddings for:
  - Graph text descriptions → `graph_embedding_path`
  - POV images → `pov_embedding_path`
- Creates embeddings compatible with training pipeline

#### `embed_layouts_with_vae.py`
- Encodes layout images using trained VAE
- Generates latent representations → `layout_embedding_path`
- Used for diffusion model training

#### `encode_layouts.py`
- Alternative encoding method
- Pre-encodes layouts for faster training

### Dataset Cleaning

**Purpose**: Filter out low-quality samples.

#### `clean_dataset.py`
- Validates layout images:
  - Checks for sufficient content (not too much black/white)
  - Validates color palette matches taxonomy
  - Checks POV image quality (if enabled)
- Adds rejection flags to manifest
- Creates cleaned manifest subset

#### `add_rejections_to_manifest.py`
- Adds rejection information to existing manifests
- Updates sample flags (is_empty, is_rejected, etc.)

### Manifest Collection

**Purpose**: Create training-ready manifest files.

#### `collect_manifest_pov_normalized.py`
- Collects all POV-normalized samples
- Creates CSV manifest with columns:
  - `layout_path`: Path to POV-normalized layout
  - `graph_embedding_path`: Path to graph text embedding
  - `pov_embedding_path`: Path to POV image embedding
  - `sample_weight`: Training weight (for class balancing)
  - `is_empty`: Empty room flag
  - `is_rejected`: Quality rejection flag

#### `collect_manifest.py`
- Creates standard manifest (non-POV-normalized)
- For scene-level or room-level training

---

## Complete Pipeline Execution

### Prerequisites

1. **3D-FRONT Dataset**: Scene JSON files
2. **3D-FUTURE Dataset**: Furniture model meshes and `model_info.json`
3. **Configuration**: Edit `paths.yaml` with your paths

### Step-by-Step Execution

#### 1. Setup Configuration

Edit `data_preparation_v2/paths.yaml`:
```yaml
output_dataset_root: "/path/to/dataset_v2"
front3d_scenes_dir: "/path/to/3D-FRONT/scenes"
front3d_model_info: "/path/to/3D-FUTURE/model_info.json"
front3d_model_dir: "/path/to/3D-FUTURE/models"
```

#### 2. Create Taxonomy

```bash
python data_preparation_v2/stage0_build_taxonomy.py \
    --model-info /path/to/model_info.json \
    --scenes-dir /path/to/3D-FRONT/scenes \
    --out /path/to/dataset_v2/taxonomy/taxonomy.json
```

#### 3. Validate Scenes (Optional)

```bash
python data_preparation_v2/stage0_validate_scenes.py \
    --scenes-dir /path/to/3D-FRONT/scenes \
    --output valid_scenes.txt
```

#### 4. Create Shards (for HPC)

```bash
python data_preparation_v2/create_shards.py \
    --scene-list valid_scenes.txt \
    --output-dir shards/ \
    --num-shards 10
```

#### 5. Run Pipeline Stages

**On HPC (recommended):**
```bash
# Launch from Stage 1
./data_preparation_v2/hpc_scripts/launch_pipeline.sh \
    --config paths.yaml \
    --stage 1

# Or launch individual stages
sbatch data_preparation_v2/hpc_scripts/run_stage2_array.sh
sbatch data_preparation_v2/hpc_scripts/run_stage3_array.sh
sbatch data_preparation_v2/hpc_scripts/run_stage4_v2_array.sh
sbatch data_preparation_v2/hpc_scripts/run_stage5_array.sh
sbatch data_preparation_v2/hpc_scripts/run_stage6_array.sh
```

**On Local Machine:**
```bash
# Stage 1: Geometry
python data_preparation_v2/stage1_reconstruct_geometry.py \
    --config paths.yaml \
    --scene-list shard_1.txt

# Stage 2: Metadata
python data_preparation_v2/stage2_compile_metadata.py \
    --config paths.yaml \
    --scene-list shard_1.txt

# Stage 3: Layouts
python data_preparation_v2/stage3_render_layouts.py \
    --dataset-root /path/to/dataset_v2 \
    --scene-list shard_1.txt

# Stage 4: POVs
python data_preparation_v2/stage4_render_povs_v2.py \
    --dataset-root /path/to/dataset_v2 \
    --scene-list shard_1.txt

# Stage 5: Graphs
python data_preparation_v2/stage5_build_graphs_v2.py \
    --dataset-root /path/to/dataset_v2 \
    --scene-list shard_1.txt

# Stage 6: Manifests
python data_preparation_v2/stage6_generate_manifests.py \
    --dataset-root /path/to/dataset_v2 \
    --scene-list shard_1.txt
```

#### 6. Generate Embeddings

```bash
# Graph and POV embeddings
python data_preparation_v2/embed_for_training.py \
    --dataset-root /path/to/dataset_v2 \
    --manifest /path/to/dataset_v2/manifests/manifest_seg_pov_normalized.csv

# Layout embeddings (after VAE training)
python data_preparation_v2/embed_layouts_with_vae.py \
    --dataset-root /path/to/dataset_v2 \
    --vae-checkpoint /path/to/vae_checkpoint.pt \
    --manifest /path/to/manifest.csv
```

#### 7. Clean Dataset

```bash
python data_preparation_v2/clean_dataset.py \
    --dataset-root /path/to/dataset_v2 \
    --manifest /path/to/manifest.csv \
    --output cleaned_manifest.csv
```

#### 8. Collect Final Manifest

```bash
python data_preparation_v2/collect_manifest_pov_normalized.py \
    --dataset-root /path/to/dataset_v2 \
    --output-dir /path/to/dataset_v2/manifests
```

---

## Output Dataset Structure

```
dataset_v2/
├── taxonomy/
│   └── taxonomy.json                    # Color mappings
│
├── geometry/
│   ├── tex/                             # Textured 3D scenes
│   │   └── {scene_id}_tex.glb
│   └── seg/                             # Segmented 3D scenes
│       └── {scene_id}_seg.glb
│
├── metadata/
│   ├── scenes/                          # Scene metadata
│   │   └── {scene_id}.json
│   └── rooms/                           # Room metadata
│       └── {scene_id}_{room_id}.json
│
├── layouts/                             # Original room layouts
│   ├── tex/
│   │   └── {scene_id}_{room_id}_tex_layout.png
│   └── seg/
│       └── {scene_id}_{room_id}_seg_layout.png
│
├── layouts_pov/                        # POV-normalized layouts
│   ├── tex/
│   │   └── {scene_id}_{room_id}_{pov_id}_tex_layout.png
│   └── seg/
│       └── {scene_id}_{room_id}_{pov_id}_seg_layout.png
│
├── povs/                                # First-person POV images
│   ├── tex/
│   │   └── {scene_id}_{room_id}_{pov_id}_tex_pov.png
│   ├── seg/
│   │   └── {scene_id}_{room_id}_{pov_id}_seg_pov.png
│   └── pov_info.json                    # Camera metadata
│
├── graphs/                              # Semantic graphs
│   ├── jsons/
│   │   └── {scene_id}_{room_id}_{pov_id}_room_graph.json
│   └── texts/
│       └── {scene_id}_{room_id}_{pov_id}_room_description.txt
│
├── embeddings/                           # Pre-computed embeddings
│   ├── graph/                            # Graph text embeddings
│   │   └── {scene_id}_{room_id}_{pov_id}_graph_emb.pt
│   ├── pov/                              # POV image embeddings
│   │   └── {scene_id}_{room_id}_{pov_id}_pov_emb.pt
│   └── layouts/                         # Layout latent embeddings
│       └── {scene_id}_{room_id}_{pov_id}_layout_emb.pt
│
└── manifests/                            # Dataset indices
    ├── scenes.json                       # Scene manifest
    ├── rooms.json                        # Room manifest
    └── manifest_seg_pov_normalized.csv   # Training manifest
```

---

## Key Concepts

### POV-Normalized Layouts

**What**: Layout images rotated to match the viewer's perspective.

**Why**: 
- Aligns layout with first-person view
- Makes spatial relationships consistent
- Enables better conditioning in diffusion models

**How**: 
- Layout rendered once in Stage 3 (world orientation)
- Rotated in Stage 4 to match POV camera orientation
- Door/window appears at bottom-center (viewer's position)

### Segmented vs Textured

**Segmented**:
- All objects colored by category (from taxonomy)
- Consistent colors across all scenes
- Used for training (easier for model to learn)

**Textured**:
- Original furniture textures/colors preserved
- More realistic appearance
- Used for visualization and evaluation

### Graph Embeddings

**Graph Text Embeddings**: CLIP embeddings of natural language descriptions
- "You are standing at the doorway. There is a sofa ahead..."

**POV Embeddings**: CLIP embeddings of first-person POV images
- Visual representation of what the viewer sees

**Layout Embeddings**: VAE latent representations of layout images
- Compressed representation for diffusion model training

---

## Troubleshooting

### Common Issues

1. **Missing 3D-FRONT files**
   - Check `front3d_scenes_dir` path in `paths.yaml`
   - Verify scene IDs match actual filenames

2. **Rendering failures (Stage 3/4)**
   - Install `xvfbwrapper` for headless rendering: `pip install xvfbwrapper`
   - Use `--hpc` flag for headless mode
   - Check OpenGL drivers

3. **Taxonomy color mismatches**
   - Regenerate taxonomy if categories changed
   - Ensure taxonomy file is loaded correctly

4. **Memory issues**
   - Process scenes in smaller batches
   - Use sharding for parallel processing

5. **Missing embeddings**
   - Run embedding generation scripts after manifests are created
   - Ensure CLIP/VAE models are loaded correctly

---

## Performance Tips

1. **Use HPC**: Stages 1-4 are computationally intensive
2. **Sharding**: Split scenes into shards for parallel processing
3. **Caching**: Geometry and metadata are cached, so re-running is fast
4. **Selective Processing**: Use `--scene-list` to process specific scenes
5. **Skip Stages**: Later stages only need `output_dataset_root`, not 3D-FRONT sources

---

## Next Steps

After completing the data preparation pipeline:

1. **Train VAE**: Use `training/train.py` with layout images
2. **Train CLIP Projections**: Use `training/train_clip_projection.py`
3. **Train Diffusion Model**: Use `training/train_diffusion.py` with embeddings
4. **Evaluate**: Use evaluation scripts with generated layouts
5. **Inference**: Generate new layouts from text/POV conditions

See `EVALUATION_METRICS_PLAN.md` for comprehensive evaluation guidelines.

