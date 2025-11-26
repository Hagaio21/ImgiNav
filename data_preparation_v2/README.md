# Mesh-Based Dataset Preparation Pipeline v2

This pipeline processes 3D scenes as meshes (not point clouds) to generate layouts, POVs, and graphs. It produces two geometry files per scene (segmented colors and textured), comprehensive metadata, layout images, POV images, and JSON graphs.

## Pipeline Overview

The pipeline consists of 5 stages:

1. **Stage 1**: Construct scene meshes (segmented and textured)
2. **Stage 2**: Create comprehensive metadata (geometrical information)
3. **Stage 3**: Layout creation (top-down views)
4. **Stage 4**: POV rendering (corner views for rooms)
5. **Stage 5**: Graph creation (scene and room graphs)

## Input Format

Scene JSON files with structure:
```json
{
  "uid": "...",
  "furniture": [...],
  "mesh": [...],
  "scene": {
    "room": [...]
  }
}
```

See `00de1e24-ab10-4aef-bb72-130ca18d017c.json` for an example.

## Stage 1: Build Scene Meshes

**Script**: `stage1_mesh_build_scenes.py`

**Purpose**: Construct scene meshes with segmented colors (taxonomy-based) and original textures.

**Input**:
- Scene JSON files
- Model directory (with furniture meshes)
- Model info JSON file
- Taxonomy JSON file

**Output**:
- `{scene_id}_segmented.{format}` - Mesh with category colors from taxonomy
- `{scene_id}_textured.{format}` - Mesh with original textures
- `{scene_id}_scene_metadata.json` - Initial scene metadata

**Usage**:
```bash
python data_preparation_v2/stage1_mesh_build_scenes.py \
    --scene_file /path/to/scenes.json \
    --out_dir /path/to/output \
    --model_dir /path/to/models \
    --model_info /path/to/model_info.json \
    --taxonomy /path/to/taxonomy.json \
    --format glb
```

**Options**:
- `--format`: Export format (obj, glb, ply) - default: glb
- `--per_scene_subdir`: Create subdirectory per scene (default: True)
- `--limit`: Maximum number of scenes to process

## Stage 2: Create Metadata

**Script**: `stage2_mesh_metadata.py`

**Purpose**: Extract comprehensive geometrical metadata including normals, bounding boxes, room locations, object locations, and transforms.

**Input**:
- Scene JSON files (same as Stage 1)
- Model directory and model info

**Output**:
- `{scene_id}_metadata.json` - Complete scene metadata
- `{scene_id}_room_{room_id}_metadata.json` - Per-room metadata

**Metadata Structure**:
```json
{
  "scene_id": "...",
  "up_direction": [0, 0, 1],
  "scene_bbox": {"min": [...], "max": [...]},
  "num_rooms": 5,
  "num_objects": 42,
  "rooms": [
    {
      "room_id": 0,
      "room_type": "Bedroom",
      "room_bbox": {"min": [...], "max": [...]},
      "room_location": [...],
      "floor_origin": [...],
      "up_direction": [...],
      "objects": [
        {
          "object_id": "...",
          "label": "chair",
          "label_id": 1234,
          "bbox": {"min": [...], "max": [...]},
          "location": [...],
          "transform": [[...], [...], [...], [...]]
        }
      ]
    }
  ]
}
```

**Usage**:
```bash
python data_preparation_v2/stage2_mesh_metadata.py \
    --in_dir /path/to/scenes \
    --out_dir /path/to/metadata \
    --model_dir /path/to/models \
    --model_info /path/to/model_info.json \
    --taxonomy /path/to/taxonomy.json
```

## Stage 3: Layout Creation

**Script**: `stage3_mesh_layouts.py`

**Purpose**: Create top-down layout images for scenes and rooms (segmented and textured versions).

**Input**:
- Scene JSON files
- Metadata from Stage 2
- Model directory and model info

**Output**:
- Scene layouts:
  - `{scene_id}/layouts/{scene_id}_scene_segmented.png`
  - `{scene_id}/layouts/{scene_id}_scene_textured.png`
- Room layouts:
  - `{scene_id}/rooms/{room_id}/layouts/{scene_id}_room_{room_id}_segmented.png`
  - `{scene_id}/rooms/{room_id}/layouts/{scene_id}_room_{room_id}_textured.png`

**Features**:
- Top-down orthographic view
- Scene layouts: entire scene visible with slight background margin
- Room layouts: tight margin around room bounding box
- Excludes ceiling elements
- Clips walls from top (configurable amount)

**Usage**:
```bash
python data_preparation_v2/stage3_mesh_layouts.py \
    --in_dir /path/to/scenes \
    --out_dir /path/to/output \
    --metadata_dir /path/to/metadata \
    --model_dir /path/to/models \
    --model_info /path/to/model_info.json \
    --taxonomy /path/to/taxonomy.json \
    --resolution 512 \
    --scene_margin 0.05 \
    --room_margin 0.02 \
    --clip_walls 0.2
```

**Options**:
- `--resolution`: Layout image resolution (default: 512)
- `--scene_margin`: Scene layout margin as fraction (default: 0.05)
- `--room_margin`: Room layout margin as fraction (default: 0.02)
- `--clip_walls`: Amount to clip walls from top in meters (default: 0.2)
- `--scene_only`: Only create scene layouts
- `--room_only`: Only create room layouts

## Stage 4: POV Rendering

**Script**: `stage4_mesh_povs.py`

**Purpose**: Render point-of-view images from room corners (8 images per room: 4 segmented + 4 textured).

**Input**:
- Scene JSON files
- Metadata from Stage 2
- Model directory and model info

**Output**:
- `{scene_id}/rooms/{room_id}/povs/seg/{scene_id}_room_{room_id}_pov_v{01-04}_seg.png`
- `{scene_id}/rooms/{room_id}/povs/tex/{scene_id}_room_{room_id}_pov_v{01-04}_tex.png`

**Features**:
- Camera positioned in room corners (4 corners)
- Camera looks at room center
- Perspective projection
- Configurable eye height and FOV
- Same ceiling exclusion and wall clipping as layouts

**Usage**:
```bash
python data_preparation_v2/stage4_mesh_povs.py \
    --in_dir /path/to/scenes \
    --out_dir /path/to/output \
    --metadata_dir /path/to/metadata \
    --model_dir /path/to/models \
    --model_info /path/to/model_info.json \
    --taxonomy /path/to/taxonomy.json \
    --width 1280 \
    --height 800 \
    --fov 70.0 \
    --eye_height 1.6 \
    --clip_walls 0.2
```

**Options**:
- `--width`: Image width (default: 1280)
- `--height`: Image height (default: 800)
- `--fov`: Field of view in degrees (default: 70.0)
- `--eye_height`: Eye height in meters (default: 1.6)
- `--clip_walls`: Amount to clip walls from top in meters (default: 0.2)

## Stage 5: Graph Creation

**Script**: `stage5_mesh_graphs.py`

**Purpose**: Generate JSON graphs for scenes (rooms as nodes) and rooms (objects as nodes).

**Input**:
- Metadata from Stage 2

**Output**:
- Scene graphs: `{scene_id}/{scene_id}_scene_graph.json`
- Room graphs: `{scene_id}/rooms/{room_id}/{scene_id}_room_{room_id}_graph.json`

**Scene Graph Structure**:
```json
{
  "scene_id": "...",
  "scene_center": [...],
  "nodes": [
    {
      "id": "room_0",
      "room_id": 0,
      "room_type": "Bedroom",
      "location": [...],
      "bbox": {...}
    }
  ],
  "edges": [
    {
      "room_a": "room_0",
      "room_b": "room_1",
      "distance_relation": "adjacent",
      "direction_relation": "north_of"
    }
  ]
}
```

**Room Graph Structure**:
```json
{
  "scene_id": "...",
  "room_id": 0,
  "room_center": [...],
  "nodes": [
    {
      "id": "obj_...",
      "object_id": "...",
      "label": "chair",
      "label_id": 1234,
      "location": [...],
      "bbox": {...}
    }
  ],
  "edges": [
    {
      "obj_a": "obj_...",
      "obj_b": "obj_...",
      "distance_relation": "near",
      "direction_relation": "left_of"
    }
  ]
}
```

**Usage**:
```bash
python data_preparation_v2/stage5_mesh_graphs.py \
    --metadata_dir /path/to/metadata \
    --out_dir /path/to/output \
    --taxonomy /path/to/taxonomy.json \
    --adjacency_threshold 0.3 \
    --proximity_threshold 0.5
```

**Options**:
- `--adjacency_threshold`: Threshold for room adjacency in meters (default: 0.3)
- `--proximity_threshold`: Threshold for object proximity in meters (default: 0.5)

## Dependencies

- **Open3D**: For mesh rendering (layouts and POVs)
- **Trimesh**: For mesh processing
- **NumPy**: For numerical operations
- **OpenCV**: For image I/O
- **SciPy**: For spatial transformations
- **Common utilities**: `common/taxonomy.py`, `common/utils.py`
- **Data preparation utilities**: `data_preparation/utils/`

## Output Structure

```
output/
├── {scene_id}/
│   ├── {scene_id}_segmented.glb
│   ├── {scene_id}_textured.glb
│   ├── {scene_id}_metadata.json
│   ├── {scene_id}_scene_graph.json
│   ├── {scene_id}_room_{room_id}_metadata.json
│   ├── layouts/
│   │   ├── {scene_id}_scene_segmented.png
│   │   └── {scene_id}_scene_textured.png
│   └── rooms/
│       └── {room_id}/
│           ├── {scene_id}_room_{room_id}_graph.json
│           ├── layouts/
│           │   ├── {scene_id}_room_{room_id}_segmented.png
│           │   └── {scene_id}_room_{room_id}_textured.png
│           └── povs/
│               ├── seg/
│               │   └── {scene_id}_room_{room_id}_pov_v{01-04}_seg.png
│               └── tex/
│                   └── {scene_id}_room_{room_id}_pov_v{01-04}_tex.png
```

## Running the Full Pipeline

```bash
# Stage 1: Build meshes
python data_preparation_v2/stage1_mesh_build_scenes.py \
    --scene_file scenes.json --out_dir output --model_dir models \
    --model_info model_info.json --taxonomy taxonomy.json --format glb

# Stage 2: Create metadata
python data_preparation_v2/stage2_mesh_metadata.py \
    --in_dir scenes --out_dir output --model_dir models \
    --model_info model_info.json --taxonomy taxonomy.json

# Stage 3: Create layouts
python data_preparation_v2/stage3_mesh_layouts.py \
    --in_dir scenes --out_dir output --metadata_dir output \
    --model_dir models --model_info model_info.json --taxonomy taxonomy.json

# Stage 4: Create POVs
python data_preparation_v2/stage4_mesh_povs.py \
    --in_dir scenes --out_dir output --metadata_dir output \
    --model_dir models --model_info model_info.json --taxonomy taxonomy.json

# Stage 5: Create graphs
python data_preparation_v2/stage5_mesh_graphs.py \
    --metadata_dir output --out_dir output --taxonomy taxonomy.json
```

## Notes

- Each stage can be run independently if previous stages are complete
- Progress tracking and error handling are built-in
- Failed scenes are logged to CSV manifests
- The pipeline uses Open3D for rendering, which requires a display server (use Xvfb on headless systems)

