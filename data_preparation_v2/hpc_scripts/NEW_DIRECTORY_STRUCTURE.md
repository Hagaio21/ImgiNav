# New Directory Structure for POV-Normalized Data

## Directory Layout

```
dataset_v2/
├── layouts_pov/              # POV-specific rotated layouts (from stage4_render_povs_v2)
│   ├── tex/
│   │   └── {scene}_{room}_{pov_id}_tex_layout.png
│   └── seg/
│       └── {scene}_{room}_{pov_id}_seg_layout.png
│
├── layouts_with_cam/         # Layouts with camera markers (optional debug output)
│   ├── tex/
│   │   └── {scene}_{room}_{pov_id}_tex_layout.png
│   └── seg/
│       └── {scene}_{room}_{pov_id}_seg_layout.png
│
├── povs/                     # POV images
│   ├── tex/
│   │   └── {scene}_{room}_{pov_id}_tex_pov.png
│   └── seg/
│       └── {scene}_{room}_{pov_id}_seg_pov.png
│
├── pov_graphs/               # POV-specific graphs (NEW location)
│   ├── jsons/
│   │   └── {scene}_{room}_{pov_id}_room_graph.json
│   └── texts/
│       └── {scene}_{room}_{pov_id}_room_description.txt
│
└── pov_info/                  # POV metadata shards (NEW location)
    ├── shard_0000.json
    ├── shard_0001.json
    ├── ...
    ├── shard_0499.json
    └── pov_info.json          # Merged file (after running merge script)
```

## Key Changes from Old Structure

1. **Graphs location**: `graphs/` → `pov_graphs/`
2. **POV info shards**: `povs/pov_info_shard_*.json` → `pov_info/shard_*.json`
3. **POV info merged**: `povs/pov_info.json` → `pov_info/pov_info.json`

## Scripts Available

### 1. Merge POV Info Shards

**Script**: `data_preparation_v2/merge_pov_info_shards.py`

**Purpose**: Merges all `shard_*.json` files from `pov_info/` into a single `pov_info.json`

**Usage**:
```bash
python data_preparation_v2/merge_pov_info_shards.py \
    --dataset-root /path/to/dataset_v2
```

**HPC Script**: `data_preparation_v2/hpc_scripts/run_merge_pov_info_shards.sh`
```bash
bsub < run_merge_pov_info_shards.sh
```

**Features**:
- Automatically detects new location (`pov_info/`) or falls back to old location (`povs/`)
- Supports both new pattern (`shard_*.json`) and old pattern (`pov_info_shard_*.json`)
- Optional cleanup: `--clean` flag to delete shard files after merge

### 2. Collect POV-Normalized Manifest

**Script**: `data_preparation_v2/collect_manifest_pov_normalized.py`

**Purpose**: Creates manifest CSV files with POV-normalized layouts and graphs

**Usage**:
```bash
python data_preparation_v2/collect_manifest_pov_normalized.py \
    --dataset-root /path/to/dataset_v2
```

**HPC Script**: `data_preparation_v2/hpc_scripts/run_collect_manifest_pov_normalized.sh`
```bash
bsub < run_collect_manifest_pov_normalized.sh
```

**Output**:
- `manifests/manifest_tex_pov_normalized.csv`
- `manifests/manifest_seg_pov_normalized.csv`

**Features**:
- Automatically detects new directory structure
- Falls back to old structure if new structure not found
- Handles both `pov_graphs/` and `graphs/` directories
- Creates one row per POV (not per room)

## Backward Compatibility

All scripts support both the new and old directory structures:
- **New structure**: `pov_info/shard_*.json`, `pov_graphs/`
- **Old structure**: `povs/pov_info_shard_*.json`, `graphs/`

Scripts will try the new location first, then fall back to the old location.

## Complete Pipeline

1. **Stage 4 v2**: Renders POVs, creates rotated layouts, generates graphs
   - Outputs shards to: `pov_info/shard_*.json`

2. **Merge shards**: Combine all shards into single file
   ```bash
   bsub < run_merge_pov_info_shards.sh
   ```
   - Output: `pov_info/pov_info.json`

3. **Collect manifest**: Create manifest CSV files
   ```bash
   bsub < run_collect_manifest_pov_normalized.sh
   ```
   - Output: `manifests/manifest_*_pov_normalized.csv`

4. **Embed POVs and graphs**: Create embeddings for training
   ```bash
   bsub < run_embed_pov_normalized.sh
   ```

5. **Train VAE**: Train on POV-normalized layouts
   ```bash
   bsub < training/hpc_scripts/vae/run_train_vae_clip_v2.sh
   ```

