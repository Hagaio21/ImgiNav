# Diffusion Experiments - New Structure

## Overview

The diffusion experiments have been reorganized into a hierarchical structure that makes it easier to manage and understand the experiment space.

## Structure

```
experiments/diffusion/v2/
├── seg/                    # Segmented layouts (uses manifest_seg.csv)
│   ├── rooms/              # Room-level experiments
│   │   ├── graph/          # Graph/text conditioning only
│   │   ├── povs/           # POV conditioning only
│   │   └── both/           # Both graph and POV conditioning
│   ├── scenes/             # Scene-level experiments
│   │   ├── graph/
│   │   ├── povs/
│   │   └── both/
│   └── both/               # Both rooms and scenes
│       ├── graph/
│       ├── povs/
│       └── both/
└── tex/                    # Textured layouts (uses manifest_tex.csv)
    ├── rooms/
    │   ├── graph/
    │   ├── povs/
    │   └── both/
    ├── scenes/
    │   ├── graph/
    │   ├── povs/
    │   └── both/
    └── both/
        ├── graph/
        ├── povs/
        └── both/
```

## Experiment Count

**Total: 36 experiments**

- **Variants**: 2 (seg, tex)
- **Scopes**: 3 (rooms, scenes, both)
- **Conditionings**: 3 (graph, povs, both)
- **Architectures**: 2 (small, medium)
- **Total**: 2 × 3 × 3 × 2 = 36

## Architecture Variants

All experiments use **bottleneck + down** attention configuration.

### Small Architecture
- `base_channels`: 48
- `depth`: 3
- `attention_heads`: 2
- `batch_size`: 48

### Medium Architecture
- `base_channels`: 64
- `depth`: 4
- `attention_heads`: 4
- `batch_size`: 48

## Manifest and VAE Checkpoints

**Note:** All generated configs use HPC paths since experiments are run on HPC.

### Segmented (seg)
- **Manifest**: `/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg.csv`
- **VAE Checkpoint**: `/work3/s233249/ImgiNav/checkpoints/vae_seg_checkpoint_best.pt`

### Textured (tex)
- **Manifest**: `/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_tex.csv`
- **VAE Checkpoint**: `/work3/s233249/ImgiNav/checkpoints/vae_tex_checkpoint_best.pt`

## Generating Configs

All configs are generated from a template using the `generate_diffusion_configs.py` script.

### Usage

```bash
# Generate all configs (dry-run to see what would be created)
python experiments/diffusion/clip/generate_diffusion_configs.py --dry-run

# Generate all configs (always uses HPC paths)
python experiments/diffusion/clip/generate_diffusion_configs.py

**Note:** 
- The script is located in `experiments/diffusion/clip/` but generates configs in `experiments/diffusion/v2/`.
- All generated configs use HPC paths (`/work3/s233249/ImgiNav/...`) since experiments are run on HPC.
- The `--env` parameter is kept for backward compatibility but is ignored - configs always use HPC paths.
```

### Custom Output Directory

```bash
python experiments/diffusion/clip/generate_diffusion_configs.py --output-dir /path/to/output
```

**Note:** By default, configs are generated in `experiments/diffusion/v2/`. The script is located in `experiments/diffusion/clip/` but outputs to the `v2/` directory.

## Config Naming Convention

Config files follow this naming pattern:
```
{arch_size}_down_bottleneck.yaml
```

Example: `small_down_bottleneck.yaml`, `medium_down_bottleneck.yaml`

Experiment names follow this pattern:
```
diff_clip_{variant}_{scope}_{arch_size}_down_bottleneck{_conditioning}
```

Examples:
- `diff_clip_seg_rooms_small_down_bottleneck` (both graph and POV)
- `diff_clip_seg_rooms_small_down_bottleneck_graph` (graph only)
- `diff_clip_tex_scenes_medium_down_bottleneck_povs` (POV only)

## Filtering

### Scope Filters

- **rooms**: `type: [room]`
- **scenes**: `type: [scene]`
- **both**: `type: []` (no filter)

### Conditioning

- **graph**: Only `text_emb: graph_embedding_path`
- **povs**: Only `pov_emb: pov_embedding_path`
- **both**: Both `text_emb` and `pov_emb`

## Template

The template file `template_diffusion.yaml` contains placeholders that are replaced during generation:
- `{EXPERIMENT_NAME}`: Generated experiment name
- `{SAVE_PATH}`: Output directory for experiment
- `{MANIFEST_PATH}`: Path to manifest CSV
- `{VAE_CHECKPOINT}`: Path to VAE checkpoint
- `{EMBEDDING_OUTPUTS}`: YAML for embedding outputs
- `{TYPE_FILTER}`: YAML for type filter
- `{BASE_CHANNELS}`: UNet base channels
- `{DEPTH}`: UNet depth
- `{ATTENTION_HEADS}`: Number of attention heads
- `{BATCH_SIZE}`: Training batch size

## Benefits of This Structure

1. **Clear Organization**: Easy to find experiments by variant, scope, and conditioning
2. **Shared Manifests**: All seg experiments use the same manifest, all tex experiments use the same manifest
3. **Template-Based**: Single source of truth for config structure
4. **Scalable**: Easy to add new variants, scopes, or architectures
5. **Environment-Aware**: Automatically uses correct paths for local vs HPC

## Migration from Old Structure

The old structure (`regular_rooms/`, `spatial_rooms/`, etc.) can be kept for reference or removed. The new structure replaces it with a more systematic organization.

