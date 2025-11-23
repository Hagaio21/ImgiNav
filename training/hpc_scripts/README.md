# CLIP Diffusion Experiments

This directory contains diffusion experiment configurations for training diffusion models with CLIP-aligned VAE embeddings. The experiments use cross-attention with CLIP embedding projections to condition the diffusion process.

## Overview

The workflow consists of three main steps:

1. **Train CLIP VAEs** - Train regular and spatial CLIP VAEs
2. **Create Embeddings** - Embed layouts using the trained VAEs
3. **Train Diffusion Models** - Train diffusion models with various configurations

## Directory Structure

```
experiments/diffusion/clip/
├── regular/          # Regular (non-spatial) CLIP VAE experiments (all types)
│   ├── small_*.yaml  # Small models (48 base_channels, depth 3)
│   ├── medium_*.yaml # Medium models (64 base_channels, depth 4)
│   └── large_*.yaml  # Large models (128 base_channels, depth 4)
├── regular_rooms/    # Regular CLIP VAE experiments (rooms only)
│   └── small_*.yaml  # Small models with type filter: room
├── regular_scenes/   # Regular CLIP VAE experiments (scenes only)
│   └── small_*.yaml  # Small models with type filter: scene
├── spatial/          # Spatial CLIP VAE experiments (all types)
│   ├── small_*.yaml
│   ├── medium_*.yaml
│   └── large_*.yaml
├── spatial_rooms/    # Spatial CLIP VAE experiments (rooms only)
│   └── small_*.yaml  # Small models with type filter: room
└── spatial_scenes/   # Spatial CLIP VAE experiments (scenes only)
    └── small_*.yaml  # Small models with type filter: scene
```

Each size has 4 attention location variants:
- `*_down.yaml` - Attention only in down path
- `*_bottleneck.yaml` - Attention only at bottleneck
- `*_up.yaml` - Attention only in up path
- `*_all.yaml` - Attention at all locations (downs, bottleneck, ups)

## Step 1: Train CLIP VAEs

Train both regular and spatial CLIP VAEs. These create the joint embedding space.

### Regular CLIP VAE

```bash
# Launch training
bsub < training/hpc_scripts/launch_train_vae_clip.sh

# Or run directly
bsub < training/hpc_scripts/run_train_vae_clip.sh
```

**Config:** `experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml`

**Output:** Checkpoint saved to `/work3/s233249/ImgiNav/experiments/clip/vae_clip/checkpoints/vae_clip_checkpoint_best.pt`

### Spatial CLIP VAE

```bash
# Launch training
bsub < training/hpc_scripts/launch_train_vae_clip_spatial.sh

# Or run directly
bsub < training/hpc_scripts/run_train_vae_clip_spatial.sh
```

**Config:** `experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip_spatial.yaml`

**Output:** Checkpoint saved to `/work3/s233249/ImgiNav/experiments/clip/vae_clip_spatial/checkpoints/vae_clip_spatial_checkpoint_best.pt`

## Step 2: Create Embeddings

After both VAEs are trained, embed layouts using both VAEs and save to shared latents directory.

```bash
# Embed layouts with both CLIP VAEs
bsub < training/hpc_scripts/run_embed_clip_vaes_shared.sh
```

**What it does:**
- Embeds layouts using regular CLIP VAE → saves to `shared_latents/latents/vae_clip/`
- Embeds layouts using spatial CLIP VAE → saves to `shared_latents/latents/vae_clip_spatial/`
- Creates/updates manifest at `shared_latents/manifest_with_latents.csv` with columns:
  - `latent_path_vae_clip` - Paths to regular CLIP VAE latents
  - `latent_path_vae_clip_spatial` - Paths to spatial CLIP VAE latents

**Input manifest:** `/work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv`

**Output manifest:** `/work3/s233249/ImgiNav/experiments/shared_latents/manifest_with_latents.csv`

## Step 3: Train Diffusion Models

Train diffusion models using the embedded latents. All experiments use cross-attention with CLIP embedding projections.

### Single Experiment

```bash
# Launch single experiment
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/small_down.yaml
```

### Multiple Experiments

```bash
# Launch all small regular experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/small_*.yaml

# Launch all medium regular experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/medium_*.yaml

# Launch all large regular experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/large_*.yaml

# Launch all regular CLIP experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/*.yaml

# Launch all spatial CLIP experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/spatial/*.yaml
```

### Type-Filtered Experiments

Type-filtered experiments train on only rooms or only scenes:

```bash
# Launch small_down for both rooms and scenes (regular CLIP)
./training/hpc_scripts/launch_train_diff_clip_small_down.sh

# Launch all regular rooms experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular_rooms/*.yaml

# Launch all regular scenes experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular_scenes/*.yaml

# Launch all spatial rooms experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/spatial_rooms/*.yaml

# Launch all spatial scenes experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/spatial_scenes/*.yaml
```

### Direct Run (without launch script)

```bash
# Run single experiment directly
bsub < training/hpc_scripts/run_train_diff_clip.sh experiments/diffusion/clip/regular/small_down.yaml
```

## Experiment Configurations

### Model Sizes

| Size | Base Channels | Depth | Attention Heads | Batch Size | Gradient Accumulation |
|------|---------------|-------|-----------------|------------|----------------------|
| Small | 48 | 3 | 2 | 4 | 1 |
| Medium | 64 | 4 | 4 | 2 | 2 |
| Large | 128 | 4 | 8 | 1 | 4 |

### Attention Locations

- **down**: Cross-attention only in down path (memory efficient)
- **bottleneck**: Cross-attention only at bottleneck
- **up**: Cross-attention only in up path
- **all**: Cross-attention at all locations (downs, bottleneck, ups)

### VAE Types

- **regular**: Uses non-spatial CLIP VAE (`latent_path_vae_clip` column)
- **spatial**: Uses spatial CLIP VAE (`latent_path_vae_clip_spatial` column)

### Type Filtering

- **regular_rooms**: Regular CLIP VAE, filtered to rooms only (`type: [room]`)
- **regular_scenes**: Regular CLIP VAE, filtered to scenes only (`type: [scene]`)
- **spatial_rooms**: Spatial CLIP VAE, filtered to rooms only (`type: [room]`)
- **spatial_scenes**: Spatial CLIP VAE, filtered to scenes only (`type: [scene]`)

Type filtering is done at the dataset level using the `type` column in the manifest. This allows training separate models for rooms vs scenes to compare performance.

## Complete Workflow Example

```bash
# 1. Train regular CLIP VAE
bsub < training/hpc_scripts/launch_train_vae_clip.sh

# 2. Train spatial CLIP VAE (can run in parallel)
bsub < training/hpc_scripts/launch_train_vae_clip_spatial.sh

# 3. Wait for both VAEs to finish, then create embeddings
bsub < training/hpc_scripts/run_embed_clip_vaes_shared.sh

# 4. Train diffusion models (example: all small regular experiments)
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/small_*.yaml
```

## Output Locations

### VAE Checkpoints
- Regular CLIP VAE: `/work3/s233249/ImgiNav/experiments/clip/vae_clip/checkpoints/`
- Spatial CLIP VAE: `/work3/s233249/ImgiNav/experiments/clip/vae_clip_spatial/checkpoints/`

### Latents
- Regular CLIP VAE latents: `/work3/s233249/ImgiNav/experiments/shared_latents/latents/vae_clip/`
- Spatial CLIP VAE latents: `/work3/s233249/ImgiNav/experiments/shared_latents/latents/vae_clip_spatial/`

### Diffusion Model Checkpoints
Each experiment saves to its own directory:
- Example: `/work3/s233249/ImgiNav/experiments/clip/diff_clip_regular_small_down/checkpoints/`

## Monitoring Jobs

```bash
# Check job status
bjobs

# Check specific job
bjobs <job_id>

# View logs
tail -f training/hpc_scripts/logs/train_diff_clip_*.out
tail -f training/hpc_scripts/logs/train_diff_clip_*.err
```

## Experiment Naming Convention

Experiments follow the pattern: `diff_clip_{vae_type}_{filter_type}_{size}_{attention_location}`

Where:
- `vae_type`: `regular` or `spatial`
- `filter_type`: `rooms` or `scenes` (only for type-filtered experiments, omitted for all-types)
- `size`: `small`, `medium`, or `large`
- `attention_location`: `down`, `bottleneck`, `up`, or `all`

Examples:
- `diff_clip_regular_small_down` - Regular CLIP, all types, small model, attention in down path
- `diff_clip_regular_rooms_small_down` - Regular CLIP, rooms only, small model, attention in down path
- `diff_clip_regular_scenes_small_down` - Regular CLIP, scenes only, small model, attention in down path
- `diff_clip_spatial_medium_all` - Spatial CLIP, all types, medium model, attention at all locations
- `diff_clip_spatial_rooms_small_bottleneck` - Spatial CLIP, rooms only, small model, attention at bottleneck

## Key Features

- **Cross-Attention**: All models use cross-attention with CLIP embedding projections
- **CLIP Alignment**: Embeddings are projected using CLIP projections from the VAE checkpoint
- **Classifier-Free Guidance**: CFG dropout rate 0.1, guidance scale 3.0
- **Resume Support**: Training automatically resumes from latest checkpoint
- **Shared Latents**: All experiments use the same shared latents manifest for consistency
- **Type Filtering**: Optional filtering by `type` column (room/scene) for specialized models

## Troubleshooting

### Embedding script fails
- Ensure both VAE checkpoints exist
- Check that input manifest exists: `experiments/shared_embeddings/manifest_with_embeddings.csv`

### Diffusion training fails
- Verify embeddings were created successfully
- Check that `shared_latents/manifest_with_latents.csv` exists
- Ensure the correct latent column is referenced in the config (`latent_path_vae_clip` or `latent_path_vae_clip_spatial`)
- For type-filtered experiments, verify the manifest has a `type` column with values "room" or "scene"

### Out of memory errors
- Reduce batch size in the config
- Increase gradient accumulation steps
- Use a smaller model size

## Notes

- All experiments use 24-hour time limit (gpuv100 queue limit)
- Training automatically uses mixed precision (AMP) for efficiency
- Checkpoints are saved every 20 epochs
- Validation and sampling occur every 10 epochs

