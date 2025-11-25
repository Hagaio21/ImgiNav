# CLIP Diffusion Experiments - HPC Scripts

This directory contains HPC scripts for training and evaluating diffusion models with CLIP-aligned VAE embeddings. The scripts are organized into logical subdirectories for better maintainability.

## Directory Structure

```
training/hpc_scripts/
├── regular/              # Regular (non-spatial) CLIP VAE experiment scripts
│   ├── launch_*.sh       # Launch scripts for regular experiments
│   └── run_train_*.sh    # Direct run scripts for regular experiments
├── spatial/              # Spatial CLIP VAE experiment scripts
│   ├── launch_*.sh       # Launch scripts for spatial experiments
│   └── run_train_*.sh   # Direct run scripts for spatial experiments
├── vae/                  # VAE training scripts
│   └── run_train_vae_*.sh
├── embedding/            # Embedding/data preparation scripts
│   ├── run_embed_*.sh
│   ├── launch_embed_*.sh
│   └── fix_manifest_csv.sh
├── evaluation/           # Evaluation scripts
│   ├── eval_diff_clip_*.sh
│   └── launch_eval_diff_clip_all.sh
├── debug/                # Debug/testing scripts
│   ├── debug_diffusion.sh
│   ├── launch_debug_diffusion.sh
│   └── run_debug_*.sh
├── analysis/             # Analysis/comparison scripts
│   ├── run_compare_all_experiments.sh
│   └── launch_compare_all_experiments.sh
├── run_train_diff_clip.sh        # General training script (used by launch scripts)
└── launch_train_diff_clip.sh     # General launch script (for any config)
```

## Experiment Configurations

The experiment configs are located in `experiments/diffusion/clip/`:

```
experiments/diffusion/clip/
├── regular/          # Regular (non-spatial) CLIP VAE experiments (all types)
│   ├── small_*.yaml  # Small models (48 base_channels, depth 3)
│   ├── medium_*.yaml # Medium models (64 base_channels, depth 4)
│   └── large_*.yaml  # Large models (128 base_channels, depth 4)
├── regular_rooms/    # Regular CLIP VAE experiments (rooms only)
│   ├── small_*.yaml
│   ├── medium_*.yaml
│   └── large_*.yaml
├── regular_scenes/   # Regular CLIP VAE experiments (scenes only)
│   ├── small_*.yaml
│   ├── medium_*.yaml
│   └── large_*.yaml
├── spatial/          # Spatial CLIP VAE experiments (all types)
│   ├── small_*.yaml
│   ├── medium_*.yaml
│   └── large_*.yaml
├── spatial_rooms/    # Spatial CLIP VAE experiments (rooms only)
│   ├── small_*.yaml
│   ├── medium_*.yaml
│   └── large_*.yaml
└── spatial_scenes/   # Spatial CLIP VAE experiments (scenes only)
    ├── small_*.yaml
    ├── medium_*.yaml
    └── large_*.yaml
```

Each size has 4 attention location variants:
- `*_down.yaml` - Attention only in down path
- `*_bottleneck.yaml` - Attention only at bottleneck (or [downs, bottleneck] for rooms/scenes)
- `*_up.yaml` - Attention only in up path
- `*_all.yaml` - Attention at all locations (downs, bottleneck, ups)

## Complete Workflow

### Step 1: Train CLIP VAEs

Train both regular and spatial CLIP VAEs. These create the joint embedding space.

#### Regular CLIP VAE

```bash
bsub < training/hpc_scripts/vae/run_train_vae_clip.sh
```

**Config:** `experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml`

**Output:** `/work3/s233249/ImgiNav/experiments/clip/vae_clip/checkpoints/vae_clip_checkpoint_best.pt`

#### Spatial CLIP VAE

```bash
bsub < training/hpc_scripts/vae/run_train_vae_clip_spatial.sh
```

**Config:** `experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip_spatial.yaml`

**Output:** `/work3/s233249/ImgiNav/experiments/clip/vae_clip_spatial/checkpoints/vae_clip_spatial_checkpoint_best.pt`

### Step 2: Create Embeddings

After both VAEs are trained, embed layouts using both VAEs and save to shared latents directory.

```bash
bsub < training/hpc_scripts/embedding/run_embed_clip_vaes_shared.sh
```

**What it does:**
- Embeds layouts using regular CLIP VAE → saves to `shared_embeddings/latents/vae_clip/`
- Embeds layouts using spatial CLIP VAE → saves to `shared_embeddings/latents/vae_clip_spatial/`
- Creates/updates manifest at `shared_embeddings/manifest_with_embeddings.csv` with columns:
  - `latent_path_vae_clip` - Paths to regular CLIP VAE latents
  - `latent_path_vae_clip_spatial` - Paths to spatial CLIP VAE latents

**Input manifest:** `/work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv`

**Output manifest:** `/work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv` (updated in place)

### Step 3: Train Diffusion Models

Train diffusion models using the embedded latents. All experiments use cross-attention with CLIP embedding projections.

#### General Launch Script (Any Config)

```bash
# Launch single experiment
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/small_down.yaml

# Launch multiple experiments
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/regular/small_*.yaml
./training/hpc_scripts/launch_train_diff_clip.sh experiments/diffusion/clip/spatial/*.yaml
```

#### Regular Experiments

**Small Models (2 jobs: rooms + scenes)**
```bash
./training/hpc_scripts/regular/launch_train_diff_clip_small_gpuv100.sh    # gpuv100 queue (24h limit)
./training/hpc_scripts/regular/launch_train_diff_clip_small_gpul40s.sh    # gpul40s queue (48h limit)
```

**Medium Models (2 jobs: rooms + scenes)**
```bash
./training/hpc_scripts/regular/launch_train_diff_clip_medium_gpuv100.sh   # gpuv100 queue (24h limit)
./training/hpc_scripts/regular/launch_train_diff_clip_medium_gpul40s.sh    # gpul40s queue (48h limit)
```

**Large Models (2 jobs: rooms + scenes)**
```bash
./training/hpc_scripts/regular/launch_train_diff_clip_large_gpuv100.sh     # gpuv100 queue (24h limit)
./training/hpc_scripts/regular/launch_train_diff_clip_large_gpul40s.sh    # gpul40s queue (48h limit)
```

**Text-Only Experiments (Rooms, Down+Bottleneck Attention)**
```bash
./training/hpc_scripts/regular/launch_rooms_text_only_all_sizes.sh
```

**Full Cross-Attention Experiments**
```bash
./training/hpc_scripts/regular/launch_both_rooms_scenes_all_sizes.sh
./training/hpc_scripts/regular/launch_medium_full_cross_attention.sh
```

#### Spatial Experiments

**Small Models (2 jobs: rooms + scenes)**
```bash
./training/hpc_scripts/spatial/launch_train_diff_clip_spatial_small_gpul40s.sh
```

**Medium Models (2 jobs: rooms + scenes)**
```bash
./training/hpc_scripts/spatial/launch_train_diff_clip_spatial_medium_gpul40s.sh
```

**Large Models (2 jobs: rooms + scenes)**
```bash
./training/hpc_scripts/spatial/launch_train_diff_clip_spatial_large_gpul40s.sh
```

**Text-Only Experiments (Rooms, Down+Bottleneck Attention)**
```bash
./training/hpc_scripts/spatial/launch_rooms_text_only_all_sizes.sh
```

**Full Cross-Attention Experiments**
```bash
./training/hpc_scripts/spatial/launch_both_rooms_scenes_all_sizes.sh
```

### Step 4: Evaluate Models

After training, evaluate the models:

```bash
# Launch all evaluation jobs (6 jobs: small/medium/large x rooms/scenes)
./training/hpc_scripts/evaluation/launch_eval_diff_clip_all.sh

# Or run individual evaluations
bsub < training/hpc_scripts/evaluation/eval_diff_clip_small_rooms.sh
bsub < training/hpc_scripts/evaluation/eval_diff_clip_small_scenes.sh
# ... etc
```

### Step 5: Compare Experiments

Compare metrics across all experiments:

```bash
./training/hpc_scripts/analysis/launch_compare_all_experiments.sh
```

This generates comparison plots and summary tables in `/work3/s233249/ImgiNav/experiments/clip/comparison_summary/`

## Model Configurations

### Model Sizes

| Size | Base Channels | Depth | Attention Heads | Batch Size (Regular) | Batch Size (Rooms/Scenes) | Gradient Accumulation |
|------|---------------|-------|-----------------|---------------------|--------------------------|----------------------|
| Small | 48 | 3 | 2 | 32 | 48 | 1 |
| Medium | 64 | 4 | 4 | 16 | 48 | 1 |
| Large | 128 | 4 | 8 | 2 | 8 | 1 |

**Note:** Rooms/scenes experiments use larger batch sizes and have cross-attention at `[downs, bottleneck]` for bottleneck variants.

### Attention Locations

- **down**: Cross-attention only in down path (memory efficient)
- **bottleneck**: Cross-attention only at bottleneck (regular configs)
- **bottleneck** (rooms/scenes): Cross-attention at `[downs, bottleneck]` for better conditioning
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

## Output Locations

### VAE Checkpoints
- Regular CLIP VAE: `/work3/s233249/ImgiNav/experiments/clip/vae_clip/checkpoints/`
- Spatial CLIP VAE: `/work3/s233249/ImgiNav/experiments/clip/vae_clip_spatial/checkpoints/`

### Latents
- Regular CLIP VAE latents: `/work3/s233249/ImgiNav/experiments/shared_embeddings/latents/vae_clip/`
- Spatial CLIP VAE latents: `/work3/s233249/ImgiNav/experiments/shared_embeddings/latents/vae_clip_spatial/`

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
- **Classifier-Free Guidance**: 
  - Regular configs: CFG dropout rate 0.1, guidance scale 5.0
  - Rooms/scenes configs: CFG dropout rate 0.15, guidance scale 5.0
- **Resume Support**: Training automatically resumes from latest checkpoint
- **Shared Latents**: All experiments use the same shared latents manifest for consistency
- **Type Filtering**: Optional filtering by `type` column (room/scene) for specialized models

## Troubleshooting

### Embedding script fails
- Ensure both VAE checkpoints exist
- Check that input manifest exists: `experiments/shared_embeddings/manifest_with_embeddings.csv`
- If CSV is misaligned, use: `./training/hpc_scripts/embedding/fix_manifest_csv.sh`

### Diffusion training fails
- Verify embeddings were created successfully
- Check that `shared_embeddings/manifest_with_embeddings.csv` exists
- Ensure the correct latent column is referenced in the config (`latent_path_vae_clip` or `latent_path_vae_clip_spatial`)
- For type-filtered experiments, verify the manifest has a `type` column with values "room" or "scene"

### Out of memory errors
- Reduce batch size in the config
- Increase gradient accumulation steps
- Use a smaller model size

## GPU Queues

- **gpuv100**: 24-hour time limit, suitable for shorter experiments
- **gpul40s**: 48-hour time limit, suitable for longer training runs
- **hpc**: CPU-only queue for analysis jobs

## Notes

- Training automatically uses mixed precision (AMP) for efficiency
- Checkpoints are saved every 20 epochs
- Validation and sampling occur every 10 epochs
- Evaluation metrics: CLIP Score, FID, and mIoU (computed during validation)
- Sample generation: 16 unconditioned samples (4x4 grid) + 16 comparison samples (target vs generated)

## Script Organization

- **regular/**: Scripts for regular (non-spatial) CLIP VAE experiments
- **spatial/**: Scripts for spatial CLIP VAE experiments
- **vae/**: Scripts for training VAEs
- **embedding/**: Scripts for creating embeddings and data preparation
- **evaluation/**: Scripts for evaluating trained models
- **debug/**: Scripts for debugging and testing
- **analysis/**: Scripts for comparing and analyzing experiments
