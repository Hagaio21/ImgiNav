# Diffusion Ablation Experiments

## Purpose
Ablation studies to understand optimal UNet capacity, scheduler type, and diffusion step count for the diffusion model.

## Experiment Structure

### UNet Capacity Ablation
Testing different base_channels and depth combinations:
- **base_channels**: 64, 128, 256
- **depth**: 3, 4, 5

### Scheduler Ablation
Testing different noise schedule types:
- **LinearScheduler**: Linear noise schedule
- **CosineScheduler**: Cosine noise schedule (recommended)
- **QuadraticScheduler**: Quadratic noise schedule

### Step Count Ablation
Testing different number of diffusion steps:
- **500 steps**: Faster training, less noise granularity
- **1000 steps**: Standard (baseline)
- **2000 steps**: More noise granularity, slower training

## Configuration Notes

- **Checkpoints**: Only `best` and `latest` are saved (no periodic checkpoints to save disk space)
- **Training time**: These experiments run for a long time due to UNet size
- **Autoencoder**: Uses frozen Phase 1.5 final autoencoder (32×32×16 latent space)
- **Early stopping**: Enabled with patience=10 to prevent overfitting

## Running Experiments

See individual shell scripts in `training/hpc_scripts/` for each experiment category.

