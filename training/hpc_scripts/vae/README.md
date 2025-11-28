# VAE Training HPC Scripts

This directory contains HPC job submission scripts for training VAE models.

## Scripts

### Individual Training Scripts

- **`run_train_vae_seg_256.sh`** - Train segmented layouts VAE (256×256)
- **`run_train_vae_tex_256.sh`** - Train textured layouts VAE (256×256)

### Launcher Script

- **`launch_train_vae_256.sh`** - Submit one or both training jobs

## Usage

### Submit Individual Jobs

**Segmented VAE:**
```bash
bsub < run_train_vae_seg_256.sh
```

**Textured VAE:**
```bash
bsub < run_train_vae_tex_256.sh
```

### Use Launcher Script

**Submit both jobs:**
```bash
./launch_train_vae_256.sh
# or
./launch_train_vae_256.sh both
```

**Submit only segmented:**
```bash
./launch_train_vae_256.sh seg
```

**Submit only textured:**
```bash
./launch_train_vae_256.sh tex
```

## Job Configuration

All scripts use:
- **Queue**: `gpuv100`
- **GPUs**: 1 GPU (exclusive process mode)
- **Memory**: 16GB
- **CPUs**: 4 cores
- **Time limit**: 24 hours
- **Conda environment**: `imginav` (fallback: `scenefactor`)

## Output

Training outputs are saved to:
- Segmented: `outputs/autoencoders/v2/vae_seg_256_clip/`
- Textured: `outputs/autoencoders/v2/vae_tex_256_clip/`

Logs are saved to:
- `training/hpc_scripts/logs/train_vae_seg_256.{JOB_ID}.out`
- `training/hpc_scripts/logs/train_vae_seg_256.{JOB_ID}.err`
- `training/hpc_scripts/logs/train_vae_tex_256.{JOB_ID}.out`
- `training/hpc_scripts/logs/train_vae_tex_256.{JOB_ID}.err`

## Monitoring

**Check job status:**
```bash
bjobs
```

**Monitor logs:**
```bash
tail -f training/hpc_scripts/logs/train_vae_seg_256.*.out
tail -f training/hpc_scripts/logs/train_vae_tex_256.*.out
```

**Cancel a job:**
```bash
bkill <JOB_ID>
```

## Prerequisites

Before submitting jobs, ensure:

1. **Manifest files exist:**
   - `dataset_v2/manifests/manifest_seg.csv`
   - `dataset_v2/manifests/manifest_tex.csv`

2. **Embeddings are created:**
   - Run `embed_for_training.py` first to create POV and graph embeddings

3. **Config files exist:**
   - `experiments/autoencoders/v2/vae_seg_256_clip.yaml`
   - `experiments/autoencoders/v2/vae_tex_256_clip.yaml`

## Training Details

- **Input size**: 256×256
- **Latent size**: 16×16×4 (after 4 downsampling steps)
- **Epochs**: 150
- **Batch size**: 32
- **Early stopping**: Patience 10, min delta 0.00005
- **Mixed precision**: Enabled (AMP)

Training automatically resumes from the latest checkpoint if available.

