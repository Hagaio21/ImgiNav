# Scripts Module

This module contains utility scripts for model inference, checkpoint migration, and testing.

## Overview

The scripts module provides:
- Model inference scripts
- Checkpoint migration utilities
- Testing and validation scripts

## Scripts

### `diffusion_inference.py`
Script for running inference with trained diffusion models.

**Features:**
- Load trained diffusion model
- Generate samples from noise
- Batch generation
- Save generated images
- Configurable sampling parameters

**Usage:**
```bash
python scripts/diffusion_inference.py \
    --checkpoint /path/to/diffusion_model.pt \
    --config /path/to/config.yaml \
    --num_samples 10 \
    --output_dir outputs/inference
```

**Parameters:**
- `--checkpoint`: Path to model checkpoint
- `--config`: Path to model config file
- `--num_samples`: Number of samples to generate
- `--output_dir`: Directory to save generated images
- `--batch_size`: Batch size for generation
- `--seed`: Random seed for reproducibility


## Usage Examples

### Generate Samples from Diffusion Model

```bash
# Single sample
python scripts/diffusion_inference.py \
    --checkpoint experiments/diffusion/ablation/capacity_unet128_d4/checkpoints/best.pt \
    --config experiments/diffusion/ablation/capacity_unet128_d4.yaml \
    --num_samples 1 \
    --output_dir outputs/samples

# Batch generation
python scripts/diffusion_inference.py \
    --checkpoint experiments/diffusion/ablation/capacity_unet128_d4/checkpoints/best.pt \
    --config experiments/diffusion/ablation/capacity_unet128_d4.yaml \
    --num_samples 100 \
    --batch_size 10 \
    --output_dir outputs/batch_samples
```


## Notes

- All scripts support both CPU and GPU inference
- Inference scripts can be used for both unconditional and conditional generation
- Scripts include error handling and validation

