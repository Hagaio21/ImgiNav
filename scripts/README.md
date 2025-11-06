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

### `migrate_dualunet_checkpoints.py`
Utility for migrating checkpoints from DualUNet to standard UNet format.

**Features:**
- Convert DualUNet checkpoints to UNet format
- Preserve model weights
- Update state dict keys
- Validation of migrated checkpoints

**Usage:**
```bash
python scripts/migrate_dualunet_checkpoints.py \
    --input_checkpoint /path/to/dualunet.pt \
    --output_checkpoint /path/to/unet.pt \
    --config /path/to/config.yaml
```

**Note:** This script is for backward compatibility when migrating from older model architectures.

### `test_checkpoint_migration.py`
Test script for validating checkpoint migration.

**Features:**
- Verify migrated checkpoints
- Compare model outputs
- Validate weight preservation
- Test forward passes

**Usage:**
```bash
python scripts/test_checkpoint_migration.py \
    --original_checkpoint /path/to/original.pt \
    --migrated_checkpoint /path/to/migrated.pt \
    --config /path/to/config.yaml
```

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

### Migrate Checkpoint

```bash
python scripts/migrate_dualunet_checkpoints.py \
    --input_checkpoint checkpoints/old_dualunet.pt \
    --output_checkpoint checkpoints/new_unet.pt \
    --config experiments/diffusion/ablation/base_config.yaml
```

### Test Migration

```bash
python scripts/test_checkpoint_migration.py \
    --original_checkpoint checkpoints/old_dualunet.pt \
    --migrated_checkpoint checkpoints/new_unet.pt \
    --config experiments/diffusion/ablation/base_config.yaml
```

## Notes

- All scripts support both CPU and GPU inference
- Checkpoint migration preserves model weights but updates architecture
- Inference scripts can be used for both unconditional and conditional generation
- Scripts include error handling and validation

