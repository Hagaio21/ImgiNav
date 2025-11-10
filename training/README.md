# Training Module

This module contains training scripts and utilities for training autoencoders and diffusion models.

## Overview

The training module provides:
- Training scripts for autoencoders (`train.py`)
- Training scripts for diffusion models (`train_diffusion.py`)
- Training scripts for ControlNet (`train_controlnet.py`)
- Training utilities and helpers

## Files

### `train.py`
Main training script for autoencoder experiments.

**Features:**
- Config-based training (YAML experiment configs)
- Multi-head loss support (RGB, Segmentation, Classification)
- Early stopping
- Checkpoint saving
- Sample generation during training
- Metrics logging (CSV)

**Usage:**
```bash
python training/train.py --config experiments/autoencoders/phase1/phase1_1_AE_S1_ch16_ds4.yaml
```

**Key Functionality:**
- Loads experiment config from YAML
- Builds model, dataset, loss, and optimizer
- Trains for specified epochs
- Saves checkpoints and metrics
- Generates sample reconstructions

### `train_diffusion.py`
Training script for diffusion models.

**Features:**
- Pre-embedded latent support (faster training)
- Noise prediction training
- Multiple scheduler support
- Weighted sampling for class balancing
- Gradient clipping for stability
- Mixed precision training (AMP)

**Usage:**
```bash
python training/train_diffusion.py --config experiments/diffusion/ablation/capacity_unet128_d4.yaml
```

**Key Functionality:**
- Loads frozen autoencoder decoder
- Trains UNet on pre-embedded latents
- Generates samples for evaluation
- Saves only best and latest checkpoints

### `train_controlnet.py`
Training script for ControlNet-based conditional diffusion.

**Features:**
- Conditional generation training
- Control signal injection
- Multiple fusion modes

### `utils.py`
Training utilities and helper functions.

**Key Functions:**
- `set_deterministic(seed)` - Set random seeds for reproducibility
- `load_config(path)` - Load YAML config files
- `build_model(config)` - Build model from config
- `build_dataset(config)` - Build dataset from config
- `build_loss(config)` - Build loss function from config
- `build_optimizer(model, config)` - Build optimizer from config
- `get_device()` - Get available device (CUDA/CPU)

**Usage:**
```python
from training.utils import load_config, build_model, build_dataset

config = load_config("experiment.yaml")
model = build_model(config)
dataset = build_dataset(config)
```

### `memorization_utils.py`
Utilities for detecting memorization in diffusion models.

**Features:**
- Compares generated samples with training data
- Computes similarity metrics
- Detects exact or near-exact matches
- Helps prevent overfitting

**Key Functions:**
- `check_memorization(model, dataloader, num_generate, num_training)` - Check for memorization
- `compute_sample_similarity(samples, training_data)` - Compute similarity metrics

## HPC Scripts

The `hpc_scripts/` directory contains shell scripts for running experiments on HPC clusters.

### Autoencoder Scripts
- `launch_phase1_1.sh` - Launch Phase 1.1 experiments
- `launch_phase1_2_and_1_3.sh` - Launch Phase 1.2 and 1.3 experiments (combined)
- `run_phase1_X_*.sh` - Individual experiment scripts

### Diffusion Scripts
- `launch_diffusion_ablation_all.sh` - Launch all diffusion ablations
- `launch_diffusion_ablation_selected.sh` - Launch selected diffusion ablations
- `launch_diffusion_ablation_attention.sh` - Launch attention ablation experiments
- `run_diffusion_capacity_*.sh` - Individual capacity ablation scripts
- `launch_stage2_*.sh` - Launch Stage 2 fine-tuning experiments
- `launch_stage3_*.sh` - Launch Stage 3 experiments

### Utility Scripts
- `preembed_latents_phase1_6.sh` - Pre-embed latents for faster diffusion training (Phase 1.6)

## Training Workflow

### Autoencoder Training

1. **Prepare Config**: Create YAML config in `experiments/autoencoders/`
2. **Launch Training**: Run training script with config
3. **Monitor**: Check logs and metrics CSV
4. **Evaluate**: Review sample reconstructions

**Example:**
```bash
# On HPC
bsub < training/hpc_scripts/run_phase1_1_v100.sh

# Or directly
python training/train.py --config experiments/autoencoders/phase1/phase1_1_AE_S1_ch16_ds4.yaml
```

### Diffusion Training

1. **Pre-embed Latents** (optional but recommended):
   ```bash
   bash training/hpc_scripts/preembed_latents_phase1_6.sh
   ```

2. **Prepare Config**: Create YAML config in `experiments/diffusion/ablation/`

3. **Launch Training**:
   ```bash
   python training/train_diffusion.py --config experiments/diffusion/ablation/capacity_unet128_d4.yaml
   ```

4. **Monitor**: Check logs and sample generations

## Output Structure

### Autoencoder Training Outputs
```
experiments/phase1/phase1_1_AE_S1_ch16_ds4/
├── checkpoints/
│   ├── phase1_1_AE_S1_ch16_ds4_checkpoint_best.pt
│   └── phase1_1_AE_S1_ch16_ds4_checkpoint_latest.pt
├── samples/
│   └── epoch_*.png
└── phase1_1_AE_S1_ch16_ds4_metrics.csv
```

### Diffusion Training Outputs
```
experiments/diffusion/ablation/capacity_unet128_d4/
├── checkpoints/
│   ├── diffusion_ablation_capacity_unet128_d4_checkpoint_best.pt
│   └── diffusion_ablation_capacity_unet128_d4_checkpoint_latest.pt
├── samples/
│   └── epoch_*.png
└── diffusion_ablation_capacity_unet128_d4_metrics.csv
```

## Key Training Parameters

### Autoencoder
- **Batch Size**: Typically 16
- **Learning Rate**: 0.0001
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Epochs**: 10-50 (depending on phase)

### Diffusion
- **Batch Size**: 32-64 (depending on model size)
- **Learning Rate**: 0.0001-0.00015 (lower for larger models)
- **Optimizer**: AdamW
- **Weight Decay**: 0.1
- **Epochs**: 500
- **Gradient Clipping**: 0.1-0.5 (more aggressive for larger models)

## Best Practices

1. **Reproducibility**: Always set seed=42
2. **Early Stopping**: Use for autoencoder training to prevent overfitting
3. **Checkpoint Saving**: Save best and latest (not all epochs for diffusion)
4. **Weighted Sampling**: Use for imbalanced datasets
5. **Gradient Clipping**: Essential for large diffusion models

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (AMP)

### Training Instability
- Reduce learning rate
- Increase gradient clipping
- Check for NaN values in loss

### Poor Convergence
- Check data loading
- Verify loss weights
- Review learning rate schedule

