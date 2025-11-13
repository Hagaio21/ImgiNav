# Stage 2: Discriminator Training (NEW)

## Overview

This is the **new Stage 2** that replaces the old Stage 2 which wasn't working well. The pipeline uses adversarial training with a discriminator to improve generation quality.

**For detailed information about the adversarial loop structure, see [ADVERSARIAL_LOOP.md](./ADVERSARIAL_LOOP.md)**

## Pipeline

The training pipeline consists of iterative adversarial training:

1. **Load Stage 1 checkpoint** - Load a pre-trained diffusion model
2. **Generate fake samples** - Generate N latents from the current model
3. **Get real samples** - Sample N real images from dataset and encode to latents
4. **Train discriminator** - Train discriminator to distinguish real vs fake latents
5. **Train diffusion model** - Train diffusion model with discriminator loss (adversarial training)
6. **Iterate** - Repeat steps 2-5 for multiple iterations

## Usage

### Basic Training

```bash
python training/train_stage2_discriminator.py experiments/diffusion/stage2_discriminator/stage2_discriminator_unet64_d4.yaml
```

### With Custom Parameters

```bash
python training/train_stage2_discriminator.py \
    experiments/diffusion/stage2_discriminator/stage2_discriminator_unet64_d4.yaml \
    --num_samples 5000 \
    --num_iterations 3 \
    --discriminator_epochs 50 \
    --diffusion_epochs 50 \
    --generation_batch_size 32 \
    --generation_steps 100
```

### Resume Training

To resume from a specific iteration:

```bash
python training/train_stage2_discriminator.py \
    experiments/diffusion/stage2_discriminator/stage2_discriminator_unet64_d4.yaml \
    --start_iteration 1
```

## Configuration

### Required Config Fields

```yaml
experiment:
  name: "diffusion_stage2_discriminator_unet64_d4"
  save_path: "/path/to/output"

diffusion:
  stage1_checkpoint: /path/to/stage1/best_checkpoint.pt  # Required

autoencoder:
  checkpoint: /path/to/autoencoder/best.pt  # Required for encoding real images
  frozen: true

discriminator:
  config:
    latent_channels: 16
    base_channels: 64
    num_layers: 4

dataset:
  manifest: "/path/to/manifest.csv"  # Required
  outputs:
    latent: "latent_path"

training:
  loss:
    type: CompositeLoss
    losses:
      - type: MSELoss
        key: "pred_noise"
        target: "noise"
        weight: 1.0
      - type: DiscriminatorLoss
        key: "latent"
        weight: 0.1  # Adjust as needed
        target: null
```

## Command Line Arguments

- `--num_samples`: Number of real and fake samples per iteration (default: 5000)
- `--num_iterations`: Number of adversarial training iterations (default: 3)
- `--discriminator_epochs`: Epochs to train discriminator per iteration (default: 50)
- `--diffusion_epochs`: Epochs to train diffusion model per iteration (default: 50)
- `--generation_batch_size`: Batch size for generating samples (default: 32)
- `--generation_steps`: Number of DDIM steps for generation (default: 100)
- `--device`: Device to use (default: cuda)
- `--seed`: Random seed (default: 42)
- `--start_iteration`: Start from this iteration (for resuming, default: 0)

## Output Structure

```
output_dir/
├── discriminator_iter_0/
│   ├── real_latents.pt
│   ├── fake_latents.pt
│   └── discriminator_history.csv
├── discriminator_iter_1/
│   └── ...
├── checkpoints/
│   ├── {exp_name}_iter_0_checkpoint_best.pt
│   ├── {exp_name}_iter_0_checkpoint_latest.pt
│   ├── {exp_name}_iter_1_checkpoint_best.pt
│   └── ...
└── {exp_name}_metrics_iter_{N}.csv
```

## Models Under 60M Parameters

The following models are suitable for this training:

- `diffusion_ablation_capacity_unet64_d4_attn` (52.04M)
- `diffusion_ablation_capacity_unet64_d4` (45.39M) ✓
- `diffusion_ablation_capacity_unet128_d3` (44.79M)
- `diffusion_ablation_capacity_unet48_d4` (25.67M) ✓
- `diffusion_ablation_scheduler_linear` (25.67M)
- `diffusion_ablation_capacity_unet48_d3` (6.49M)
- `diffusion_ablation_capacity_unet32_d3_attn` (3.39M)
- `diffusion_ablation_capacity_unet32_d3` (2.97M) ✓

Example configs are provided for the marked models (✓).

## Tips

1. **Start with fewer iterations** (2-3) to test the pipeline
2. **Adjust discriminator weight** (0.05-0.2) based on training stability
3. **Monitor discriminator accuracy** - should be around 0.5-0.7 (not too high, not too low)
4. **Use smaller batch sizes** for larger models to avoid OOM errors
5. **Check generated samples** after each iteration to see improvement

## Differences from Old Stage 2

- **Old Stage 2**: Used semantic losses (segmentation + perceptual) on decoded images
- **New Stage 2**: Uses adversarial training with discriminator in latent space
- **Old Stage 2**: Single training pass
- **New Stage 2**: Iterative adversarial training (multiple iterations)

