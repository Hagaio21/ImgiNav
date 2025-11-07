# Discriminator Training Guide

> **Note**: The discriminator is now used in **Stage 3** (adversarial training), not Stage 2. 
> Stage 2 uses semantic losses (segmentation + perceptual) only.
> 
> For the complete discriminator workflow, see: `experiments/diffusion/stage3/README.md`

## Overview

The discriminator is a latent-space classifier that distinguishes between viable (real) and non-viable (fake) layout latents. It's used as an adversarial loss in **Stage 3** training to ensure generated layouts are viable.

## Quick Start

### Step 1: Create Discriminator Dataset

Create the complete discriminator dataset in one step:

```bash
python scripts/create_discriminator_dataset.py \
    --manifest /path/to/augmented/manifest.csv \
    --autoencoder_checkpoint /path/to/autoencoder/best.pt \
    --diffusion_checkpoint /path/to/stage1_or_stage2/best_checkpoint.pt \
    --diffusion_config /path/to/stage1_or_stage2/config.yaml \
    --output_dir /path/to/discriminator_dataset \
    --num_samples 5000
```

This creates:
- Real latents from 5000 non-augmented images
- Bad latents from diffusion model
- Manifest with labels

### Step 2: Train Discriminator

```bash
python training/train_discriminator.py \
    --real_latents /path/to/discriminator_dataset/real_latents_all.pt \
    --fake_latents /path/to/discriminator_dataset/bad_latents_all.pt \
    --output_dir /path/to/discriminator_output
```

### Step 3: Use in Stage 3 Training

Add discriminator to Stage 3 config and train:

```yaml
discriminator:
  checkpoint: /path/to/discriminator_output/discriminator_best.pt
  weight: 0.1
```

```bash
python training/train_diffusion.py experiments/diffusion/stage3/stage3_unet128_d4.yaml
```

## Discriminator Architecture

- **Input**: Latents [B, 16, 32, 32]
- **Output**: Viability score [B, 1] in [0, 1]
  - 1.0 = Highly viable
  - 0.0 = Non-viable
- **Architecture**: CNN with 4 downsampling layers
- **Parameters**: ~100K-500K (depending on base_channels)

## Loss Function

The discriminator loss in Stage 3 training:

```python
viability_score = discriminator(latents)  # [0, 1]
discriminator_loss = -log(viability_score.mean()) * weight
```

This pushes the model to generate latents with high viability scores.

## Monitoring

During Stage 3 training, monitor:
- `discriminator_loss`: Should decrease over time
- `viability_score`: Should increase over time (toward 1.0)

If viability_score > 0.8 consistently, consider:
- Regenerating bad layouts from current model
- Retraining discriminator with new bad layouts

## Iterative Training (Optional)

For best results, you can iteratively improve:

1. **Round 1**: Create dataset from Stage 1 → Train discriminator → Train Stage 3
2. **Round 2**: Create dataset from Stage 3 (epoch 25) → Retrain discriminator → Continue Stage 3
3. **Round 3**: Create dataset from Stage 3 (epoch 50) → Retrain discriminator → Continue Stage 3

This keeps the discriminator challenging as the model improves.

## Tips

1. **Start small**: Use 0.1 weight initially, increase if needed
2. **Monitor scores**: If viability_score > 0.9, discriminator might be too easy
3. **Balance losses**: Ensure discriminator_loss doesn't dominate
4. **Quality over quantity**: 5,000 samples is usually enough
5. **Diversity**: Generate bad layouts with different random seeds

## Troubleshooting

**Discriminator loss too high:**
- Reduce discriminator weight (e.g., 0.05 instead of 0.1)
- Check if discriminator is too strong

**Viability score not improving:**
- Discriminator might be too weak
- Regenerate bad layouts and retrain discriminator
- Increase discriminator weight slightly

**Training unstable:**
- Reduce discriminator weight
- Ensure discriminator is frozen (requires_grad=False)

