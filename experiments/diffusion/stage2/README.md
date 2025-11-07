# Stage 2: Diffusion Viability Fine-tuning

## Overview

Stage 2 fine-tunes diffusion models trained in Stage 1 (ablation experiments) to ensure decoded layouts are semantically viable. This addresses the issue where diffusion models generate latents that decode to images with correct colors and structure but are not viable layouts.

## Approach

**Stage 1**: Learn latent distribution efficiently
- Loss: MSE on noise prediction (latent space only)
- Fast training (no decoding overhead)
- Result: Model learns to denoise latents

**Stage 2**: Ensure decoded layouts are semantically viable
- Loss: Noise prediction + Semantic losses (segmentation + perceptual)
- Direct constraints on decoded images
- Decode every batch → slower but focused
- Result: Model learns to generate semantically viable layouts

**Stage 3**: Add learned viability constraints (adversarial)
- Loss: Noise prediction + Semantic losses + Discriminator loss
- Learned constraints from discriminator (data-driven)
- Requires pre-trained discriminator
- Result: Model learns to generate layouts that pass discriminator

## Configuration

### Required Config Fields

```yaml
diffusion:
  stage1_checkpoint: /path/to/stage1/best_checkpoint.pt  # Required: Stage 1 checkpoint to load

dataset:
  outputs:
    latent: "latent_path"      # Pre-embedded latents
    rgb: "layout_path"         # RGB images for perceptual loss
    segmentation: "layout_path" # Segmentation maps for semantic loss

training:
  loss:
    noise_loss:              # Primary loss (noise prediction)
      type: MSELoss
      key: "pred_noise"
      target: "noise"
      weight: 1.0
    semantic_loss:           # Semantic constraint loss
      type: SemanticLoss
      segmentation_loss:
        type: CrossEntropyLoss
        key: "segmentation"
        target: "segmentation"
        weight: 0.1
      perceptual_loss:
        type: PerceptualLoss
        key: "rgb"
        target: "rgb"
        weight: 0.05
```

### Key Differences from Stage 1

1. **Decoder Frozen**: `autoencoder.frozen: true` - decoder stays frozen, only UNet is trained
2. **Lower Learning Rate**: Typically 3-5x lower (e.g., 0.00005 vs 0.00015)
3. **Smaller Batch Size**: 32 instead of 64 (due to decoding overhead)
4. **Fewer Epochs**: 100 instead of 500 (fine-tuning)
5. **Semantic Losses**: Segmentation + perceptual losses on decoded images

## Usage

### Training

```bash
python training/train_diffusion.py experiments/diffusion/stage2/stage2_unet128_d4.yaml
```

### Resume Training

```bash
python training/train_diffusion.py experiments/diffusion/stage2/stage2_unet128_d4.yaml --resume
```

### Start Fresh (ignore checkpoints)

```bash
python training/train_diffusion.py experiments/diffusion/stage2/stage2_unet128_d4.yaml --no-resume
```

## Creating Stage 2 Configs

To create a Stage 2 config for a Stage 1 experiment:

1. Copy `base_config.yaml` or an existing Stage 2 config
2. Update `experiment.name` and `experiment.save_path`
3. Update `diffusion.stage1_checkpoint` to point to your Stage 1 best checkpoint
4. Adjust loss weights if needed (start with defaults: seg=0.1, perc=0.05)

Example:
```yaml
experiment:
  name: "diffusion_stage2_unet256_d4"
  save_path: "/work3/s233249/ImgiNav/experiments/diffusion/stage2/stage2_unet256_d4"

diffusion:
  stage1_checkpoint: /work3/s233249/ImgiNav/experiments/diffusion/ablation/capacity_unet256_d4/capacity_unet256_d4_checkpoint_best.pt
```

## Loss Weights

Default weights:
- **Noise loss**: 1.0 (primary)
- **Segmentation loss**: 0.1 (semantic constraint)
- **Perceptual loss**: 0.05 (semantic features)

Tuning tips:
- If layouts are visually good but semantically invalid → increase segmentation weight
- If layouts are semantically valid but blurry → increase perceptual weight
- If training is unstable → reduce semantic loss weights

## Output Structure

```
experiments/diffusion/stage2/stage2_unet128_d4/
├── checkpoints/
│   ├── diffusion_stage2_unet128_d4_checkpoint_best.pt
│   └── diffusion_stage2_unet128_d4_checkpoint_latest.pt
├── samples/
│   └── diffusion_stage2_unet128_d4_epoch_*.png
└── diffusion_stage2_unet128_d4_metrics.csv
```

## Notes

- Stage 2 training is slower than Stage 1 due to decoding overhead
- Monitor both noise_loss and semantic_loss components
- Best checkpoint is saved based on validation loss
- Decoder stays frozen - only UNet is updated based on semantic losses

