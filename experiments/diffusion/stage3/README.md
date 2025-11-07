# Stage 3: Diffusion Adversarial Fine-tuning

## Overview

Stage 3 fine-tunes diffusion models trained in Stage 2 with adversarial training using a discriminator. This adds learned viability constraints on top of Stage 2's direct semantic constraints (segmentation + perceptual losses).

## Approach

**Stage 1**: Learn latent distribution efficiently
- Loss: MSE on noise prediction (latent space only)
- Fast training (no decoding overhead)
- Result: Model learns to denoise latents

**Stage 2**: Ensure decoded layouts are semantically viable
- Loss: Noise prediction + Semantic losses (segmentation + perceptual)
- Direct constraints on decoded images
- Result: Model learns to generate semantically viable layouts

**Stage 3**: Add learned viability constraints (adversarial)
- Loss: Noise prediction + Semantic losses + Discriminator loss
- Learned constraints from discriminator (data-driven)
- Result: Model learns to generate layouts that pass discriminator (learned viability)

## Workflow

### Prerequisites

1. **Stage 1 checkpoint**: Diffusion model trained on noise prediction
2. **Stage 2 checkpoint**: Diffusion model fine-tuned with semantic losses
3. **Discriminator**: Trained to distinguish viable vs non-viable layouts

### Step 1: Create Discriminator Dataset

Create the discriminator dataset (real latents + bad latents + manifest) in one step:

```bash
python scripts/create_discriminator_dataset.py \
    --manifest /path/to/augmented/manifest.csv \
    --autoencoder_checkpoint /path/to/autoencoder/best.pt \
    --diffusion_checkpoint /path/to/stage1_or_stage2/best_checkpoint.pt \
    --diffusion_config /path/to/stage1_or_stage2/config.yaml \
    --output_dir /path/to/discriminator_dataset \
    --num_samples 5000 \
    --batch_size 32 \
    --num_steps 100
```

This script:
- Selects 5000 real (non-augmented) images from manifest
- Encodes them to get real latents
- Generates 5000 bad latents from diffusion model
- Creates a manifest with good/bad labels

**Note**: You can use Stage 1 checkpoint (easier to distinguish) or Stage 2 checkpoint (harder, better discriminator).

Output:
- `discriminator_dataset/real_latents/` - Individual real latent files
- `discriminator_dataset/real_latents_all.pt` - All real latents in one file
- `discriminator_dataset/bad_latents/` - Individual bad latent files
- `discriminator_dataset/bad_latents_all.pt` - All bad latents in one file
- `discriminator_dataset/discriminator_manifest.csv` - Manifest with labels

### Step 2: Train Discriminator

Train the discriminator to distinguish real vs fake latents:

```bash
python training/train_discriminator.py \
    --real_latents /path/to/discriminator_dataset/real_latents_all.pt \
    --fake_latents /path/to/discriminator_dataset/bad_latents_all.pt \
    --output_dir /path/to/discriminator_output \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.0002
```

This creates:
- `discriminator_output/discriminator_best.pt` - Best checkpoint
- `discriminator_output/discriminator_history.csv` - Training history

### Step 3: Train Stage 3

Add discriminator to Stage 3 config and train:

```yaml
discriminator:
  checkpoint: /path/to/discriminator_output/discriminator_best.pt
  weight: 0.1  # Start with 0.1, adjust as needed
```

```bash
python training/train_diffusion_stage3.py experiments/diffusion/stage3/stage3_unet128_d4.yaml
```

## Configuration

### Required Config Fields

```yaml
diffusion:
  stage2_checkpoint: /path/to/stage2/best_checkpoint.pt  # Required: Stage 2 checkpoint

discriminator:
  checkpoint: /path/to/discriminator_best.pt  # Required: Trained discriminator
  weight: 0.1  # Discriminator loss weight

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
    semantic_loss:           # Semantic constraint loss (from Stage 2)
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

### Key Differences from Stage 2

1. **Loads Stage 2 checkpoint** (not Stage 1)
2. **Requires discriminator** (must be trained first)
3. **Lower learning rate**: 0.00003 (even lower than Stage 2)
4. **Fewer epochs**: 50 (refinement stage)
5. **Adversarial loss**: Discriminator loss added to semantic losses

## Usage

### Training

```bash
python training/train_diffusion_stage3.py experiments/diffusion/stage3/stage3_unet128_d4.yaml
```

### Resume Training

```bash
python training/train_diffusion_stage3.py experiments/diffusion/stage3/stage3_unet128_d4.yaml --resume
```

## Creating Stage 3 Configs

To create a Stage 3 config for a Stage 2 experiment:

1. Copy `base_config.yaml` or an existing Stage 3 config
2. Update `experiment.name` and `experiment.save_path`
3. Update `diffusion.stage2_checkpoint` to point to your Stage 2 best checkpoint
4. Update `discriminator.checkpoint` to point to your trained discriminator
5. Adjust discriminator weight if needed (start with 0.1)

Example:
```yaml
experiment:
  name: "diffusion_stage3_unet128_d4"
  save_path: "/work3/s233249/ImgiNav/experiments/diffusion/stage3/stage3_unet128_d4"

diffusion:
  stage2_checkpoint: /work3/s233249/ImgiNav/experiments/diffusion/stage2/stage2_unet128_d4/diffusion_stage2_unet128_d4_checkpoint_best.pt

discriminator:
  checkpoint: /work3/s233249/ImgiNav/discriminator_output/discriminator_best.pt
  weight: 0.1
```

## Loss Weights

Default weights:
- **Noise loss**: 1.0 (primary)
- **Segmentation loss**: 0.1 (semantic constraint)
- **Perceptual loss**: 0.05 (semantic features)
- **Discriminator loss**: 0.1 (adversarial constraint)

Tuning tips:
- If layouts are good but not passing discriminator → increase discriminator weight
- If training is unstable → reduce discriminator weight
- If viability_score > 0.9 consistently → discriminator might be too easy (regenerate bad layouts)

## Monitoring

During Stage 3 training, monitor:
- `discriminator_loss`: Should decrease over time
- `viability_score`: Should increase over time (toward 1.0)
- `noise_loss`: Should remain stable or decrease slightly
- `semantic_loss`: Should remain stable

If viability_score > 0.8 consistently, consider:
- Regenerating bad layouts from current model
- Retraining discriminator with new bad layouts
- This is the iterative/adversarial training approach

## Output Structure

```
experiments/diffusion/stage3/stage3_unet128_d4/
├── checkpoints/
│   ├── diffusion_stage3_unet128_d4_checkpoint_best.pt
│   └── diffusion_stage3_unet128_d4_checkpoint_latest.pt
├── samples/
│   └── diffusion_stage3_unet128_d4_epoch_*.png
└── diffusion_stage3_unet128_d4_metrics.csv
```

## Iterative Training (Optional)

For best results, you can iteratively improve:

1. **Round 1**: Generate bad layouts from Stage 1 → Train discriminator → Train Stage 3
2. **Round 2**: Generate bad layouts from Stage 3 (epoch 25) → Retrain discriminator → Continue Stage 3
3. **Round 3**: Generate bad layouts from Stage 3 (epoch 50) → Retrain discriminator → Continue Stage 3

This keeps the discriminator challenging as the model improves.

## Notes

- Stage 3 training is slower than Stage 2 due to discriminator overhead
- Discriminator operates in latent space (fast, no decoding needed)
- Discriminator is frozen during training (adversarial loss only)
- Best checkpoint is saved based on validation loss
- Decoder stays frozen - only UNet is updated based on semantic and discriminator losses

