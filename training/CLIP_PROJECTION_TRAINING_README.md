# Standalone CLIP Projection Training (Experiment 3)

This document describes how to train a standalone CLIP projection that maps textured and segmented POV embeddings (along with graph embeddings) to segmented layout latents.

## Overview

After training two VAEs:
1. One on segmented layouts and segmented POVs
2. One on textured layouts and textured POVs

You can train a special CLIP projection (Experiment 3) that maps:
- Textured and segmented POV embeddings
- Graph embeddings
to segmented layout latents.

The key insight is that layout latents are pre-encoded using the trained VAE, and only the CLIP projection is trained to align the conditions to these latents.

## Workflow

### Step 1: Encode Layouts to Latents

First, encode the segmented layouts to latents using the trained segmented VAE:

```bash
python data_preparation_v2/encode_layouts.py \
    --vae-checkpoint outputs/autoencoders/v2/vae_seg_256_clip/checkpoints/vae_seg_256_clip_checkpoint_best.pt \
    --manifest dataset_v2/manifests/manifest_seg.csv \
    --dataset-root dataset_v2 \
    --variant seg \
    --batch-size 32 \
    --device cuda
```

This will create latents in:
```
dataset_v2/layouts/latents/seg_vae_seg_256_clip/{scene_id}_{room_id}_layout.pt
```

### Step 2: Train CLIP Projection

Train the standalone CLIP projection:

```bash
python training/train_clip_projection.py \
    --config experiments/diffusion/clip/experiment3_clip_projection.yaml \
    --layout-latents-dir dataset_v2/layouts/latents/seg_vae_seg_256_clip
```

The script will:
1. Load pre-encoded layout latents
2. Load POV embeddings (from manifest - can be textured or segmented)
3. Load graph embeddings (from manifest)
4. Train only the CLIP projection to align conditions to layout latents

## Configuration

See `experiments/diffusion/clip/experiment3_clip_projection.yaml` for configuration options.

Key settings:
- `clip_projection`: CLIP projection configuration
  - `projection_dim`: Dimension of joint space (default: 256)
  - `text_dim`: Graph embedding dimension (default: 384)
  - `pov_dim`: POV embedding dimension (default: 512)
  - `latent_dim`: Will be inferred automatically from sample latents
- `loss`: CLIPLoss configuration
  - `temperature`: Temperature for contrastive loss (default: 0.07)
  - `combine_method`: How to combine text and POV ("add", "concat", "average")

## Output

The training script saves:
- Checkpoints: `outputs/clip_projections/experiment3/checkpoints/`
  - `{exp_name}_clip_projection_best.pt`: Best checkpoint
  - `{exp_name}_clip_projection_latest.pt`: Latest checkpoint for resuming
  - `{exp_name}_clip_projection_epoch_{epoch}.pt`: Periodic checkpoints
- Metrics: `outputs/clip_projections/experiment3/{exp_name}_metrics.csv`

## Using the Trained Projection

The trained CLIP projection can be used in diffusion models by loading it as a checkpoint:

```yaml
embedding_projection:
  type: CLIPEmbeddingToSpatial
  clip_projections: "outputs/clip_projections/experiment3/checkpoints/clip_projection_experiment3_clip_projection_best.pt"
  output_channels: 48
  spatial_size: [16, 16]
  combine_method: average
```

## Notes

- The layout latents are pre-encoded, so the VAE encoder is not needed during training
- Only the CLIP projection parameters are trained (text_proj, pov_proj, latent_proj)
- The graph embeddings are the same for both segmented and textured VAEs
- POV embeddings can come from either segmented or textured manifests (or both)

