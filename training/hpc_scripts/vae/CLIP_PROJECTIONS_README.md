# CLIP Projections: Saving and Loading

## Overview

When training VAE models with CLIP loss, the CLIP projection layers are saved **separately** from the VAE checkpoint. This document explains how they're saved and how to use them with diffusion models.

## How CLIP Projections Are Saved

### During VAE Training

1. **VAE Checkpoint** (`{exp_name}_checkpoint_best.pt`):
   - Contains: Encoder, Decoder, optimizer state, training history
   - **Excludes**: CLIP projections (saved separately)

2. **CLIP Projection Checkpoint** (`{exp_name}_clip_projection_epoch_{epoch}.pt`):
   - Contains: CLIP projection weights (text_proj, pov_proj, spatial projections if used)
   - Saved in: `outputs/autoencoders/v2/{exp_name}/checkpoints/`
   - Example: `vae_seg_256_clip_clip_projection_epoch_150.pt`

### Checkpoint Structure

```
outputs/autoencoders/v2/vae_seg_256_clip/
├── checkpoints/
│   ├── vae_seg_256_clip_checkpoint_best.pt      # VAE (no CLIP projections)
│   ├── vae_seg_256_clip_checkpoint_latest.pt    # VAE (no CLIP projections)
│   ├── vae_seg_256_clip_clip_projection_epoch_150.pt  # CLIP projections only
│   └── ...
```

## Loading CLIP Projections in Diffusion Models

Diffusion models can load CLIP projections in two ways:

### Method 1: From Standalone CLIP Projection Checkpoint (Recommended)

Point to the separate CLIP projection checkpoint file:

```yaml
embedding_projection:
  type: CLIPEmbeddingToSpatial
  clip_projections: "outputs/autoencoders/v2/vae_seg_256_clip/checkpoints/vae_seg_256_clip_clip_projection_epoch_150.pt"
  output_channels: 48
  spatial_size: [16, 16]
  combine_method: average
```

### Method 2: From VAE Checkpoint (Fallback)

**Note**: This only works if the VAE checkpoint was saved with `exclude_projections=False`. By default, VAE checkpoints exclude projections.

If you need to load from VAE checkpoint, you would need to:
1. Load the VAE checkpoint (which loads the model with CLIP projections in memory)
2. Extract the CLIP projections from the loaded model

However, the current code tries this as a fallback if the standalone checkpoint fails.

## Finding the CLIP Projection Checkpoint

After VAE training completes, look for files matching:
```
{exp_name}_clip_projection*.pt
```

In the checkpoints directory. The latest/best one is typically:
- `{exp_name}_clip_projection_epoch_{last_epoch}.pt` (from latest checkpoint)
- Or use the one saved when the best checkpoint was created

## Example: Using with Diffusion Model

```yaml
# Diffusion config
diffusion:
  vae_checkpoint: "outputs/autoencoders/v2/vae_seg_256_clip/checkpoints/vae_seg_256_clip_checkpoint_best.pt"
  
  embedding_projection:
    type: CLIPEmbeddingToSpatial
    clip_projections: "outputs/autoencoders/v2/vae_seg_256_clip/checkpoints/vae_seg_256_clip_clip_projection_epoch_150.pt"
    output_channels: 48
    spatial_size: [16, 16]  # Match VAE latent spatial size
    combine_method: average
```

## Important Notes

1. **VAE checkpoint excludes projections by design** - This keeps the VAE checkpoint focused on encoder/decoder only.

2. **CLIP projections are saved separately** - This allows you to:
   - Use the same VAE with different projection configurations
   - Update projections without retraining the VAE
   - Share projections across different experiments

3. **Spatial vs Global projections** - If your VAE was trained with spatial CLIP alignment, the projection checkpoint will include spatial projection layers. The diffusion model will automatically detect and use them.

4. **Checkpoint compatibility** - Make sure the CLIP projection checkpoint matches the VAE checkpoint (same training run, same epoch or best checkpoint).

## Troubleshooting

**Error: "VAE checkpoint does not have CLIP projections"**
- This is expected! Use the separate CLIP projection checkpoint instead.

**Error: "Failed to load CLIP projections"**
- Check that the path to the CLIP projection checkpoint is correct
- Verify the checkpoint file exists
- Ensure the checkpoint was saved during VAE training (check logs)

**Projection dimensions don't match**
- Ensure `output_channels` and `spatial_size` in diffusion config match your VAE architecture
- For 256×256 input with 4 downsampling steps: `spatial_size: [16, 16]`

