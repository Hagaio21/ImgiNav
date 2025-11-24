# CLIP-Conditioned Diffusion Model Experiments

This directory contains experiments for training diffusion models conditioned on CLIP-aligned embeddings (text/graph and POV embeddings) using cross-attention.

## Overview

The diffusion model learns to generate layout latents conditioned on:
- **Text/Graph embeddings**: Semantic embeddings of the scene graph
- **POV embeddings**: Point-of-view embeddings for camera positioning

These embeddings are projected through CLIP joint space (from the pre-trained VAE) and converted to spatial features for cross-attention in the UNet.

## Architecture

### Components

1. **VAE Decoder** (frozen)
   - Loaded from pre-trained CLIP VAE checkpoint
   - Decodes latents back to images
   - Frozen during diffusion training

2. **CLIP Embedding Projection** (`CLIPEmbeddingToSpatial`)
   - **CLIP Projections** (frozen): From pre-trained VAE
     - Projects text embeddings → CLIP joint space (256-dim)
     - Projects POV embeddings → CLIP joint space (256-dim)
   - **Spatial Projection** (trainable): Simple Linear layer
     - Projects CLIP joint space (256-dim) → `output_channels`
     - Reshapes to spatial features `[B, output_channels, H, W]`
     - ~12k parameters

3. **UNet** (trainable)
   - Denoises latents conditioned on spatial features
   - Uses cross-attention at specified locations (downs, bottleneck, ups)
   - Cross-attention: Q from latents, K/V from conditioning signal
   - ~7.4M parameters (for small model)

### Conditioning Flow

```
text_emb [B, text_dim] ──┐
                         ├─→ CLIP Projections (frozen) ─→ Joint Space [B, 256]
pov_emb  [B, pov_dim]  ──┘
                                    ↓
                         Spatial Projection (trainable) ─→ [B, output_channels]
                                    ↓
                         Reshape + Interpolate ─→ [B, output_channels, H, W]
                                    ↓
                         Cross-Attention in UNet (K, V)
                                    ↓
                         Q (from latents) attends to K, V
```

### Cross-Attention Mechanism

- **Query (Q)**: From noisy latents (self-attention on latents)
- **Key (K)**: From conditioning signal (spatial features)
- **Value (V)**: From conditioning signal (spatial features)

The UNet learns to use the conditioning signal to guide denoising. When `conditioning_signal=None` (CFG dropout), it falls back to self-attention.

## Training

### Trainable Components

- **UNet**: All parameters (~7.4M for small model)
- **Spatial Projection**: Linear layer in `CLIPEmbeddingToSpatial` (~12k parameters)

### Frozen Components

- **VAE Decoder**: From pre-trained checkpoint
- **CLIP Projections**: From pre-trained VAE (define joint embedding space)

### Training Details

- **Loss**: MSE on predicted noise
- **CFG Dropout**: Randomly drops conditioning signal (default: 10% probability)
  - Teaches model to work with/without conditioning
  - Enables classifier-free guidance at inference
- **Guidance Scale**: Used at inference (default: 3.0)
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 (default)
- **Batch Size**: 32 (for small models)

### Training Process

1. Load pre-computed latents from manifest
2. Extract text_emb and pov_emb from manifest
3. Apply CFG dropout (randomly set embeddings to None)
4. Forward pass:
   - Project embeddings through CLIP → spatial features
   - Add noise to latents
   - UNet predicts noise using cross-attention
5. Compute loss (MSE between predicted and actual noise)
6. Backprop and update UNet + spatial_proj

## Configuration

### Key Config Parameters

```yaml
embedding_projection:
  type: CLIPEmbeddingToSpatial
  output_channels: 48        # Must match UNet base_channels
  spatial_size: [16, 16]     # Spatial resolution of conditioning
  combine_method: average     # How to combine text + POV in CLIP space

unet:
  type: UnetWithAttention
  base_channels: 48
  depth: 3
  enable_cross_attention: true
  attention_at:              # Where to apply cross-attention
    - downs
    - bottleneck

training:
  batch_size: 32
  learning_rate: 0.0001
  cfg_dropout_rate: 0.1      # Probability of dropping conditioning
  guidance_scale: 5.0        # For inference
```

### Experiment Variants

- **regular_rooms/**: Rooms only (type filter: room)
- **regular_scenes/**: Scenes only (type filter: scene)
- **regular/**: All types (no filter)
- **spatial_*/**: Uses spatial CLIP VAE (preserves spatial structure)

### Attention Locations

- `*_down.yaml`: Attention only in downsampling path
- `*_bottleneck.yaml`: Attention only at bottleneck
- `*_up.yaml`: Attention only in upsampling path
- `*_all.yaml`: Attention at all locations (downs, bottleneck, ups)

## Data Requirements

### Manifest Columns

- `latent_path_vae_clip`: Path to pre-computed latents
- `graph_embedding_path`: Path to text/graph embeddings
- `pov_embedding_path`: Path to POV embeddings

### Latents

Latents should be pre-computed using the embedding script:
```bash
# Embed layouts with CLIP VAE
python training/embed_controlnet_dataset.py \
  --ae-checkpoint <vae_checkpoint> \
  --input-manifest <manifest> \
  --output-manifest <manifest> \
  --latent-dir <output_dir> \
  --column-name latent_path_vae_clip
```

## Running Experiments

### Launch Training

```bash
# Submit jobs for rooms and scenes experiments
bash training/hpc_scripts/launch_train_diff_clip_regular_rooms_scenes_down_bottleneck.sh
```

### Individual Training

```bash
# Train rooms experiment
bsub < training/hpc_scripts/run_train_diff_clip_regular_rooms_bottleneck.sh

# Train scenes experiment
bsub < training/hpc_scripts/run_train_diff_clip_regular_scenes_bottleneck.sh
```

## Architecture Details

### UNet Structure

- **Input**: Noisy latents `[B, 4, H, W]` (4 channels from VAE)
- **Output**: Predicted noise `[B, 4, H, W]`
- **Conditioning**: Spatial features `[B, output_channels, H_cond, W_cond]`
  - Interpolated to match UNet resolution at each layer
  - Channel-projected if needed via `ctrl_proj`

### Cross-Attention Blocks

- **SelfAttentionBlock** with `enable_cross_attention=True`
- Q from input latents (self-attention)
- K, V from conditioning signal (cross-attention)
- Falls back to self-attention when `conditioning_signal=None` (CFG dropout)

### Parameter Counts (Small Model)

- UNet: ~7.4M trainable parameters
- Spatial Projection: ~12k trainable parameters
- Total: ~7.4M trainable parameters
- CLIP Projections: ~600k parameters (frozen)
- Decoder: ~309M parameters (frozen)

## Key Design Decisions

1. **CLIP Joint Space**: Uses pre-trained CLIP projections from VAE to ensure embeddings are in the same semantic space as latents
2. **Spatial Features**: Converts 1D embeddings to spatial features for cross-attention (preserves spatial structure)
3. **Trainable Spatial Projection**: Learns to map CLIP embeddings to features useful for UNet cross-attention
4. **Frozen CLIP Projections**: Preserves the joint embedding space learned during VAE training
5. **CFG Dropout**: Enables classifier-free guidance for better control at inference

## Validation

The training script validates:
- CLIP projections are loaded from VAE checkpoint
- CLIPEmbeddingToSpatial is properly initialized
- Forward pass works correctly
- Only UNet and spatial_proj are trainable

If CLIP projections are missing, training exits with an error (this experiment requires CLIP projections).

