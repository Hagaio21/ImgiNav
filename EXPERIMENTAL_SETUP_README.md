# Experimental Setup

This document describes the complete experimental setup for the ImgiNav project, including the VAE training configuration and all diffusion model experiments.

---

## 1. VAE Experiment: `vae_clip_v2`

### Overview

The VAE (Variational Autoencoder) is trained to encode 2D layout images into a compressed latent representation while maintaining semantic alignment with text and POV embeddings through CLIP loss.

**Purpose**: Create a compressed, semantically-aligned latent space for layout images that can be used for diffusion model training.

### Architecture Parameters

#### Encoder
- **Input channels**: 3 (RGB)
- **Base channels**: 64
- **Downsampling steps**: 3
  - Resolution progression: 256 → 128 → 64 → 32
- **Latent channels**: 4
- **Normalization**: GroupNorm (8 groups)
- **Activation**: SiLU

#### Decoder
- **Latent channels**: 4
- **Base channels**: 64
- **Upsampling steps**: 3
  - Resolution progression: 32 → 64 → 128 → 256
- **Normalization**: GroupNorm (8 groups)
- **Activation**: SiLU
- **Output head**: RGBHead with tanh activation (outputs in [-1, 1] range)

#### CLIP Projections
- **Projection dimension**: 256 (joint embedding space)
- **Text embedding dimension**: 384 (from CLIP text encoder)
- **POV embedding dimension**: 512 (from CLIP image encoder)

### Training Parameters

- **Epochs**: 150
- **Batch size**: 32
- **Learning rate**: 0.0001
- **Optimizer**: AdamW
- **Weight decay**: 0.02
- **Train/validation split**: 80/20
- **Seed**: 42
- **Mixed precision (AMP)**: Enabled
- **Early stopping**: 
  - Patience: 15 epochs
  - Min delta: 0.00005
  - Restores best checkpoint

### Dataset

- **Manifest**: `manifest_seg_pov_normalized.csv`
- **Input**: POV-normalized segmented layout images (256×256)
- **Image preprocessing**:
  - Resize to 256×256 (nearest neighbor interpolation to preserve sharp edges)
  - Normalize to [-1, 1] range (mean=0.5, std=0.5)
- **Sample weighting**: 
  - Uses precomputed weights from manifest
  - Non-empty rooms weighted 2× higher
  - Max weight: 10.0

### Loss Function Composition

The VAE uses a **CompositeLoss** combining four loss components:

#### 1. L1 Reconstruction Loss
- **Type**: `L1Loss`
- **Weight**: 1.0
- **Purpose**: Primary reconstruction objective
- **Why**: L1 loss is more robust to outliers than L2 and encourages sharper reconstructions. It directly measures pixel-wise accuracy between input and reconstructed layout images.

#### 2. KLD Loss (KL Divergence)
- **Type**: `KLDLoss`
- **Weight**: 0.0001
- **Purpose**: Regularize latent distribution to be close to standard normal
- **Why**: Ensures the latent space follows a standard normal distribution N(0,1), which is crucial for:
  - Enabling sampling from the latent space
  - Making the latent space smooth and interpolatable
  - Preparing for diffusion model training (which expects well-behaved latents)
- **Note**: Very small weight (0.0001) to prevent over-regularization that would hurt reconstruction quality

#### 3. Latent Standardization Loss
- **Type**: `LatentStandardizationLoss`
- **Weight**: 0.1
- **Key**: `mu` (mean of latent distribution)
- **Mean penalty type**: L2
- **Std penalty type**: L2
- **Per channel**: True
- **Purpose**: Additional regularization to ensure latent statistics match expected distribution
- **Why**: Complements KLD loss by explicitly penalizing deviations from zero mean and unit variance per channel. This helps:
  - Stabilize training
  - Ensure consistent latent statistics across batches
  - Improve latent space quality for downstream diffusion models

#### 4. CLIP Loss
- **Type**: `CLIPLoss`
- **Weight**: 0.2
- **Temperature**: 0.07
- **Projection dimension**: 256
- **Text dimension**: 384
- **POV dimension**: 512
- **Combine method**: Average (averages text and POV embeddings before alignment)
- **Use model projections**: True (uses VAE's CLIPProjections)
- **Purpose**: Align latent features with semantic embeddings (text and POV)
- **Why**: Creates a semantically meaningful latent space where:
  - Layouts with similar semantic content (text descriptions or POV views) are close in latent space
  - The latent representation captures semantic information, not just visual appearance
  - Enables conditioning diffusion models on text/POV embeddings
- **How it works**: 
  - Projects VAE latent features to CLIP joint space (256-dim)
  - Projects text/POV embeddings to same joint space
  - Uses contrastive loss to pull matching pairs together and push non-matching pairs apart
  - Temperature (0.07) controls the sharpness of the similarity distribution

### Total Loss Formula

```
L_total = 1.0 × L_L1 + 0.0001 × L_KLD + 0.1 × L_standardization + 0.2 × L_CLIP
```

### Key Design Decisions

1. **L1 over L2**: Better for preserving sharp edges in segmented layouts
2. **Small KLD weight**: Balances reconstruction quality with latent regularity
3. **CLIP loss**: Enables semantic conditioning in diffusion models
4. **POV-normalized layouts**: Ensures consistent spatial relationships
5. **Segmented layouts**: Easier for model to learn category-based patterns

---

## 2. Diffusion Model Experiments

### Overview

Six diffusion model experiments are trained to generate layout latents conditioned on different input types. All experiments use the same frozen VAE encoder/decoder from `vae_clip_v2`.

**Experiment Matrix**:
- **3 Conditioning Types**: POV-only, Graph-only, Both
- **3 Model Sizes**: Small, Medium, Large
- **Total**: 6 experiments (Large model only exists for POV-only)

### Shared Configuration

All experiments share these settings:

#### Dataset
- **Manifest**: `vae_seg_256_clip_latent_manifest_seg_with_rejections.csv`
- **Filters**: 
  - Type: `room` (room-level samples only)
  - Rejected: `false` (excludes low-quality samples)
- **Input**: Pre-encoded layout latents (from VAE encoder)

#### Autoencoder (Frozen)
- **Checkpoint**: `vae_seg_256_clip_checkpoint_best.pt`
- **Frozen**: True (encoder and decoder not trained)
- **Latent statistics**:
  - Clamp min: -3.233157
  - Clamp max: 3.712635
  - Scale factor: 1.830329

#### Embedding Projection
- **Type**: `CLIPEmbeddingToSpatial`
- **CLIP projections**: Loaded from VAE checkpoint (frozen)
- **Spatial size**: 16×16 (matches latent resolution)
- **Combine method**: Average (for "both" experiments)

#### Scheduler
- **Type**: `LinearScheduler`
- **Num steps**: 1000 (diffusion timesteps)

#### Training
- **Epochs**: 500
- **Train/validation split**: 80/20
- **Seed**: 4242
- **Learning rate**: 0.0001
- **Optimizer**: AdamW
- **Weight decay**: 0.01
- **Learning rate scheduler**: Cosine annealing
- **Mixed precision (AMP)**: Enabled
- **Max gradient norm**: 0.5 (gradient clipping)
- **Gradient accumulation steps**: 1
- **Sample weighting**: Uses precomputed weights from manifest

#### Classifier-Free Guidance (CFG)
- **CFG dropout rate**: 0.25 (25% of samples trained unconditionally)
- **Guidance scale**: 7.0 (at inference)

#### Loss Function
- **Type**: `MSELoss`
- **Key**: `pred_noise`
- **Target**: `noise`
- **Weight**: 1.0
- **Purpose**: Standard diffusion loss - predicts the noise added to latents at each timestep

---

## 3. Experiment Details

### 3.1 POV-Only Experiments

Conditioned only on POV (point-of-view) image embeddings.

#### Small: `diff_clip_seg_rooms_small_down_bottleneck_povs`

**UNet Architecture**:
- **Base channels**: 48
- **Depth**: 3 (down/up blocks)
- **Residual blocks per level**: 2
- **Time embedding dimension**: 256
- **Normalization**: GroupNorm (8 groups)
- **Dropout**: 0.1
- **Attention**:
  - Enabled: True
  - Heads: 2
  - Locations: downs, bottleneck
  - Window size: 8×8 (windowed attention for efficiency)
- **Cross-attention**: Enabled
- **Batch size**: 48

**Embedding Projection**:
- **Output channels**: 48 (matches UNet base channels)

#### Medium: `diff_clip_seg_rooms_medium_down_bottleneck_povs`

**UNet Architecture**:
- **Base channels**: 64
- **Depth**: 4
- **Residual blocks per level**: 2
- **Time embedding dimension**: 256
- **Normalization**: GroupNorm (8 groups)
- **Dropout**: 0.1
- **Attention**:
  - Enabled: True
  - Heads: 4
  - Locations: downs, bottleneck
  - Window size: 8×8
- **Cross-attention**: Enabled
- **Batch size**: 48

**Embedding Projection**:
- **Output channels**: 64

#### Large: `diff_clip_seg_rooms_large_down_bottleneck_povs`

**UNet Architecture**:
- **Base channels**: 128
- **Depth**: 4
- **Residual blocks per level**: 2
- **Time embedding dimension**: 256
- **Normalization**: GroupNorm (8 groups)
- **Dropout**: 0.1
- **Attention**:
  - Enabled: True
  - Heads: 8
  - Locations: downs, bottleneck
  - Window size: 8×8
- **Cross-attention**: Enabled
- **Batch size**: 32 (reduced due to larger model)

**Embedding Projection**:
- **Output channels**: 128

---

### 3.2 Graph-Only Experiments

Conditioned only on graph text embeddings (natural language descriptions).

#### Small: `diff_clip_seg_rooms_small_down_bottleneck_graph`

**UNet Architecture**:
- **Base channels**: 48
- **Depth**: 3
- **Residual blocks per level**: 2
- **Time embedding dimension**: 256
- **Normalization**: GroupNorm (8 groups)
- **Dropout**: 0.1
- **Attention**:
  - Enabled: True
  - Heads: 2
  - Locations: downs, bottleneck
  - Window size: 8×8
- **Cross-attention**: Enabled
- **Batch size**: 48

**Embedding Projection**:
- **Output channels**: 48

#### Medium: `diff_clip_seg_rooms_medium_down_bottleneck_graph`

**UNet Architecture**:
- **Base channels**: 64
- **Depth**: 4
- **Residual blocks per level**: 2
- **Time embedding dimension**: 256
- **Normalization**: GroupNorm (8 groups)
- **Dropout**: 0.1
- **Attention**:
  - Enabled: True
  - Heads: 4
  - Locations: downs, bottleneck
  - Window size: 8×8
- **Cross-attention**: Enabled
- **Batch size**: 48

**Embedding Projection**:
- **Output channels**: 64

---

### 3.3 Both (POV + Graph) Experiments

Conditioned on both POV and graph text embeddings (averaged).

#### Small: `diff_clip_seg_rooms_small_down_bottleneck_both`

**UNet Architecture**:
- **Base channels**: 48
- **Depth**: 3
- **Residual blocks per level**: 2
- **Time embedding dimension**: 256
- **Normalization**: GroupNorm (8 groups)
- **Dropout**: 0.1
- **Attention**:
  - Enabled: True
  - Heads: 2
  - Locations: downs, bottleneck
  - Window size: 8×8
- **Cross-attention**: Enabled
- **Batch size**: 48

**Embedding Projection**:
- **Output channels**: 48
- **Combine method**: Average (averages POV and text embeddings)

#### Medium: `diff_clip_seg_rooms_medium_down_bottleneck_both`

**UNet Architecture**:
- **Base channels**: 64
- **Depth**: 4
- **Residual blocks per level**: 2
- **Time embedding dimension**: 256
- **Normalization**: GroupNorm (8 groups)
- **Dropout**: 0.1
- **Attention**:
  - Enabled: True
  - Heads: 4
  - Locations: downs, bottleneck
  - Window size: 8×8
- **Cross-attention**: Enabled
- **Batch size**: 48

**Embedding Projection**:
- **Output channels**: 64
- **Combine method**: Average

---

## 4. Architecture Comparison

### Model Size Scaling

| Size | Base Channels | Depth | Attention Heads | Batch Size | Parameters (approx) |
|------|--------------|-------|-----------------|------------|---------------------|
| Small | 48 | 3 | 2 | 48 | ~2-3M |
| Medium | 64 | 4 | 4 | 48 | ~7-8M |
| Large | 128 | 4 | 8 | 32 | ~15-20M |

### Conditioning Comparison

| Experiment Type | Input Embeddings | Embedding Dimensions | Projection Output |
|----------------|------------------|---------------------|-------------------|
| POV-only | POV image embedding | 512 | Matches UNet base channels |
| Graph-only | Graph text embedding | 384 | Matches UNet base channels |
| Both | POV + Graph (averaged) | 512 + 384 → 256 (after CLIP projection) | Matches UNet base channels |

---

## 5. Key Design Decisions

### VAE Design

1. **Small latent space (4 channels)**: Balances compression with reconstruction quality
2. **CLIP alignment**: Enables semantic conditioning in diffusion models
3. **L1 reconstruction loss**: Better for sharp, categorical layouts
4. **Small KLD weight**: Prioritizes reconstruction over perfect latent distribution

### Diffusion Design

1. **Frozen VAE**: Only UNet is trained, reducing computational cost
2. **Windowed attention**: 8×8 windows for efficiency (vs full attention)
3. **Cross-attention**: Enables conditioning on embeddings
4. **CFG dropout**: 25% unconditional training enables classifier-free guidance
5. **High guidance scale (7.0)**: Strong conditioning at inference
6. **MSE loss**: Standard for noise prediction in diffusion models

### Conditioning Strategy

1. **POV-only**: Tests visual conditioning capability
2. **Graph-only**: Tests text/language conditioning capability
3. **Both**: Tests combined multimodal conditioning
4. **Averaging embeddings**: Simple but effective combination method

---

## 6. Training Schedule

### Checkpointing
- **Save interval**: Every 20 epochs
- **Evaluation interval**: Every 10 epochs
- **Sample generation**: Every 10 epochs (for qualitative assessment)

### Evaluation
- **Num conditioned samples per type**: 16
- **Guidance scale**: 7.0 (at inference)
- **Sampling method**: DDIM (50 steps)

---

## 7. Expected Outcomes

### VAE
- Compressed latent representation (4 channels, 32×32 resolution)
- Semantic alignment with text/POV embeddings
- High-quality reconstruction of layout images

### Diffusion Models
- **POV-only**: Should generate layouts matching visual POV perspective
- **Graph-only**: Should generate layouts matching text descriptions
- **Both**: Should combine visual and textual information for best results
- **Size comparison**: Larger models should capture more complex patterns but may overfit

---

## 8. Evaluation Metrics

See `EVALUATION_METRICS_PLAN.md` for comprehensive evaluation metrics including:
- Object-level metrics (precision, recall, F1)
- Spatial metrics (IoU, centroid distance)
- Pixel-level metrics (pixel accuracy, mIoU)
- Layout-level metrics (room coverage, object density)

