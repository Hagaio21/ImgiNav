# CLIP-Conditioned Diffusion Model Architecture

This document provides a detailed visual overview of how all components integrate in the CLIP-conditioned diffusion model.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Training vs Inference](#training-vs-inference)
5. [Component Relationships](#component-relationships)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DIFFUSION MODEL SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────┘

INPUTS:
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Graph Embed  │  │ POV Embed     │  │ Layout Image │
  │ [B, 384]     │  │ [B, 512]      │  │ [B, 3, H, W] │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                  │                  │
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │   CLIP PROJECTIONS (Frozen)          │
         │   - text_proj: 384 → 256             │
         │   - pov_proj: 512 → 256             │
         │   - Combines in joint space          │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │   CLIP JOINT SPACE                   │
         │   [B, 256]                           │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │   CLIPEmbeddingToSpatial             │
         │   (Trainable Spatial Projection)    │
         │   - spatial_proj: 256 → output_ch    │
         │   - Broadcast to [B, C, H, W]       │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │   SPATIAL CONDITIONING SIGNAL         │
         │   [B, output_channels, H_cond, W]   │
         └──────────────┬───────────────────────┘
                        │
                        │
         ┌──────────────┴───────────────────────┐
         │                                       │
         ▼                                       ▼
┌────────────────────┐              ┌────────────────────┐
│   VAE ENCODER      │              │   DIFFUSION UNET    │
│   (Pre-encode)     │              │   (Trainable)      │
│                    │              │                    │
│ Layout Image       │              │ Noisy Latents      │
│ → Latents          │              │ + Conditioning     │
│ [B, 4, H, W]       │              │ → Predicted Noise  │
└────────┬───────────┘              └────────┬───────────┘
         │                                    │
         │                                    │
         └────────────────┬───────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   VAE DECODER        │
              │   (Frozen)            │
              │                       │
              │ Latents → Layout Image│
              │ [B, 4, H, W] →       │
              │ [B, 3, H, W]          │
              └───────────────────────┘
```

---

## Component Details

### 1. CLIP Projections (Separate Component, Frozen)

```
┌─────────────────────────────────────────────────────────────┐
│              CLIP PROJECTIONS (Standalone)                   │
│              Loaded from VAE checkpoint or standalone        │
│              Status: FROZEN (not trained with diffusion)     │
└─────────────────────────────────────────────────────────────┘

Inputs:
  ┌──────────────┐      ┌──────────────┐
  │ text_emb     │      │ pov_emb      │
  │ [B, 384]     │      │ [B, 512]     │
  └──────┬───────┘      └──────┬───────┘
         │                     │
         ▼                     ▼
  ┌──────────────┐      ┌──────────────┐
  │ text_proj    │      │ pov_proj     │
  │ Linear       │      │ Linear       │
  │ 384 → 256    │      │ 512 → 256    │
  └──────┬───────┘      └──────┬───────┘
         │                     │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ combine_embeddings()  │
         │ (average/add/concat)  │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Combined Embedding   │
         │ [B, 256]             │
         │ (CLIP Joint Space)   │
         └──────────────────────┘

Components:
  - text_proj: TextProjection (384 → 256)
  - pov_proj: ImageProjection (512 → 256)
  - latent_proj: LatentProjection (for VAE latents, not used in diffusion)
```

**Key Points:**
- Separate from diffusion model
- Loaded from checkpoint (VAE or standalone)
- Frozen during diffusion training
- Defines the joint embedding space (256-dim)

---

### 2. CLIPEmbeddingToSpatial (Spatial Projection, Trainable)

```
┌─────────────────────────────────────────────────────────────┐
│         CLIPEmbeddingToSpatial (Trainable Component)        │
│         Part of Diffusion Model                             │
│         Status: TRAINABLE                                   │
└─────────────────────────────────────────────────────────────┘

Inputs:
  ┌──────────────┐      ┌──────────────┐
  │ text_emb     │      │ pov_emb      │
  │ [B, 384]     │      │ [B, 512]     │
  └──────┬───────┘      └──────┬───────┘
         │                     │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ CLIP Projections     │
         │ (Frozen, loaded)     │
         │ text_proj + pov_proj │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ CLIP Joint Space      │
         │ [B, 256]              │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ spatial_proj         │
         │ (Trainable)          │
         │ Linear: 256 → C_out  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ [B, C_out]           │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Reshape & Interpolate│
         │ [B, C_out, 1, 1]     │
         │ → [B, C_out, H, W]   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Spatial Features     │
         │ [B, C_out, H, W]      │
         │ (For Cross-Attention) │
         └──────────────────────┘

Components:
  - clip_projections: CLIPProjections (frozen, loaded from checkpoint)
  - spatial_proj: nn.Linear(256, output_channels) (trainable)
  - spatial_proj_conv: nn.Conv2d(256, output_channels, 1) (for spatial mode)
```

**Key Points:**
- Part of diffusion model (not separate)
- Contains frozen CLIP projections (loaded)
- Has trainable spatial projection layer
- Converts 1D CLIP embeddings to spatial features

---

### 3. Diffusion UNet with Cross-Attention

```
┌─────────────────────────────────────────────────────────────┐
│                    DIFFUSION UNET                           │
│                    (Trainable)                              │
└─────────────────────────────────────────────────────────────┘

Input:
  ┌──────────────────────┐
  │ Noisy Latents        │
  │ [B, 4, H, W]         │
  │ + Timestep [B]       │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Time Embedding       │
  │ [B, time_dim]        │
  └──────────┬───────────┘
             │
             │
  ┌──────────┴──────────────────────────────────┐
  │                                              │
  ▼                                              ▼
┌─────────────┐                          ┌─────────────┐
│ Down Blocks │                          │ Conditioning│
│ (Downsample)│                          │ Signal      │
│             │                          │ [B,C,H,W]   │
│ - ResBlock  │                          └──────┬──────┘
│ - Attention │                                  │
│   (if enabled)                                │
└──────┬──────┘                                  │
       │                                         │
       ▼                                         │
┌─────────────┐                                  │
│ Bottleneck  │                                  │
│             │                                  │
│ - ResBlock  │                                  │
│ - Attention │◄─────────────────────────────────┘
│   (Cross-Att)                                  │
└──────┬──────┘                                  │
       │                                         │
       ▼                                         │
┌─────────────┐                                  │
│ Up Blocks   │                                  │
│ (Upsample)  │                                  │
│             │                                  │
│ - ResBlock  │                                  │
│ - Attention │◄─────────────────────────────────┘
│   (Cross-Att)                                  │
└──────┬──────┘                                  │
       │                                         │
       ▼                                         │
┌─────────────┐
│ Final Conv  │
│ [B, 4, H, W]│
└─────────────┘

Cross-Attention Mechanism:
  ┌─────────────────────────────────────────┐
  │  SelfAttentionBlock                    │
  │  (enable_cross_attention=True)          │
  │                                         │
  │  Q = q_proj(latents)  ← From latents   │
  │  K = k_proj(conditioning) ← From cond  │
  │  V = v_proj(conditioning) ← From cond  │
  │                                         │
  │  Attention(Q, K, V)                    │
  └─────────────────────────────────────────┘
```

**Key Points:**
- Fully trainable
- Uses cross-attention at specified locations (downs, bottleneck, ups)
- Q from latents, K/V from conditioning signal
- Falls back to self-attention when conditioning is None (CFG dropout)

---

## Data Flow

### Complete Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA FLOW                          │
└─────────────────────────────────────────────────────────────────┘

1. INPUT PREPARATION
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Graph Embed  │  │ POV Embed    │  │ Layout Image │
   │ [B, 384]     │  │ [B, 512]     │  │ [B, 3, H, W] │
   │ (Pre-computed)│ │ (Pre-computed)│ │ (From dataset)│
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                  │                  │
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                             ▼
2. CLIP PROJECTION (Frozen)
   ┌──────────────────────────────────────┐
   │ CLIPProjections.forward()            │
   │ - text_proj(text_emb) → [B, 256]     │
   │ - pov_proj(pov_emb) → [B, 256]       │
   │ - combine → [B, 256]                 │
   └──────────────┬───────────────────────┘
                  │
                  ▼
3. SPATIAL PROJECTION (Trainable)
   ┌──────────────────────────────────────┐
   │ CLIPEmbeddingToSpatial.forward()      │
   │ - Uses CLIP projections (frozen)      │
   │ - spatial_proj([B, 256]) → [B, C]     │
   │ - Reshape → [B, C, 1, 1]             │
   │ - Interpolate → [B, C, H, W]         │
   └──────────────┬───────────────────────┘
                  │
                  ▼
4. LAYOUT ENCODING (Pre-computed or on-the-fly)
   ┌──────────────────────────────────────┐
   │ VAE Encoder (Frozen)                 │
   │ Layout Image → Latents               │
   │ [B, 3, H, W] → [B, 4, H, W]         │
   └──────────────┬───────────────────────┘
                  │
                  ▼
5. DIFFUSION PROCESS
   ┌──────────────────────────────────────┐
   │ Add Noise                             │
   │ latents + noise → noisy_latents      │
   │ [B, 4, H, W]                         │
   └──────────────┬───────────────────────┘
                  │
                  ▼
6. UNET FORWARD PASS
   ┌──────────────────────────────────────┐
   │ UNet.forward(noisy_latents, t,        │
   │              conditioning_signal)     │
   │                                      │
   │ Down Blocks:                         │
   │   - ResBlock + Cross-Attention        │
   │   - conditioning_signal used for K,V │
   │                                      │
   │ Bottleneck:                          │
   │   - ResBlock + Cross-Attention        │
   │                                      │
   │ Up Blocks:                           │
   │   - ResBlock + Cross-Attention        │
   │   - Skip connections                 │
   │                                      │
   │ Output: pred_noise [B, 4, H, W]      │
   └──────────────┬───────────────────────┘
                  │
                  ▼
7. LOSS COMPUTATION
   ┌──────────────────────────────────────┐
   │ MSE Loss                              │
   │ loss = MSE(pred_noise, true_noise)   │
   └──────────────┬───────────────────────┘
                  │
                  ▼
8. BACKPROPAGATION
   ┌──────────────────────────────────────┐
   │ Gradients flow to:                    │
   │ - UNet parameters (trainable)        │
   │ - spatial_proj (trainable)            │
   │                                      │
   │ Gradients do NOT flow to:            │
   │ - CLIP projections (frozen)          │
   │ - VAE encoder/decoder (frozen)        │
   └──────────────────────────────────────┘
```

---

## Training vs Inference

### Training Mode

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING: All components active, gradients flow          │
└─────────────────────────────────────────────────────────┘

Inputs:
  - text_emb: [B, 384]
  - pov_emb: [B, 512]
  - layout_image: [B, 3, H, W]

Flow:
  1. CLIP Projections (frozen) → [B, 256]
  2. Spatial Projection (trainable) → [B, C, H, W]
  3. VAE Encoder (frozen) → [B, 4, H, W]
  4. Add noise → [B, 4, H, W]
  5. UNet (trainable) → pred_noise [B, 4, H, W]
  6. Loss: MSE(pred_noise, true_noise)
  7. Backprop: Update UNet + spatial_proj

CFG Dropout:
  - Randomly set conditioning_signal = None (10% probability)
  - UNet falls back to self-attention
  - Enables classifier-free guidance at inference
```

### Inference Mode

```
┌─────────────────────────────────────────────────────────┐
│ INFERENCE: Generate new layouts from conditions         │
└─────────────────────────────────────────────────────────┘

Inputs:
  - text_emb: [B, 384]
  - pov_emb: [B, 512]
  - Random noise: [B, 4, H, W]

Flow:
  1. CLIP Projections (frozen) → [B, 256]
  2. Spatial Projection (frozen) → [B, C, H, W]
  3. Start with random noise [B, 4, H, W]
  4. Iterative denoising (DDPM/DDIM):
     For each timestep t:
       - UNet predicts noise
       - Remove predicted noise
       - Use conditioning_signal for cross-attention
  5. Final latents [B, 4, H, W]
  6. VAE Decoder (frozen) → Layout Image [B, 3, H, W]

Classifier-Free Guidance (CFG):
  - Run UNet twice:
    1. With conditioning: pred_noise_cond
    2. Without conditioning: pred_noise_uncond
  - Combine: pred_noise = pred_noise_uncond + 
              guidance_scale * (pred_noise_cond - pred_noise_uncond)
  - guidance_scale typically 3.0-5.0
```

---

## Component Relationships

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPONENT DEPENDENCIES                   │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  CLIP Projections│
                    │  (Standalone)   │
                    │  - text_proj    │
                    │  - pov_proj     │
                    │  Status: Frozen │
                    └────────┬────────┘
                             │
                             │ Loaded from checkpoint
                             │
                             ▼
                    ┌─────────────────┐
                    │ CLIPEmbedding   │
                    │ ToSpatial       │
                    │                 │
                    │ - clip_projections (frozen)
                    │ - spatial_proj (trainable)
                    │ Status: Part of Diffusion Model
                    └────────┬────────┘
                             │
                             │ Provides conditioning_signal
                             │
                             ▼
                    ┌─────────────────┐
                    │  Diffusion UNet │
                    │                 │
                    │ - Uses conditioning_signal
                    │   for cross-attention
                    │ Status: Trainable
                    └────────┬────────┘
                             │
                             │ Predicts noise
                             │
                             ▼
                    ┌─────────────────┐
                    │  VAE Decoder    │
                    │                 │
                    │ - Decodes latents
                    │   to images
                    │ Status: Frozen
                    └─────────────────┘
```

### Checkpoint Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    CHECKPOINT ORGANIZATION                    │
└─────────────────────────────────────────────────────────────┘

VAE Checkpoint (from VAE training):
  ├── encoder.state_dict
  ├── decoder.state_dict
  └── clip_projection.state_dict (optional, saved separately)
      ├── text_proj.*
      ├── pov_proj.*
      └── latent_proj.* (not used in diffusion)

Standalone CLIP Projection Checkpoint (from Experiment 3):
  └── clip_projection.state_dict
      ├── text_proj.*
      ├── pov_proj.*
      └── (latent_proj.* if trained)

Diffusion Model Checkpoint:
  ├── unet.state_dict
  ├── embedding_projection.state_dict
  │   ├── clip_projections.* (frozen, loaded from VAE/standalone)
  │   └── spatial_proj.* (trainable)
  ├── decoder.state_dict (frozen, loaded from VAE)
  └── scheduler.state_dict
```

---

## Key Design Principles

1. **Separation of Concerns**
   - CLIP Projections: Define joint embedding space (trained with VAE)
   - Spatial Projection: Maps CLIP space to UNet features (trained with diffusion)
   - UNet: Denoising model (trained with diffusion)

2. **Frozen vs Trainable**
   - **Frozen**: CLIP projections, VAE encoder/decoder
   - **Trainable**: UNet, spatial_proj in CLIPEmbeddingToSpatial

3. **Cross-Attention Mechanism**
   - Q from latents (what to attend to)
   - K, V from conditioning (what to attend with)
   - Enables conditional generation

4. **Flexibility**
   - CLIP projections can be loaded from VAE or standalone checkpoint
   - Supports both global and spatial CLIP alignment modes
   - CFG dropout enables classifier-free guidance

---

## Summary

The architecture consists of three main components:

1. **CLIP Projections** (separate, frozen): Projects embeddings to joint space
2. **CLIPEmbeddingToSpatial** (part of diffusion model): Converts CLIP embeddings to spatial features
3. **Diffusion UNet** (trainable): Denoises latents using cross-attention with spatial features

The key insight is that CLIP projections are **separate** from the diffusion model but are **loaded into** CLIPEmbeddingToSpatial, which is part of the diffusion model. This allows the diffusion model to use the pre-trained joint embedding space while learning how to map it to spatial features useful for cross-attention.

---

## Diagram Review

### Your Diagram Analysis

Your diagram correctly shows the overall flow, but here are some clarifications and corrections:

#### ✅ **Correct Aspects:**

1. **Layout → Encoder → z0 → +noise → zt → Denoiser**: ✅ Correct
   - Layout encoder is VAE encoder (frozen during diffusion training)
   - z0 is clean latent, zt is noisy latent
   - Denoiser is the UNet

2. **POV + Graph → M → Denoiser**: ✅ Correct concept
   - M block represents CLIPEmbeddingToSpatial
   - Combines POV and Graph embeddings
   - Provides conditioning signal to Denoiser

3. **Denoiser → predicted noise → L_MSE**: ✅ Correct
   - UNet predicts noise
   - MSE loss compares predicted vs actual noise

#### ⚠️ **Clarifications Needed:**

1. **POV/Graph "Encoders"**: 
   - These are **pre-computed embeddings** (not trainable encoders during diffusion)
   - Should be labeled as "Pre-computed Embeddings" or "Load Embeddings"
   - No gradients flow to these during diffusion training
   - They were created by separate encoders (ResNet18 for POV, sentence-transformers for Graph) in a preprocessing step

2. **M Block (CLIPEmbeddingToSpatial)**:
   - **Partially trainable**, not fully trainable:
     - CLIP projections (text_proj, pov_proj): **Frozen** (loaded from checkpoint)
     - spatial_proj: **Trainable** (learns to map CLIP space to spatial features)
   - Should show two parts: frozen CLIP projections + trainable spatial projection

3. **Layout Encoder**:
   - **Frozen** during diffusion training
   - Should NOT receive gradients from L_MSE
   - Only used to encode layout images to latents (pre-computed or on-the-fly)

4. **Loss Backpropagation**:
   - L_MSE → Denoiser: ✅ Correct (UNet is trainable)
   - L_MSE → M: ✅ Partially correct (only spatial_proj receives gradients, not CLIP projections)
   - L_MSE → Layout Encoder: ❌ Incorrect (encoder is frozen)
   - L_MSE → POV/Graph Encoders: ❌ Incorrect (these are pre-computed, not trainable)
   - L_MSE → +noise: ❌ Incorrect (noise is random, not trainable)

### Corrected Diagram Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    CORRECTED TRAINING FLOW                  │
└─────────────────────────────────────────────────────────────┘

Layout Image
    │
    ▼
┌──────────────┐
│ VAE Encoder  │ (Frozen - no gradients)
│              │
└──────┬───────┘
       │
       ▼
     z0 ──┐
         │
         ▼
    ┌────────┐
    │ +noise │ (Random process - no gradients)
    └────┬───┘
         │
         ▼
        zt ────────────────────┐
                                │
                                ▼
                         ┌──────────────┐
                         │   Denoiser   │ (Trainable)
                         │   (UNet)     │◄───┐
                         └──────┬───────┘    │
                                │            │
                                ▼            │
                         predicted noise     │
                                │            │
                                └────────────┘
                                         │
                                         ▼
                                    ┌────────┐
                                    │ L_MSE  │
                                    └────┬───┘
                                         │
                                         │ Gradients flow to:
                                         │ ✅ Denoiser (UNet)
                                         │ ✅ M.spatial_proj (trainable part)
                                         │ ❌ M.CLIP_projections (frozen)
                                         │ ❌ VAE Encoder (frozen)
                                         │ ❌ POV/Graph (pre-computed)

POV Embedding (Pre-computed) ──┐
                                │
Graph Embedding (Pre-computed) ─┤
                                │
                                ▼
                         ┌──────────────┐
                         │      M      │ (Partially Trainable)
                         │              │
                         │ ┌──────────┐│
                         │ │CLIP Proj ││ (Frozen - loaded)
                         │ │text_proj ││
                         │ │pov_proj  ││
                         │ └────┬─────┘│
                         │      │      │
                         │      ▼      │
                         │ ┌──────────┐│
                         │ │spatial_  ││ (Trainable)
                         │ │proj      ││◄───┐
                         │ └────┬─────┘│    │
                         └──────┼──────┘    │
                                │           │
                                ▼           │
                    Spatial Conditioning   │
                    Signal [B,C,H,W]       │
                                │           │
                                └───────────┘
```

### Key Corrections Summary

1. **POV/Graph Encoders**: Show as "Pre-computed Embeddings" (not trainable)
2. **M Block**: Show as two parts - frozen CLIP projections + trainable spatial_proj
3. **Layout Encoder**: Mark as frozen (no gradients)
4. **Loss Backprop**: Only show gradients to trainable components:
   - Denoiser (UNet) ✅
   - M.spatial_proj ✅
   - NOT to: CLIP projections, VAE encoder, POV/Graph embeddings

