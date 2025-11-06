# Experiments Summary

This document provides a comprehensive overview of all experiments conducted in the ImgiNav project, organized by phase.

## Table of Contents
1. [Autoencoder Experiments (Phase 1.1-1.6)](#autoencoder-experiments-phase-11-16)
2. [Diffusion Ablation Experiments](#diffusion-ablation-experiments)

---

## Autoencoder Experiments (Phase 1.1-1.6)

### Phase 1.1: Latent Shape Sweep (Channel × Spatial Resolution)

Systematic exploration of different latent space configurations to find optimal balance between reconstruction quality and efficiency.

| Experiment | Latent Channels | Downsampling Steps | Latent Shape | Latent Dims | Base Channels | Epochs | Batch Size | LR | Loss Weights (RGB/Seg/Cls) |
|------------|----------------|-------------------|-------------|-------------|--------------|--------|------------|----|---------------------------|
| S1 | 16 | 4 | 32×32×16 | 16,384 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S2 | 8 | 3 | 64×64×8 | 32,768 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S3 | 4 | 3 | 64×64×4 | 16,384 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S4 | 8 | 4 | 32×32×8 | 8,192 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S5 | 16 | 5 | 16×16×16 | 4,096 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S6 | 32 | 4 | 32×32×32 | 32,768 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S7 | 4 | 4 | 32×32×4 | 4,096 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S8 | 8 | 5 | 16×16×8 | 2,048 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S9 | 16 | 3 | 64×64×16 | 65,536 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S10 | 2 | 3 | 64×64×2 | 8,192 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S11 | 8 | 6 | 8×8×8 | 512 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |
| S12 | 4 | 2 | 128×128×4 | 65,536 | 32 | 10 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 |

**Common Settings:**
- Optimizer: AdamW
- Weight Decay: 0.01
- Activation: SiLU
- Norm Groups: 8
- Variational: false
- Early Stopping: patience=3, min_delta=0.0001

**Winner:** S1 (32×32×16 = 16,384 dims) - Selected for subsequent phases

---

### Phase 1.2: VAE Test

Comparison between deterministic autoencoder and variational autoencoder to determine if VAE regularization improves latent space quality.

| Experiment | Variational | Latent Shape | Base Channels | Epochs | Batch Size | LR | Loss Weights (RGB/KLD/Seg/Cls) |
|------------|-------------|--------------|---------------|--------|------------|----|-------------------------------|
| V1 (Deterministic) | false | 32×32×16 | 32 | 20 | 16 | 0.0001 | 1.0 / - / 0.05 / 0.01 |
| V2 (VAE Light) | true | 32×32×16 | 32 | 20 | 16 | 0.0001 | 1.0 / 0.0001 / 0.05 / 0.01 |

**Common Settings:**
- Optimizer: AdamW
- Weight Decay: 0.01
- Activation: SiLU
- Norm Groups: 8
- Early Stopping: patience=3, min_delta=0.0001

**Winner:** V1 (Deterministic AE) - Selected for subsequent phases

---

### Phase 1.3: Loss Tuning

Testing different loss weight configurations to optimize multi-head autoencoder performance.

| Experiment | Latent Shape | Variational | Epochs | Batch Size | LR | Loss Weights (RGB/Seg/Cls) | Description |
|------------|--------------|-------------|--------|------------|----|----------------------------|-------------|
| F1 (RGB Only) | 32×32×16 | false | 20 | 16 | 0.0001 | 1.0 / 0.0 / 0.0 | Pure RGB reconstruction |
| F2 (RGB + Seg) | 32×32×16 | false | 20 | 16 | 0.0001 | 1.0 / 0.05 / 0.0 | RGB + segmentation |
| F3 (Full Multi-head) | 32×32×16 | false | 20 | 16 | 0.0001 | 1.0 / 0.05 / 0.01 | Full multi-head |

**Common Settings:**
- Optimizer: AdamW
- Weight Decay: 0.01
- Activation: SiLU
- Norm Groups: 8
- Base Channels: 32
- Early Stopping: patience=3, min_delta=0.0001

**Note:** VAE variants (F1_vae, F2_vae, F3_vae) also exist with same loss weights but variational=true

**Winner:** F2 (RGB + Segmentation) - Selected for Phase 1.5

---

### Phase 1.5: Final Training

Full training with best configuration from previous phases.

| Experiment | Latent Shape | Variational | Base Channels | Epochs | Batch Size | LR | Loss Weights (RGB/Seg) |
|------------|--------------|-------------|---------------|--------|------------|----|------------------------|
| Final | 32×32×16 | false | 32 | 50 | 16 | 0.0001 | 1.0 / 0.1 |

**Settings:**
- Optimizer: AdamW
- Weight Decay: 0.01
- Activation: SiLU
- Norm Groups: 8
- Early Stopping: patience=5, min_delta=0.0001
- **Architecture:** Based on Phase 1.1 winner (S1)
- **Encoder:** Deterministic (Phase 1.2 winner: V1)
- **Loss:** RGB + Segmentation (Phase 1.3 winner: F2)

---

### Phase 1.6: Normalized Training

Training with latent standardization loss to ensure latents are standardized (zero mean, unit variance) for diffusion compatibility.

| Experiment | Latent Shape | Variational | Base Channels | Epochs | Batch Size | LR | Loss Weights (RGB/Seg/LatentStd) |
|------------|--------------|-------------|---------------|--------|------------|----|----------------------------------|
| Normalized | 32×32×16 | false | 32 | 50 | 16 | 0.0001 | 1.0 / 0.2 / 0.1 |

**Settings:**
- Optimizer: AdamW
- Weight Decay: 0.01
- Activation: SiLU
- Norm Groups: 8
- Early Stopping: patience=5, min_delta=0.0001
- **Architecture:** Same as Phase 1.5
- **Additional Loss:** LatentStandardizationLoss (weight=0.1) to ensure latents ~N(0,1)

**Purpose:** Ensures latents are standardized for diffusion model compatibility

---

## Diffusion Ablation Experiments

Systematic ablation studies to determine optimal UNet capacity, depth, and training configuration for the diffusion model.

### UNet Capacity Ablations

| Experiment | Base Channels | Depth | Res Blocks | Time Dim | Batch Size | LR | Weight Decay | Max Grad Norm | Weighted Sampling | Scheduler Steps |
|------------|---------------|-------|------------|----------|------------|----|--------------|---------------|------------------|-----------------|
| unet32_d3 | 32 | 3 | 2 | 128 | 64 | 0.00015 | 0.1 | 0.5 | false | 500 |
| unet48_d3 | 48 | 3 | 2 | 128 | 64 | 0.00015 | 0.1 | 0.5 | false | 500 |
| unet48_d4 | 48 | 4 | 2 | 128 | 64 | 0.00015 | 0.1 | 0.5 | false | 500 |
| unet64_d4 | 64 | 4 | 2 | 128 | 64 | 0.00015 | 0.1 | 0.5 | false | 500 |
| unet64_d5 | 64 | 5 | 2 | 128 | 64 | 0.00015 | 0.1 | 0.5 | false | 500 |
| unet128_d3 | 128 | 3 | 2 | 128 | 64 | 0.00015 | 0.1 | 0.5 | true | 500 |
| unet128_d4 | 128 | 4 | 2 | 128 | 64 | 0.00015 | 0.1 | 0.5 | true | 500 |
| unet256_d4 | 256 | 4 | 2 | 128 | 32 | 0.0001 | 0.1 | 0.1 | true | 500 |
| unet256_d4_rb3 | 256 | 4 | 3 | 256 | 32 | 0.0001 | 0.1 | 0.1 | true | 500 |

**Common Settings:**
- **Autoencoder:** Phase 1.6 normalized autoencoder (frozen)
- **Latent Space:** 32×32×16 (from Phase 1.6)
- **UNet In/Out Channels:** 16
- **Scheduler:** CosineScheduler
- **Optimizer:** AdamW
- **Epochs:** 500
- **Dropout:** 0.2
- **Norm Groups:** 8
- **Use AMP:** true
- **Loss:** MSE on noise prediction (weight=1.0)
- **Memorization Check:** every 5 epochs (100 generated vs 1000 training)
- **Eval Interval:** 5 epochs
- **Sample Interval:** 10 epochs
- **Save Interval:** 99999 (only best and latest checkpoints)

**Key Variations:**
- **Batch Size:** Reduced to 32 for larger models (256 base_channels)
- **Learning Rate:** Reduced to 0.0001 for 256 base_channels models
- **Max Grad Norm:** More aggressive (0.1) for 256 base_channels to prevent gradient explosion
- **Weighted Sampling:** Enabled for larger models (128+ base_channels) to balance room_id distribution
- **Res Blocks:** Increased to 3 for unet256_d4_rb3
- **Time Dim:** Increased to 256 for unet256_d4_rb3

**Dataset:**
- **Manifest:** `/work3/s233249/ImgiNav/datasets/augmented/manifest.csv`
- **Input:** Pre-embedded latents (for faster training)
- **Filters:** Non-empty samples only

---

## Summary

### Autoencoder Phases
- **Phase 1.1:** 12 experiments testing latent space configurations
- **Phase 1.2:** 2 experiments comparing deterministic vs VAE
- **Phase 1.3:** 6 experiments (3 deterministic + 3 VAE) testing loss weights
- **Phase 1.5:** 1 experiment for final full training
- **Phase 1.6:** 1 experiment with latent standardization

### Diffusion Ablations
- **9 experiments** testing UNet capacity (base_channels: 32-256) and depth (3-5)
- Focus on preventing memorization and optimizing model capacity

### Final Selected Configuration
- **Autoencoder:** Phase 1.6 normalized (32×32×16 latent, deterministic, RGB+Seg loss with latent standardization)
- **Diffusion:** Best performing UNet configuration from ablations (to be determined from results)

---

## Notes

- All experiments use seed=42 for reproducibility
- Training split: 80% train, 20% validation
- Early stopping enabled for autoencoder experiments
- Diffusion experiments use memorization checks to prevent overfitting
- Phase 1.4 was skipped (loss tuning moved to Phase 1.3)

