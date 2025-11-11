# ControlNet Ablation Experiments

This directory contains ControlNet training configs that correspond to the diffusion ablation experiments. Each ControlNet config uses a frozen UNet from a pretrained diffusion model checkpoint.

## Overview

ControlNet trains a lightweight adapter on top of a frozen pretrained diffusion UNet. The adapter learns to inject conditioning information (graph embeddings + POV embeddings) into the UNet's skip connections.

## Config Structure

Each config file:
- **Loads a frozen UNet** from a diffusion checkpoint (placeholder until diffusion training completes)
- **Trains only the adapter** to convert conditioning inputs to control features
- **Matches UNet architecture** (base_channels, depth) from the corresponding diffusion experiment

## Experiments

### Capacity Ablations

These test different UNet sizes as frozen backbones for ControlNet:

1. **controlnet_unet32_d3** - Small UNet (32 base channels, depth 3)
2. **controlnet_unet48_d3** - Small-medium UNet (48 base channels, depth 3)
3. **controlnet_unet48_d4** - Small-medium UNet (48 base channels, depth 4)
4. **controlnet_unet64_d4** - Medium UNet (64 base channels, depth 4)
5. **controlnet_unet64_d5** - Medium UNet, Deep (64 base channels, depth 5)
6. **controlnet_unet128_d3** - Medium-large UNet, Shallow (128 base channels, depth 3)
7. **controlnet_unet128_d4** - Medium-large UNet ⭐ BASELINE (128 base channels, depth 4)
8. **controlnet_unet128_d5_rb3** - Medium-large UNet, Deep (128 base channels, depth 5, 3 res blocks)
9. **controlnet_unet256_d4** - Large UNet (256 base channels, depth 4)
10. **controlnet_unet256_d4_rb3** - Large UNet (256 base channels, depth 4, 3 res blocks)

### Attention Variants

11. **controlnet_unet32_d3_attn** - Small UNet with attention (32 base channels, depth 3)
12. **controlnet_unet64_d4_attn** - Medium UNet with attention (64 base channels, depth 4)

## Setup

### Prerequisites

1. **Diffusion checkpoints**: Train diffusion models first, then update checkpoint paths in configs
2. **ControlNet manifest**: Create training manifest with aligned embeddings:
   ```bash
   python data_preparation/create_controlnet_manifest.py \
       --layouts-manifest datasets/layouts_with_latents.csv \
       --pov-embeddings-manifest datasets/povs_with_embeddings.csv \
       --graph-embeddings-manifest datasets/graphs_with_embeddings.csv \
       --output datasets/controlnet_training_manifest.csv
   ```

3. **Embeddings**: Ensure you have:
   - Layout embeddings (latents): Created with `create_embeddings.py --type layout`
   - POV embeddings: Created with `create_embeddings.py --type pov`
   - Graph embeddings: Created with `create_embeddings.py --type graph`

### Updating Checkpoint Paths

After diffusion training completes, update the `diffusion.checkpoint` placeholder in each config:

```yaml
diffusion:
  checkpoint: "/work3/s233249/ImgiNav/experiments/diffusion/ablation/capacity_unet128_d4/diffusion_ablation_capacity_unet128_d4_checkpoint_best.pt"
```

## Training

```bash
python training/train_controlnet.py --config experiments/controlnet/ablation/controlnet_unet128_d4.yaml
```

## Configuration Details

### Adapter Configuration

- **text_dim**: 384 (graph embedding dimension from all-MiniLM-L6-v2)
- **pov_dim**: 512 (POV embedding dimension from ResNet18)
- **base_channels**: Matches UNet base_channels from diffusion checkpoint
- **depth**: Matches UNet depth from diffusion checkpoint
- **pov_is_spatial**: false (POV embeddings are non-spatial [B, C])

### Training Parameters

- **Batch size**: 32 (smaller than diffusion due to conditioning overhead)
- **Learning rate**: 0.0001 (similar to diffusion fine-tuning)
- **Epochs**: 100 (fewer than diffusion since adapter is smaller)
- **Gradient clipping**: 1.0 (0.5 for large models like unet256)

### Loss

SNR-weighted MSE on noise prediction (same as diffusion training).

## Expected Outcomes

1. Identify optimal frozen UNet capacity for ControlNet
2. Understand if attention in UNet helps ControlNet performance
3. Compare adapter training efficiency across different UNet sizes

## Notes

- The frozen UNet backbone is the most critical architectural decision
- Adapter is much smaller than UNet, so training is faster
- All configs use the same adapter architecture (SimpleAdapter) for fair comparison
- Fusion mode is "add" (can experiment with "concat" or "cross_attn" later)

