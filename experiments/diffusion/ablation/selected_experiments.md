# Selected Diffusion Ablation Experiments (6 total)

## Rationale
For ControlNet, we need to find the optimal **frozen UNet architecture**. The UNet capacity is the most critical factor since it will be frozen and reused. Scheduler and step count are secondary considerations.

## Selected Experiments (6 total)

### 1. UNet Capacity Ablation (5 experiments) - PRIORITY FOR CONTROLNET
These test different UNet sizes to find the optimal **frozen backbone** for ControlNet:

1. **capacity_unet64_d4** - Small UNet (64 base channels, depth 4)
   - Lightweight, fast training
   - Tests lower bound of capacity

2. **capacity_unet128_d4** - Medium UNet (128 base channels, depth 4) ⭐ BASELINE
   - Standard size, most commonly used in diffusion models
   - Balanced capacity vs efficiency

3. **capacity_unet256_d4** - Large UNet (256 base channels, depth 4)
   - Maximum capacity test
   - Tests if more capacity helps (may be overkill)

4. **capacity_unet128_d3** - Medium UNet, Shallow (128 base channels, depth 3)
   - Tests depth vs width trade-off
   - Fewer parameters, different architecture

5. **capacity_unet64_d5** - Small UNet, Deep (64 base channels, depth 5)
   - Tests if depth compensates for narrow width
   - Alternative architecture design

### 2. Scheduler Ablation (1 experiment)
6. **scheduler_linear** - Linear noise schedule
   - Alternative to cosine scheduler
   - Tests if simpler schedule works as well for ControlNet training
   - (Cosine is baseline in all capacity experiments)

## Strategy
Focus on **UNet capacity** since it will be frozen for ControlNet. The frozen UNet backbone is the most critical architectural decision.

## Configuration
- All use: Cosine scheduler (except scheduler_linear), 1000 steps (except steps experiments)
- All use: Frozen Phase 1.5 autoencoder (32×32×16 latent space)
- Checkpoints: Only `best` and `latest` (no periodic saves)

## Expected Outcomes
1. Identify optimal UNet capacity for frozen ControlNet backbone
2. Understand if scheduler choice matters significantly
3. Confirm step count baseline is appropriate

