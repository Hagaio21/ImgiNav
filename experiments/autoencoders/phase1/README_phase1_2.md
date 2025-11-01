# Phase 1.2: VAE Test

## Purpose
Compare Variational Autoencoder (VAE) vs deterministic encoder to determine if smooth latent space regularization improves diffusion and ControlNet performance.

## Experiments

| Experiment | Encoder Type | KL Weight | Description |
|------------|--------------|----------|-------------|
| **V1** | Deterministic | 0.0 | No regularization, direct latent encoding |
| **V2** | VAE | 0.0001 | Light KL regularization for smooth latents |

## Configuration
- **Architecture**: Use **best config from Phase 1.1**:
  - Best `latent_channels`
  - Best `base_channels`
  - Best `downsampling_steps`
- **Epochs**: 10 (quick test)
- **Loss**: 
  - V1: MSE=1.0, Seg=0.05, Cls=0.01
  - V2: MSE=1.0, **KLD=0.0001**, Seg=0.05, Cls=0.01
- **Batch size**: 16

## Before Running

⚠️ **IMPORTANT**: Update both Phase 1.2 config files with the winning architecture from Phase 1.1:
1. Update `latent_channels` in encoder/decoder
2. Update `base_channels` in encoder/decoder
3. Update `downsampling_steps` in encoder/decoder
4. Set `variational: true` for V2, `variational: false` for V1

## Running Experiments

### Launch all experiments:
```bash
bash training/hpc_scripts/launch_phase1_2.sh
```

### Or submit individually:
```bash
# V100 (1 experiment: V1)
bsub < training/hpc_scripts/run_phase1_2_v100.sh

# L40s (1 experiment: V2)
bsub < training/hpc_scripts/run_phase1_2_l40s.sh
```

## Output Locations

### Individual experiment outputs:
- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_2_AE_V*_*/`
- Metrics CSV: `{exp_dir}/phase1_2_AE_V*_*_metrics.csv`

### Shared phase outputs:
- Metrics: `outputs/phase1_2_vae_test/*_metrics.csv`
- Samples: `outputs/phase1_2_vae_test/samples/`

## Analysis

After experiments complete:
```bash
bash analysis/hpc_scripts/launch_phase1_2_analysis.sh
```

### Key Metrics to Compare:
1. **Validation MSE** - Reconstruction quality
2. **KLD Loss** (V2 only) - Should be low but non-zero (smoothness)
3. **Latent space smoothness** - VAE should have smoother interpolations
4. **Training stability** - Check loss curves

### Decision Criteria:
- ✅ **VAE advantages**:
  - Smoother latent space (better for diffusion)
  - Better interpolation between samples
  - Regularized representation
- ✅ **Deterministic advantages**:
  - Simpler architecture
  - Faster training (no reparameterization)
  - More direct encoding
- ⚠️ **For ControlNet**: VAE may help with conditioning stability
- ⚠️ **Check KLD**: Should be small (0.0001-0.001) to avoid posterior collapse

### Visual Inspection:
- Compare sample quality between V1 and V2
- Check if VAE produces smoother latent interpolations
- Verify VAE doesn't blur images (KLD too high)

## Next Steps

Once analysis is complete:
1. **Select encoder type** (VAE vs deterministic)
2. **Update Phase 1.3 configs** with:
   - Best `latent_channels` from Phase 1.1
   - Best `base_channels` from Phase 1.1
   - Best `downsampling_steps` from Phase 1.1
   - Best `variational` setting from Phase 1.2
   - If VAE chosen: include `KLDLoss` in loss config
3. **Proceed to Phase 1.3**: Loss Tuning

## Expected Timeline
- **Training time**: ~1-2 hours per experiment (10 epochs)
- **Total time**: ~2-3 hours for both experiments
- **Analysis time**: 30 minutes

## Notes
- VAE adds complexity but may improve latent space quality
- KLD weight (0.0001) is light - increase if reconstruction degrades
- Deterministic is faster and simpler, typically chosen if quality is similar
