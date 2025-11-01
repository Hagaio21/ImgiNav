# Phase 1.2: Spatial Resolution Test

## Purpose
Test different latent spatial resolutions to find the optimal balance between spatial detail (better for ControlNet) and computational efficiency (faster diffusion).

## Experiments

| Experiment | Downsampling Steps | Latent Resolution | Compression Ratio | Description |
|------------|-------------------|-------------------|-------------------|-------------|
| **R1** | 3 | 64×64 | 8× | Higher resolution, better spatial control |
| **R2** | 4 | 32×32 | 16× | Baseline - standard compression |
| **R3** | 5 | 16×16 | 32× | Very compact, fastest diffusion |

## Configuration
- **Architecture**: Use **best config from Phase 1.1** (update `latent_channels` and `base_channels`)
- **Epochs**: 10 (quick test)
- **Loss**: MSE=1.0, Seg=0.05, Cls=0.01
- **Batch size**: 16

## Before Running

⚠️ **IMPORTANT**: Update all Phase 1.2 config files with the winning architecture from Phase 1.1:
1. Update `latent_channels` in encoder/decoder
2. Update `base_channels` in encoder/decoder
3. Check that `downsampling_steps` matches the experiment (3, 4, or 5)

## Running Experiments

### Launch all experiments:
```bash
bash training/hpc_scripts/launch_phase1_2.sh
```

### Or submit individually:
```bash
# V100 (2 experiments: R1, R2)
bsub < training/hpc_scripts/run_phase1_2_v100.sh

# L40s (1 experiment: R3)
bsub < training/hpc_scripts/run_phase1_2_l40s.sh
```

## Output Locations

### Individual experiment outputs:
- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_2_AE_R*_*/`
- Metrics CSV: `{exp_dir}/phase1_2_AE_R*_*_metrics.csv`

### Shared phase outputs:
- Metrics: `outputs/phase1_2_spatial_resolution/*_metrics.csv`
- Samples: `outputs/phase1_2_spatial_resolution/samples/`

## Analysis

### Key Metrics to Compare:
1. **Validation MSE** - Reconstruction quality
2. **Spatial detail preservation** - Check sample images for fine details
3. **Training speed** - Higher resolution = slower training
4. **Memory usage** - Higher resolution = more GPU memory

### Decision Criteria:
- ✅ **For ControlNet**: Higher resolution (64×64) is preferred for better spatial control
- ✅ **Balance**: Consider if 64×64 improves quality significantly vs 32×32
- ⚠️ **Performance**: 16×16 may be too compressed, losing spatial information
- ⚠️ **Speed**: 64×64 will slow down diffusion training

### Visual Inspection:
- Check `samples/` folder for reconstruction quality
- Compare fine detail preservation across resolutions
- Look for artifacts at different resolutions

## Next Steps

Once analysis is complete:
1. **Select best resolution** (likely 64×64 or 32×32 for ControlNet)
2. **Update Phase 1.3 configs** with:
   - Best `latent_channels` from Phase 1.1
   - Best `base_channels` from Phase 1.1
   - Best `downsampling_steps` from Phase 1.2
3. **Proceed to Phase 1.3**: VAE Test

## Expected Timeline
- **Training time**: ~1-2 hours per experiment (10 epochs)
- **Total time**: ~3-4 hours for all 3 experiments
- **Analysis time**: 30 minutes

## Notes
- Higher resolution (64×64) is typically better for ControlNet spatial conditioning
- All experiments use deterministic encoding (VAE tested in Phase 1.3)
- Loss weights remain fixed - tuned in Phase 1.4

