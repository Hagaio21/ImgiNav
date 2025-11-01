# Phase 1: Autoencoder Selection

## Overview

Phase 1 systematically tests different autoencoder architectures to find the optimal latent space representation for ControlNet-based diffusion models. The goal is to balance:
- **Reconstruction quality** (high fidelity RGB output)
- **Latent space compactness** (faster diffusion training)
- **Spatial control** (important for ControlNet)

## Phase Structure

| Phase | Focus | Experiments | Timeline |
|-------|-------|-------------|----------|
| **1.1** | Channel × Spatial Resolution Sweep | 12 experiments | 1-2 days |
| **1.2** | VAE Test | 2 experiments | 1 day |
| **1.3** | Loss Tuning | 3 experiments | 1 day |
| **1.5** | Final Training | 1 experiment | 1 day |

**Total Phase 1 Timeline**: ~4-5 days

## Quick Start

### Phase 1.1 (Start Here):
```bash
# Launch all latent channel sweep experiments
bash training/hpc_scripts/launch_phase1_1.sh
```

### Analysis Between Phases:
After each sub-phase completes:
1. Check shared metrics: `outputs/phase1_X_*/`
2. Compare validation MSE, PSNR, accuracy
3. Update next phase configs with winners
4. Proceed to next sub-phase

## Phase Dependencies

Each phase depends on the previous phase's results:

```
Phase 1.1 (Channel × Spatial Resolution Sweep)
    ↓ [Select: latent_channels, downsampling_steps, base_channels]
Phase 1.2 (VAE Test)
    ↓ [Select: variational true/false]
Phase 1.3 (Loss Tuning)
    ↓ [Select: loss weights]
Phase 1.5 (Final Training)
    ↓ [Final Autoencoder]
Phase 2 (Diffusion Architecture)
```

## Experiment Outputs

### Per-Experiment Outputs:
- **Checkpoints**: Individual experiment folders with model checkpoints
- **Metrics**: CSV files with training/validation metrics
- **Samples**: Visual reconstructions for quality inspection

### Shared Phase Outputs:
- **Metrics**: All experiments' metrics in `outputs/phase1_X_*/` for easy comparison
- **Samples**: Smaller sample batches for quick visual comparison

## Analysis Workflow

### After Each Sub-Phase:

1. **Load metrics**:
   ```python
   import pandas as pd
   import glob
   
   # Load all metrics for comparison
   metrics_files = glob.glob("outputs/phase1_X_*/*_metrics.csv")
   dfs = [pd.read_csv(f) for f in metrics_files]
   # Compare final validation losses
   ```

2. **Key decisions**:
   - **Phase 1.1**: Which channel count × spatial resolution combination? (efficiency focus)
   - **Phase 1.2**: VAE or deterministic encoder?
   - **Phase 1.3**: Which loss weight configuration?

3. **Update configs**: Edit next phase's YAML files with winners

## GPU Resources

- **V100**: 4 parallel jobs
- **L40s**: 2 parallel jobs (can run 1 job)
- **Total**: 5-6 experiments can run simultaneously

## Decision Criteria Summary

| Decision | Primary Metric | Secondary Considerations |
|----------|----------------|------------------------|
| **Latent Shape** | Validation MSE per dimension | Efficiency, computational cost |
| **VAE** | Validation MSE + KLD | Latent smoothness, complexity |
| **Loss** | Validation MSE | Auxiliary task performance |

## Success Criteria

A good autoencoder for ControlNet should have:
- ✅ **Low validation MSE** (< 0.01 for 512×512 images)
- ✅ **Good visual quality** (PSNR > 30 dB)
- ✅ **Stable training** (smooth loss curves)
- ✅ **Reasonable latent size** (not too large for diffusion speed)
- ✅ **Spatial detail preservation** (important for ControlNet)

## Troubleshooting

### Experiments failing:
- Check logs: `training/hpc_scripts/logs/phase1_X_*.out`
- Verify config paths and dataset manifest
- Check GPU memory usage

### Poor reconstruction quality:
- Increase model capacity (base_channels)
- Increase latent channels
- Check loss weights aren't conflicting

### Training instability:
- Reduce learning rate
- Check batch size
- Verify dataset is loading correctly

## Next Steps After Phase 1

Once Phase 1.5 completes:
1. ✅ Document final architecture
2. ✅ Save best checkpoint path
3. ✅ Note latent dimensions (H×W×C) for Phase 2
4. → **Proceed to Phase 2**: Diffusion Architecture Finalization

## Additional Resources

- Individual phase READMEs: `README_phase1_X.md`
- Config files: `experiments/autoencoders/phase1/phase1_X_*.yaml`
- Training script: `training/train.py`
- HPC scripts: `training/hpc_scripts/run_phase1_X_*.sh`

