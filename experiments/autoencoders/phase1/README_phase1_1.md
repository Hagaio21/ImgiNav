# Phase 1.1: Channel × Spatial Resolution Sweep

## Purpose
Comprehensive sweep of latent space configurations to find the optimal **efficiency-quality trade-off**. Tests combinations of channel counts and spatial resolutions to identify the smallest "good enough" latent representation for ControlNet-based diffusion.

## Experiments

12 experiments covering full parameter space:

| Experiment | Channels | Downsampling | Spatial | Total Dims | Description |
|------------|----------|--------------|---------|------------|-------------|
| **S1** | 16 | 4 | 32×32 | 16K | Baseline (Phase 1.1 winner equivalent) |
| **S2** | 8 | 3 | 64×64 | 32K | Higher res, fewer channels |
| **S3** | 4 | 3 | 64×64 | 16K | Same size as S1, different shape |
| **S4** | 8 | 4 | 32×32 | 8K | Smaller, fewer channels |
| **S5** | 16 | 5 | 16×16 | 4K | Much smaller spatial |
| **S6** | 32 | 4 | 32×32 | 32K | More channels, same spatial |
| **S7** | 4 | 4 | 32×32 | 4K | Very compact |
| **S8** | 8 | 5 | 16×16 | 2K | Extremely compact |
| **S9** | 16 | 3 | 64×64 | 65K | High quality ceiling test |
| **S10** | 2 | 3 | 64×64 | 8K | Minimal channels, high res |
| **S11** | 8 | 6 | 8×8 | 512 | Efficiency extreme |
| **S12** | 4 | 2 | 128×128 | 65K | Very high res, few channels |

## Configuration
- **Base channels**: 32 (fixed)
- **Epochs**: 10 (quick screening)
- **Loss**: MSE=1.0, Seg=0.05, Cls=0.01
- **Batch size**: 16

## Running Experiments

### Launch all experiments:
```bash
bash training/hpc_scripts/launch_phase1_1.sh
```

Jobs will queue automatically:
- **V100**: 10 experiments (S1-S10) - runs 4 at a time
- **L40s**: 2 experiments (S11-S12) - runs as slots become available

### Check job status:
```bash
bjobs
```

### Monitor logs:
```bash
# V100 logs
tail -f training/hpc_scripts/logs/phase1_1_v100.*.out

# L40s logs
tail -f training/hpc_scripts/logs/phase1_1_l40s.*.out
```

## Output Locations

### Individual experiment outputs:
- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_1_AE_S*_*/`
- Metrics CSV: `{exp_dir}/phase1_1_AE_S*_*_metrics.csv`
- Sample images: `{exp_dir}/samples/`

### Shared phase outputs (for analysis):
- Metrics: `outputs/phase1_1_latent_shape_sweep/*_metrics.csv`
- Samples: `outputs/phase1_1_latent_shape_sweep/samples/`

## Analysis

After all experiments complete, run analysis:

```bash
bash analysis/hpc_scripts/launch_phase1_1_analysis.sh
```

### Key Metrics:
1. **Validation MSE** - Reconstruction quality
2. **MSE per dimension** - Efficiency metric
3. **Total latent dimensions** - Computational cost
4. **Visual quality** - Sample image comparison

### Decision Framework:
- ✅ **Primary**: Find smallest latent that meets quality threshold
- ✅ **Efficiency**: Compare MSE per dimension
- ✅ **Cost-Benefit**: Is improvement worth 2×/4× dimension increase?
- ⚠️ **For ControlNet**: Spatial resolution may be more important than channels
- ⚠️ **Rule of thumb**: If <10% MSE improvement for 4× dimensions, choose smaller

### Expected Results:
- Smaller latents (2K-4K dims) may show acceptable quality
- Higher resolution (64×64) with fewer channels may outperform
- Total dimension count matters more than individual channel/spatial split

## Next Steps

Once analysis is complete:
1. **Select winner** based on efficiency-quality trade-off
2. **Extract winning config**: `latent_channels`, `downsampling_steps`, `base_channels`
3. **Update Phase 1.2 configs** with winner's parameters
4. **Proceed to Phase 1.2**: VAE Test

## Expected Timeline
- **Training time**: ~1 hour per experiment (10 epochs)
- **Total time**: ~12 hours for all 12 experiments (queued automatically)
- **Analysis time**: 30 minutes

## Notes
- All experiments use deterministic encoding (VAE tested in Phase 1.2)
- Focus on **efficiency frontier**: smallest "good enough" latent
- Quality ceiling (S9, S12) helps understand upper bound
- Extreme efficiency (S8, S11) tests lower bound
