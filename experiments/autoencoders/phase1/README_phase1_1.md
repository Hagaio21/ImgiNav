# Phase 1.1: Latent Channel Sweep

## Purpose
Test different latent channel counts and model capacities to find the optimal balance between compression (faster diffusion) and information retention (better reconstruction quality).

## Experiments

| Experiment | Latent Channels | Base Channels | Latent Size | Description |
|------------|----------------|---------------|-------------|-------------|
| **L1** | 4 | 32 | 32×32×4 (4K dims) | Very compact, small model |
| **L2** | 8 | 32 | 32×32×8 (8K dims) | Baseline - standard compression |
| **L3** | 16 | 32 | 32×32×16 (16K dims) | More capacity, same model size |
| **L4** | 8 | 64 | 32×32×8 (8K dims) | Baseline channels, larger model |
| **L5** | 4 | 64 | 32×32×4 (4K dims) | Compact channels, powerful model |

## Configuration
- **Epochs**: 10 (quick screening)
- **Loss**: MSE=1.0, Seg=0.05, Cls=0.01
- **Batch size**: 16
- **Spatial resolution**: 32×32 (4 downsampling steps)

## Running Experiments

### Launch all experiments:
```bash
bash training/hpc_scripts/launch_phase1_1.sh
```

### Or submit individually:
```bash
# V100 (4 experiments: L1-L4)
bsub < training/hpc_scripts/run_phase1_1_v100.sh

# L40s (1 experiment: L5)
bsub < training/hpc_scripts/run_phase1_1_l40s.sh
```

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
- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_1_AE_L*_*/`
- Metrics CSV: `{exp_dir}/phase1_1_AE_L*_*_metrics.csv`
- Sample images: `{exp_dir}/samples/`

### Shared phase outputs (for analysis):
- Metrics: `outputs/phase1_1_latent_channels/*_metrics.csv`
- Samples: `outputs/phase1_1_latent_channels/samples/`

## Analysis

After all experiments complete, analyze results in the shared phase folder:

### Key Metrics to Compare:
1. **Validation MSE** (lower is better) - RGB reconstruction quality
2. **Validation PSNR** (higher is better) - Image quality
3. **Segmentation accuracy** - Semantic preservation
4. **Classification accuracy** - Room/scene discrimination
5. **Training stability** - Loss convergence smoothness
6. **Model size** - Parameter count (affects inference speed)

### Decision Criteria:
- ✅ **Lowest validation MSE** = best reconstruction
- ✅ **Highest PSNR** = best image quality
- ✅ **Stable training curves** = reliable model
- ⚠️ **Consider latent size** - smaller = faster diffusion but may lose detail
- ⚠️ **Consider model capacity** - larger models may overfit

## Next Steps

Once analysis is complete:
1. **Select top 2 configurations** from Phase 1.1
2. **Update Phase 1.2 configs** with the winner's `latent_channels` and `base_channels`
3. **Proceed to Phase 1.2**: Spatial Resolution Test

## Expected Timeline
- **Training time**: ~1-2 hours per experiment (10 epochs)
- **Total time**: ~6-8 hours for all 5 experiments
- **Analysis time**: 30 minutes

## Notes
- All experiments use deterministic encoding (no VAE)
- Spatial resolution fixed at 32×32 for this phase
- Loss weights fixed - will be tuned in Phase 1.4

