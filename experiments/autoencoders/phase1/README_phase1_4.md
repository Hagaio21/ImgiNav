# Phase 1.4: Loss Tuning

## Purpose
Determine the optimal loss weight configuration for multi-head autoencoder. Test if auxiliary losses (segmentation, classification) help or hurt the primary RGB reconstruction task.

## Experiments

| Experiment | MSE Weight | Seg Weight | Cls Weight | Description |
|------------|------------|------------|------------|-------------|
| **F1** | 1.0 | 0.0 | 0.0 | RGB only - pure reconstruction |
| **F2** | 1.0 | 0.05 | 0.0 | RGB + segmentation - semantic guidance |
| **F3** | 1.0 | 0.05 | 0.01 | Full multi-head - current best |

## Configuration
- **Architecture**: Use **best config from Phase 1.3**:
  - Best `latent_channels`
  - Best `base_channels`
  - Best `downsampling_steps`
  - Best `variational` setting
- **Epochs**: 10 (quick test)
- **Batch size**: 16

## Before Running

⚠️ **IMPORTANT**: Update all Phase 1.4 config files with the winning architecture from Phase 1.3:
1. Update `latent_channels` in encoder/decoder
2. Update `base_channels` in encoder/decoder
3. Update `downsampling_steps` in encoder/decoder
4. Update `variational` setting
5. Loss weights are already set correctly in each config

## Running Experiments

### Launch all experiments:
```bash
bash training/hpc_scripts/launch_phase1_4.sh
```

### Or submit individually:
```bash
# V100 (2 experiments: F1, F2)
bsub < training/hpc_scripts/run_phase1_4_v100.sh

# L40s (1 experiment: F3)
bsub < training/hpc_scripts/run_phase1_4_l40s.sh
```

## Output Locations

### Individual experiment outputs:
- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_4_AE_F*_*/`
- Metrics CSV: `{exp_dir}/phase1_4_AE_F*_*_metrics.csv`

### Shared phase outputs:
- Metrics: `outputs/phase1_4_loss_tuning/*_metrics.csv`
- Samples: `outputs/phase1_4_loss_tuning/samples/`

## Analysis

### Key Metrics to Compare:
1. **Validation MSE** - Primary metric: RGB reconstruction quality
2. **Segmentation accuracy** - Only relevant for F2, F3
3. **Classification accuracy** - Only relevant for F3
4. **Training stability** - Check if auxiliary losses help convergence

### Decision Criteria:
- ✅ **Best validation MSE** = primary winner
- ✅ **Auxiliary losses help if**:
  - MSE is similar or better than RGB-only
  - Segmentation/classification accuracy is good
  - Training is more stable
- ⚠️ **Auxiliary losses hurt if**:
  - MSE is significantly worse than RGB-only
  - Training is unstable or converges slower
- 🎯 **For ControlNet**: Semantic guidance (segmentation) may help spatial understanding

### Visual Inspection:
- Compare RGB reconstruction quality across all three
- Check if segmentation helps preserve semantic boundaries
- Verify classification doesn't degrade reconstruction

## Next Steps

Once analysis is complete:
1. **Select best loss configuration**
2. **Update Phase 1.5 config** with:
   - Best `latent_channels` from Phase 1.1
   - Best `base_channels` from Phase 1.1
   - Best `downsampling_steps` from Phase 1.2
   - Best `variational` setting from Phase 1.3
   - **Best loss weights** from Phase 1.4
3. **Proceed to Phase 1.5**: Final Training (full 50 epochs)

## Expected Timeline
- **Training time**: ~1-2 hours per experiment (10 epochs)
- **Total time**: ~3-4 hours for all 3 experiments
- **Analysis time**: 30 minutes

## Notes
- RGB-only (F1) is the baseline - others should match or improve it
- Segmentation loss may help preserve spatial semantics for ControlNet
- Classification loss is very light (0.01) - mainly for regularization
- If F1 wins clearly, simplify to RGB-only for final training

