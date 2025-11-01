# Phase 1.3: Loss Tuning

## Purpose
Test different loss weight configurations to find optimal balance between RGB reconstruction (primary task) and auxiliary tasks (segmentation, classification) that guide the latent space.

## Experiments

| Experiment | MSE | Seg | Cls | Description |
|------------|-----|-----|-----|-------------|
| **F1** | 1.0 | 0.0 | 0.0 | RGB only - pure reconstruction |
| **F2** | 1.0 | 0.05 | 0.0 | RGB + Segmentation - semantic guidance |
| **F3** | 1.0 | 0.05 | 0.01 | Full multi-head - current best |

## Configuration
- **Architecture**: Use **best config from Phase 1.2**:
  - Best `latent_channels` from Phase 1.1
  - Best `base_channels` from Phase 1.1
  - Best `downsampling_steps` from Phase 1.1
  - Best `variational` setting from Phase 1.2
- **Epochs**: 10 (quick test)
- **Batch size**: 16

## Before Running

⚠️ **IMPORTANT**: Update all Phase 1.3 config files with the winning architecture from Phase 1.2:
1. Update `latent_channels` in encoder/decoder
2. Update `base_channels` in encoder/decoder
3. Update `downsampling_steps` in encoder/decoder
4. Set `variational` based on Phase 1.2 winner
5. Update loss weights according to experiment (F1, F2, F3)
6. If VAE was chosen in Phase 1.2, include `KLDLoss` in loss config

## Running Experiments

### Launch all experiments:
```bash
bash training/hpc_scripts/launch_phase1_3.sh
```

### Or submit individually:
```bash
# V100 (2 experiments: F1, F2)
bsub < training/hpc_scripts/run_phase1_3_v100.sh

# L40s (1 experiment: F3)
bsub < training/hpc_scripts/run_phase1_3_l40s.sh
```

## Output Locations

### Individual experiment outputs:
- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_3_AE_F*_*/`
- Metrics CSV: `{exp_dir}/phase1_3_AE_F*_*_metrics.csv`

### Shared phase outputs:
- Metrics: `outputs/phase1_3_loss_tuning/*_metrics.csv`
- Samples: `outputs/phase1_3_loss_tuning/samples/`

## Analysis

After experiments complete:
```bash
bash analysis/hpc_scripts/launch_phase1_3_analysis.sh
```

### Key Metrics to Compare:
1. **Validation MSE** - Primary: RGB reconstruction quality
2. **Segmentation loss** (F2, F3) - Semantic preservation
3. **Classification loss** (F3) - Room/scene discrimination
4. **Overall training stability** - Loss curve smoothness
5. **KLD loss** (if VAE chosen) - Should remain stable

### Decision Criteria:
- ✅ **Primary task**: RGB reconstruction (MSE) must be high quality
- ✅ **Auxiliary tasks**: Should help, not hurt, reconstruction
- ⚠️ **Segmentation**: Light weight (0.05) provides semantic guidance
- ⚠️ **Classification**: Very light weight (0.01) for regularization
- ⚠️ **Trade-off**: More losses = richer latent but potential conflicts
- ⚠️ **VAE compatibility**: If VAE chosen, ensure loss weights work well with KLD

### Visual Inspection:
- Compare RGB reconstruction quality across F1, F2, F3
- Check segmentation quality (F2, F3)
- Verify auxiliary tasks don't degrade RGB reconstruction
- Check that loss weights don't cause training instability

## Next Steps

Once analysis is complete:
1. **Select best loss configuration** (likely F2 or F3)
2. **Update Phase 1.5 config** with:
   - All winning parameters from Phases 1.1-1.3:
     - Best `latent_channels` from Phase 1.1
     - Best `base_channels` from Phase 1.1
     - Best `downsampling_steps` from Phase 1.1
     - Best `variational` setting from Phase 1.2
   - Best loss weights from Phase 1.3
3. **Proceed to Phase 1.5**: Final Training (full 50 epochs)

## Expected Timeline
- **Training time**: ~1-2 hours per experiment (10 epochs)
- **Total time**: ~3-4 hours for all 3 experiments
- **Analysis time**: 30 minutes

## Notes
- F1 (RGB only) tests baseline reconstruction without auxiliary guidance
- F2 tests if segmentation helps without classification
- F3 (full multi-head) is current best practice with all auxiliary tasks
- Loss weights are relative - only ratios matter
- If VAE was chosen in Phase 1.2, KLD loss should already be included with appropriate weight
