# Phase 1.5: Final Training

## Purpose
Train the best autoencoder configuration (determined from Phases 1.1-1.4) to full convergence. This will be the final autoencoder used for diffusion model training.

## Configuration

⚠️ **IMPORTANT**: This config must be updated with all winning parameters from previous phases:

### From Phase 1.1 (Latent Channels):
- `encoder.latent_channels`: [Update with winner]
- `decoder.latent_channels`: [Update with winner]
- `encoder.base_channels`: [Update with winner]
- `decoder.base_channels`: [Update with winner]

### From Phase 1.2 (Spatial Resolution):
- `encoder.downsampling_steps`: [Update with winner]
- `decoder.upsampling_steps`: [Update with winner]

### From Phase 1.3 (VAE):
- `encoder.variational`: [Update with winner: true/false]
- If VAE: Add `KLDLoss` to training.loss.losses

### From Phase 1.4 (Loss Tuning):
- `training.loss.losses[].weight`: [Update all weights with winners]

## Training Configuration
- **Epochs**: 50 (full training)
- **Early stopping**: patience=5, min_delta=0.0001
- **Batch size**: 16
- **Learning rate**: 0.0001
- **Optimizer**: AdamW
- **Weight decay**: 0.01

## Running Experiment

### Submit job:
```bash
bsub < training/hpc_scripts/run_phase1_5.sh
```

### Or if no HPC script exists, run directly:
```bash
python training/train.py experiments/autoencoders/phase1/phase1_5_AE_final.yaml
```

### Monitor progress:
```bash
# Check job status
bjobs

# Monitor logs
tail -f training/hpc_scripts/logs/phase1_5_final.*.out

# Watch metrics
watch -n 5 "tail -1 outputs/phase1_5_final_training/phase1_5_AE_final_metrics.csv"
```

## Output Locations

### Experiment outputs:
- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_5_AE_final/`
- Best checkpoint: `phase1_5_AE_final_checkpoint_best.pt`
- Latest checkpoint: `phase1_5_AE_final_checkpoint_latest.pt`
- Metrics CSV: `phase1_5_AE_final_metrics.csv`
- Sample images: `samples/epoch_*.png`

### Shared phase outputs:
- Metrics: `outputs/phase1_5_final_training/phase1_5_AE_final_metrics.csv`
- Samples: `outputs/phase1_5_final_training/samples/`

## Evaluation

After training completes, evaluate the final model:

### Metrics to check:
1. **Final validation MSE** - Should be lowest from all phases
2. **PSNR** - Image quality metric
3. **Training curves** - Should show stable convergence
4. **Sample quality** - Visual inspection of reconstructions

### Validation tests:
1. **Reconstruction quality**: Check `samples/` folder
2. **Latent space smoothness**: Test interpolation between samples
3. **Model size**: Note parameter count for reference
4. **Inference speed**: Time encoding/decoding for ControlNet planning

## Final Deliverables

After successful training, you should have:

✅ **Best checkpoint**: `phase1_5_AE_final_checkpoint_best.pt`  
✅ **Complete metrics**: `phase1_5_AE_final_metrics.csv`  
✅ **Sample visualizations**: `samples/epoch_*.png`  
✅ **Model configuration**: Saved in checkpoint

## Next Steps (Phase 2)

Once Phase 1.5 is complete:

1. **Document final architecture**:
   - Latent size (H×W×C)
   - Model parameters
   - Training metrics

2. **Prepare for Phase 2**:
   - Use this autoencoder's encoder for diffusion latent encoding
   - Use decoder for diffusion output decoding
   - Note latent channels for UNet configuration

3. **Proceed to Phase 2**: Diffusion Architecture Finalization

## Expected Timeline
- **Training time**: ~8-12 hours (50 epochs with early stopping)
- **Evaluation time**: 30 minutes

## Notes
- This is the final autoencoder - ensure it's fully trained before moving to Phase 2
- Early stopping will restore best checkpoint automatically
- Save the best checkpoint path for Phase 2 config
- Consider running inference tests to verify latent space quality

