# Phase 1.6: Normalized Autoencoder Training

## Purpose
Train a normalized version of the Phase 1.5 autoencoder with latent normalization enabled. This ensures that latents are standardized (approximately zero mean, unit variance), making them compatible with standard diffusion models that use N(0,1) noise.

## Key Changes from Phase 1.5

1. **Latent Normalization**: Added `normalize_latents: true` to encoder config
2. **Latent Denormalization**: Added `denormalize_latents: true` to decoder config
3. **Same Architecture**: All other parameters match Phase 1.5 exactly

## Architecture

- **Latent Space**: 32×32×16 = 16,384 dims
- **Encoder**: Deterministic AE with normalization
- **Decoder**: Deterministic decoder with denormalization
- **Normalization**: Learnable shift and scale parameters trained end-to-end

## Normalization Details

The normalization layer:
- **Normalizes** latents in the encoder: `z_norm = (z - shift) / (scale + eps)`
- **Denormalizes** latents in the decoder: `z = z_norm * scale + shift`
- Uses learnable parameters that are trained end-to-end with the autoencoder
- Parameters are shared between encoder and decoder to ensure consistency

## Benefits for Diffusion

1. **Standard Noise**: Diffusion can use standard N(0,1) noise without `noise_offset` and `noise_scale`
2. **Stability**: Standardized latents reduce training instability
3. **Compatibility**: Works with standard diffusion implementations

## Training Configuration

- **Epochs**: 50 (full training)
- **Early stopping**: patience=5, min_delta=0.0001
- **Batch size**: 16
- **Learning rate**: 0.0001
- **Optimizer**: AdamW
- **Weight decay**: 0.01
- **Loss**: MSE=1.0, Segmentation=0.1

## Running Experiment

### Submit job:
```bash
python training/train.py experiments/autoencoders/phase1/phase1_6_AE_normalized.yaml
```

### Monitor progress:
```bash
# Watch metrics
tail -f outputs/phase1_6_normalized_training/phase1_6_AE_normalized_metrics.csv
```

## Output Locations

- Checkpoints: `/work3/s233249/ImgiNav/experiments/phase1/phase1_6_AE_normalized/`
- Best checkpoint: `phase1_6_AE_normalized_checkpoint_best.pt`
- Metrics CSV: `phase1_6_AE_normalized_metrics.csv`
- Sample images: `samples/epoch_*.png`

## Expected Results

After training:
- Latents should have approximately zero mean and unit variance
- Reconstruction quality should match Phase 1.5
- Normalizer parameters should converge to match the latent distribution

## Verification

To verify normalization is working:

1. **Check latent statistics** after encoding:
   ```python
   latents = model.encode(images)
   mean = latents["latent"].mean()
   std = latents["latent"].std()
   # Should be approximately 0 and 1
   ```

2. **Check normalizer parameters**:
   ```python
   shift = model.encoder.latent_normalizer.shift
   scale = torch.exp(model.encoder.latent_normalizer.log_scale)
   # These should match the original latent distribution (before normalization)
   ```

## Next Steps

Once Phase 1.6 is complete:

1. **Verify normalization**: Check that latents are properly standardized
2. **Use for diffusion**: This autoencoder can be used with standard N(0,1) noise
3. **Update diffusion configs**: Remove `noise_offset` and `noise_scale` from diffusion scheduler configs

## Notes

- This is a retraining of Phase 1.5 with normalization enabled
- The normalizer parameters are learned end-to-end, so they adapt to the latent distribution
- Normalization happens inside the encoder/decoder, so it's transparent to the training loop
- The decoder automatically denormalizes, so the reconstruction quality should be preserved

