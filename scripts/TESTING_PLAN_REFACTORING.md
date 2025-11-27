# Testing Plan - Post-Refactoring Verification

## Overview

This testing plan verifies that VAE and Diffusion models work correctly after the major codebase refactoring. It tests all experiment configurations with fake data to ensure:
- Models can be instantiated from configs
- Forward passes work correctly
- Sampling works for diffusion models
- No regressions were introduced

## Test Strategy

### 1. **Scope**
- Test all experiment configs in `experiments/` directory
- Run 1 batch of fake data through each model
- Verify outputs are valid (correct shapes, no NaNs, no errors)

### 2. **Test Types**

#### A. VAE/Autoencoder Tests
- **Configs**: `experiments/autoencoders/new_layouts/*.yaml`
- **Test**: 
  - Build model from config
  - Create fake RGB input (batch_size=2, 256×256×3)
  - Run forward pass
  - Verify output structure
  - Test encode/decode methods

#### B. Diffusion Model Tests
- **Configs**: `experiments/diffusion/clip/**/*.yaml`
- **Test**:
  - Build model from config
  - Create fake latents (batch_size=2, matching latent shape)
  - Create fake embeddings (text_emb, pov_emb)
  - Run forward pass
  - Test sampling (short, 5 steps)
  - Verify outputs are valid

### 3. **Fake Data Generation**

For each config, generate appropriate fake data:
- **VAE**: RGB images `[B, 3, 256, 256]` in [-1, 1] range
- **Diffusion**: 
  - Latents `[B, C, H, W]` matching decoder output shape
  - Text embeddings `[B, 384]` (CLIP text dim)
  - POV embeddings `[B, 512]` (CLIP image dim)
  - Timesteps `[B]` random integers

## Test Execution

### Quick Test (Single Config)
```bash
python scripts/test_refactored_models.py experiments/diffusion/clip/regular/both_small_down.yaml
```

### Test All Configs
```bash
# Test all diffusion configs
python scripts/test_refactored_models.py --all-diffusion

# Test all VAE configs
python scripts/test_refactored_models.py --all-vae

# Test everything
python scripts/test_refactored_models.py --all
```

### Test Specific Directory
```bash
python scripts/test_refactored_models.py --directory experiments/diffusion/clip/regular
```

## Expected Results

### Success Criteria
- ✅ All models instantiate without errors
- ✅ Forward passes complete without errors
- ✅ Outputs have correct shapes
- ✅ No NaN or Inf values in outputs
- ✅ Sampling works for diffusion models
- ✅ Memory usage is reasonable

### Failure Indicators
- ❌ Model instantiation fails
- ❌ Forward pass raises exceptions
- ❌ Output shapes don't match expected
- ❌ NaN or Inf values in outputs
- ❌ Memory errors (OOM)
- ❌ Sampling fails

## Test Report

The script generates a report with:
- **Summary**: Total configs tested, passed, failed
- **Per-config results**: Status, errors (if any), output shapes
- **Timing**: How long each test took
- **Memory**: Peak memory usage per test

## Troubleshooting

### Common Issues

1. **Missing checkpoint dependencies**
   - Some configs reference checkpoints that may not exist
   - Solution: Script skips checkpoint loading, uses config-only instantiation

2. **Missing manifest files**
   - Configs reference manifest CSVs that may not exist
   - Solution: Script doesn't load datasets, only tests model instantiation and forward pass

3. **Device/CUDA issues**
   - Some configs may specify GPU when CPU is available
   - Solution: Script forces CPU mode for testing

4. **Memory issues**
   - Large models may OOM on test machine
   - Solution: Script uses small batch size (batch_size=2) and reports memory usage

## Integration with CI/CD

This testing plan can be integrated into CI/CD:
- Run on every PR to verify refactoring doesn't break models
- Run before releases to ensure compatibility
- Run after major refactoring to catch regressions

## Next Steps After Testing

1. **If all tests pass**: Proceed with confidence that refactoring is successful
2. **If tests fail**: 
   - Review error messages
   - Check if failures are due to refactoring or pre-existing issues
   - Fix issues and re-run tests
   - Document any known limitations

## Related Documents

- **Migration Plan**: `scripts/MIGRATION_PLAN_REFACTORING.md`
- **Migration Guide**: `scripts/MIGRATION_GUIDE.md`

