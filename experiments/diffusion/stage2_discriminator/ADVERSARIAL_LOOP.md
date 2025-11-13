# Adversarial Training Loop Structure

## Overview

The Stage 2 Discriminator Training uses an **iterative adversarial training** approach, similar to GAN training. The process alternates between training a discriminator and training the generator (diffusion model).

## Adversarial Loop Structure

### Default Configuration

- **Number of Iterations**: 3 (configurable with `--num_iterations`)
- **Samples per Iteration**: 5000 (configurable with `--num_samples`)
- **Discriminator Epochs per Iteration**: 50 (configurable with `--discriminator_epochs`)
- **Diffusion Epochs per Iteration**: 50 (configurable with `--diffusion_epochs`)

### Single Iteration Breakdown

Each iteration consists of **4 steps**:

```
ITERATION N:
├── Step 1: Generate Fake Latents
│   ├── Load current diffusion model
│   ├── Generate N fake latents using DDIM (100 steps default)
│   └── Save fake latents [N, 16, 32, 32]
│
├── Step 2: Get Real Latents
│   ├── Sample N real images from dataset (non-augmented)
│   ├── Encode to latents using autoencoder
│   └── Save real latents [N, 16, 32, 32]
│
├── Step 3: Train Discriminator (50 epochs)
│   ├── Load real latents (label=1) and fake latents (label=0)
│   ├── Train discriminator to distinguish real vs fake
│   ├── Validate every epoch
│   └── Save best discriminator checkpoint
│
└── Step 4: Train Diffusion Model (50 epochs)
    ├── Load discriminator checkpoint
    ├── Train diffusion model with discriminator loss
    ├── Validate every 5 epochs (eval_interval)
    ├── Generate samples every 10 epochs (sample_interval)
    └── Save best model checkpoint
```

## Detailed Epoch Structure

### Discriminator Training Epochs (Step 3)

For each of the **50 discriminator epochs**:

1. **Training Phase**:
   - Shuffle real and fake latents
   - Split into train/val (80/20)
   - For each batch:
     - Forward pass through discriminator
     - Compute BCE loss (real=1, fake=0)
     - Backward pass and optimizer step
     - Track accuracy and loss

2. **Validation Phase**:
   - Evaluate on validation set
   - Compute validation loss and accuracy
   - Update learning rate scheduler

3. **Checkpointing**:
   - Save best checkpoint if validation loss improves
   - Save final checkpoint at end

**Output per epoch**:
```
Epoch 1/50: Train Loss=0.6234, Train Acc=0.7123, Val Loss=0.5891, Val Acc=0.7234
Epoch 2/50: Train Loss=0.6012, Train Acc=0.7345, Val Loss=0.5678, Val Acc=0.7456
...
```

### Diffusion Model Training Epochs (Step 4)

For each of the **50 diffusion epochs**:

1. **Training Phase**:
   - For each batch from dataset:
     - Sample random timesteps t
     - Add noise to latents
     - Forward pass through UNet
     - Compute loss:
       - Noise prediction loss (MSE)
       - Discriminator loss (adversarial)
     - Backward pass and optimizer step
     - Track all loss components

2. **Validation Phase** (every 5 epochs):
   - Evaluate on validation set
   - Compute validation loss
   - Track discriminator viability scores

3. **Sampling** (every 10 epochs):
   - Generate 64 samples using DDPM
   - Save sample grid image

4. **Checkpointing**:
   - Save latest checkpoint every epoch
   - Save best checkpoint if validation loss improves

**Output per epoch**:
```
Epoch 1/50
Train Loss: 0.123456
  noise_loss: 0.100000
  discriminator_loss: 0.023456
  viability_score: 0.654321

Epoch 5/50
Train Loss: 0.112345
  noise_loss: 0.098765
  discriminator_loss: 0.013580
  viability_score: 0.712345
Val Loss: 0.108765
  val_noise_loss: 0.097654
  val_discriminator_loss: 0.011111
  val_viability_score: 0.723456
  Saved samples to samples/
```

## Complete Training Timeline

For **3 iterations** with default settings:

```
Total Time: ~72 hours (estimated)

Iteration 1:
  Step 1: Generate 5000 fake latents        (~30 min)
  Step 2: Encode 5000 real latents          (~30 min)
  Step 3: Train discriminator (50 epochs)   (~2-3 hours)
  Step 4: Train diffusion (50 epochs)       (~20-24 hours)
  Total: ~24 hours

Iteration 2:
  Step 1: Generate 5000 fake latents        (~30 min)
  Step 2: Encode 5000 real latents          (~30 min)
  Step 3: Train discriminator (50 epochs)   (~2-3 hours)
  Step 4: Train diffusion (50 epochs)       (~20-24 hours)
  Total: ~24 hours

Iteration 3:
  Step 1: Generate 5000 fake latents        (~30 min)
  Step 2: Encode 5000 real latents          (~30 min)
  Step 3: Train discriminator (50 epochs)   (~2-3 hours)
  Step 4: Train diffusion (50 epochs)       (~20-24 hours)
  Total: ~24 hours
```

## Output Structure

After training completes, you'll have:

```
output_dir/
├── discriminator_iter_0/
│   ├── real_latents.pt              # Real latents from iteration 0
│   ├── fake_latents.pt              # Fake latents from iteration 0
│   ├── discriminator_history.csv    # Discriminator training history
│   └── (discriminator checkpoints saved to models/losses/checkpoints/)
│
├── discriminator_iter_1/
│   ├── real_latents.pt
│   ├── fake_latents.pt
│   └── discriminator_history.csv
│
├── discriminator_iter_2/
│   ├── real_latents.pt
│   ├── fake_latents.pt
│   └── discriminator_history.csv
│
├── checkpoints/
│   ├── {exp_name}_iter_0_checkpoint_best.pt      # Best model from iter 0
│   ├── {exp_name}_iter_0_checkpoint_latest.pt    # Latest model from iter 0
│   ├── {exp_name}_iter_1_checkpoint_best.pt      # Best model from iter 1
│   ├── {exp_name}_iter_1_checkpoint_latest.pt
│   ├── {exp_name}_iter_2_checkpoint_best.pt      # Best model from iter 2
│   └── {exp_name}_iter_2_checkpoint_latest.pt
│
├── samples/
│   ├── {exp_name}_iter_0_epoch_010_samples.png
│   ├── {exp_name}_iter_0_epoch_020_samples.png
│   ├── {exp_name}_iter_1_epoch_010_samples.png
│   └── ...
│
└── {exp_name}_metrics_iter_{N}.csv  # Training metrics per iteration
```

## Key Metrics to Monitor

### Discriminator Training
- **Train/Val Accuracy**: Should be around 0.5-0.7 (not too high, not too low)
  - Too high (>0.9): Discriminator too strong, generator can't learn
  - Too low (<0.5): Discriminator too weak, not providing useful signal
- **Train/Val Loss**: Should decrease and stabilize

### Diffusion Model Training
- **Noise Loss**: Should decrease (model learning to denoise)
- **Discriminator Loss**: Should decrease (model learning to fool discriminator)
- **Viability Score**: Should increase (generated samples becoming more viable)
  - Target: >0.7 by end of training

## Resuming Training

If training is interrupted, you can resume from a specific iteration:

```bash
python training/train_stage2_discriminator.py \
    experiments/diffusion/stage2_discriminator/stage2_discriminator_unet64_d4.yaml \
    --start_iteration 1  # Resume from iteration 1 (0-indexed)
```

This will:
- Skip iterations 0
- Start from iteration 1
- Load the latest checkpoint from iteration 0 if available

## Tips for Successful Training

1. **Start with fewer iterations** (2-3) to test the pipeline
2. **Monitor discriminator accuracy** - adjust discriminator weight if needed
3. **Check generated samples** after each iteration to see improvement
4. **Adjust discriminator epochs** if discriminator converges too fast/slow
5. **Adjust diffusion epochs** based on convergence speed
6. **Use smaller batch sizes** for larger models to avoid OOM errors

