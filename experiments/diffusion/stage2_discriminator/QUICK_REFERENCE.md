# Quick Reference: Stage 2 Discriminator Training

## Adversarial Loop Summary

### How Many Loops?
- **Default: 3 iterations** (configurable with `--num_iterations`)
- Each iteration is a complete adversarial cycle

### What Happens in Each Iteration?

```
Iteration N:
1. Generate 5000 fake latents from current model (~30 min)
2. Encode 5000 real latents from dataset (~30 min)
3. Train discriminator for 50 epochs (~2-3 hours)
4. Train diffusion model for 50 epochs (~20-24 hours)
```

### Discriminator Training

- **Epochs**: 50 per iteration (default, configurable with `--discriminator_epochs`)
- **What it does**: Learns to distinguish real vs fake latents
- **Output per epoch**: Train/Val loss and accuracy
- **Checkpoint**: Saved to `models/losses/checkpoints/discriminator_best.pt`

### Diffusion Model Training

- **Epochs**: 50 per iteration (default, configurable with `--diffusion_epochs`)
- **What it does**: Learns to generate latents that fool the discriminator
- **Loss components**:
  - Noise prediction loss (MSE)
  - Discriminator loss (adversarial)
- **Validation**: Every 5 epochs (eval_interval)
- **Sampling**: Every 10 epochs (sample_interval)
- **Checkpoint**: Saved per iteration to `checkpoints/{exp_name}_iter_{N}_checkpoint_best.pt`

## Output Structure

```
output_dir/
├── discriminator_iter_0/          # Iteration 0 discriminator data
│   ├── real_latents.pt
│   ├── fake_latents.pt
│   └── discriminator_history.csv
├── discriminator_iter_1/          # Iteration 1 discriminator data
├── discriminator_iter_2/          # Iteration 2 discriminator data
├── checkpoints/
│   ├── {exp_name}_iter_0_checkpoint_best.pt
│   ├── {exp_name}_iter_1_checkpoint_best.pt
│   └── {exp_name}_iter_2_checkpoint_best.pt
├── samples/
│   └── {exp_name}_iter_{N}_epoch_{M}_samples.png
└── {exp_name}_metrics_iter_{N}.csv
```

## Running on HPC

```bash
# Submit job
bsub < training/hpc_scripts/run_stage2_discriminator_unet64_d4.sh

# Monitor
bjobs
tail -f training/hpc_scripts/logs/stage2_discriminator_unet64_d4.*.out
```

## Key Metrics

- **Discriminator Accuracy**: Should be 0.5-0.7 (balanced)
- **Viability Score**: Should increase over iterations (target >0.7)
- **Discriminator Loss**: Should decrease (model learning to fool discriminator)

## Time Estimates

- **Per iteration**: ~24 hours
- **Total (3 iterations)**: ~72 hours
- **Discriminator training**: ~2-3 hours per iteration
- **Diffusion training**: ~20-24 hours per iteration

