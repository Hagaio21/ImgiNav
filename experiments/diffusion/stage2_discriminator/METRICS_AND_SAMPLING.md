# Metrics and Sampling Documentation

## Overview

The Stage 2 Discriminator Training pipeline now tracks extensive metrics and saves samples with discriminator evaluation.

## Sampling with Discriminator Evaluation

### What Gets Saved

When sampling (every `--diffusion_sample_interval` steps, default: 5000):

1. **Best Samples** (16 samples with highest discriminator scores)
   - Saved to: `samples/{exp_name}_step_{N:06d}_best_samples.png`
   - Shows the most viable generated layouts

2. **Worst Samples** (16 samples with lowest discriminator scores)
   - Saved to: `samples/{exp_name}_step_{N:06d}_worst_samples.png`
   - Shows the least viable generated layouts

3. **All Samples** (all generated samples in a grid)
   - Saved to: `samples/{exp_name}_step_{N:06d}_all_samples.png`
   - Complete view of all generated samples

4. **Sample Metrics** (JSON file with statistics)
   - Saved to: `samples/{exp_name}_step_{N:06d}_sample_metrics.json`
   - Contains:
     - `mean_discriminator_score`: Average score across all samples
     - `std_discriminator_score`: Standard deviation
     - `min_discriminator_score`: Minimum score
     - `max_discriminator_score`: Maximum score
     - `median_discriminator_score`: Median score
     - `best_mean_score`: Mean of best 16 samples
     - `worst_mean_score`: Mean of worst 16 samples
     - `num_samples`: Total number of samples generated

## Metrics Tracked

### Discriminator Training Metrics

Per evaluation (every `--discriminator_eval_interval` steps, default: 500):

- `step`: Training step
- `train_loss`: Training loss (BCE)
- `train_acc`: Training accuracy
- `train_f1`: Training F1 score
- `val_loss`: Validation loss
- `val_acc`: Validation accuracy
- `val_f1`: Validation F1 score
- `learning_rate`: Current learning rate
- `steps_without_improvement`: Steps since last improvement

**Saved to**: `discriminator_iter_{N}/discriminator_history.csv`

**Plotted to**: `discriminator_iter_{N}/discriminator_iter_{N}_discriminator_metrics.png`

### Diffusion Training Metrics

Per evaluation (every `--diffusion_eval_interval` steps, default: 1000):

- `step`: Training step
- `iteration`: Current iteration number
- `train_loss`: Total training loss
- `val_loss`: Total validation loss
- `learning_rate`: Current learning rate
- `evals_without_improvement`: Evaluations since last improvement
- `train_noise_loss`: Noise prediction loss (MSE)
- `train_discriminator_loss`: Discriminator adversarial loss
- `train_viability_score`: Viability score from discriminator
- `val_noise_loss`: Validation noise loss
- `val_discriminator_loss`: Validation discriminator loss
- `val_viability_score`: Validation viability score
- Sample metrics (when sampling occurs):
  - `mean_discriminator_score`
  - `std_discriminator_score`
  - `min_discriminator_score`
  - `max_discriminator_score`
  - `median_discriminator_score`
  - `best_mean_score`
  - `worst_mean_score`

**Saved to**: `{exp_name}_metrics_iter_{N}.csv`

**Plotted to**: `{exp_name}_iter_{N}_diffusion_metrics.png`

## Plots Generated

### Discriminator Metrics Plot

`discriminator_iter_{N}/discriminator_iter_{N}_discriminator_metrics.png`

Contains 4 subplots:
1. **Discriminator Loss**: Train vs Val loss over steps
2. **Discriminator Accuracy**: Train vs Val accuracy over steps
3. **Overfitting Indicator (Loss)**: Val - Train loss difference
4. **Overfitting Indicator (Accuracy)**: Train - Val accuracy difference

### Diffusion Metrics Plot

`{exp_name}_iter_{N}_diffusion_metrics.png`

Contains multiple subplots (dynamically generated based on available metrics):
1. **Total Loss**: Train vs Val total loss
2. **Noise Prediction Loss**: MSE loss on noise prediction
3. **Discriminator Adversarial Loss**: Adversarial loss from discriminator
4. **Viability Score**: Score from discriminator (should increase)
5. **Generated Sample Discriminator Scores**: Mean, best, and worst sample scores
6. **Learning Rate Schedule**: Learning rate over training

## Resume Functionality

### Resuming Discriminator Training

If discriminator training is interrupted, it will automatically resume from:
- `discriminator_iter_{N}/discriminator_checkpoint_latest.pt`

This checkpoint contains:
- Model state dict
- Optimizer state dict
- Training history
- Best validation loss
- Current step

### Resuming Diffusion Training

If diffusion training is interrupted, it will automatically resume from:
- `checkpoints/{exp_name}_iter_{N}_checkpoint_latest.pt`

This checkpoint contains:
- Model state dict
- Optimizer state dict
- Training history
- Best validation loss
- Current step
- Iteration number

### Resuming Full Pipeline

To resume from a specific iteration:

```bash
python training/train_stage2_discriminator.py \
    experiments/diffusion/stage2_discriminator/stage2_discriminator_unet64_d4.yaml \
    --start_iteration 1
```

This will:
- Skip iterations 0
- Load the latest checkpoint from iteration 0 if available
- Start from iteration 1

## Early Stopping

Both discriminator and diffusion training use early stopping:

- **Patience**: Number of evaluations without improvement (default: 10)
- **Min Delta**: Minimum change to count as improvement (default: 0.0001)

Training stops when:
- No improvement for `patience` evaluations
- Best checkpoint is automatically saved
- Training history and plots are saved

## Output Structure

```
output_dir/
├── discriminator_iter_0/
│   ├── real_latents.pt
│   ├── fake_latents.pt
│   ├── discriminator_history.csv
│   ├── discriminator_checkpoint_latest.pt  # For resume
│   └── discriminator_iter_0_discriminator_metrics.png
├── discriminator_iter_1/
│   └── ...
├── checkpoints/
│   ├── {exp_name}_iter_0_checkpoint_best.pt
│   ├── {exp_name}_iter_0_checkpoint_latest.pt  # For resume
│   └── ...
├── samples/
│   ├── {exp_name}_step_0005000_best_samples.png
│   ├── {exp_name}_step_0005000_worst_samples.png
│   ├── {exp_name}_step_0005000_all_samples.png
│   ├── {exp_name}_step_0005000_sample_metrics.json
│   └── ...
├── {exp_name}_metrics_iter_0.csv
├── {exp_name}_metrics_iter_1.csv
├── {exp_name}_iter_0_diffusion_metrics.png
└── {exp_name}_iter_1_diffusion_metrics.png
```

## Key Metrics to Monitor

### Discriminator Training
- **Accuracy**: Should be 0.5-0.7 (balanced)
- **Loss Difference**: Should be small (not overfitting)
- **Learning Rate**: Should decrease when plateau detected

### Diffusion Training
- **Noise Loss**: Should decrease (learning to denoise)
- **Discriminator Loss**: Should decrease (learning to fool discriminator)
- **Viability Score**: Should increase (target >0.7)
- **Sample Scores**: 
  - Mean should increase over time
  - Best samples should have high scores (>0.8)
  - Worst samples should improve over iterations

