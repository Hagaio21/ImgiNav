# Baseline Models for Comparison

This directory contains scripts for creating baseline models using pretrained Stable Diffusion for comparison with your custom diffusion model.

## Overview

The baseline workflow:
1. **Sample from pretrained SD** (zero-shot) - Quick baseline without training
2. **Fine-tune SD on layouts** - Better baseline that learns your domain
3. **Sample from fine-tuned SD** - Generate layouts for comparison

## Installation

Install required dependencies:

```bash
pip install diffusers transformers accelerate
```

## Usage

### 1. Quick Baseline: Sample from Pretrained SD

Generate samples from pretrained Stable Diffusion without any training:

```bash
# Unconditional sampling (no text prompt)
python baseline/sample_sd_baseline.py \
    --num_samples 64 \
    --num_steps 50 \
    --output_dir outputs/baseline_sd_unconditional \
    --unconditional

# With text prompt (optional)
python baseline/sample_sd_baseline.py \
    --prompt "room layout" \
    --num_samples 64 \
    --num_steps 50 \
    --output_dir outputs/baseline_sd_prompted
```

**Note:** Pretrained SD is trained on natural images, so layout quality will be poor. This is mainly for reference.

### 2. Fine-tune SD on Layout Dataset

Fine-tune Stable Diffusion's UNet on your layout images:

```bash
python baseline/finetune_sd.py \
    --dataset_dir /path/to/layout/images \
    --output_dir outputs/baseline_sd_finetuned \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-5
```

**Dataset format:**
- Directory containing PNG/JPG images of layouts
- Images will be resized to 512x512 automatically

**Training details:**
- Only UNet is trained (VAE and text encoder are frozen)
- Uses DDPMScheduler for training
- Saves checkpoints every 500 steps
- Final model saved to `output_dir/unet` and `output_dir/pipeline`

### 3. Sample from Fine-tuned SD

Generate samples from the fine-tuned model:

```bash
python baseline/sample_finetuned_sd.py \
    --model_dir outputs/baseline_sd_finetuned/pipeline \
    --num_samples 64 \
    --num_steps 50 \
    --output_dir outputs/baseline_sd_finetuned_samples
```

## Comparison Workflow

1. **Generate baseline samples:**
   ```bash
   python baseline/sample_finetuned_sd.py \
       --model_dir outputs/baseline_sd_finetuned/pipeline \
       --num_samples 64 \
       --output_dir outputs/baseline_comparison/sd
   ```

2. **Generate custom model samples:**
   ```bash
   python scripts/diffusion_inference.py \
       --output outputs/baseline_comparison/custom/samples.png
   ```

3. **Compare results:**
   - Visual comparison of grids
   - Quantitative metrics (FID, IS, etc.) if desired

## Architecture Differences

**Stable Diffusion:**
- Transformer-UNet hybrid with attention
- Large model (~860M parameters)
- Built-in text conditioning
- Trained on natural images

**Your Custom Model:**
- Pure convolutional UNet (no attention)
- Smaller, domain-specific
- External ControlNet for conditioning
- Trained specifically on layouts

## Expected Results

- **Pretrained SD (zero-shot):** Poor layout quality (trained on natural images)
- **Fine-tuned SD:** Better, but may still struggle with structured/geometric layouts
- **Your Custom Model:** Should perform best (designed for layouts)

## Notes

- Fine-tuning SD requires significant GPU memory (16GB+ recommended)
- Training is slower than your custom model due to SD's size
- For fair comparison, use same number of DDIM steps (e.g., 50 or 100)
- Fine-tuned SD may still generate natural image artifacts

## Troubleshooting

**Out of memory:**
- Reduce `batch_size` (try 2 or 1)
- Use `torch.float16` (automatic on CUDA)

**Poor quality:**
- Train for more epochs
- Try different learning rates (1e-5 to 5e-5)
- SD may fundamentally struggle with structured layouts

