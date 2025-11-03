# Quick inference script - PowerShell
# Usage: .\scripts\inference_quick.ps1

python scripts/inference.py `
    --checkpoint checkpoints/diffusion_ablation_capacity_unet64_d4_checkpoint_best.pt `
    --config experiments/diffusion/ablation/capacity_unet64_d4.yaml `
    --output outputs/inference_samples `
    --num_samples 16 `
    --method ddpm

