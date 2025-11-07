#!/usr/bin/env python3
"""
Sample from fine-tuned Stable Diffusion model.

Usage:
    python baseline/sample_finetuned_sd.py \
        --model_dir outputs/baseline_sd_finetuned/pipeline \
        --num_samples 64 \
        --num_steps 50 \
        --output_dir outputs/baseline_sd_finetuned_samples
"""

import argparse
import torch
from pathlib import Path
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not installed. Install with: pip install diffusers transformers accelerate")


def sample_finetuned_sd(
    model_dir,
    num_samples=64,
    num_inference_steps=50,
    guidance_scale=1.0,  # Lower for unconditional
    seed=42,
    output_dir="outputs/baseline_sd_finetuned_samples",
    device="cuda"
):
    """
    Sample from fine-tuned Stable Diffusion model.
    
    Args:
        model_dir: Directory containing fine-tuned pipeline
        num_samples: Number of samples to generate
        num_inference_steps: DDIM steps
        guidance_scale: Classifier-free guidance scale (1.0 for unconditional)
        seed: Random seed
        output_dir: Output directory
        device: Device to use
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Loading fine-tuned model from: {model_dir}")
    print(f"Device: {device}")
    
    # Load fine-tuned pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print(f"Generating {num_samples} samples...")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print("Unconditional generation (empty prompt)")
    
    # Generate samples in batches
    batch_size = 4
    all_images = []
    
    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)
        print(f"  Generating batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size} ({current_batch} samples)...")
        
        with torch.no_grad():
            images = pipe(
                prompt=[""] * current_batch,  # Empty prompt for unconditional
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images
        
        all_images.extend(images)
    
    # Save images
    print(f"\nSaving {len(all_images)} samples...")
    
    for idx, img in enumerate(all_images):
        img_path = output_dir / f"sd_finetuned_{idx:04d}.png"
        img.save(img_path)
    
    # Create grid
    img_tensors = []
    for img in all_images:
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensors.append(img_tensor)
    
    grid_tensor = torch.stack(img_tensors)
    grid_n = int(np.sqrt(num_samples))
    grid_path = output_dir / "sd_finetuned_grid.png"
    save_image(grid_tensor, grid_path, nrow=grid_n, normalize=False)
    
    print(f"\n{'='*60}")
    print("Sampling Complete!")
    print(f"Individual samples: {output_dir}/sd_finetuned_*.png")
    print(f"Grid: {grid_path}")
    print(f"{'='*60}")
    
    return all_images


def main():
    parser = argparse.ArgumentParser(description="Sample from fine-tuned Stable Diffusion")
    parser.add_argument("--model_dir", type=Path, required=True,
                       help="Directory containing fine-tuned pipeline")
    parser.add_argument("--num_samples", type=int, default=64,
                       help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of DDIM steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=Path, default="outputs/baseline_sd_finetuned_samples",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    sample_finetuned_sd(
        model_dir=args.model_dir,
        num_samples=args.num_samples,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()

