#!/usr/bin/env python3
"""
Baseline sampling using pretrained Stable Diffusion model.
Useful for comparison with custom diffusion model.

Usage:
    # Unconditional sampling
    python baseline/sample_sd_baseline.py \
        --num_samples 64 \
        --num_steps 50 \
        --output_dir outputs/baseline_sd_unconditional \
        --unconditional
    
    # With text prompt
    python baseline/sample_sd_baseline.py \
        --prompt "room layout" \
        --num_samples 64 \
        --num_steps 50 \
        --output_dir outputs/baseline_sd_prompted
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


def sample_sd_baseline(
    model_id="runwayml/stable-diffusion-v1-5",
    prompt=None,
    num_samples=64,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    output_dir="outputs/baseline_sd",
    device="cuda"
):
    """
    Sample from pretrained Stable Diffusion model.
    
    Args:
        model_id: HuggingFace model ID (e.g., "runwayml/stable-diffusion-v1-5")
        prompt: Text prompt (None for unconditional)
        num_samples: Number of samples to generate
        num_inference_steps: DDIM steps
        guidance_scale: Classifier-free guidance scale
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
    
    print(f"Loading Stable Diffusion model: {model_id}")
    print(f"Device: {device}")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # Disable safety checker for layouts
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print(f"Generating {num_samples} samples...")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    if prompt:
        print(f"Prompt: {prompt}")
    else:
        print("Unconditional generation (empty prompt)")
    
    # Generate samples in batches
    batch_size = 4  # SD can be memory intensive
    all_images = []
    
    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)
        print(f"  Generating batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size} ({current_batch} samples)...")
        
        with torch.no_grad():
            images = pipe(
                prompt=[prompt] * current_batch if prompt else [""] * current_batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images
        
        all_images.extend(images)
    
    # Convert PIL images to tensors and save
    print(f"\nSaving {len(all_images)} samples...")
    
    # Save individual images
    for idx, img in enumerate(all_images):
        img_path = output_dir / f"sd_baseline_{idx:04d}.png"
        img.save(img_path)
    
    # Create grid
    # Convert PIL to tensor for grid
    img_tensors = []
    for img in all_images:
        # Convert PIL to tensor [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensors.append(img_tensor)
    
    grid_tensor = torch.stack(img_tensors)
    grid_n = int(np.sqrt(num_samples))  # 8x8 for 64 samples
    
    grid_path = output_dir / "sd_baseline_grid.png"
    save_image(grid_tensor, grid_path, nrow=grid_n, normalize=False)
    
    print(f"\n{'='*60}")
    print("Baseline Sampling Complete!")
    print(f"Individual samples: {output_dir}/sd_baseline_*.png")
    print(f"Grid: {grid_path}")
    print(f"{'='*60}")
    
    return all_images


def sample_sd_unconditional(
    model_id="runwayml/stable-diffusion-v1-5",
    num_samples=64,
    num_inference_steps=50,
    seed=42,
    output_dir="outputs/baseline_sd_unconditional",
    device="cuda"
):
    """
    Unconditional sampling (no text prompt).
    Uses empty prompt with guidance_scale=1.0.
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Loading Stable Diffusion model: {model_id}")
    print("Unconditional generation (no text prompt)")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # For unconditional, use empty prompt with guidance_scale=1.0
    print(f"Generating {num_samples} unconditional samples...")
    
    batch_size = 4
    all_images = []
    
    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)
        print(f"  Batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}...")
        
        with torch.no_grad():
            images = pipe(
                prompt=[""] * current_batch,  # Empty prompt
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0,  # No guidance for unconditional
                height=512,
                width=512
            ).images
        
        all_images.extend(images)
    
    # Save images
    for idx, img in enumerate(all_images):
        img_path = output_dir / f"sd_unconditional_{idx:04d}.png"
        img.save(img_path)
    
    # Create grid
    img_tensors = []
    for img in all_images:
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensors.append(img_tensor)
    
    grid_tensor = torch.stack(img_tensors)
    grid_n = int(np.sqrt(num_samples))
    grid_path = output_dir / "sd_unconditional_grid.png"
    save_image(grid_tensor, grid_path, nrow=grid_n, normalize=False)
    
    print(f"\nSamples saved to: {output_dir}")
    return all_images


def main():
    parser = argparse.ArgumentParser(description="Generate baseline samples using pretrained Stable Diffusion")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt (None for unconditional)")
    parser.add_argument("--num_samples", type=int, default=64,
                       help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of DDIM steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=Path, default="outputs/baseline_sd",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--unconditional", action="store_true",
                       help="Unconditional generation (ignore prompt)")
    
    args = parser.parse_args()
    
    if args.unconditional or args.prompt is None:
        sample_sd_unconditional(
            model_id=args.model_id,
            num_samples=args.num_samples,
            num_inference_steps=args.num_steps,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device
        )
    else:
        sample_sd_baseline(
            model_id=args.model_id,
            prompt=args.prompt,
            num_samples=args.num_samples,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device
        )


if __name__ == "__main__":
    main()

