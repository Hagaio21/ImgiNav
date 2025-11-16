#!/usr/bin/env python3
"""
Utility to use ControlNet-style conditioning with fine-tuned Stable Diffusion.

NOTE: This script provides a workaround since the custom ControlNet architecture
is designed for the custom UNet, not diffusers' UNet. The finetuned SD uses diffusers
which has a different architecture.

This script demonstrates how to:
1. Load the finetuned SD model
2. Use it with additional conditioning (if needed)
3. Potentially convert weights or use as a base for further training

Usage:
    python baseline/use_controlnet_with_finetuned_sd.py \
        --base_dir outputs/baseline_sd_finetuned \
        --num_samples 16
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
    from diffusers import StableDiffusionPipeline, DDIMScheduler, ControlNetModel
    from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not installed. Install with: pip install diffusers transformers accelerate")


def find_model_dir(base_dir):
    """
    Auto-detect model directory, preferring best checkpoint over final model.
    
    Args:
        base_dir: Base directory containing fine-tuned model outputs
        
    Returns:
        Path to model directory, or None if not found
    """
    base_path = Path(base_dir)
    
    # Prefer best checkpoint
    best_checkpoint = base_path / "checkpoint-best" / "pipeline"
    if best_checkpoint.exists():
        return best_checkpoint
    
    # Fallback to final model
    final_model = base_path / "pipeline"
    if final_model.exists():
        return final_model
    
    return None


def load_finetuned_sd(model_dir, device="cuda"):
    """
    Load fine-tuned Stable Diffusion pipeline.
    
    Args:
        model_dir: Directory containing fine-tuned pipeline
        device: Device to load on
        
    Returns:
        Loaded pipeline
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Loading fine-tuned model from: {model_dir}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    return pipe


def sample_with_finetuned_sd(
    model_dir=None,
    base_dir=None,
    num_samples=16,
    num_inference_steps=50,
    guidance_scale=1.0,
    seed=42,
    output_dir="outputs/baseline_sd_finetuned_samples",
    device="cuda"
):
    """
    Sample from fine-tuned Stable Diffusion model.
    
    This is a basic sampling function. For ControlNet-style conditioning,
    you would need to:
    1. Train a ControlNet adapter on top of the finetuned SD UNet
    2. Or convert the finetuned SD UNet weights to match your custom UNet architecture
    
    Args:
        model_dir: Directory containing fine-tuned pipeline (if None, auto-detect from base_dir)
        base_dir: Base directory to search for model (used if model_dir is None)
        num_samples: Number of samples to generate
        num_inference_steps: DDIM steps
        guidance_scale: Classifier-free guidance scale (1.0 for unconditional)
        seed: Random seed
        output_dir: Output directory
        device: Device to use
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    # Auto-detect model directory if not provided
    if model_dir is None:
        if base_dir is None:
            raise ValueError("Either model_dir or base_dir must be provided")
        model_dir = find_model_dir(base_dir)
        if model_dir is None:
            raise FileNotFoundError(
                f"Could not find model in {base_dir}. "
                f"Expected either {base_dir}/checkpoint-best/pipeline or {base_dir}/pipeline"
            )
        print(f"Auto-detected model: {model_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Device: {device}")
    
    # Load fine-tuned pipeline
    pipe = load_finetuned_sd(model_dir, device)
    
    print(f"Generating {num_samples} samples...")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print("Unconditional generation (empty prompt)")
    print("\nNOTE: For ControlNet-style conditioning with this finetuned SD:")
    print("  - The custom ControlNet architecture is designed for your custom UNet")
    print("  - The finetuned SD uses diffusers' UNet which has a different architecture")
    print("  - Options:")
    print("    1. Train a new ControlNet adapter specifically for the finetuned SD UNet")
    print("    2. Convert the finetuned SD UNet weights to match your custom UNet (complex)")
    print("    3. Use diffusers' ControlNetPipeline with a compatible ControlNet model")
    
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


def create_controlnet_from_finetuned_sd(
    finetuned_sd_dir,
    controlnet_checkpoint_path=None,
    output_path=None,
    device="cuda"
):
    """
    Attempt to create a ControlNet-compatible model from finetuned SD.
    
    WARNING: This is a placeholder function. The architectures are incompatible,
    so this would require significant conversion work or training a new adapter.
    
    Args:
        finetuned_sd_dir: Directory with finetuned SD pipeline
        controlnet_checkpoint_path: Path to existing ControlNet checkpoint (if any)
        output_path: Where to save the combined model
        device: Device to use
    """
    print("\n" + "="*60)
    print("ControlNet + Finetuned SD Integration")
    print("="*60)
    print("\nIMPORTANT: Architecture Incompatibility")
    print("  - Your custom ControlNet uses a custom UNet architecture")
    print("  - The finetuned SD uses diffusers' UNet architecture")
    print("  - These are NOT directly compatible")
    print("\nOptions:")
    print("  1. Use diffusers' ControlNetPipeline:")
    print("     - Train a ControlNet using diffusers' ControlNetModel")
    print("     - Use StableDiffusionControlNetPipeline")
    print("  2. Convert weights (complex, may lose quality):")
    print("     - Map diffusers UNet weights to custom UNet structure")
    print("     - Requires careful weight mapping and may not work perfectly")
    print("  3. Train new ControlNet adapter:")
    print("     - Start from finetuned SD UNet")
    print("     - Train ControlNet adapter using diffusers' framework")
    print("\nFor now, use the finetuned SD directly for unconditional generation.")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Use ControlNet-style conditioning with fine-tuned Stable Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample from finetuned SD:
  python baseline/use_controlnet_with_finetuned_sd.py --base_dir outputs/baseline_sd_finetuned --num_samples 16
  
  # Specify model directory explicitly:
  python baseline/use_controlnet_with_finetuned_sd.py --model_dir outputs/baseline_sd_finetuned/checkpoint-best/pipeline --num_samples 16
        """
    )
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_dir", type=Path,
                            help="Directory containing fine-tuned pipeline")
    model_group.add_argument("--base_dir", type=Path,
                            help="Base directory to auto-detect model (prefers checkpoint-best/pipeline)")
    
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to generate (default: 16)")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of DDIM steps (default: 50)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                       help="Classifier-free guidance scale (default: 1.0 for unconditional)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=Path, default="outputs/baseline_sd_finetuned_samples",
                       help="Output directory (default: outputs/baseline_sd_finetuned_samples)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu, default: cuda)")
    parser.add_argument("--explain_controlnet", action="store_true",
                       help="Show explanation about ControlNet integration")
    
    args = parser.parse_args()
    
    if args.explain_controlnet:
        create_controlnet_from_finetuned_sd(
            finetuned_sd_dir=args.model_dir or args.base_dir,
            device=args.device
        )
    else:
        sample_with_finetuned_sd(
            model_dir=args.model_dir,
            base_dir=args.base_dir,
            num_samples=args.num_samples,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device
        )


if __name__ == "__main__":
    main()

