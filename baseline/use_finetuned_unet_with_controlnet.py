#!/usr/bin/env python3
"""
Load fine-tuned Stable Diffusion UNet and use it with ControlNet.

This script demonstrates how to:
1. Load the fine-tuned UNet from the best checkpoint
2. Create a ControlNet from the fine-tuned UNet
3. Use it with StableDiffusionControlNetPipeline

Usage:
    python baseline/use_finetuned_unet_with_controlnet.py \
        --checkpoint_dir outputs/baseline_sd_finetuned_full/checkpoint-best \
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
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UNet2DConditionModel,
        DDIMScheduler,
        AutoencoderKL
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not installed. Install with: pip install diffusers transformers accelerate")


def load_finetuned_unet(checkpoint_dir, device="cuda"):
    """
    Load fine-tuned UNet from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint-best/unet
        device: Device to load on
        
    Returns:
        Loaded UNet model
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    checkpoint_dir = Path(checkpoint_dir)
    unet_dir = checkpoint_dir / "unet"
    
    if not unet_dir.exists():
        raise FileNotFoundError(f"UNet directory not found: {unet_dir}")
    
    print(f"Loading fine-tuned UNet from: {unet_dir}")
    unet = UNet2DConditionModel.from_pretrained(
        str(unet_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    unet = unet.to(device)
    
    return unet


def create_controlnet_from_unet(unet, device="cuda"):
    """
    Create a ControlNet model from a fine-tuned UNet.
    
    Args:
        unet: Fine-tuned UNet2DConditionModel
        device: Device to create on
        
    Returns:
        ControlNet model
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    print("Creating ControlNet from fine-tuned UNet...")
    # ControlNet uses the same config as UNet but with additional control layers
    # We can create it from the UNet's config
    unet_config = unet.config
    
    # Create ControlNet with the same config as the UNet
    controlnet = ControlNetModel.from_config(unet_config)
    
    # Copy encoder weights from UNet to ControlNet (ControlNet shares encoder with UNet)
    # Note: ControlNet has additional control layers that are randomly initialized
    controlnet_state_dict = controlnet.state_dict()
    unet_state_dict = unet.state_dict()
    
    # Copy matching weights (encoder blocks)
    copied = 0
    for key in controlnet_state_dict.keys():
        if key in unet_state_dict and controlnet_state_dict[key].shape == unet_state_dict[key].shape:
            controlnet_state_dict[key] = unet_state_dict[key]
            copied += 1
    
    controlnet.load_state_dict(controlnet_state_dict, strict=False)
    controlnet = controlnet.to(device)
    
    print(f"✓ ControlNet created (copied {copied} encoder weights from UNet)")
    return controlnet


def load_base_components(model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Load base VAE and text encoder from pretrained SD.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load on
        
    Returns:
        Tuple of (vae, text_encoder, tokenizer)
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
    
    print(f"Loading base components from: {model_id}")
    
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    vae = vae.to(device)
    
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    text_encoder = text_encoder.to(device)
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )
    
    print("✓ Base components loaded")
    return vae, text_encoder, tokenizer


def create_controlnet_pipeline(
    checkpoint_dir,
    controlnet_input=None,
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda"
):
    """
    Create a ControlNet pipeline using the fine-tuned UNet.
    
    Args:
        checkpoint_dir: Directory containing checkpoint-best/unet
        controlnet_input: Optional control image (PIL Image or tensor)
        model_id: Base SD model ID for VAE/text encoder
        device: Device to use
        
    Returns:
        StableDiffusionControlNetPipeline
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    # Load fine-tuned UNet
    unet = load_finetuned_unet(checkpoint_dir, device)
    
    # Create ControlNet from UNet
    controlnet = create_controlnet_from_unet(unet, device)
    
    # Load base components
    vae, text_encoder, tokenizer = load_base_components(model_id, device)
    
    # Create scheduler
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Create pipeline
    print("Creating ControlNet pipeline...")
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    print("✓ ControlNet pipeline created")
    return pipe


def sample_with_controlnet(
    checkpoint_dir,
    num_samples=16,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    output_dir="outputs/controlnet_finetuned_samples",
    control_image=None,
    prompt=None,
    device="cuda"
):
    """
    Sample from ControlNet pipeline using fine-tuned UNet.
    
    Args:
        checkpoint_dir: Directory containing checkpoint-best
        num_samples: Number of samples to generate
        num_inference_steps: DDIM steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed
        output_dir: Output directory
        control_image: Optional control image (PIL Image)
        prompt: Text prompt (if None, uses empty prompt)
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
    
    # Create pipeline
    pipe = create_controlnet_pipeline(checkpoint_dir, control_image, device=device)
    
    # Prepare prompts
    if prompt is None:
        prompt = [""] * num_samples  # Empty prompt for unconditional
    elif isinstance(prompt, str):
        prompt = [prompt] * num_samples
    
    print(f"Generating {num_samples} samples...")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    if control_image is not None:
        print("Using control image for conditioning")
    else:
        print("No control image (unconditional generation)")
    
    # Generate samples in batches
    batch_size = 4
    all_images = []
    
    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)
        print(f"  Generating batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size} ({current_batch} samples)...")
        
        batch_prompts = prompt[i:i+current_batch]
        
        with torch.no_grad():
            if control_image is not None:
                # Repeat control image for batch
                control_images = [control_image] * current_batch
                images = pipe(
                    prompt=batch_prompts,
                    image=control_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=512,
                    width=512
                ).images
            else:
                # Unconditional generation (no control image)
                images = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=512,
                    width=512
                ).images
        
        all_images.extend(images)
    
    # Save images
    print(f"\nSaving {len(all_images)} samples...")
    
    for idx, img in enumerate(all_images):
        img_path = output_dir / f"controlnet_finetuned_{idx:04d}.png"
        img.save(img_path)
    
    # Create grid
    img_tensors = []
    for img in all_images:
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensors.append(img_tensor)
    
    grid_tensor = torch.stack(img_tensors)
    grid_n = int(np.sqrt(num_samples))
    grid_path = output_dir / "controlnet_finetuned_grid.png"
    save_image(grid_tensor, grid_path, nrow=grid_n, normalize=False)
    
    print(f"\n{'='*60}")
    print("Sampling Complete!")
    print(f"Individual samples: {output_dir}/controlnet_finetuned_*.png")
    print(f"Grid: {grid_path}")
    print(f"{'='*60}")
    
    return all_images


def main():
    parser = argparse.ArgumentParser(
        description="Use fine-tuned UNet with ControlNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unconditional generation:
  python baseline/use_finetuned_unet_with_controlnet.py \\
      --checkpoint_dir outputs/baseline_sd_finetuned_full/checkpoint-best \\
      --num_samples 16
  
  # With control image:
  python baseline/use_finetuned_unet_with_controlnet.py \\
      --checkpoint_dir outputs/baseline_sd_finetuned_full/checkpoint-best \\
      --num_samples 16 \\
      --control_image path/to/control.png \\
      --prompt "room layout"
        """
    )
    
    parser.add_argument("--checkpoint_dir", type=Path, required=True,
                       help="Directory containing checkpoint-best (should have 'unet' subdirectory)")
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to generate (default: 16)")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of DDIM steps (default: 50)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=Path, default="outputs/controlnet_finetuned_samples",
                       help="Output directory (default: outputs/controlnet_finetuned_samples)")
    parser.add_argument("--control_image", type=Path, default=None,
                       help="Optional control image path (PIL Image)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt (default: empty for unconditional)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu, default: cuda)")
    
    args = parser.parse_args()
    
    # Load control image if provided
    control_image = None
    if args.control_image is not None:
        control_image = Image.open(args.control_image).convert("RGB")
        print(f"Loaded control image: {args.control_image}")
    
    sample_with_controlnet(
        checkpoint_dir=args.checkpoint_dir,
        num_samples=args.num_samples,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_dir=args.output_dir,
        control_image=control_image,
        prompt=args.prompt,
        device=args.device
    )


if __name__ == "__main__":
    main()

