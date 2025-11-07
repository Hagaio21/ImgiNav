#!/usr/bin/env python3
"""
Compare custom diffusion model with Stable Diffusion baseline.

Generates samples from both models with the same parameters and saves
them side-by-side for easy comparison.

Usage:
    # Compare Stage 1 checkpoint with pretrained SD
    python baseline/compare_models.py \
        --custom_checkpoint /path/to/stage1/checkpoint_best.pt \
        --custom_config /path/to/stage1/config.yaml \
        --sd_model_id runwayml/stable-diffusion-v1-5 \
        --num_samples 64 \
        --num_steps 50 \
        --output_dir outputs/comparison_stage1_vs_sd
    
    # Compare Stage 3 checkpoint with fine-tuned SD
    python baseline/compare_models.py \
        --custom_checkpoint /path/to/stage3/checkpoint_best.pt \
        --custom_config /path/to/stage3/config.yaml \
        --sd_model_dir /path/to/finetuned_sd/pipeline \
        --num_samples 64 \
        --num_steps 50 \
        --output_dir outputs/comparison_stage3_vs_sd_finetuned
"""

import argparse
import torch
from pathlib import Path
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import load_config
from models.diffusion import DiffusionModel

try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not installed. Install with: pip install diffusers transformers accelerate")


def sample_custom_model(
    checkpoint_path,
    config_path,
    num_samples=64,
    num_steps=50,
    device="cuda",
    seed=42
):
    """
    Sample from custom diffusion model.
    
    Args:
        checkpoint_path: Path to checkpoint
        config_path: Path to config YAML
        num_samples: Number of samples
        num_steps: DDIM steps
        device: Device
        seed: Random seed
    
    Returns:
        List of PIL Images
    """
    print(f"\n{'='*60}")
    print("Custom Model Sampling")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device_obj = torch.device(device)
    
    # Load config
    config = load_config(config_path)
    
    # Load model
    print("Loading custom model...")
    model = DiffusionModel.load_checkpoint(
        checkpoint_path,
        map_location=device,
        config=config
    )
    model = model.to(device_obj)
    model.eval()
    
    print(f"Model loaded. Generating {num_samples} samples with {num_steps} DDIM steps...")
    
    # Generate samples in batches
    batch_size = 8  # Adjust based on GPU memory
    all_images = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Custom model sampling"):
        current_batch = min(batch_size, num_samples - i)
        
        with torch.no_grad():
            sample_output = model.sample(
                batch_size=current_batch,
                num_steps=num_steps,
                method="ddim",
                eta=0.0,
                device=device_obj,
                verbose=False
            )
        
        # Decode latents to images
        # sample_output is a dict with "latent" key
        latents = sample_output.get("latent")
        decoded = model.decoder({"latent": latents})
        decoded_rgb = decoded.get("rgb")
        
        # Convert to [0, 1] range for PIL
        if decoded_rgb.min() < 0:
            decoded_rgb = (decoded_rgb + 1.0) / 2.0
        decoded_rgb = torch.clamp(decoded_rgb, 0.0, 1.0)
        
        # Convert to PIL images
        for j in range(current_batch):
            img_tensor = decoded_rgb[j].cpu()
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            all_images.append(img_pil)
    
    print(f"Generated {len(all_images)} samples from custom model")
    return all_images


def sample_sd_model(
    model_id=None,
    model_dir=None,
    num_samples=64,
    num_steps=50,
    guidance_scale=1.0,
    device="cuda",
    seed=42
):
    """
    Sample from Stable Diffusion (pretrained or fine-tuned).
    
    Args:
        model_id: HuggingFace model ID (for pretrained)
        model_dir: Directory with fine-tuned pipeline (for fine-tuned)
        num_samples: Number of samples
        num_steps: DDIM steps
        guidance_scale: Guidance scale (1.0 for unconditional)
        device: Device
        seed: Random seed
    
    Returns:
        List of PIL Images
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required")
    
    print(f"\n{'='*60}")
    print("Stable Diffusion Sampling")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device_obj = torch.device(device)
    
    # Load pipeline
    if model_dir:
        print(f"Loading fine-tuned model from: {model_dir}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        model_name = "Fine-tuned SD"
    elif model_id:
        print(f"Loading pretrained model: {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        model_name = f"Pretrained SD ({model_id})"
    else:
        raise ValueError("Must provide either model_id or model_dir")
    
    pipe = pipe.to(device_obj)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print(f"Model: {model_name}")
    print(f"Generating {num_samples} samples with {num_steps} DDIM steps...")
    print(f"Guidance scale: {guidance_scale} (unconditional)")
    
    # Generate samples in batches
    batch_size = 4
    all_images = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc="SD sampling"):
        current_batch = min(batch_size, num_samples - i)
        
        with torch.no_grad():
            images = pipe(
                prompt=[""] * current_batch,  # Empty prompt for unconditional
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images
        
        all_images.extend(images)
    
    print(f"Generated {len(all_images)} samples from SD")
    return all_images


def create_comparison_grid(
    custom_images,
    sd_images,
    output_path,
    nrow=8
):
    """
    Create side-by-side comparison grid.
    
    Args:
        custom_images: List of PIL Images from custom model
        sd_images: List of PIL Images from SD
        output_path: Path to save grid
        nrow: Number of images per row
    """
    # Ensure same number of images
    num_images = min(len(custom_images), len(sd_images))
    custom_images = custom_images[:num_images]
    sd_images = sd_images[:num_images]
    
    # Convert PIL to tensors
    custom_tensors = []
    sd_tensors = []
    
    for img in custom_images:
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            if img_array.ndim == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
        else:
            img_tensor = img
        custom_tensors.append(img_tensor)
    
    for img in sd_images:
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            if img_array.ndim == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
        else:
            img_tensor = img
        sd_tensors.append(img_tensor)
    
    # Stack tensors
    custom_grid = torch.stack(custom_tensors)
    sd_grid = torch.stack(sd_tensors)
    
    # Create side-by-side comparison
    # Reshape to grid format
    num_rows = (num_images + nrow - 1) // nrow
    
    comparison_rows = []
    for row_idx in range(num_rows):
        start_idx = row_idx * nrow
        end_idx = min(start_idx + nrow, num_images)
        
        # Get row from each model
        custom_row = custom_grid[start_idx:end_idx]
        sd_row = sd_grid[start_idx:end_idx]
        
        # Pad if needed
        if len(custom_row) < nrow:
            padding = torch.zeros(
                nrow - len(custom_row),
                *custom_row.shape[1:]
            )
            custom_row = torch.cat([custom_row, padding], dim=0)
            sd_row = torch.cat([sd_row, padding], dim=0)
        
        # Concatenate side-by-side
        comparison_row = torch.cat([custom_row, sd_row], dim=0)  # [2*nrow, C, H, W]
        comparison_rows.append(comparison_row)
    
    # Stack rows
    comparison_grid = torch.stack(comparison_rows)  # [num_rows, 2*nrow, C, H, W]
    # Reshape to [num_rows * 2*nrow, C, H, W]
    comparison_grid = comparison_grid.view(-1, *comparison_grid.shape[2:])
    
    # Save grid
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(comparison_grid, output_path, nrow=nrow * 2, normalize=False)
    
    print(f"\nComparison grid saved to: {output_path}")
    print(f"  Left side: Custom model")
    print(f"  Right side: Stable Diffusion")


def compare_models(
    custom_checkpoint,
    custom_config,
    sd_model_id=None,
    sd_model_dir=None,
    num_samples=64,
    num_steps=50,
    seed=42,
    output_dir="outputs/comparison",
    device="cuda"
):
    """
    Compare custom model with SD baseline.
    
    Args:
        custom_checkpoint: Path to custom model checkpoint
        custom_config: Path to custom model config
        sd_model_id: HuggingFace model ID (for pretrained SD)
        sd_model_dir: Directory with fine-tuned SD pipeline
        num_samples: Number of samples to generate
        num_steps: DDIM steps
        seed: Random seed
        output_dir: Output directory
        device: Device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print(f"DDIM steps: {num_steps}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    
    # Sample from custom model
    custom_images = sample_custom_model(
        checkpoint_path=custom_checkpoint,
        config_path=custom_config,
        num_samples=num_samples,
        num_steps=num_steps,
        device=device,
        seed=seed
    )
    
    # Sample from SD
    sd_images = sample_sd_model(
        model_id=sd_model_id,
        model_dir=sd_model_dir,
        num_samples=num_samples,
        num_steps=num_steps,
        guidance_scale=1.0,
        device=device,
        seed=seed
    )
    
    # Save individual images
    print(f"\nSaving individual images...")
    custom_dir = output_dir / "custom_model"
    sd_dir = output_dir / "stable_diffusion"
    custom_dir.mkdir(parents=True, exist_ok=True)
    sd_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, img in enumerate(custom_images):
        img_path = custom_dir / f"custom_{idx:04d}.png"
        img.save(img_path)
    
    for idx, img in enumerate(sd_images):
        img_path = sd_dir / f"sd_{idx:04d}.png"
        img.save(img_path)
    
    # Create separate grids
    grid_n = int(np.sqrt(num_samples))
    
    custom_grid_path = output_dir / "custom_model_grid.png"
    sd_grid_path = output_dir / "sd_grid.png"
    
    # Custom model grid
    custom_tensors = []
    for img in custom_images:
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        else:
            img_tensor = img
        custom_tensors.append(img_tensor)
    custom_grid = torch.stack(custom_tensors)
    save_image(custom_grid, custom_grid_path, nrow=grid_n, normalize=False)
    
    # SD grid
    sd_tensors = []
    for img in sd_images:
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        else:
            img_tensor = img
        sd_tensors.append(img_tensor)
    sd_grid = torch.stack(sd_tensors)
    save_image(sd_grid, sd_grid_path, nrow=grid_n, normalize=False)
    
    # Create side-by-side comparison grid
    comparison_path = output_dir / "comparison_grid.png"
    create_comparison_grid(custom_images, sd_images, comparison_path, nrow=grid_n)
    
    print(f"\n{'='*60}")
    print("Comparison Complete!")
    print(f"{'='*60}")
    print(f"Custom model samples: {custom_dir}")
    print(f"SD samples: {sd_dir}")
    print(f"Custom grid: {custom_grid_path}")
    print(f"SD grid: {sd_grid_path}")
    print(f"Comparison grid: {comparison_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Compare custom diffusion model with SD baseline")
    
    # Custom model args
    parser.add_argument("--custom_checkpoint", type=Path, required=True,
                       help="Path to custom model checkpoint")
    parser.add_argument("--custom_config", type=Path, required=True,
                       help="Path to custom model config YAML")
    
    # SD model args (one of these required)
    parser.add_argument("--sd_model_id", type=str, default=None,
                       help="HuggingFace model ID for pretrained SD (e.g., 'runwayml/stable-diffusion-v1-5')")
    parser.add_argument("--sd_model_dir", type=Path, default=None,
                       help="Directory with fine-tuned SD pipeline")
    
    # Sampling args
    parser.add_argument("--num_samples", type=int, default=64,
                       help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of DDIM steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output args
    parser.add_argument("--output_dir", type=Path, default="outputs/comparison",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate SD model args
    if not args.sd_model_id and not args.sd_model_dir:
        parser.error("Must provide either --sd_model_id or --sd_model_dir")
    
    compare_models(
        custom_checkpoint=args.custom_checkpoint,
        custom_config=args.custom_config,
        sd_model_id=args.sd_model_id,
        sd_model_dir=args.sd_model_dir,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()

