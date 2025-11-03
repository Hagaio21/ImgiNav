#!/usr/bin/env python3
"""
Generate samples from a trained diffusion model checkpoint.

Usage:
    python scripts/inference.py \
        --checkpoint checkpoints/diffusion_ablation_capacity_unet64_d4_checkpoint_best.pt \
        --config experiments/diffusion/ablation/capacity_unet64_d4.yaml \
        --output outputs/samples \
        --num_samples 16 \
        --method ddpm
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained diffusion model"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to diffusion model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to diffusion model config YAML file (optional - checkpoint contains all config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference_samples",
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="Sampling method: ddpm or ddim"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of sampling steps (default: full for ddpm, 50 for ddim)"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0.0 = deterministic, 1.0 = stochastic)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Also save latent representations"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("="*60)
    print("Diffusion Model Inference")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config if args.config else '(using checkpoint config)'}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"Num samples: {args.num_samples}")
    print("="*60)
    
    # Load checkpoint - it contains all necessary config
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Config file is optional - checkpoint contains decoder config
    # If config is provided, it can override some settings, but decoder will come from checkpoint
    diffusion_cfg = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract diffusion config - handle nested structure
        if "model" in config and "diffusion" in config["model"]:
            diffusion_cfg = config["model"]["diffusion"]
        elif "diffusion" in config:
            diffusion_cfg = config["diffusion"]
        else:
            # Assume top-level config is already the diffusion config
            diffusion_cfg = {
                "autoencoder": config.get("autoencoder"),
                "unet": config.get("unet", {}),
                "scheduler": config.get("scheduler", {})
            }
            # If autoencoder not at top level, check if it's nested
            if not diffusion_cfg["autoencoder"] and "model" in config:
                diffusion_cfg["autoencoder"] = config["model"].get("autoencoder")
        
        print(f"Note: Config file provided, but decoder will be loaded from checkpoint")
    
    print(f"\nLoading model from checkpoint...")
    print(f"Checkpoint contains all model components (decoder, UNet, scheduler)")
    model = DiffusionModel.load_checkpoint(
        checkpoint_path,
        map_location=device,
        config=diffusion_cfg  # Optional - checkpoint has full config
    )
    
    model = model.to(device).eval()
    print("Model loaded successfully")
    
    # Determine number of steps
    if args.num_steps is None:
        if args.method == "ddim":
            num_steps = 50
        else:
            num_steps = model.scheduler.num_steps
    else:
        num_steps = args.num_steps
    
    print(f"\nSampling parameters:")
    print(f"  Method: {args.method}")
    print(f"  Steps: {num_steps}")
    print(f"  Eta: {args.eta}")
    print(f"  Batch size: {args.batch_size}")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    all_latents = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, args.num_samples, args.batch_size):
            batch_size = min(args.batch_size, args.num_samples - i)
            print(f"  Generating batch {i//args.batch_size + 1} ({batch_size} samples)...")
            
            result = model.sample(
                batch_size=batch_size,
                num_steps=num_steps,
                method=args.method,
                eta=args.eta,
                device=device,
                verbose=False
            )
            
            if "rgb" in result:
                all_samples.append(result["rgb"])
            if "latent" in result and args.save_latents:
                all_latents.append(result["latent"])
    
    # Concatenate all samples
    if all_samples:
        samples = torch.cat(all_samples, dim=0)
        print(f"Generated {len(samples)} RGB samples")
        
        # Save individual samples
        samples_dir = output_dir / "individual"
        samples_dir.mkdir(exist_ok=True)
        for idx, sample in enumerate(samples):
            save_image(sample, samples_dir / f"sample_{idx:04d}.png", normalize=False)
        
        # Save grid
        grid = make_grid(samples, nrow=4, padding=2)
        grid_path = output_dir / "samples_grid.png"
        save_image(grid, grid_path, normalize=False)
        print(f"Saved grid to: {grid_path}")
        print(f"Saved individual samples to: {samples_dir}/")
    
    if all_latents and args.save_latents:
        latents = torch.cat(all_latents, dim=0)
        latents_path = output_dir / "latents.pt"
        torch.save(latents, latents_path)
        print(f"Saved latents to: {latents_path}")
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

