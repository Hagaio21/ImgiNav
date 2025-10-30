#!/usr/bin/env python3
"""
Simple inference script for diffusion models.
"""

import argparse
import time
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import yaml

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import LatentDiffusion


def main():
    parser = argparse.ArgumentParser(description='Generate images from diffusion models')
    parser.add_argument('config', type=str, help='Path to the inference config YAML file')
    parser.add_argument('--output', type=str, default='generated.png', help='Output image path')
    parser.add_argument('--method', type=str, choices=['ddim', 'ddpm'], default='ddim', 
                       help='Sampling method (default: ddim)')
    parser.add_argument('--num_steps', type=int, default=None, 
                       help='Number of sampling steps (uses method default if not specified)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading model from config: {args.config}")
    
    # Build model
    model = LatentDiffusion.from_config(config, device=args.device)
    
    # Load checkpoint
    checkpoint_path = config.get('checkpoint')
    if not checkpoint_path:
        raise ValueError("No checkpoint path specified in config")
    
    state = torch.load(checkpoint_path, map_location=args.device)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load into model
    if 'state_dict' in state:
        model.backbone.load_state_dict(state['state_dict'], strict=False)
    elif 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    
    print("Model loaded successfully")
    
    # Determine number of steps
    if args.num_steps is None:
        if args.method == "ddim":
            num_steps = 50
        else:
            num_steps = model.scheduler.num_steps
        print(f"Using {args.method.upper()} default steps: {num_steps}")
    else:
        num_steps = args.num_steps
        print(f"Using user-specified steps: {num_steps}")
    
    # Generate image
    print(f"Generating image using {args.method.upper()} with {num_steps} steps...")
    start_time = time.time()
    
    model.eval()
    with torch.no_grad():
        image = model.sample(
            batch_size=1, 
            num_steps=num_steps, 
            method=args.method,
            device=args.device,
            image=True,
            verbose=True
        )
    
    generation_time = time.time() - start_time
    print(f"{args.method.upper()} generation time: {generation_time:.2f} seconds")
    
    # Convert to PIL Image
    image = image.squeeze(0).cpu()
    if image.min() < 0:
        image = (image + 1) / 2
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)
    
    # Save image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    print(f"Image saved to: {output_path}")
    print(f"Image size: {image.size}")


if __name__ == "__main__":
    main()
