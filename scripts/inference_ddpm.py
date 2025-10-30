#!/usr/bin/env python3
"""
DDPM inference script for diffusion experiments.

This script loads a diffusion experiment and generates images using DDPM sampling.
"""

import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import time

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.builder import build_model
from models.autoencoder import AutoEncoder
import yaml


def load_experiment_config(experiment_path: Path) -> dict:
    """Load experiment configuration."""
    config_path = experiment_path / "output" / "experiment_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def update_config_paths(config: dict, experiment_path: Path) -> dict:
    """Update absolute paths in config to be relative to current working directory."""
    # Update autoencoder paths
    if 'autoencoder' in config['model']:
        ae_config = config['model']['autoencoder']
        
        # Update autoencoder config path
        if 'config' in ae_config:
            old_config_path = ae_config['config']
            if '/work3/s233249/ImgiNav/' in old_config_path:
                # Replace with local path
                new_config_path = old_config_path.replace('/work3/s233249/ImgiNav/', '')
                if Path(new_config_path).exists():
                    ae_config['config'] = new_config_path
                    print(f"Updated autoencoder config path: {new_config_path}")
        
        # Update autoencoder checkpoint path
        if 'checkpoint' in ae_config:
            old_checkpoint_path = ae_config['checkpoint']
            if '/work3/s233249/ImgiNav/' in old_checkpoint_path:
                # Replace with local path
                new_checkpoint_path = old_checkpoint_path.replace('/work3/s233249/ImgiNav/', '')
                if Path(new_checkpoint_path).exists():
                    ae_config['checkpoint'] = new_checkpoint_path
                    print(f"Updated autoencoder checkpoint path: {new_checkpoint_path}")
    
    return config


def build_diffusion_model(config: dict, device: str) -> tuple:
    """Build diffusion model from config."""
    # Add model type
    config['model']['type'] = 'diffusion'
    
    # Build autoencoder first
    if 'autoencoder' in config['model']:
        ae_config = config['model']['autoencoder']
        ae_checkpoint = ae_config.get('checkpoint')
        
        # Load autoencoder config
        ae_config_path = ae_config.get('config')
        if ae_config_path and Path(ae_config_path).exists():
            with open(ae_config_path, 'r') as f:
                ae_config_dict = yaml.safe_load(f)
            
            # Build autoencoder using the config
            autoencoder = AutoEncoder.from_config(ae_config_dict)
            
            # Load checkpoint if available
            if ae_checkpoint and Path(ae_checkpoint).exists():
                ae_state = torch.load(ae_checkpoint, map_location=device)
                autoencoder.load_state_dict(ae_state.get("model", ae_state), strict=False)
            
            autoencoder.eval().to(device)
            
            # Update the model config to pass the built autoencoder
            config['model']['autoencoder'] = autoencoder
        else:
            raise FileNotFoundError(f"Autoencoder config not found: {ae_config_path}")
    
    # Build model using the builder
    model, aux_model = build_model(config['model'], device=device)
    
    return model, aux_model


def load_checkpoint(model, checkpoint_path: Path, device: str) -> bool:
    """Load checkpoint into model."""
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
        
    state = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {list(state.keys())}")
    
    # Check if this is a state_dict checkpoint
    if 'state_dict' in state:
        print("Found 'state_dict' key, loading from it")
        state_dict = state['state_dict']
        print(f"State dict keys: {list(state_dict.keys())[:10]}...")  # Show first 10 keys
        
        # Try to load into the backbone (UNet) part of the model
        try:
            missing_keys, unexpected_keys = model.backbone.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                print(f"First few missing keys: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"First few unexpected keys: {unexpected_keys[:5]}")
        except Exception as e:
            print(f"Error loading into backbone: {e}")
            return False
    elif 'model' in state:
        print("Loading from 'model' key")
        model.load_state_dict(state['model'], strict=False)
    elif 'unet' in state:
        print("Loading from 'unet' key")
        model.load_state_dict(state['unet'], strict=False)
    else:
        print("Loading from root state")
        model.load_state_dict(state, strict=False)
    
    print(f"Successfully loaded checkpoint: {checkpoint_path}")
    return True


def find_latest_checkpoint(experiment_path: Path) -> Path:
    """Find the latest checkpoint in the experiment directory."""
    checkpoint_dir = experiment_path / "checkpoints"
    
    # Look for common checkpoint patterns
    patterns = ['unet_latest.pt', 'unet_best.pt', '*.pt']
    
    for pattern in patterns:
        checkpoints = list(checkpoint_dir.glob(pattern))
        if checkpoints:
            # Sort by modification time and return the latest
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return checkpoints[0]
            
    return None


def generate_image(model, autoencoder, device: str, num_steps: int = None) -> Image.Image:
    """Generate an image using DDPM sampling."""
    model.eval()
    autoencoder.eval()
    
    with torch.no_grad():
        # Use the model's sample method with image=True to get decoded images directly
        if hasattr(model, 'sample'):
            # Time the generation
            start_time = time.time()
            
            # Generate image using DDPM sampling
            image = model.sample(
                batch_size=1, 
                num_steps=num_steps, 
                method='ddpm',
                device=device,
                image=True,  # This will automatically decode using the autoencoder
                verbose=True
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"DDPM generation time: {generation_time:.2f} seconds ({generation_time/num_steps:.3f} seconds per step)")
        else:
            raise AttributeError("Model does not support sampling")
        
        # Convert to PIL Image
        # The image is already decoded and should be in range [0, 1] or [-1, 1]
        image = image.squeeze(0).cpu()  # Remove batch dimension
        
        # Normalize to [0, 1] if needed
        if image.min() < 0:
            image = (image + 1) / 2
        
        # Clamp to [0, 1]
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy and then to PIL
        image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
        image_np = (image_np * 255).astype(np.uint8)
        
        return Image.fromarray(image_np)


def load_experiment_and_model(experiment_path: str, checkpoint_path: str = None, device: str = "cpu"):
    """Load experiment and build model."""
    experiment_path = Path(experiment_path)
    
    # Load experiment config
    config = load_experiment_config(experiment_path)
    print(f"Loaded experiment: {config['experiment']['name']}")
    
    # Update paths
    config = update_config_paths(config, experiment_path)
    
    # Build model
    model, autoencoder = build_diffusion_model(config, device)
    
    # Load checkpoint if specified
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        success = load_checkpoint(model, checkpoint_path, device)
        if not success:
            raise RuntimeError("Failed to load checkpoint")
    else:
        # Try to find latest checkpoint
        checkpoint = find_latest_checkpoint(experiment_path)
        if checkpoint:
            success = load_checkpoint(model, checkpoint, device)
            if not success:
                print("Warning: Failed to load checkpoint")
        else:
            print("Warning: No checkpoint found")
    
    # Debug: Print model info
    print(f"Model type: {type(model)}")
    print(f"Model scheduler: {type(model.scheduler)}")
    print(f"Scheduler num_steps: {model.scheduler.num_steps}")
    print(f"Model latent_shape: {model.latent_shape}")
    print(f"Autoencoder type: {type(autoencoder)}")
    
    # Check if model parameters are actually loaded
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check a few parameter values to see if they look reasonable
    backbone_params = list(model.backbone.parameters())
    if backbone_params:
        first_param = backbone_params[0]
        print(f"First backbone parameter shape: {first_param.shape}")
        print(f"First backbone parameter mean: {first_param.mean().item():.6f}")
        print(f"First backbone parameter std: {first_param.std().item():.6f}")
    
    return model, autoencoder


def main():
    parser = argparse.ArgumentParser(description='Generate images from diffusion experiments using DDPM')
    parser.add_argument('experiment_path', type=str, help='Path to the diffusion experiment directory')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file (default: latest checkpoint)')
    parser.add_argument('--output', type=str, default='ddpm_generated.png', help='Output image path')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of sampling steps (uses scheduler default if not specified)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")
    
    print(f"Loading experiment from: {experiment_path}")
    
    # Load experiment and build model
    print("Building diffusion model...")
    model, autoencoder = load_experiment_and_model(
        str(experiment_path), 
        args.checkpoint, 
        args.device
    )
    
    # Use scheduler's default steps if not specified
    if args.num_steps is None:
        num_steps = model.scheduler.num_steps
        print(f"Using scheduler's default steps: {num_steps}")
    else:
        num_steps = args.num_steps
        print(f"Using user-specified steps: {num_steps}")
    
    # Generate image
    print(f"Generating image using DDPM with {num_steps} steps...")
    image = generate_image(model, autoencoder, args.device, num_steps)
    
    # Save image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    print(f"DDPM image saved to: {output_path}")
    print(f"Image size: {image.size}")


if __name__ == "__main__":
    main()
