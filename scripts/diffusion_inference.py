#!/usr/bin/env python3
"""
Diffusion model inference script.

Loads a trained diffusion model and autoencoder from config files and checkpoints,
then generates samples using either DDIM or DDPM sampling.
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
import torch
import numpy as np
from torchvision.utils import save_image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import LatentDiffusion
from models.autoencoder import AutoEncoder
from models.components.unet import DualUNet
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler


def load_autoencoder(ae_config_path: str, ae_checkpoint_path: str, device: str = "cuda"):
    """Load autoencoder from config and checkpoint."""
    print(f"[Loading] Autoencoder config: {ae_config_path}")
    
    with open(ae_config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    
    # Extract model section from config (handle both flat and nested formats)
    # The autoencoder_config.yaml from experiments has encoder/decoder directly
    if "encoder" in full_cfg and "decoder" in full_cfg:
        # This is already in the right format (encoder/decoder sections)
        model_cfg = full_cfg
    elif "model" in full_cfg:
        model_cfg = full_cfg["model"]
    else:
        model_cfg = full_cfg
    
    # Build autoencoder
    ae = AutoEncoder.from_config(model_cfg, legacy_mode=False)
    
    # Load checkpoint
    if ae_checkpoint_path and Path(ae_checkpoint_path).exists():
        print(f"[Loading] Autoencoder checkpoint: {ae_checkpoint_path}")
        state = torch.load(ae_checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        if "model" in state:
            loaded_state = state["model"]
        else:
            loaded_state = state
        
        # Handle DataParallel prefix if present
        if loaded_state and list(loaded_state.keys())[0].startswith('module.'):
            loaded_state = {k[7:]: v for k, v in loaded_state.items()}
        
        missing_keys, unexpected_keys = ae.load_state_dict(loaded_state, strict=False)
        if missing_keys:
            print(f"[Warning] Missing keys when loading autoencoder: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"[Warning] Unexpected keys when loading autoencoder: {len(unexpected_keys)} keys")
    else:
        print(f"[Warning] Autoencoder checkpoint not found: {ae_checkpoint_path}")
    
    ae.eval().to(device)
    return ae


def build_scheduler(sched_cfg: dict):
    """Build scheduler from config."""
    sched_type = sched_cfg.get("type", "cosine").lower()
    num_steps = sched_cfg.get("num_steps", 1000)
    
    if sched_type == "cosine":
        return CosineScheduler(num_steps=num_steps)
    elif sched_type == "linear":
        return LinearScheduler(num_steps=num_steps)
    elif sched_type == "quadratic":
        return QuadraticScheduler(num_steps=num_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}. Choose 'cosine', 'linear', or 'quadratic'.")


def build_unet(unet_cfg: dict):
    """Build UNet from config."""
    return DualUNet.from_config(unet_cfg)


def load_diffusion_model(
    diff_config_path: str,
    diff_checkpoint_path: str,
    ae_config_path: str,
    ae_checkpoint_path: str,
    device: str = "cuda"):
    """Load diffusion model from configs and checkpoints."""
    print(f"[Loading] Diffusion config: {diff_config_path}")
    
    # Load diffusion config
    with open(diff_config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    
    # Handle nested config structure (like in training configs)
    if "model" in full_cfg and "diffusion" in full_cfg["model"]:
        model_cfg = full_cfg["model"]
        diff_cfg = model_cfg["diffusion"].copy()
        
        # If autoencoder info is in the main config, use it
        if "autoencoder" not in diff_cfg and "autoencoder" in model_cfg:
            diff_cfg["autoencoder"] = model_cfg["autoencoder"]
    else:
        # Assume flat structure
        diff_cfg = full_cfg.get("diffusion", full_cfg)
    
    # Load autoencoder - always use command-line arguments if provided (they take precedence)
    # Only fall back to config paths if CLI args are not provided
    if ae_config_path and Path(ae_config_path).exists():
        ae_cfg_path = ae_config_path
    elif "autoencoder" in diff_cfg:
        ae_cfg_path = diff_cfg["autoencoder"].get("config", ae_config_path)
        # Convert absolute Linux paths to relative if they don't exist
        if not Path(ae_cfg_path).exists() and ae_cfg_path.startswith("/"):
            # Try to extract relative path or use the CLI arg
            ae_cfg_path = ae_config_path if ae_config_path else ae_cfg_path
    else:
        ae_cfg_path = ae_config_path
    
    if ae_checkpoint_path and Path(ae_checkpoint_path).exists():
        ae_ckpt_path = ae_checkpoint_path
    elif "autoencoder" in diff_cfg:
        ae_ckpt_path = diff_cfg["autoencoder"].get("checkpoint", ae_checkpoint_path)
        # Convert absolute Linux paths to relative if they don't exist
        if not Path(ae_ckpt_path).exists() and ae_ckpt_path.startswith("/"):
            # Try to extract relative path or use the CLI arg
            ae_ckpt_path = ae_checkpoint_path if ae_checkpoint_path else ae_ckpt_path
    else:
        ae_ckpt_path = ae_checkpoint_path
    
    autoencoder = load_autoencoder(ae_cfg_path, ae_ckpt_path, device)
    
    # Build scheduler
    scheduler_cfg = diff_cfg.get("scheduler", {}).copy()
    # Normalize scheduler type to lowercase (build_scheduler expects lowercase)
    sched_type = scheduler_cfg.get("type", "cosine")
    if isinstance(sched_type, str):
        sched_type = sched_type.lower()
        # Handle class names (e.g., "CosineScheduler" -> "cosine")
        if "cosine" in sched_type.lower():
            sched_type = "cosine"
        elif "linear" in sched_type.lower():
            sched_type = "linear"
        elif "quadratic" in sched_type.lower():
            sched_type = "quadratic"
        scheduler_cfg["type"] = sched_type
    
    scheduler = build_scheduler(scheduler_cfg)
    
    # Build UNet
    unet_cfg = diff_cfg.get("unet", {})
    # Handle nested unet config (from_config expects either config dict or path)
    if isinstance(unet_cfg, dict) and "config" in unet_cfg:
        unet_config_dict = unet_cfg["config"]
    else:
        unet_config_dict = unet_cfg
    
    unet = build_unet(unet_config_dict)
    
    # Calculate latent shape from autoencoder
    latent_shape = (
        autoencoder.encoder.latent_channels,
        autoencoder.encoder.latent_base,
        autoencoder.encoder.latent_base,
    )
    
    # Create LatentDiffusion model
    diffusion = LatentDiffusion(
        backbone=unet,
        scheduler=scheduler,
        autoencoder=autoencoder,
        latent_shape=latent_shape,
    ).to(device)
    
    # Load diffusion checkpoint
    if diff_checkpoint_path and Path(diff_checkpoint_path).exists():
        print(f"[Loading] Diffusion checkpoint: {diff_checkpoint_path}")
        state = torch.load(diff_checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if "state_dict" in state:
            loaded_state = state["state_dict"]
        elif "model" in state:
            loaded_state = state["model"]
        else:
            loaded_state = state
        
        # Handle DataParallel prefix if present
        if loaded_state and list(loaded_state.keys())[0].startswith('module.'):
            loaded_state = {k[7:]: v for k, v in loaded_state.items()}
        
        missing_keys, unexpected_keys = diffusion.backbone.load_state_dict(loaded_state, strict=False)
        if missing_keys:
            print(f"[Warning] Missing keys when loading diffusion model: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"[Warning] Unexpected keys when loading diffusion model: {len(unexpected_keys)} keys")
        print(f"[Loaded] Diffusion model checkpoint loaded successfully")
    else:
        print(f"[Warning] Diffusion checkpoint not found: {diff_checkpoint_path}")
    
    diffusion.eval()
    return diffusion


def save_samples(samples: torch.Tensor, output_path: Path, nrow: int = 4):
    """Save generated samples as images."""

    save_image(
        samples,
        output_path,
        nrow=nrow,
        normalize=False  # Trust the [0, 1] range from the AE
    )

    
    print(f"[Saved] Generated samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with DDPM
  python scripts/diffusion_inference.py \\
    --diff-config config/models/diffusion/E2_Cosine_64.yaml \\
    --diff-checkpoint experiments/Diffusion_Uncond/E2_Cosine_64/checkpoints/unet_latest.pt \\
    --ae-config config/models/autoencoders/AE_small_latent.yaml \\
    --ae-checkpoint experiments/AEVAE_sweep/AE_small_latent/checkpoints/ae_latest.pt

  # Use DDIM with fewer steps
  python scripts/diffusion_inference.py \\
    --diff-config config/models/diffusion/E2_Cosine_64.yaml \\
    --diff-checkpoint experiments/.../unet_latest.pt \\
    --ae-config config/models/autoencoders/AE_small_latent.yaml \\
    --ae-checkpoint experiments/.../ae_latest.pt \\
    --method ddim --num-steps 50
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--diff-config",
        type=str,
        required=True,
        help="Path to diffusion model config YAML file"
    )
    parser.add_argument(
        "--diff-checkpoint",
        type=str,
        required=True,
        help="Path to diffusion model checkpoint file"
    )
    parser.add_argument(
        "--ae-config",
        type=str,
        required=True,
        help="Path to autoencoder config YAML file"
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=str,
        required=True,
        help="Path to autoencoder checkpoint file"
    )
    
    # Sampling arguments
    parser.add_argument(
        "--method",
        type=str,
        choices=["ddim", "ddpm"],
        default="ddim",
        help="Sampling method: 'ddim' (default, faster) or 'ddpm' (full stochastic)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of sampling steps (default: 50 for DDIM, full steps for DDPM)"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0.0 = deterministic DDIM, 1.0 = DDPM-like stochastic)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples to generate (default: 4)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0 = no guidance)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/diffusion_samples.png",
        help="Output path for generated samples (default: outputs/diffusion_samples.png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save intermediate images during diffusion sampling to process/ directory"
    )
    parser.add_argument(
        "--history-rate",
        type=int,
        default=10,
        help="Save every Nth intermediate step when --save-history is enabled (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        print(f"[Seed] Set random seed to {args.seed}")
    
    # Validate paths and permissions
    paths = {
        "diff_config": args.diff_config,
        "diff_checkpoint": args.diff_checkpoint,
        "ae_config": args.ae_config,
        "ae_checkpoint": args.ae_checkpoint,
    }
    
    for name, path in paths.items():
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"{name.replace('_', ' ').title()}: {path}")
        if not path_obj.is_file():
            raise ValueError(f"{name.replace('_', ' ').title()} is not a file: {path}")
        if not os.access(path_obj, os.R_OK):
            raise PermissionError(f"No read permission for {name.replace('_', ' ').title()}: {path}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading Diffusion Model")
    print("="*60)
    diffusion = load_diffusion_model(
        diff_config_path=args.diff_config,
        diff_checkpoint_path=args.diff_checkpoint,
        ae_config_path=args.ae_config,
        ae_checkpoint_path=args.ae_checkpoint,
        device=args.device
    )
    
    # Determine number of steps
    if args.num_steps is None:
        if args.method == "ddim":
            num_steps = 50  # Default for DDIM
        else:
            num_steps = diffusion.scheduler.num_steps  # Full steps for DDPM
    else:
        num_steps = args.num_steps
    
    print(f"\n[Sampling] Method: {args.method.upper()}")
    print(f"[Sampling] Steps: {num_steps}")
    print(f"[Sampling] Batch size: {args.batch_size}")
    if args.method == "ddim":
        print(f"[Sampling] Eta: {args.eta}")
    if args.guidance_scale != 1.0:
        print(f"[Sampling] Guidance scale: {args.guidance_scale}")
    
    # Generate samples
    print("\n" + "="*60)
    print("Generating Samples")
    print("="*60)
    
    # Setup process directory if saving history
    process_dir = None
    if args.save_history:
        output_path = Path(args.output)
        process_dir = output_path.parent / "process"
        process_dir.mkdir(parents=True, exist_ok=True)
        print(f"[History] Will save intermediate images to: {process_dir}")
    
    with torch.no_grad():
        if args.save_history:
            # Get full history of intermediate latents
            samples, history = diffusion.sample(
                batch_size=args.batch_size,
                image=False,  # Keep as latents to decode manually for history
                cond=None,   # Unconditional generation
                num_steps=num_steps,
                method=args.method,
                eta=args.eta,
                device=args.device,
                guidance_scale=args.guidance_scale,
                return_full_history=True,
                verbose=True
            )
            
            # Save intermediate images
            latents_history = history.get("latents", [])
            print(f"\n[History] Saving {len(latents_history)} intermediate latents...")
            
            for step_idx, latent in enumerate(latents_history):
                # Only save every Nth step (and always save the last one)
                if step_idx % args.history_rate != 0 and step_idx != len(latents_history) - 1:
                    continue
                
                # Decode latent to image
                with torch.no_grad():
                    rgb_out, _ = diffusion.autoencoder.decode(latent, from_latent=True)
                
                # Save intermediate image
                history_path = process_dir / f"step_{step_idx:04d}.png"
                save_samples(rgb_out, history_path, nrow=args.batch_size)
            
            # Decode final sample
            decoded = diffusion.autoencoder.decode(samples, from_latent=True)[0]
            samples = decoded
        else:
            # Normal sampling (no history)
            samples = diffusion.sample(
                batch_size=args.batch_size,
                image=True,  # Decode to RGB images via autoencoder
                cond=None,   # Unconditional generation
                num_steps=num_steps,
                method=args.method,
                eta=args.eta,
                device=args.device,
                guidance_scale=args.guidance_scale,
                verbose=True
            )
            
            # Training code always checks and decodes if needed (line 293 in diffusion_trainer.py)
            decoded = samples if samples.shape[1] != diffusion.autoencoder.encoder.latent_channels else diffusion.autoencoder.decode(samples, from_latent=True)[0]
            samples = decoded

    # Save final samples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_samples(samples, output_path)
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print(f"Samples saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()