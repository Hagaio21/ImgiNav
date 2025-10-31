#!/usr/bin/env python3

import argparse
import sys
import yaml
from pathlib import Path
import torch
from torchvision.utils import save_image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import LatentDiffusion


def save_samples(samples: torch.Tensor, output_path: Path, nrow: int = 4):
    save_image(samples, output_path, nrow=nrow, normalize=False)
    print(f"[Saved] Generated samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained diffusion model"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/diffusion_samples.png",
        help="Output path for generated samples"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ddim", "ddpm"],
        default="ddim",
        help="Sampling mode"
    )
    
    args = parser.parse_args()
    
    config_path = "config/inference_config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        inference_cfg = yaml.safe_load(f)
    
    diff_config = inference_cfg["diffusion"]["config"]
    diff_checkpoint = inference_cfg["diffusion"]["checkpoint"]
    ae_config = inference_cfg["autoencoder"]["config"]
    ae_checkpoint = inference_cfg["autoencoder"]["checkpoint"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(diff_config, "r", encoding="utf-8") as f:
        diffusion_cfg = yaml.safe_load(f)
    
    if "model" in diffusion_cfg and "diffusion" in diffusion_cfg["model"]:
        diffusion_cfg = diffusion_cfg["model"]["diffusion"]
    
    diffusion_cfg["autoencoder"] = {"config": ae_config, "checkpoint": ae_checkpoint}
    
    diffusion = LatentDiffusion.from_config(diffusion_cfg, device=device)
    
    state = torch.load(diff_checkpoint, map_location=device)
    loaded_state = state.get("state_dict", state.get("model", state))
    if loaded_state and list(loaded_state.keys())[0].startswith('module.'):
        loaded_state = {k[7:]: v for k, v in loaded_state.items()}
    diffusion.backbone.load_state_dict(loaded_state, strict=False)
    
    diffusion.eval()
    
    method = args.mode
    if method == "ddim":
        num_steps = 50
    else:
        num_steps = diffusion.scheduler.num_steps
    
    batch_size = 4
    eta = 0.0
    guidance_scale = 1.0
    
    with torch.no_grad():
        samples = diffusion.sample(
            batch_size=batch_size,
            image=True,
            cond=None,
            num_steps=num_steps,
            method=method,
            eta=eta,
            device=device,
            guidance_scale=guidance_scale,
            verbose=True
        )

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