#!/usr/bin/env python3

import argparse
import sys
import yaml
from pathlib import Path
import torch
from torchvision.utils import save_image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel


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
    parser.add_argument(
        "--text-emb",
        type=str,
        default=None,
        help="Path to text embedding file (.pt)"
    )
    parser.add_argument(
        "--pov-emb",
        type=str,
        default=None,
        help="Path to POV embedding file (.pt)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale for classifier-free guidance"
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
    
    # Load config - for standalone checkpoints, config may come from checkpoint
    from training.utils import load_config
    config = load_config(diff_config)
    
    # If autoencoder config is provided, merge it
    if ae_config and ae_checkpoint:
        config["autoencoder"] = {"config": ae_config, "checkpoint": ae_checkpoint}
    
    # Load model from checkpoint (standalone checkpoint contains all needed info)
    diffusion = DiffusionModel.load_checkpoint(diff_checkpoint, map_location=device, config=config)
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    method = args.mode
    if method == "ddim":
        num_steps = 50
    else:
        num_steps = diffusion.scheduler.num_steps
    
    batch_size = args.batch_size
    eta = 0.0
    guidance_scale = args.guidance_scale
    
    # Load text and POV embeddings if provided
    text_emb = None
    pov_emb = None
    
    if args.text_emb:
        text_emb = torch.load(args.text_emb, map_location=device)
        if text_emb.dim() > 2:
            text_emb = text_emb.flatten(start_dim=1)
        # Expand to batch size if needed
        if text_emb.shape[0] == 1 and batch_size > 1:
            text_emb = text_emb.repeat(batch_size, 1)
        print(f"Loaded text embedding: {text_emb.shape}")
    
    if args.pov_emb:
        pov_emb = torch.load(args.pov_emb, map_location=device)
        if pov_emb.dim() > 2:
            pov_emb = pov_emb.flatten(start_dim=1)
        # Expand to batch size if needed
        if pov_emb.shape[0] == 1 and batch_size > 1:
            pov_emb = pov_emb.repeat(batch_size, 1)
        print(f"Loaded POV embedding: {pov_emb.shape}")
    
    with torch.no_grad():
        samples = diffusion.sample(
            batch_size=batch_size,
            num_steps=num_steps,
            method=method,
            eta=eta,
            device=device,
            guidance_scale=guidance_scale,
            text_emb=text_emb,
            pov_emb=pov_emb,
            verbose=True
        )
        
        # Extract RGB from samples dict
        if isinstance(samples, dict) and "rgb" in samples:
            samples = samples["rgb"]

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