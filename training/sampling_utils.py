"""
Sampling utilities for diffusion model inference during training.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image


@torch.no_grad()
def generate_samples_unconditioned(diffusion_model, exp_dir, epoch, num_samples, device):
    """
    Generate and save sample images without conditioning using LatentDiffusion.
    Uses fixed noise for consistency across epochs.
    
    Args:
        diffusion_model: LatentDiffusion model
        exp_dir: Experiment directory
        epoch: Current epoch number
        num_samples: Number of samples to generate
        device: Device string
    """
    diffusion_model.eval()
    exp_dir = Path(exp_dir)
    
    # Load or create fixed latents for reproducibility
    fixed_latents_path = exp_dir / "fixed_latents.pt"
    if fixed_latents_path.exists():
        # Load existing fixed noise
        fixed_noise = torch.load(fixed_latents_path).to(device)
    else:
        # Create and save fixed noise
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        C, H, W = diffusion_model.latent_shape
        fixed_noise = torch.randn(num_samples, C, H, W)
        torch.save(fixed_noise.cpu(), fixed_latents_path)
        fixed_noise = fixed_noise.to(device)
    
    # Generate samples using LatentDiffusion
    images = diffusion_model.sample(
        batch_size=num_samples,
        image=True,  # Decode to images
        cond=None,
        device=device
    )
    
    # Save images
    sample_path = exp_dir / 'samples' / f'epoch_{epoch+1}.png'
    save_image(images, sample_path, nrow=int(num_samples**0.5), normalize=True)
    print(f"Saved samples to {sample_path}", flush=True)
    
    diffusion_model.train()


@torch.no_grad()
def generate_samples_conditioned(diffusion_model, mixer, samples, exp_dir, epoch, device):
    """
    Generate conditioned samples and compare with targets using LatentDiffusion.
    """
    diffusion_model.eval()
    diffusion_model.autoencoder.eval()
    exp_dir = Path(exp_dir)
    num_samples = len(samples)

    # --------------------------------------------------------
    # Prepare conditioning tensors (preserve alignment)
    cond_povs = [s["pov"] for s in samples]          # may contain None
    cond_graphs = [s["graph"] for s in samples]

    # Replace None POVs with zero tensors of the correct shape
    if any(p is None for p in cond_povs):
        valid_pov = next(p for p in cond_povs if p is not None)
        zero_pov = torch.zeros_like(valid_pov)
        cond_povs = [p if p is not None else zero_pov for p in cond_povs]

    cond_pov = torch.stack(cond_povs).to(device) if cond_povs else None
    cond_graph = torch.stack(cond_graphs).to(device)

    # Keep both entries even if one is None
    conds = [cond_pov, cond_graph]
    cond = mixer(conds)
    # --------------------------------------------------------

    # Target latents
    target_latents = torch.stack([s["layout"] for s in samples]).to(device)

    # CFG setup
    uncond_cond = torch.zeros_like(cond)
    guidance_scale = samples[0].get("guidance_scale", 5.0) if isinstance(samples, list) else 5.0

    # Generate latents using LatentDiffusion
    latents = diffusion_model.sample(
        batch_size=num_samples,
        image=False,
        cond=cond,
        uncond_cond=uncond_cond,
        guidance_scale=guidance_scale,
        device=device
    )


    print(f"Sampled latent stats - min: {latents.min():.4f}, max: {latents.max():.4f}, "
          f"mean: {latents.mean():.4f}, std: {latents.std():.4f}", flush=True)

    mse = F.mse_loss(latents, target_latents).item()
    print(f"MSE between generated and target latents: {mse:.6f}", flush=True)

    # Decode both generated and target
    pred_images = diffusion_model.autoencoder.decoder(latents)
    target_images = diffusion_model.autoencoder.decoder(target_latents)

    # Save side-by-side (pred on top, target on bottom)
    both = torch.cat([pred_images, target_images], dim=0)
    save_path = exp_dir / "samples" / f"epoch_{epoch+1:04d}_samples.png"
    save_image(both, save_path, nrow=num_samples, normalize=True)
    print(f"Saved samples with targets to {save_path}", flush=True)

    diffusion_model.train()
    return mse
