import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image, make_grid
import os

def normalize_diff_map(diff_map):
    """Normalizes a difference map globally to [0, 1] for visualization."""
    vmin = diff_map.min()
    vmax = diff_map.max()
    scale = vmax - vmin
    if scale < 1e-6:
        scale = 1.0  # Prevent division by zero
    return (diff_map - vmin) / scale
    
# --- Helper functions (ensure these are present) ---
def normalize_latents_for_viz(latents):
    """Normalizes latents channel-wise to [0, 1] for visualization."""
    b, c, h, w = latents.shape
    latents_flat = latents.view(b * c, -1)
    vmin = latents_flat.min(dim=1, keepdim=True)[0]
    vmax = latents_flat.max(dim=1, keepdim=True)[0]
    scale = vmax - vmin
    scale[scale < 1e-6] = 1.0 # Prevent division by zero
    normalized = (latents_flat - vmin) / scale
    return normalized.view(b, c, h, w)

# (normalize_diff_map is not needed for unconditioned sampling)

# --- Updated Unconditioned Sampler ---
@torch.no_grad()
def generate_samples_unconditioned(diffusion_model, exp_dir, epoch, num_samples, device):
    """
    Generate and save unconditioned sample images and latent visualizations
    using LatentDiffusion and fixed noise.

    Args:
        diffusion_model: LatentDiffusion model instance.
        exp_dir: Path to the experiment directory.
        epoch: Current epoch number (for naming files).
        num_samples: Number of samples to generate.
        device: Torch device ('cuda' or 'cpu').
    """
    diffusion_model.eval() # Set model to evaluation mode
    exp_dir = Path(exp_dir)
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(exist_ok=True) # Ensure samples directory exists

    # --- 1. Load or Create Fixed Noise ---
    fixed_noise_path = exp_dir / "fixed_noise.pt"
    C, H, W = diffusion_model.latent_shape

    if fixed_noise_path.exists():
        try:
            fixed_noise = torch.load(fixed_noise_path, weights_only=False).to(device)
            if fixed_noise.shape[0] != num_samples or fixed_noise.shape[1:] != (C, H, W):
                 print(f"Warning: Fixed noise shape mismatch (found {fixed_noise.shape}, expected {(num_samples, C, H, W)}). Regenerating.")
                 fixed_noise = None
        except Exception as e:
            print(f"Warning: Failed to load fixed noise: {e}. Regenerating.")
            fixed_noise = None
    else:
        fixed_noise = None

    if fixed_noise is None:
        print(f"Generating new fixed noise for {num_samples} samples.")
        torch.manual_seed(1234)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(1234)
        fixed_noise = torch.randn(num_samples, C, H, W, device=device)
        try:
            torch.save(fixed_noise.cpu(), fixed_noise_path)
            print(f"Saved new fixed noise to {fixed_noise_path}")
        except Exception as e:
            print(f"Error saving fixed noise: {e}")

    # --- 2. Generate Latents ---
    print("Generating unconditioned samples...")
    # Use the fixed noise as the starting point
    latents = diffusion_model.sample(
        batch_size=num_samples, # May still be needed by sample method
        image=False,           # Output latents
        cond=None,             # No conditioning
        start_noise=fixed_noise, # Start from fixed noise
        device=device
        # Assuming guidance_scale=0 or is ignored when cond=None
    )
    print(
        f"  Unconditioned latent stats - min: {latents.min():.4f}, max: {latents.max():.4f}, "
        f"mean: {latents.mean():.4f}, std: {latents.std():.4f}",
        flush=True
    )

    # --- 3. Decode Images ---
    images = diffusion_model.autoencoder.decoder(latents)

    # --- 4. Save Decoded Images ---
    img_save_path = samples_dir / f"epoch_{epoch+1:04d}_uncond_images.png"
    # Determine nrow for a roughly square grid
    nrow = int(num_samples**0.5)
    if nrow * nrow < num_samples: # Adjust if not a perfect square
        nrow += 1
    save_image(images, img_save_path, nrow=nrow, normalize=True, value_range=(-1, 1)) # Adjust value_range if needed
    print(f"Saved unconditioned images to {img_save_path}", flush=True)

    # --- 5. Save Latent Visualization ---
    latents_viz = normalize_latents_for_viz(latents)
    # Handle channel numbers for visualization (e.g., take first 3 if C>3)
    if C > 3:
        latents_viz = latents_viz[:, :3, ...]
    elif C == 1:
        latents_viz = latents_viz.repeat(1, 3, 1, 1)

    latent_save_path = samples_dir / f"epoch_{epoch+1:04d}_uncond_latents.png"
    save_image(latents_viz, latent_save_path, nrow=nrow, normalize=False) # Already normalized [0,1]
    print(f"Saved unconditioned latent visualization to {latent_save_path}", flush=True)

    diffusion_model.train() # Set model back to training mode

@torch.no_grad()
def generate_samples_conditioned(diffusion_model, mixer, samples, exp_dir, epoch, device, config=None):
    """
    Generate conditioned AND unconditioned samples from FIXED NOISE.
    Saves:
        1. Comparison grid: [Unconditioned | Conditioned | Target] (Decoded Images)
        2. Latent grid: [Unconditioned | Conditioned | Target] (Normalized Latents)
        3. Difference maps: [|C-T|, |U-T|, |C-U|] (Normalized Latent Diffs)
    """
    diffusion_model.eval()
    diffusion_model.autoencoder.eval()
    exp_dir = Path(exp_dir)
    num_samples = len(samples)
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(exist_ok=True) # Ensure samples directory exists

    # --- 1. Load or Create Fixed Noise ---
    fixed_noise_path = exp_dir / "fixed_noise.pt"
    C, H, W = diffusion_model.latent_shape

    if fixed_noise_path.exists():
        try:
            fixed_noise = torch.load(fixed_noise_path, weights_only=False).to(device)
            if fixed_noise.shape != (num_samples, C, H, W):
                 print(f"Warning: Fixed noise shape mismatch. Regenerating.")
                 fixed_noise = None
        except Exception as e:
            print(f"Warning: Failed to load fixed noise: {e}. Regenerating.")
            fixed_noise = None
    else:
        fixed_noise = None

    if fixed_noise is None:
        print(f"Generating new fixed noise for {num_samples} samples.")
        torch.manual_seed(1234)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(1234)
        fixed_noise = torch.randn(num_samples, C, H, W, device=device)
        try:
            torch.save(fixed_noise.cpu(), fixed_noise_path)
            print(f"Saved new fixed noise to {fixed_noise_path}")
        except Exception as e:
            print(f"Error saving fixed noise: {e}")

    # --- 2. Prepare Conditioning Tensors & Targets ---
    cond_povs = [s.get("pov") for s in samples]
    cond_graphs = [s.get("graph") for s in samples]
    target_latents = torch.stack([s["layout"] for s in samples]).to(device)

    # --- (Robust None POV Handling - simplified assuming valid_pov exists if needed) ---
    valid_pov = next((p for p in cond_povs if p is not None), None)
    if any(p is None for p in cond_povs) and valid_pov is not None:
         zero_pov = torch.zeros_like(valid_pov)
         cond_povs = [p if p is not None else zero_pov for p in cond_povs]
    # Note: Assumes pov_dim > 0 implies at least one valid_pov exists in fixed samples
    # More robust check might be needed if fixed_samples could *all* have None POV.

    cond_pov = torch.stack(cond_povs).to(device) if cond_povs and config["mixer"].get("pov_dim", 0) > 0 else None
    cond_graph = torch.stack(cond_graphs).to(device) if cond_graphs and config["mixer"].get("graph_dim", 0) > 0 else None

    cond_list = [cond_pov, cond_graph] # Mixer expects list

    cond = mixer(cond_list)
    uncond_cond = torch.zeros_like(cond)

    # --- 3. CFG Scale ---
    guidance_scale = config.get("training", {}).get("cfg", {}).get("guidance_scale", 5.0)

    # --- 4. Generate Conditioned Latents ---
    print("Generating conditioned samples...")
    conditioned_latents = diffusion_model.sample(
        image=False, cond=cond, uncond_cond=uncond_cond,
        guidance_scale=guidance_scale, device=device, start_noise=fixed_noise
    )
    mse_conditioned = F.mse_loss(conditioned_latents, target_latents).item()
    print(f"  MSE (Conditioned vs Target): {mse_conditioned:.6f}", flush=True)

    # --- 5. Generate Unconditioned Latents ---
    print("Generating unconditioned samples...")
    unconditioned_latents = diffusion_model.sample(
        image=False, cond=uncond_cond, guidance_scale=0, # Use uncond_cond, no guidance
        device=device, start_noise=fixed_noise
    )
    mse_unconditioned = F.mse_loss(unconditioned_latents, target_latents).item()
    print(f"  MSE (Unconditioned vs Target): {mse_unconditioned:.6f}", flush=True)

    # --- 6. Decode Images ---
    unconditioned_images = diffusion_model.autoencoder.decoder(unconditioned_latents)
    conditioned_images = diffusion_model.autoencoder.decoder(conditioned_latents)
    target_images = diffusion_model.autoencoder.decoder(target_latents)

    # --- 7. Save Image Comparison Grid ---
    image_comparison_grid = torch.cat([unconditioned_images, conditioned_images, target_images], dim=0)
    img_save_path = samples_dir / f"epoch_{epoch+1:04d}_comparison_images.png"
    save_image(image_comparison_grid, img_save_path, nrow=num_samples, normalize=True, value_range=(-1,1)) # Adjust value_range if AE output differs
    print(f"Saved image comparison grid to {img_save_path}", flush=True)

    # --- 8. Save Latent Visualization Grid ---
    # Normalize latents for visualization
    uncond_latents_viz = normalize_latents_for_viz(unconditioned_latents)
    cond_latents_viz = normalize_latents_for_viz(conditioned_latents)
    target_latents_viz = normalize_latents_for_viz(target_latents)
    # Treat channels as RGB - only works well if C=1 or C=3. If C=4, maybe take first 3?
    if C > 3:
         uncond_latents_viz = uncond_latents_viz[:,:3,...]
         cond_latents_viz = cond_latents_viz[:,:3,...]
         target_latents_viz = target_latents_viz[:,:3,...]
    elif C == 1: # If grayscale, repeat channel 3 times for save_image
        uncond_latents_viz = uncond_latents_viz.repeat(1,3,1,1)
        cond_latents_viz = cond_latents_viz.repeat(1,3,1,1)
        target_latents_viz = target_latents_viz.repeat(1,3,1,1)

    latent_comparison_grid = torch.cat([uncond_latents_viz, cond_latents_viz, target_latents_viz], dim=0)
    latent_save_path = samples_dir / f"epoch_{epoch+1:04d}_comparison_latents.png"
    save_image(latent_comparison_grid, latent_save_path, nrow=num_samples, normalize=False) # Already normalized
    print(f"Saved latent visualization grid to {latent_save_path}", flush=True)

    # --- 9. Save Difference Maps ---
    diff_cond_target = torch.abs(conditioned_latents - target_latents)
    diff_uncond_target = torch.abs(unconditioned_latents - target_latents)
    diff_cond_uncond = torch.abs(conditioned_latents - unconditioned_latents)

    # Normalize globally and take mean across channels for grayscale viz
    diff_ct_viz = normalize_diff_map(diff_cond_target).mean(dim=1, keepdim=True).repeat(1,3,1,1) # Repeat for RGB
    diff_ut_viz = normalize_diff_map(diff_uncond_target).mean(dim=1, keepdim=True).repeat(1,3,1,1)
    diff_cu_viz = normalize_diff_map(diff_cond_uncond).mean(dim=1, keepdim=True).repeat(1,3,1,1)

    # Create grid for difference maps (optional, could save separately)
    # Row 1: |C-T|, Row 2: |U-T|, Row 3: |C-U|
    diff_grid = torch.cat([diff_ct_viz, diff_ut_viz, diff_cu_viz], dim=0)
    diff_save_path = samples_dir / f"epoch_{epoch+1:04d}_comparison_diffs.png"
    save_image(diff_grid, diff_save_path, nrow=num_samples, normalize=False) # Already normalized
    print(f"Saved difference map grid (|C-T|, |U-T|, |C-U|) to {diff_save_path}", flush=True)


    diffusion_model.train() # Set back to train mode
    return mse_conditioned # Return conditioned MSE