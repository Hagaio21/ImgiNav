import torch
import torch.nn.functional as F
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

# --- Matplotlib Helper (with white background fix) ---
def _save_grid_with_titles_matplotlib(rows_of_tensors: list[list[torch.Tensor]], 
                                      row_titles: list[str], 
                                      save_path: Path, 
                                      is_grayscale: bool = False):
    """
    Saves a grid of images using Matplotlib with titles for each row.
    This version explicitly sets a white background.
    """
    num_rows = len(rows_of_tensors)
    if num_rows == 0:
        print("Warning: No rows to plot.")
        return
    num_cols = len(rows_of_tensors[0])
    if num_cols == 0:
        print("Warning: No columns to plot.")
        return

    # --- 1. Determine Image Shape ---
    first_tensor = rows_of_tensors[0][0].cpu()
    C, H, W = first_tensor.shape
    
    # --- 2. Create Figure and GridSpec ---
    title_col_width = 1.5
    image_col_width = W / H
    width_ratios = [title_col_width] + [image_col_width] * num_cols
    
    scale = H / 100
    fig_width_inches = sum(width_ratios) * scale * 0.7
    fig_height_inches = num_rows * scale * 1.1
    
    # Explicitly set figure background color to white
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), facecolor='white')
    
    gs = GridSpec(num_rows, num_cols + 1, width_ratios=width_ratios, 
                  wspace=0.05, hspace=0.05)

    # --- 3. Plot Titles and Images ---
    for i in range(num_rows):
        # --- Plot Title ---
        ax_title = fig.add_subplot(gs[i, 0])
        ax_title.set_facecolor('white') # Set title background to white
        ax_title.text(0.5, 0.5, row_titles[i], 
                      horizontalalignment='center', 
                      verticalalignment='center', 
                      fontsize=16, # Font size control
                      rotation='horizontal')
        ax_title.axis('off')
        
        # --- Plot Images ---
        for j in range(num_cols):
            ax_img = fig.add_subplot(gs[i, j + 1])
            ax_img.set_facecolor('white') # Set image background to white
            tensor_img = rows_of_tensors[i][j].cpu()
            
            if is_grayscale or C == 1:
                plot_img = tensor_img.squeeze().numpy()
                ax_img.imshow(plot_img, cmap='gray', vmin=0, vmax=1)
            else:
                plot_img = tensor_img.permute(1, 2, 0).numpy()
                ax_img.imshow(plot_img) 

            ax_img.axis('off')

    # --- 4. Save Figure ---
    try:
        # Set facecolor in savefig as well
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    except Exception as e:
        print(f"Error saving matplotlib figure: {e}")
    finally:
        plt.close(fig) # Ensure figure is closed to save memory


# --- Unconditioned Sampler (Matplotlib) ---
@torch.no_grad()
def generate_samples_unconditioned(diffusion_model, exp_dir, epoch, num_samples, device):
    """
    Generate and save unconditioned sample images and latent visualizations
    using LatentDiffusion and fixed noise. (Matplotlib version)
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
    # --- NO CLAMPING OR NORMALIZING ---

    # --- 4. Save Decoded Images (Matplotlib) ---
    img_save_path = samples_dir / f"epoch_{epoch+1:04d}_uncond_images.png"
    _save_grid_with_titles_matplotlib(
        rows_of_tensors=[[img for img in images]],
        row_titles=["Unconditioned"],
        save_path=img_save_path,
        is_grayscale=False
    )
    print(f"Saved unconditioned images to {img_save_path}", flush=True)

    # --- 5. Save Latent Visualization (Matplotlib) ---
    latents_viz = normalize_latents_for_viz(latents)
    if C > 3:
        latents_viz = latents_viz[:, :3, ...]
    elif C == 1:
        latents_viz = latents_viz.repeat(1, 3, 1, 1)

    latent_save_path = samples_dir / f"epoch_{epoch+1:04d}_uncond_latents.png"
    _save_grid_with_titles_matplotlib(
        rows_of_tensors=[[lat for lat in latents_viz]],
        row_titles=["Unconditioned Latents"],
        save_path=latent_save_path,
        is_grayscale=False # Plotted as RGB
    )
    print(f"Saved unconditioned latent visualization to {latent_save_path}", flush=True)

    diffusion_model.train() # Set model back to training mode


# --- Conditioned Sampler (Matplotlib) ---
@torch.no_grad()
def generate_samples_conditioned(diffusion_model, mixer, samples, exp_dir, epoch, device, config=None):
    """
    Generate conditioned AND unconditioned samples from FIXED NOISE.
    Saves grids using Matplotlib.
    """
    diffusion_model.eval()
    diffusion_model.autoencoder.eval()
    exp_dir = Path(exp_dir)
    num_samples = len(samples)
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(exist_ok=True) 

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
    use_pov = config.get("mixer", {}).get("use_pov", True)
    use_graph = config.get("mixer", {}).get("use_graph", True)

    cond_povs, cond_graphs = [], []
    
    if use_pov:
        cond_povs = [s.get("pov") for s in samples]
        valid_pov = next((p for p in cond_povs if p is not None), None)
        if any(p is None for p in cond_povs) and valid_pov is not None:
            zero_pov = torch.zeros_like(valid_pov)
            cond_povs = [p if p is not None else zero_pov for p in cond_povs]
        
    if use_graph:
        cond_graphs = [s.get("graph") for s in samples]

    target_latents = torch.stack([s["layout"] for s in samples]).to(device)

    cond_pov = torch.stack(cond_povs).to(device) if use_pov and cond_povs else None
    cond_graph = torch.stack(cond_graphs).to(device) if use_graph and cond_graphs else None

    cond_list = [cond_pov, cond_graph]
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
        image=False, cond=uncond_cond, guidance_scale=0,
        device=device, start_noise=fixed_noise
    )
    mse_unconditioned = F.mse_loss(unconditioned_latents, target_latents).item()
    print(f"  MSE (Unconditioned vs Target): {mse_unconditioned:.6f}", flush=True)

    # --- 6. Decode Images ---
    # Decoder output is assumed to be [0, 1] and not processed further
    unconditioned_images = diffusion_model.autoencoder.decoder(unconditioned_latents)
    conditioned_images = diffusion_model.autoencoder.decoder(conditioned_latents)
    target_images = diffusion_model.autoencoder.decoder(target_latents)

    # --- 7. Save Image Comparison Grid (Matplotlib) ---
    img_save_path = samples_dir / f"epoch_{epoch+1:04d}_comparison_images.png"
    _save_grid_with_titles_matplotlib(
        rows_of_tensors=[
            [img for img in unconditioned_images],
            [img for img in conditioned_images],
            [img for img in target_images]
        ],
        row_titles=["Unconditioned", "Conditioned", "Target"],
        save_path=img_save_path,
        is_grayscale=False # Decoded images are RGB
    )
    print(f"Saved image comparison grid to {img_save_path}", flush=True)

    # --- 8. Save Latent Visualization Grid (Matplotlib) ---
    uncond_latents_viz = normalize_latents_for_viz(unconditioned_latents)
    cond_latents_viz = normalize_latents_for_viz(conditioned_latents)
    target_latents_viz = normalize_latents_for_viz(target_latents)
    
    C_lat, _, _ = uncond_latents_viz.shape[1:]
    
    if C_lat > 3:
         uncond_latents_viz = uncond_latents_viz[:,:3,...]
         cond_latents_viz = cond_latents_viz[:,:3,...]
         target_latents_viz = target_latents_viz[:,:3,...]
    elif C_lat == 1: 
        # Repeat channels to plot as RGB
        uncond_latents_viz = uncond_latents_viz.repeat(1,3,1,1)
        cond_latents_viz = cond_latents_viz.repeat(1,3,1,1)
        target_latents_viz = target_latents_viz.repeat(1,3,1,1)
    
    latent_save_path = samples_dir / f"epoch_{epoch+1:04d}_comparison_latents.png"
    _save_grid_with_titles_matplotlib(
        rows_of_tensors=[
            [lat for lat in uncond_latents_viz],
            [lat for lat in cond_latents_viz],
            [lat for lat in target_latents_viz]
        ],
        row_titles=["Unconditioned", "Conditioned", "Target"],
        save_path=latent_save_path,
        is_grayscale=False # We plot them as RGB
    )
    print(f"Saved latent visualization grid to {latent_save_path}", flush=True)

    # --- 9. Save Difference Maps (Matplotlib) ---
    diff_cond_target = torch.abs(conditioned_latents - target_latents)
    diff_uncond_target = torch.abs(unconditioned_latents - target_latents)
    diff_cond_uncond = torch.abs(conditioned_latents - unconditioned_latents)

    # Normalize globally and take mean across channels
    diff_ct_viz = normalize_diff_map(diff_cond_target).mean(dim=1, keepdim=True)
    diff_ut_viz = normalize_diff_map(diff_uncond_target).mean(dim=1, keepdim=True)
    diff_cu_viz = normalize_diff_map(diff_cond_uncond).mean(dim=1, keepdim=True)
    
    diff_save_path = samples_dir / f"epoch_{epoch+1:04d}_comparison_diffs.png"
    _save_grid_with_titles_matplotlib(
        rows_of_tensors=[
            [diff for diff in diff_ct_viz],
            [diff for diff in diff_ut_viz],
            [diff for diff in diff_cu_viz]
        ],
        row_titles=["|Cond - Tgt|", "|Uncond - Tgt|", "|Cond - Uncond|"],
        save_path=diff_save_path,
        is_grayscale=True # These are grayscale
    )
    print(f"Saved difference map grid to {diff_save_path}", flush=True)

    diffusion_model.train()
    return mse_conditioned