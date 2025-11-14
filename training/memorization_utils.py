

from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from training.utils import to_device

# Try to import LPIPS for perceptual similarity
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def load_training_samples(dataset, num_samples=None, device="cuda", batch_size=1000, load_rgb=False):

    if num_samples is None:
        num_samples = len(dataset)
        print(f"Loading entire dataset: {num_samples} training samples...")
        indices = np.arange(len(dataset))  # Use all indices
    else:
        print(f"Loading {num_samples} training samples...")
        num_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    training_latents = []
    training_rgb = []
    training_metadata = []
    
    # Process in batches to manage memory
    num_batches = (len(indices) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        
        batch_latents = []
        batch_rgb = []
        batch_metadata = []
        
        for idx in tqdm(batch_indices, desc=f"Loading batch {batch_idx+1}/{num_batches}", leave=False):
            try:
                sample = dataset[idx]
                
                # Get latent if available
                if "latent" in sample:
                    lat = sample["latent"]
                    if isinstance(lat, torch.Tensor):
                        batch_latents.append(lat.cpu())  # Keep on CPU initially
                    else:
                        batch_latents.append(torch.tensor(lat))
                
                # Get RGB if available and requested
                if load_rgb and "rgb" in sample:
                    rgb = sample["rgb"]
                    if isinstance(rgb, torch.Tensor):
                        batch_rgb.append(rgb.cpu())  # Keep on CPU initially
                    else:
                        batch_rgb.append(torch.tensor(rgb))
                
                # Get metadata
                metadata = {}
                if hasattr(dataset, 'df') and idx < len(dataset.df):
                    row = dataset.df.iloc[idx]
                    metadata = {
                        'index': idx,
                        'scene_id': row.get('scene_id', 'unknown'),
                        'room_id': row.get('room_id', 'unknown'),
                        'path': row.get('path', 'unknown')
                    }
                batch_metadata.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to load sample {idx}: {e}")
                continue
        
        # Stack batch and move to device (more memory efficient)
        if batch_latents:
            training_latents.append(torch.stack(batch_latents))
        if batch_rgb:
            training_rgb.append(torch.stack(batch_rgb))
        training_metadata.extend(batch_metadata)
        
        # Clear batch from CPU memory
        del batch_latents, batch_rgb
    
    # Concatenate all batches and move to device
    result = {'metadata': training_metadata}
    if training_latents:
        print(f"Concatenating {len(training_latents)} batches of latents...")
        result['latents'] = torch.cat(training_latents, dim=0).to(device)
        del training_latents
    if training_rgb:
        print(f"Concatenating {len(training_rgb)} batches of RGB...")
        result['rgb'] = torch.cat(training_rgb, dim=0).to(device)
        del training_rgb
    
    return result


def generate_samples(model, num_samples, batch_size=16, device="cuda", method="ddpm"):
    """Generate samples from the diffusion model."""
    print(f"Generating {num_samples} samples using {method}...")
    all_latents = []
    all_rgb = []
    
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, batch_size), desc="Generating"):
            batch_size_actual = min(batch_size, num_samples - len(all_latents))
            
            sample_output = model.sample(
                batch_size=batch_size_actual,
                num_steps=model.scheduler.num_steps if method == "ddpm" else 50,
                method=method,
                eta=0.0,
                device=device,
                verbose=False
            )
            
            if "latent" in sample_output:
                all_latents.append(sample_output["latent"])
            if "rgb" in sample_output:
                all_rgb.append(sample_output["rgb"])
    
    result = {}
    if all_latents:
        result['latents'] = torch.cat(all_latents, dim=0)
    if all_rgb:
        result['rgb'] = torch.cat(all_rgb, dim=0)
    
    return result


def compute_latent_distances(generated, training, gen_batch_size=32, train_batch_size=1000):

    device = generated.device
    
    # Flatten latents
    gen_flat = generated.view(generated.size(0), -1)
    train_flat = training.view(training.size(0), -1)
    
    # Normalize for cosine similarity (do this once)
    gen_norm = F.normalize(gen_flat, p=2, dim=1)
    train_norm = F.normalize(train_flat, p=2, dim=1)
    
    # Initialize result arrays
    num_generated = len(gen_flat)
    min_l2_distances = torch.full((num_generated,), float('inf'), device=device)
    min_l2_indices = torch.zeros(num_generated, dtype=torch.long, device=device)
    max_cosine_similarities = torch.full((num_generated,), float('-inf'), device=device)
    max_cosine_indices = torch.zeros(num_generated, dtype=torch.long, device=device)
    
    # Process generated samples in batches
    for gen_start in range(0, num_generated, gen_batch_size):
        gen_end = min(gen_start + gen_batch_size, num_generated)
        gen_batch = gen_flat[gen_start:gen_end]
        gen_norm_batch = gen_norm[gen_start:gen_end]
        
        # Process training samples in chunks to avoid large matrices
        for train_start in range(0, len(train_flat), train_batch_size):
            train_end = min(train_start + train_batch_size, len(train_flat))
            train_batch = train_flat[train_start:train_end]
            train_norm_batch = train_norm[train_start:train_end]
            
            # Compute L2 distances for this chunk
            distances = torch.cdist(gen_batch, train_batch, p=2)  # [gen_batch, train_batch]
            chunk_min_dist, chunk_min_idx = distances.min(dim=1)  # [gen_batch]
            chunk_min_idx = chunk_min_idx + train_start  # Adjust indices to global
            
            # Update global minimums (use proper indexing to avoid view issues)
            gen_slice = slice(gen_start, gen_end)
            update_mask = chunk_min_dist < min_l2_distances[gen_slice]
            if update_mask.any():
                min_l2_distances[gen_slice] = torch.where(
                    update_mask, chunk_min_dist, min_l2_distances[gen_slice]
                )
                min_l2_indices[gen_slice] = torch.where(
                    update_mask, chunk_min_idx, min_l2_indices[gen_slice]
                )
            
            # Compute cosine similarities for this chunk
            similarities = gen_norm_batch @ train_norm_batch.T  # [gen_batch, train_batch]
            chunk_max_sim, chunk_max_idx = similarities.max(dim=1)  # [gen_batch]
            chunk_max_idx = chunk_max_idx + train_start  # Adjust indices to global
            
            # Update global maximums (use proper indexing to avoid view issues)
            update_mask = chunk_max_sim > max_cosine_similarities[gen_slice]
            if update_mask.any():
                max_cosine_similarities[gen_slice] = torch.where(
                    update_mask, chunk_max_sim, max_cosine_similarities[gen_slice]
                )
                max_cosine_indices[gen_slice] = torch.where(
                    update_mask, chunk_max_idx, max_cosine_indices[gen_slice]
                )
            
            # Clear intermediate tensors
            del distances, similarities, chunk_min_dist, chunk_min_idx, chunk_max_sim, chunk_max_idx
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Clear batch tensors
        del gen_batch, gen_norm_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Move to CPU and convert to numpy
    l2_distances = min_l2_distances.cpu().numpy()
    nn_indices_l2 = min_l2_indices.cpu().numpy()
    cosine_similarities = max_cosine_similarities.cpu().numpy()
    nn_indices_cosine = max_cosine_indices.cpu().numpy()
    
    return {
        'l2_distances': l2_distances,
        'nn_indices_l2': nn_indices_l2,
        'cosine_similarities': cosine_similarities,
        'nn_indices_cosine': nn_indices_cosine
    }


def compute_diversity_metrics_latent(latents, max_samples=5000):
    """Compute diversity metrics on latents with memory-efficient processing.
    
    Args:
        latents: Latent tensors [N, C, H, W] or [N, D]
        max_samples: Maximum number of samples to use for pairwise computation.
                     If N > max_samples, randomly sample max_samples for computation.
    """
    device = latents.device
    
    # Flatten spatial dimensions if needed
    if latents.dim() > 2:
        flat = latents.view(latents.size(0), -1)
    else:
        flat = latents
    
    # If too many samples, randomly sample a subset for pairwise computation
    num_samples = len(flat)
    if num_samples > max_samples:
        indices = torch.randperm(num_samples, device=device)[:max_samples]
        flat = flat[indices]
        num_samples = max_samples
    
    # Compute pairwise distances in chunks (only upper triangle to avoid duplicates)
    batch_size = 100  # Process 100 samples at a time
    all_distances = []
    
    for i in range(0, num_samples, batch_size):
        end_i = min(i + batch_size, num_samples)
        batch_i = flat[i:end_i]
        
        # Only compare against samples j >= i (upper triangle)
        for j in range(i, num_samples, batch_size):
            end_j = min(j + batch_size, num_samples)
            batch_j = flat[j:end_j]
            
            # Compute distances
            distances = torch.cdist(batch_i, batch_j, p=2)  # [batch_i, batch_j]
            
            # Remove diagonal if comparing same batch
            if i == j:
                # Upper triangle only (exclude diagonal)
                mask = torch.triu(torch.ones(len(batch_i), len(batch_j), dtype=bool, device=device), diagonal=1)
                distances = distances[mask]
            else:
                # Keep all distances (i < j, so this is upper triangle)
                distances = distances.flatten()
            
            all_distances.append(distances.cpu())
            
            # Clear intermediate tensors
            del distances
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Clear batch
        del batch_i
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate all distances
    if all_distances:
        pairwise_dist = torch.cat(all_distances)
        del all_distances
    else:
        # Fallback if no distances computed
        pairwise_dist = torch.tensor([0.0])
    
    mean_pairwise_dist = pairwise_dist.mean().item()
    std_pairwise_dist = pairwise_dist.std().item()
    min_pairwise_dist = pairwise_dist.min().item()
    
    # Unique ratio (samples with distance > threshold)
    threshold = 0.01
    unique_ratio = (pairwise_dist > threshold).float().mean().item()
    
    return {
        'mean_pairwise_distance': mean_pairwise_dist,
        'std_pairwise_distance': std_pairwise_dist,
        'min_pairwise_distance': min_pairwise_dist,
        'unique_ratio': unique_ratio
    }


def sample_from_latent(model, z0, num_steps=None, method="ddpm", device="cuda"):
    """
    Sample from a specific starting latent z0.
    
    Args:
        model: DiffusionModel instance
        z0: Starting latent tensor [B, C, H, W]
        num_steps: Number of sampling steps (default: full schedule)
        method: Sampling method ("ddpm" or "ddim")
        device: Device to use
    
    Returns:
        Dictionary with "latent" and "rgb" keys
    """
    model.eval()
    device_obj = to_device(device)
    z0 = z0.to(device_obj)
    
    if num_steps is None:
        num_steps = model.scheduler.num_steps if method == "ddpm" else 50
    
    # Start from the provided latent (not noise)
    # We need to add noise to match the diffusion forward process
    # Start from timestep T (full noise) and denoise
    model.scheduler = model.scheduler.to(device_obj)
    
    # Create timestep schedule
    if method == "ddim":
        step_size = model.scheduler.num_steps // num_steps
        timesteps = torch.arange(0, model.scheduler.num_steps, step_size, device=device_obj).long()
    else:
        timesteps = torch.arange(model.scheduler.num_steps - 1, -1, -1, device=device_obj).long()
    
    # Start from timestep T (full noise) - add noise to z0
    t_start = timesteps[0] if len(timesteps) > 0 else model.scheduler.num_steps - 1
    noise = model.scheduler.randn_like(z0)
    alpha_bars = model.scheduler.alpha_bars.to(device_obj)
    alpha_bar_t = alpha_bars[t_start].view(-1, 1, 1, 1)
    latents = alpha_bar_t.sqrt() * z0 + (1 - alpha_bar_t).sqrt() * noise
    
    # Denoising loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(latents.shape[0]).to(device_obj)
            pred_noise = model.unet(latents, t_batch, cond=None)
            
            if method == "ddim":
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                if i > 0:
                    alpha_bar_prev = alpha_bars[timesteps[i-1]].view(-1, 1, 1, 1)
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device_obj, dtype=alpha_bar_t.dtype).view(-1, 1, 1, 1)
                
                pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
                latents = alpha_bar_prev.sqrt() * pred_x0 + (1 - alpha_bar_prev).sqrt() * pred_noise
            else:
                # DDPM step
                alpha_bars = model.scheduler.alpha_bars.to(device_obj)
                alphas = model.scheduler.alphas.to(device_obj)
                betas = model.scheduler.betas.to(device_obj)
                
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                beta_t = betas[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i+1]
                    alpha_bar_prev = alpha_bars[t_prev].view(-1, 1, 1, 1)
                    alpha_t = alphas[t].view(-1, 1, 1, 1)
                    
                    pred_mean = (1.0 / alpha_t.sqrt()) * (latents - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise)
                    posterior_variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)) * beta_t
                    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
                    noise_step = model.scheduler.randn_like(latents)
                    latents = pred_mean + posterior_variance.sqrt() * noise_step
                else:
                    pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                    pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
                    latents = pred_x0
    
    # Decode to RGB
    result = {"latent": latents}
    with torch.no_grad():
        decoded_out = model.decoder({"latent": latents})
        if "rgb" in decoded_out:
            rgb = decoded_out["rgb"]
            # RGBHead uses tanh activation by default, outputting in [-1, 1] range
            # Denormalize from [-1, 1] to [0, 1] for image saving
            # This matches how images are saved during autoencoder training (train.py line 186-187)
            rgb = (rgb + 1.0) / 2.0
            # Clamp to ensure valid [0, 1] range (some values might be slightly outside due to numerical precision)
            rgb = torch.clamp(rgb, 0.0, 1.0)
            result["rgb"] = rgb
    
    return result


def perturbation_test(model, z0_samples, noise_stds=[0.01, 0.02, 0.05, 0.1], device="cuda", method="ddpm"):
    """
    Test memorization by perturbing latents and comparing outputs.
    
    Args:
        model: DiffusionModel instance
        z0_samples: List of latent tensors [B, C, H, W] to test
        noise_stds: List of noise standard deviations to test
        device: Device to use
        method: Sampling method
    
    Returns:
        Dictionary with results for each noise level
    """
    print("\n" + "="*60)
    print("Perturbation Test: Assessing Memorization vs Generalization")
    print("="*60)
    
    device_obj = to_device(device)
    
    # Initialize LPIPS if available
    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net='alex').to(device_obj)
            lpips_model.eval()
            print("Using LPIPS (AlexNet) for perceptual similarity")
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS: {e}")
    
    results = {}
    
    # Generate baseline samples from z0
    print(f"\nGenerating baseline samples from {len(z0_samples)} latents...")
    baseline_samples = []
    for z0 in tqdm(z0_samples, desc="Baseline generation"):
        z0_batch = z0.unsqueeze(0) if z0.dim() == 3 else z0
        sample = sample_from_latent(model, z0_batch, method=method, device=device_obj)
        baseline_samples.append(sample)
    
    # Test each noise level
    for noise_std in noise_stds:
        print(f"\nTesting noise level: σ={noise_std}")
        perturbed_samples = []
        noise_magnitudes = []
        
        for i, z0 in enumerate(tqdm(z0_samples, desc=f"Perturbation σ={noise_std}")):
            z0_batch = z0.unsqueeze(0) if z0.dim() == 3 else z0
            z0_batch = z0_batch.to(device_obj)
            
            # Add perturbation
            epsilon = torch.randn_like(z0_batch) * noise_std
            z_perturbed = z0_batch + epsilon
            
            # Sample from perturbed latent
            sample = sample_from_latent(model, z_perturbed, method=method, device=device_obj)
            perturbed_samples.append(sample)
            
            # Record noise magnitude
            noise_mag = epsilon.norm(p=2).item()
            noise_magnitudes.append(noise_mag)
        
        # Compute metrics
        lpips_scores = []
        l2_distances = []
        
        for baseline, perturbed in zip(baseline_samples, perturbed_samples):
            if "rgb" in baseline and "rgb" in perturbed:
                rgb_base = baseline["rgb"]
                rgb_pert = perturbed["rgb"]
                
                # L2 distance
                l2 = F.mse_loss(rgb_base, rgb_pert).item()
                l2_distances.append(l2)
                
                # LPIPS (perceptual similarity)
                if lpips_model is not None:
                    # LPIPS expects images in [-1, 1] range
                    rgb_base_lpips = rgb_base * 2.0 - 1.0
                    rgb_pert_lpips = rgb_pert * 2.0 - 1.0
                    with torch.no_grad():
                        lpips_score = lpips_model(rgb_base_lpips, rgb_pert_lpips).item()
                    lpips_scores.append(lpips_score)
        
        results[noise_std] = {
            "noise_magnitudes": noise_magnitudes,
            "lpips_scores": lpips_scores if lpips_scores else None,
            "l2_distances": l2_distances,
            "mean_noise_magnitude": np.mean(noise_magnitudes),
            "mean_lpips": np.mean(lpips_scores) if lpips_scores else None,
            "mean_l2": np.mean(l2_distances),
            "std_lpips": np.std(lpips_scores) if lpips_scores else None,
            "std_l2": np.std(l2_distances),
        }
        
        print(f"  Mean noise magnitude: {results[noise_std]['mean_noise_magnitude']:.6f}")
        print(f"  Mean L2 distance: {results[noise_std]['mean_l2']:.6f}")
        if results[noise_std]['mean_lpips'] is not None:
            print(f"  Mean LPIPS: {results[noise_std]['mean_lpips']:.6f}")
    
    return results


def plot_perturbation_results(perturbation_results, output_dir):
    """Plot perturbation test results showing LPIPS/L2 vs noise magnitude."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    noise_stds = sorted([float(k) for k in perturbation_results.keys()])
    mean_lpips = [perturbation_results[str(s)]['mean_lpips'] for s in noise_stds]
    mean_l2 = [perturbation_results[str(s)]['mean_l2'] for s in noise_stds]
    mean_noise_mag = [perturbation_results[str(s)]['mean_noise_magnitude'] for s in noise_stds]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot LPIPS vs noise magnitude
    ax1 = axes[0]
    if any(lp is not None for lp in mean_lpips):
        ax1.plot(mean_noise_mag, [lp if lp is not None else 0 for lp in mean_lpips], 
                'o-', label='LPIPS', linewidth=2, markersize=8)
        ax1.set_xlabel('Noise Magnitude ||ε||')
        ax1.set_ylabel('LPIPS Score')
        ax1.set_title('LPIPS vs Noise Magnitude\n(Lower = more memorization)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'LPIPS not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('LPIPS vs Noise Magnitude')
    
    # Plot L2 vs noise magnitude
    ax2 = axes[1]
    ax2.plot(mean_noise_mag, mean_l2, 's-', label='L2 Distance', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Noise Magnitude ||ε||')
    ax2.set_ylabel('L2 Distance (MSE)')
    ax2.set_title('L2 Distance vs Noise Magnitude\n(Lower = more memorization)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'perturbation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved perturbation visualization to {output_dir / 'perturbation_analysis.png'}")


def check_memorization(model, training_samples, generated_samples, output_dir, 
                      latent_perturbation_std=0.0, run_perturbation_test=False, 
                      num_perturbation_samples=20, method="ddpm", device="cuda",
                      gen_batch_size=32, train_batch_size=1000):
    """
    Core memorization check function that computes metrics and saves results.
    
    This is the main function that should be called from training scripts or CLI.
    It expects model, training_samples, and generated_samples to already be loaded.
    
    Args:
        model: DiffusionModel instance (must have decoder)
        training_samples: Dict with 'latents' key (and optionally 'metadata')
        generated_samples: Dict with 'latents' and optionally 'rgb' keys
        output_dir: Path to save results
        latent_perturbation_std: If > 0, adds noise to training latents before comparison
        run_perturbation_test: If True, run perturbation test
        num_perturbation_samples: Number of latents to test in perturbation test
        method: Sampling method ("ddpm" or "ddim")
        device: Device to use
        gen_batch_size: Batch size for generated samples in distance computation (smaller = less memory)
        train_batch_size: Batch size for training samples in distance computation (smaller = less memory)
    
    Returns:
        Dictionary with summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear CUDA cache before starting
    device_obj = to_device(device)
    if device_obj.type == 'cuda':
        torch.cuda.empty_cache()
    
    results = {}
    
    # Latent space distances (only comparison we do)
    if generated_samples.get('latents') is not None and training_samples.get('latents') is not None:
        print("Computing latent space distances...")
        
        # Apply latent perturbation to training samples if specified (for robustness testing)
        training_latents = training_samples['latents']
        if latent_perturbation_std > 0.0:
            print(f"  Applying latent perturbation (std={latent_perturbation_std}) to training samples...")
            perturbation = torch.randn_like(training_latents) * latent_perturbation_std
            training_latents = training_latents + perturbation
        
        latent_results = compute_latent_distances(
            generated_samples['latents'], training_latents,
            gen_batch_size=gen_batch_size,
            train_batch_size=train_batch_size
        )
        results['latent_l2_distances'] = latent_results['l2_distances']
        results['latent_nn_indices_l2'] = latent_results['nn_indices_l2']
        results['latent_cosine_similarities'] = latent_results['cosine_similarities']
        results['latent_nn_indices_cosine'] = latent_results['nn_indices_cosine']
    
    # Diversity metrics (compute on latents instead of RGB)
    if generated_samples.get('latents') is not None:
        print("Computing diversity metrics (latent space)...")
        results['diversity'] = compute_diversity_metrics_latent(generated_samples['latents'])
    
    # Compute statistics and check thresholds
    print("\n" + "="*60)
    print("Memorization Analysis Results (Latent Space)")
    print("="*60)
    
    summary = {}
    
    # Latent space analysis (primary metric)
    if 'latent_l2_distances' in results:
        distances = results['latent_l2_distances']
        similarities = results['latent_cosine_similarities']
        
        mean_dist = np.mean(distances)
        mean_sim = np.mean(similarities)
        max_sim = np.max(similarities)
        
        # Threshold for memorization in latent space
        sim_threshold = 0.99  # Very high cosine similarity
        memorized_count = (similarities > sim_threshold).sum()
        memorized_ratio = memorized_count / len(similarities)
        
        summary['latent'] = {
            'mean_l2_distance': float(mean_dist),
            'mean_cosine_similarity': float(mean_sim),
            'max_cosine_similarity': float(max_sim),
            'memorized_count': int(memorized_count),
            'memorized_ratio': float(memorized_ratio),
            'similarity_threshold': sim_threshold
        }
        
        print(f"\nLatent Space Analysis:")
        print(f"  Mean L2 distance:         {mean_dist:.6f}")
        print(f"  Mean cosine similarity:    {mean_sim:.4f}")
        print(f"  Max cosine similarity:     {max_sim:.4f}")
        print(f"  Memorized samples:         {memorized_count}/{len(similarities)} ({memorized_ratio:.2%})")
        print(f"  Similarity threshold:      {sim_threshold}")
    
    # Diversity analysis
    if 'diversity' in results:
        div = results['diversity']
        summary['diversity'] = div
        
        print(f"\nDiversity Metrics:")
        print(f"  Mean pairwise distance:   {div['mean_pairwise_distance']:.6f}")
        print(f"  Std pairwise distance:    {div['std_pairwise_distance']:.6f}")
        print(f"  Min pairwise distance:   {div['min_pairwise_distance']:.6f}")
        print(f"  Unique ratio:             {div['unique_ratio']:.2%}")
    
    # Overall assessment
    print("\n" + "="*60)
    print("Assessment:")
    print("="*60)
    
    max_memorized_ratio = 0.0
    if 'latent' in summary:
        max_memorized_ratio = summary['latent']['memorized_ratio']
    
    if max_memorized_ratio < 0.01:
        print("✓ Model shows no significant memorization (< 1% samples)")
    elif max_memorized_ratio < 0.05:
        print("⚠ Model shows minimal memorization (1-5% samples)")
    else:
        print("✗ Model shows significant memorization (> 5% samples)")
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    # Save summary
    import json
    with open(output_dir / 'memorization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output_dir / 'memorization_summary.json'}")
    
    # Save detailed results
    num_generated = len(generated_samples.get('latents', generated_samples.get('rgb', [])))
    results_df = pd.DataFrame({
        'generated_idx': range(num_generated)
    })
    
    if 'latent_l2_distances' in results:
        results_df['latent_l2_distance'] = results['latent_l2_distances']
        results_df['latent_cosine_similarity'] = results['latent_cosine_similarities']
        results_df['latent_nn_index'] = results['latent_nn_indices_cosine']
    
    results_df.to_csv(output_dir / 'memorization_results.csv', index=False)
    print(f"Saved detailed results to {output_dir / 'memorization_results.csv'}")
    
    # Run perturbation test if requested
    if run_perturbation_test:
        print("\n" + "="*60)
        print("Running Perturbation Test")
        print("="*60)
        
        # Select random training latents as z0 samples
        if training_samples.get('latents') is not None:
            num_samples = min(num_perturbation_samples, len(training_samples['latents']))
            indices = np.random.choice(len(training_samples['latents']), num_samples, replace=False)
            z0_samples = [training_samples['latents'][idx] for idx in indices]
            
            # Run perturbation test
            perturbation_results = perturbation_test(
                model, z0_samples, 
                noise_stds=[0.01, 0.02, 0.05, 0.1], 
                device=device, 
                method=method
            )
            
            # Save perturbation results
            perturbation_output = output_dir / "perturbation_test"
            perturbation_output.mkdir(exist_ok=True)
            
            # Save results as JSON
            with open(perturbation_output / 'perturbation_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for noise_std, data in perturbation_results.items():
                    json_results[str(noise_std)] = {
                        "mean_noise_magnitude": float(data["mean_noise_magnitude"]),
                        "mean_lpips": float(data["mean_lpips"]) if data["mean_lpips"] is not None else None,
                        "mean_l2": float(data["mean_l2"]),
                        "std_lpips": float(data["std_lpips"]) if data["std_lpips"] is not None else None,
                        "std_l2": float(data["std_l2"]),
                    }
                json.dump(json_results, f, indent=2)
            
            # Create visualization
            plot_perturbation_results(perturbation_results, perturbation_output)
            
            # Add to summary
            summary['perturbation_test'] = {
                "num_samples": num_samples,
                "noise_levels": list(perturbation_results.keys()),
            }
            
            print(f"\nPerturbation test results saved to: {perturbation_output}")
        else:
            print("Warning: No training latents available for perturbation test")
    
    return summary

