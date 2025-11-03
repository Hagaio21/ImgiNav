#!/usr/bin/env python3
"""
Check for memorization in diffusion models by comparing generated samples to training data.

This script generates samples from a trained diffusion model and compares them to training data
to detect if the model is simply memorizing training examples rather than learning generalizable patterns.

Usage:
    python scripts/check_memorization.py \
        --config experiments/diffusion/ablation/base_config.yaml \
        --checkpoint outputs/experiment_name/experiment_name_checkpoint_best.pt \
        --manifest datasets/layouts_latents.csv \
        --output outputs/memorization_check \
        --num_generate 100 \
        --num_training 5000

Metrics computed:
    - RGB space L2 distances: Pixel-level similarity
    - Latent space L2 distances: Feature-level similarity  
    - Latent space cosine similarity: Direction similarity in latent space
    - Diversity metrics: Pairwise distances between generated samples

Thresholds:
    - RGB L2 distance < 0.001: Likely memorized
    - Latent cosine similarity > 0.99: Likely memorized
    - Memorized ratio > 5%: Significant memorization concern
"""

import argparse
import sys
import yaml
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
from scipy.stats import entropy

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.datasets.datasets import ManifestDataset
from training.utils import load_config, build_dataset


def load_training_samples(dataset, num_samples=None, device="cuda", batch_size=1000, load_rgb=False):
    """Load a subset of training samples for comparison.
    
    Args:
        dataset: Dataset to load from
        num_samples: Number of samples to load. If None, loads entire dataset.
        device: Device to load samples to
        batch_size: Process in batches to manage memory
        load_rgb: If True, load RGB images. If False, only load latents (memory efficient).
    """
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


def compute_pixel_distances(generated, training):
    """Compute L2 distances in pixel/RGB space."""
    # Flatten spatial dimensions
    gen_flat = generated.view(generated.size(0), -1)
    train_flat = training.view(training.size(0), -1)
    
    # Compute pairwise distances (batch for memory efficiency)
    min_distances = []
    nn_indices = []
    
    batch_size = 64  # Process in batches to avoid OOM
    for i in range(0, len(gen_flat), batch_size):
        gen_batch = gen_flat[i:i+batch_size]
        distances = torch.cdist(gen_batch, train_flat, p=2)
        min_dist, nn_idx = distances.min(dim=1)
        min_distances.append(min_dist.cpu())
        nn_indices.append(nn_idx.cpu())
    
    return torch.cat(min_distances).numpy(), torch.cat(nn_indices).numpy()


def compute_latent_distances(generated, training):
    """Compute distances in latent space."""
    # Flatten latents
    gen_flat = generated.view(generated.size(0), -1)
    train_flat = training.view(training.size(0), -1)
    
    # L2 distance
    min_distances = []
    nn_indices = []
    
    batch_size = 64
    for i in range(0, len(gen_flat), batch_size):
        gen_batch = gen_flat[i:i+batch_size]
        distances = torch.cdist(gen_batch, train_flat, p=2)
        min_dist, nn_idx = distances.min(dim=1)
        min_distances.append(min_dist.cpu())
        nn_indices.append(nn_idx.cpu())
    
    l2_distances = torch.cat(min_distances).numpy()
    nn_indices_l2 = torch.cat(nn_indices).numpy()
    
    # Cosine similarity
    gen_norm = F.normalize(gen_flat, p=2, dim=1)
    train_norm = F.normalize(train_flat, p=2, dim=1)
    
    max_similarities = []
    nn_indices_cosine = []
    
    for i in range(0, len(gen_norm), batch_size):
        gen_batch = gen_norm[i:i+batch_size]
        similarities = gen_batch @ train_norm.T
        max_sim, nn_idx = similarities.max(dim=1)
        max_similarities.append(max_sim.cpu())
        nn_indices_cosine.append(nn_idx.cpu())
    
    cosine_similarities = torch.cat(max_similarities).numpy()
    nn_indices_cosine = torch.cat(nn_indices_cosine).numpy()
    
    return {
        'l2_distances': l2_distances,
        'nn_indices_l2': nn_indices_l2,
        'cosine_similarities': cosine_similarities,
        'nn_indices_cosine': nn_indices_cosine
    }


def compute_diversity_metrics(samples):
    """Compute diversity metrics among generated samples."""
    flat = samples.view(samples.size(0), -1)
    
    # Pairwise distances
    pairwise_dist = torch.cdist(flat, flat, p=2)
    
    # Remove diagonal
    mask = ~torch.eye(len(pairwise_dist), dtype=bool, device=pairwise_dist.device)
    pairwise_dist = pairwise_dist[mask]
    
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


def compute_diversity_metrics_latent(latents):
    """Compute diversity metrics on latents (flattened internally)."""
    # Flatten spatial dimensions if needed
    if latents.dim() > 2:
        flat = latents.view(latents.size(0), -1)
    else:
        flat = latents
    return compute_diversity_metrics(flat)


def plot_distributions(results, output_dir):
    """Plot distance distributions and statistics using seaborn."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seaborn style (with fallback if style not available)
    try:
        sns.set_style("darkgrid")
    except (OSError, FileNotFoundError):
        # Fallback to default style if seaborn styles not available
        plt.style.use('default')
        # Manually set darkgrid-like appearance
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
    
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create 2x2 grid: L2 distances, Cosine similarities, Diversity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Latent L2 distances
    if 'latent_l2_distances' in results:
        ax = axes[0, 0]
        distances = results['latent_l2_distances']
        sns.histplot(distances, bins=50, ax=ax, kde=True, stat='density')
        ax.axvline(np.mean(distances), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.6f}')
        ax.axvline(np.median(distances), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.6f}')
        ax.set_xlabel('L2 Distance (Latent space)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Nearest Neighbor Distances (Latent)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        axes[0, 0].set_visible(False)
    
    # Cosine similarities
    if 'latent_cosine_similarities' in results:
        ax = axes[0, 1]
        similarities = results['latent_cosine_similarities']
        sns.histplot(similarities, bins=50, ax=ax, kde=True, stat='density')
        ax.axvline(np.mean(similarities), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.4f}')
        ax.axvline(np.median(similarities), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(similarities):.4f}')
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Nearest Neighbor Cosine Similarities (Latent)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        axes[0, 1].set_visible(False)
    
    # Diversity metrics
    if 'diversity' in results:
        ax = axes[1, 0]
        div = results['diversity']
        metrics = ['Mean\nPairwise\nDist', 'Std\nPairwise\nDist', 'Min\nPairwise\nDist', 'Unique\nRatio']
        values = [
            div.get('mean_pairwise_distance', 0),
            div.get('std_pairwise_distance', 0),
            div.get('min_pairwise_distance', 0),
            div.get('unique_ratio', 0)
        ]
        sns.barplot(x=metrics, y=values, ax=ax, palette='muted', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Generated Samples Diversity (Latent)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
    else:
        axes[1, 0].set_visible(False)
    
    # Hide last subplot (not needed)
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memorization_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_dir / 'memorization_analysis.png'}")
    plt.close()


def save_closest_matches(generated_samples, training_samples, results, output_dir, model=None, dataset=None, top_k=20, use_latent=False):
    """Save visualizations of the closest matches.
    
    Args:
        generated_samples: Dictionary with 'rgb' or 'latents'
        training_samples: Dictionary with 'rgb' or 'latents'
        results: Dictionary with distance/similarity metrics
        output_dir: Output directory
        model: Diffusion model (for decoding latents to RGB)
        dataset: Dataset (for loading training RGB if needed)
        top_k: Number of closest matches to save
        use_latent: If True, use latent similarities instead of RGB distances
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_dir = output_dir / 'closest_matches'
    matches_dir.mkdir(exist_ok=True)
    
    closest = []
    
    # Find top K closest matches (always use latent space)
    if 'latent_cosine_similarities' in results:
        similarities = results['latent_cosine_similarities']
        nn_indices = results['latent_nn_indices_cosine']
        
        # Get top K most similar (highest cosine similarity)
        top_k_actual = min(top_k, len(similarities))
        # Sort by similarity descending (most similar first)
        top_indices = np.argsort(similarities)[::-1][:top_k_actual]
        
        # Save closest matches
        for gen_idx in top_indices:
            train_idx = nn_indices[gen_idx]
            closest.append({
                'generated_idx': int(gen_idx),
                'training_idx': int(train_idx),
                'cosine_similarity': float(similarities[gen_idx]),
                'l2_distance': float(results.get('latent_l2_distances', [0] * len(similarities))[gen_idx]) if 'latent_l2_distances' in results else None
            })
    else:
        print("Warning: No distance/similarity metrics available for closest matches")
        return
    
    # Decode RGB for visualization
    print(f"\nDecoding RGB for top {min(10, len(closest))} closest matches...")
    
    gen_latents = generated_samples.get('latents')
    train_latents = training_samples.get('latents')
    
    # Decode generated latents to RGB (if model available)
    if gen_latents is not None and model is not None and hasattr(model, 'decoder'):
        gen_rgb_decoded = []
        with torch.no_grad():
            # Decode only top matches
            for match in closest[:min(10, len(closest))]:
                gen_idx = match['generated_idx']
                if gen_idx < len(gen_latents):
                    latent = gen_latents[gen_idx:gen_idx+1]  # Keep batch dim
                    decoded = model.decoder({"latent": latent})
                    if "rgb" in decoded:
                        rgb = decoded["rgb"][0]  # Remove batch dim
                        # Denormalize from [-1, 1] to [0, 1]
                        rgb = (rgb + 1.0) / 2.0
                        rgb = torch.clamp(rgb, 0.0, 1.0)
                        gen_rgb_decoded.append(rgb)
        
        # Decode training latents to RGB for matched samples
        train_rgb_decoded = []
        if train_latents is not None and model is not None and hasattr(model, 'decoder'):
            with torch.no_grad():
                for match in closest[:min(10, len(closest))]:
                    train_idx = match['training_idx']
                    if train_idx < len(train_latents):
                        latent = train_latents[train_idx:train_idx+1]  # Keep batch dim
                        decoded = model.decoder({"latent": latent})
                        if "rgb" in decoded:
                            rgb = decoded["rgb"][0]  # Remove batch dim
                            # Denormalize from [-1, 1] to [0, 1]
                            rgb = (rgb + 1.0) / 2.0
                            rgb = torch.clamp(rgb, 0.0, 1.0)
                            train_rgb_decoded.append(rgb)
        
        # Create visualization if we have both
        if len(gen_rgb_decoded) == len(train_rgb_decoded) and len(gen_rgb_decoded) > 0:
            rows = []
            for gen_img, train_img in zip(gen_rgb_decoded, train_rgb_decoded):
                # Side by side
                pair = torch.cat([gen_img, train_img], dim=2)  # Concatenate horizontally
                rows.append(pair)
            
            if rows:
                grid = make_grid(torch.stack(rows), nrow=1, padding=2)
                save_image(grid, matches_dir / 'top_closest_matches.png', normalize=False)
                print(f"Saved closest matches visualization to {matches_dir / 'top_closest_matches.png'}")
    elif generated_samples.get('rgb') is not None and training_samples.get('rgb') is not None:
        # Fallback: use RGB if already available
        gen_rgb = generated_samples['rgb']
        train_rgb = training_samples['rgb']
        
        rows = []
        for i, match in enumerate(closest[:min(10, len(closest))]):
            gen_idx = match['generated_idx']
            train_idx = match['training_idx']
            if gen_idx < len(gen_rgb) and train_idx < len(train_rgb):
                gen_img = gen_rgb[gen_idx]
                train_img = train_rgb[train_idx]
                # Side by side
                pair = torch.cat([gen_img, train_img], dim=2)  # Concatenate horizontally
                rows.append(pair)
        
        if rows:
            grid = make_grid(torch.stack(rows), nrow=1, padding=2)
            save_image(grid, matches_dir / 'top_closest_matches.png', normalize=False)
            print(f"Saved closest matches visualization to {matches_dir / 'top_closest_matches.png'}")
    else:
        print("Warning: Cannot decode RGB for visualization (no decoder available)")
    
    # Save matches to CSV (always create this)
    if closest:
        matches_df = pd.DataFrame(closest)
        matches_df.to_csv(matches_dir / 'closest_matches.csv', index=False)
        print(f"Saved {len(closest)} closest matches to {matches_dir / 'closest_matches.csv'}")
    else:
        print("Warning: No closest matches found to save")


def check_memorization(config_path, checkpoint_path, manifest_path, output_dir,
                       num_generate=1000, num_training=None, method="ddpm"):
    """Main function to check for memorization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract diffusion config - handle nested structure
    if "model" in config and "diffusion" in config["model"]:
        diffusion_cfg = config["model"]["diffusion"]
    elif "diffusion" in config:
        diffusion_cfg = config["diffusion"]
    else:
        # Assume top-level config is already the diffusion config
        # Extract relevant parts
        diffusion_cfg = {
            "autoencoder": config.get("autoencoder"),
            "unet": config.get("unet", {}),
            "scheduler": config.get("scheduler", {})
        }
        # If autoencoder not at top level, check if it's nested
        if not diffusion_cfg["autoencoder"] and "model" in config:
            diffusion_cfg["autoencoder"] = config["model"].get("autoencoder")
    
    # Ensure autoencoder config has checkpoint
    if not diffusion_cfg.get("autoencoder") or not diffusion_cfg["autoencoder"].get("checkpoint"):
        raise ValueError("Config must contain 'autoencoder.checkpoint' path")
    
    model = DiffusionModel.from_config(diffusion_cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    
    # Handle DataParallel prefix
    if state_dict and len(state_dict) > 0:
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        elif first_key.startswith('unet.'):
            # Already prefixed with unet, use as-is
            pass
        else:
            # Add unet prefix if not present
            state_dict = {f'unet.{k}': v for k, v in state_dict.items()}
    
    # Load into unet
    missing_keys, unexpected_keys = model.unet.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)}")
    model = model.to(device).eval()
    print("Model loaded successfully")
    
    # Load training dataset
    print("\n" + "="*60)
    print("Loading training data...")
    print("="*60)
    # Override manifest path in config with the provided one
    if "dataset" not in config:
        config["dataset"] = {}
    config["dataset"]["manifest"] = manifest_path
    print(f"Using manifest: {manifest_path}")
    
    # Only load latents for comparison (memory efficient)
    # RGB will be decoded later only for visualization of closest matches
    import pandas as pd
    manifest_df = pd.read_csv(manifest_path)
    has_latent_path = "latent_path" in manifest_df.columns
    
    # Update outputs to only include latent (we'll decode RGB for visualization later)
    if "outputs" not in config["dataset"]:
        config["dataset"]["outputs"] = {}
    
    # Only load latent for comparison (saves memory)
    if has_latent_path and "latent" not in config["dataset"]["outputs"]:
        config["dataset"]["outputs"]["latent"] = "latent_path"
    
    # Store layout_path info for later RGB decoding if needed
    has_layout_path = "layout_path" in manifest_df.columns
    
    dataset = build_dataset(config)
    # If num_training is None, use entire dataset
    if num_training is None:
        num_training = len(dataset)
        print(f"Checking against entire dataset: {num_training} samples")
    else:
        print(f"Checking against {num_training} training samples (dataset has {len(dataset)} total)")
    # Only load latents (not RGB) to save memory
    training_samples = load_training_samples(dataset, num_samples=num_training, device=device, load_rgb=False)
    print(f"Loaded {len(training_samples['metadata'])} training samples (latents only)")
    
    if training_samples.get('latents') is None:
        print("Error: Could not load training samples (no latents found)")
        return
    
    # Generate samples
    print("\n" + "="*60)
    print("Generating samples...")
    print("="*60)
    generated_samples = generate_samples(model, num_generate, device=device, method=method)
    
    if generated_samples.get('latents') is None and generated_samples.get('rgb') is None:
        print("Error: Could not generate samples")
        return
    
    print(f"Generated {len(generated_samples.get('latents', generated_samples.get('rgb', [])))} samples")
    
    # Compute distances and metrics (only in latent space for memory efficiency)
    print("\n" + "="*60)
    print("Computing memorization metrics (latent space only)...")
    print("="*60)
    
    results = {}
    
    # Latent space distances (only comparison we do)
    if generated_samples.get('latents') is not None and training_samples.get('latents') is not None:
        print("Computing latent space distances...")
        latent_results = compute_latent_distances(
            generated_samples['latents'], training_samples['latents']
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
    summary['num_generated'] = num_generate
    summary['num_training'] = num_training
    summary['method'] = method
    
    with open(output_dir / 'memorization_summary.json', 'w') as f:
        import json
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
    
    # Create plots
    plot_distributions(results, output_dir)
    
    # Save closest matches (decode RGB from latents for visualization)
    if 'latent_cosine_similarities' in results:
        save_closest_matches(generated_samples, training_samples, results, output_dir, 
                            model=model, dataset=dataset, top_k=20, use_latent=True)
    
    print("\n" + "="*60)
    print("Memorization check complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Check for memorization in diffusion models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to diffusion model config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to diffusion model checkpoint"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to training dataset manifest CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/memorization_check",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_generate",
        type=int,
        default=1000,
        help="Number of samples to generate for testing (default: 1000)"
    )
    parser.add_argument(
        "--num_training",
        type=int,
        default=None,
        help="Number of training samples to compare against (default: None = entire dataset). Omit to use entire dataset."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="Sampling method to use"
    )
    
    args = parser.parse_args()
    
    check_memorization(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output,
        num_generate=args.num_generate,
        num_training=args.num_training,
        method=args.method
    )


if __name__ == "__main__":
    main()

