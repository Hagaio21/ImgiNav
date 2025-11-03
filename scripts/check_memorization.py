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
from scipy.stats import entropy

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.datasets.datasets import ManifestDataset
from training.utils import load_config, build_dataset


def load_training_samples(dataset, num_samples=5000, device="cuda"):
    """Load a subset of training samples for comparison."""
    print(f"Loading {num_samples} training samples...")
    training_latents = []
    training_rgb = []
    training_metadata = []
    
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in tqdm(indices, desc="Loading training samples"):
        try:
            sample = dataset[idx]
            
            # Get latent if available
            if "latent" in sample:
                lat = sample["latent"]
                if isinstance(lat, torch.Tensor):
                    training_latents.append(lat)
                else:
                    training_latents.append(torch.tensor(lat))
            
            # Get RGB if available
            if "rgb" in sample:
                rgb = sample["rgb"]
                if isinstance(rgb, torch.Tensor):
                    training_rgb.append(rgb)
                else:
                    training_rgb.append(torch.tensor(rgb))
            
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
            training_metadata.append(metadata)
        except Exception as e:
            print(f"Warning: Failed to load sample {idx}: {e}")
            continue
    
    result = {'metadata': training_metadata}
    if training_latents:
        result['latents'] = torch.stack(training_latents).to(device)
    if training_rgb:
        result['rgb'] = torch.stack(training_rgb).to(device)
    
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


def plot_distributions(results, output_dir):
    """Plot distance distributions and statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RGB L2 distances
    if 'rgb_l2_distances' in results:
        ax = axes[0, 0]
        distances = results['rgb_l2_distances']
        ax.hist(distances, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(distances), color='r', linestyle='--', label=f'Mean: {np.mean(distances):.6f}')
        ax.axvline(np.median(distances), color='g', linestyle='--', label=f'Median: {np.median(distances):.6f}')
        ax.set_xlabel('L2 Distance (RGB space)')
        ax.set_ylabel('Frequency')
        ax.set_title('Nearest Neighbor Distances (RGB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Latent L2 distances
    if 'latent_l2_distances' in results:
        ax = axes[0, 1]
        distances = results['latent_l2_distances']
        ax.hist(distances, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(distances), color='r', linestyle='--', label=f'Mean: {np.mean(distances):.6f}')
        ax.axvline(np.median(distances), color='g', linestyle='--', label=f'Median: {np.median(distances):.6f}')
        ax.set_xlabel('L2 Distance (Latent space)')
        ax.set_ylabel('Frequency')
        ax.set_title('Nearest Neighbor Distances (Latent)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Cosine similarities
    if 'latent_cosine_similarities' in results:
        ax = axes[1, 0]
        similarities = results['latent_cosine_similarities']
        ax.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(similarities), color='r', linestyle='--', label=f'Mean: {np.mean(similarities):.4f}')
        ax.axvline(np.median(similarities), color='g', linestyle='--', label=f'Median: {np.median(similarities):.4f}')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('Nearest Neighbor Cosine Similarities (Latent)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Diversity metrics
    if 'diversity' in results:
        ax = axes[1, 1]
        div = results['diversity']
        metrics = ['Mean\nPairwise\nDist', 'Std\nPairwise\nDist', 'Min\nPairwise\nDist', 'Unique\nRatio']
        values = [
            div.get('mean_pairwise_distance', 0),
            div.get('std_pairwise_distance', 0),
            div.get('min_pairwise_distance', 0),
            div.get('unique_ratio', 0)
        ]
        ax.bar(metrics, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Value')
        ax.set_title('Generated Samples Diversity')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memorization_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir / 'memorization_analysis.png'}")
    plt.close()


def save_closest_matches(generated_samples, training_samples, results, output_dir, top_k=20):
    """Save visualizations of the closest matches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_dir = output_dir / 'closest_matches'
    matches_dir.mkdir(exist_ok=True)
    
    # Find top K closest matches
    if 'rgb_l2_distances' in results:
        distances = results['rgb_l2_distances']
        nn_indices = results['rgb_nn_indices']
        
        # Get top K closest
        top_k_actual = min(top_k, len(distances))
        top_indices = np.argsort(distances)[:top_k_actual]
        
        # Save closest matches
        closest = []
        for gen_idx in top_indices:
            train_idx = nn_indices[gen_idx]
            closest.append({
                'generated_idx': int(gen_idx),
                'training_idx': int(train_idx),
                'distance': float(distances[gen_idx])
            })
        
        # Create visualization grid for top matches
        if generated_samples.get('rgb') is not None and training_samples.get('rgb') is not None:
            gen_rgb = generated_samples['rgb']
            train_rgb = training_samples['rgb']
            
            rows = []
            for i, match in enumerate(closest[:min(10, len(closest))]):
                gen_img = gen_rgb[match['generated_idx']]
                train_img = train_rgb[match['training_idx']]
                # Side by side
                pair = torch.cat([gen_img, train_img], dim=2)  # Concatenate horizontally
                rows.append(pair)
            
            if rows:
                grid = make_grid(torch.stack(rows), nrow=1, padding=2)
                save_image(grid, matches_dir / 'top_closest_matches.png', normalize=False)
                print(f"Saved closest matches visualization to {matches_dir / 'top_closest_matches.png'}")
        
        # Save matches to CSV
        matches_df = pd.DataFrame(closest)
        matches_df.to_csv(matches_dir / 'closest_matches.csv', index=False)
        print(f"Saved {len(closest)} closest matches to {matches_dir / 'closest_matches.csv'}")


def check_memorization(config_path, checkpoint_path, manifest_path, output_dir,
                       num_generate=100, num_training=5000, method="ddpm"):
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
    dataset = build_dataset(config)
    training_samples = load_training_samples(dataset, num_samples=num_training, device=device)
    print(f"Loaded {len(training_samples['metadata'])} training samples")
    
    if training_samples.get('latents') is None and training_samples.get('rgb') is None:
        print("Error: Could not load training samples (no latents or RGB found)")
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
    
    # Compute distances and metrics
    print("\n" + "="*60)
    print("Computing memorization metrics...")
    print("="*60)
    
    results = {}
    
    # RGB space distances
    if generated_samples.get('rgb') is not None and training_samples.get('rgb') is not None:
        print("Computing RGB space distances...")
        rgb_distances, rgb_nn_indices = compute_pixel_distances(
            generated_samples['rgb'], training_samples['rgb']
        )
        results['rgb_l2_distances'] = rgb_distances
        results['rgb_nn_indices'] = rgb_nn_indices
    
    # Latent space distances
    if generated_samples.get('latents') is not None and training_samples.get('latents') is not None:
        print("Computing latent space distances...")
        latent_results = compute_latent_distances(
            generated_samples['latents'], training_samples['latents']
        )
        results['latent_l2_distances'] = latent_results['l2_distances']
        results['latent_nn_indices_l2'] = latent_results['nn_indices_l2']
        results['latent_cosine_similarities'] = latent_results['cosine_similarities']
        results['latent_nn_indices_cosine'] = latent_results['nn_indices_cosine']
    
    # Diversity metrics
    if generated_samples.get('rgb') is not None:
        print("Computing diversity metrics...")
        results['diversity'] = compute_diversity_metrics(generated_samples['rgb'])
    
    # Compute statistics and check thresholds
    print("\n" + "="*60)
    print("Memorization Analysis Results")
    print("="*60)
    
    summary = {}
    
    # RGB space analysis
    if 'rgb_l2_distances' in results:
        distances = results['rgb_l2_distances']
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Threshold for memorization (adjust based on your data scale)
        threshold = 0.001  # Very small distance suggests exact match
        memorized_count = (distances < threshold).sum()
        memorized_ratio = memorized_count / len(distances)
        
        summary['rgb'] = {
            'mean_distance': float(mean_dist),
            'median_distance': float(median_dist),
            'min_distance': float(min_dist),
            'max_distance': float(max_dist),
            'memorized_count': int(memorized_count),
            'memorized_ratio': float(memorized_ratio),
            'threshold': threshold
        }
        
        print(f"\nRGB Space Analysis:")
        print(f"  Mean distance:     {mean_dist:.6f}")
        print(f"  Median distance:    {median_dist:.6f}")
        print(f"  Min distance:       {min_dist:.6f}")
        print(f"  Max distance:       {max_dist:.6f}")
        print(f"  Memorized samples:  {memorized_count}/{len(distances)} ({memorized_ratio:.2%})")
        print(f"  Threshold:          {threshold}")
    
    # Latent space analysis
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
    if 'rgb' in summary:
        max_memorized_ratio = max(max_memorized_ratio, summary['rgb']['memorized_ratio'])
    if 'latent' in summary:
        max_memorized_ratio = max(max_memorized_ratio, summary['latent']['memorized_ratio'])
    
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
    results_df = pd.DataFrame({
        'generated_idx': range(len(generated_samples.get('rgb', generated_samples.get('latents', []))))
    })
    
    if 'rgb_l2_distances' in results:
        results_df['rgb_l2_distance'] = results['rgb_l2_distances']
        results_df['rgb_nn_index'] = results['rgb_nn_indices']
    
    if 'latent_l2_distances' in results:
        results_df['latent_l2_distance'] = results['latent_l2_distances']
        results_df['latent_cosine_similarity'] = results['latent_cosine_similarities']
        results_df['latent_nn_index'] = results['latent_nn_indices_cosine']
    
    results_df.to_csv(output_dir / 'memorization_results.csv', index=False)
    print(f"Saved detailed results to {output_dir / 'memorization_results.csv'}")
    
    # Create plots
    plot_distributions(results, output_dir)
    
    # Save closest matches
    if 'rgb_l2_distances' in results:
        save_closest_matches(generated_samples, training_samples, results, output_dir, top_k=20)
    
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
        default=100,
        help="Number of samples to generate for testing"
    )
    parser.add_argument(
        "--num_training",
        type=int,
        default=5000,
        help="Number of training samples to compare against"
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

