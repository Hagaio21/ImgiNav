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
from training.memorization_utils import (
    load_training_samples,
    generate_samples,
    compute_latent_distances,
    compute_diversity_metrics_latent,
    check_memorization as check_memorization_core,
    plot_perturbation_results,
    perturbation_test,
    sample_from_latent
)


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
        sns.barplot(x=metrics, y=values, ax=ax, hue=metrics, palette='muted', edgecolor='black', linewidth=1.5, legend=False)
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


def check_memorization(config_path, manifest_path, output_dir,
                       num_generate=1000, num_training=None, method="ddpm", latent_perturbation_std=0.0,
                       model=None, checkpoint_path=None, run_perturbation_test=False, num_perturbation_samples=20):
    """Main function to check for memorization.
    
    Args:
        model: Optional model object to use directly (if provided, checkpoint_path is ignored)
        checkpoint_path: Path to checkpoint (only used if model is None)
        latent_perturbation_std: If > 0, adds Gaussian noise to training latents before comparison.
                                  This tests if the model memorizes exact latents vs. learning patterns.
                                  Small values (0.01-0.05) are recommended.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model if not provided
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either model or checkpoint_path must be provided")
        
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
        
        # Load checkpoint using proper class method
        # The checkpoint may contain its own config, but we'll use the provided config
        # to ensure consistency with the current setup (autoencoder path, etc.)
        print(f"Loading checkpoint from: {checkpoint_path}")
        model = DiffusionModel.load_checkpoint(
            checkpoint_path, 
            map_location=device,
            config=diffusion_cfg  # Use config from file (ensures correct autoencoder path)
        )
        
        model = model.to(device).eval()
        print("Model loaded successfully")
    else:
        # Use provided model (set to eval mode and ensure on correct device)
        print("\n" + "="*60)
        print("Using provided model...")
        print("="*60)
        model = model.to(device).eval()
        print("Model ready (already loaded)")
        
        # Still need config for dataset loading
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
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
    
    # Filter out empty images from memorization check
    # Empty images are similar by default and skew memorization metrics
    if hasattr(dataset, 'manifest') or has_layout_path:
        import pandas as pd
        manifest_df_check = pd.read_csv(manifest_path)
        if "is_empty" in manifest_df_check.columns:
            # Count empty samples
            empty_count = (manifest_df_check["is_empty"] == True).sum()
            total_count = len(manifest_df_check)
            print(f"\nFiltering empty images for memorization check:")
            print(f"  Total samples in manifest: {total_count}")
            print(f"  Empty samples (will be excluded): {empty_count}")
            print(f"  Non-empty samples (will be used): {total_count - empty_count}")
            
            # Update dataset filters to exclude empty images
            if "filters" not in config["dataset"]:
                config["dataset"]["filters"] = {}
            config["dataset"]["filters"]["is_empty"] = [False]
            # Rebuild dataset with filter
            dataset = build_dataset(config)
    
    # If num_training is None, use entire dataset (after filtering)
    if num_training is None:
        num_training = len(dataset)
        print(f"\nChecking against entire dataset (after filtering): {num_training} samples")
    else:
        print(f"Checking against {num_training} training samples (dataset has {len(dataset)} total after filtering)")
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
    
    # Use the core memorization check function from utils
    summary = check_memorization_core(
        model=model,
        training_samples=training_samples,
        generated_samples=generated_samples,
        output_dir=output_dir,
        latent_perturbation_std=latent_perturbation_std,
        run_perturbation_test=run_perturbation_test,
        num_perturbation_samples=num_perturbation_samples,
        method=method,
        device=device
    )
    
    # Add metadata to summary
    summary['num_generated'] = num_generate
    summary['num_training'] = num_training
    summary['method'] = method
    
    # Save updated summary
    import json
    with open(output_dir / 'memorization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create plots and save closest matches
    # Load results from saved files for plotting
    results_df = pd.read_csv(output_dir / 'memorization_results.csv')
    results = {}
    if 'latent_l2_distance' in results_df.columns:
        results['latent_l2_distances'] = results_df['latent_l2_distance'].values
        results['latent_cosine_similarities'] = results_df['latent_cosine_similarity'].values
        results['latent_nn_indices_cosine'] = results_df['latent_nn_index'].values
    
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
        required=False,
        default=None,
        help="Path to diffusion model checkpoint (required if not using --model)"
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
    parser.add_argument(
        "--latent_perturbation_std",
        type=float,
        default=0.0,
        help="Standard deviation of latent perturbation to apply to training samples (default: 0.0 = no perturbation). Use small values (e.g., 0.01-0.05) to test robustness."
    )
    parser.add_argument(
        "--run_perturbation_test",
        action="store_true",
        help="Run perturbation test: add noise to latents and compare outputs (tests memorization vs generalization)"
    )
    parser.add_argument(
        "--num_perturbation_samples",
        type=int,
        default=20,
        help="Number of latents to test in perturbation test (default: 20)"
    )
    
    args = parser.parse_args()
    
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when running from command line")
    
    check_memorization(
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=args.output,
        num_generate=args.num_generate,
        num_training=args.num_training,
        method=args.method,
        latent_perturbation_std=args.latent_perturbation_std,
        checkpoint_path=args.checkpoint,
        run_perturbation_test=args.run_perturbation_test,
        num_perturbation_samples=args.num_perturbation_samples
    )


if __name__ == "__main__":
    main()

