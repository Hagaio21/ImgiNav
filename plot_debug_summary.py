#!/usr/bin/env python3
"""
Plot summary of debug results across all architectures.

Creates comparison plots for:
- Loss curves across attention ablations
- VAE reconstruction MSE
- Loss reduction percentages
- Comparison across data types (rooms/scenes/both)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

# Color palette for different attention types
ATTENTION_COLORS = {
    "down": "#1f77b4",  # blue
    "bottleneck": "#ff7f0e",  # orange
    "up": "#2ca02c",  # green
    "all": "#d62728"  # red
}

# Line styles for different data types
DATA_TYPE_STYLES = {
    "regular": "-",  # solid
    "rooms": "--",  # dashed
    "scenes": "-."  # dash-dot
}


def load_results(summary_path):
    """Load summary results from JSON."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(all_results, output_dir):
    """Plot loss curves for all architectures."""
    # Separate regular and spatial VAE results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Regular VAE types
    regular_data_types = ["regular", "regular_rooms", "regular_scenes"]
    # Spatial VAE types
    spatial_data_types = ["spatial", "spatial_rooms", "spatial_scenes"]
    
    # Plot regular VAE results
    for idx, data_type in enumerate(regular_data_types):
        ax = axes[0, idx]
        
        if data_type not in all_results:
            ax.text(0.5, 0.5, f"No data for {data_type}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Regular VAE - {data_type.replace('regular_', '').capitalize()}")
            continue
        
        for attn_type in ["down", "bottleneck", "up", "all"]:
            if attn_type not in all_results[data_type]:
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics is None or "overfit_test" not in metrics:
                continue
            
            losses = metrics["overfit_test"].get("losses", [])
            if not losses:
                continue
            
            iterations = np.arange(len(losses))
            ax.plot(iterations, losses, 
                   color=ATTENTION_COLORS[attn_type],
                   linestyle=DATA_TYPE_STYLES.get(data_type, "-"),
                   label=attn_type,
                   linewidth=2,
                   alpha=0.8)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Regular VAE - {data_type.replace('regular_', '').capitalize()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Plot spatial VAE results
    for idx, data_type in enumerate(spatial_data_types):
        ax = axes[1, idx]
        
        if data_type not in all_results:
            ax.text(0.5, 0.5, f"No data for {data_type}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Spatial VAE - {data_type.replace('spatial_', '').capitalize()}")
            continue
        
        for attn_type in ["down", "bottleneck", "up", "all"]:
            if attn_type not in all_results[data_type]:
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics is None or "overfit_test" not in metrics:
                continue
            
            losses = metrics["overfit_test"].get("losses", [])
            if not losses:
                continue
            
            iterations = np.arange(len(losses))
            ax.plot(iterations, losses, 
                   color=ATTENTION_COLORS[attn_type],
                   linestyle=DATA_TYPE_STYLES.get(data_type, "-"),
                   label=attn_type,
                   linewidth=2,
                   alpha=0.8)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Spatial VAE - {data_type.replace('spatial_', '').capitalize()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        for attn_type in ["down", "bottleneck", "up", "all"]:
            if attn_type not in all_results[data_type]:
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics is None or "overfit_test" not in metrics:
                continue
            
            losses = metrics["overfit_test"].get("losses", [])
            if not losses:
                continue
            
            iterations = np.arange(len(losses))
            ax.plot(iterations, losses, 
                   color=ATTENTION_COLORS[attn_type],
                   linestyle=DATA_TYPE_STYLES[data_type],
                   label=attn_type,
                   linewidth=2,
                   alpha=0.8)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"{data_type.capitalize()} - Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / "summary_loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to: {output_path}")


def plot_batch_size_results(all_results, output_dir):
    """Plot batch size test results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Separate regular and spatial
    regular_data_types = ["regular", "regular_rooms", "regular_scenes"]
    spatial_data_types = ["spatial", "spatial_rooms", "spatial_scenes"]
    attention_types = ["down", "bottleneck", "up", "all"]
    
    x_pos = np.arange(len(attention_types))
    width = 0.25
    
    # 1. Optimal Batch Size - Regular VAE
    ax = axes[0, 0]
    for i, data_type in enumerate(regular_data_types):
        if data_type not in all_results:
            continue
        
        optimal_bs = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                optimal_bs.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("batch_size_test"):
                optimal_bs.append(metrics["batch_size_test"].get("optimal_batch_size", 0))
            else:
                optimal_bs.append(0)
        
        label = data_type.replace("regular_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, optimal_bs, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Optimal Batch Size")
    ax.set_title("Regular VAE - Optimal Batch Size")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Optimal Batch Size - Spatial VAE
    ax = axes[0, 1]
    for i, data_type in enumerate(spatial_data_types):
        if data_type not in all_results:
            continue
        
        optimal_bs = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                optimal_bs.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("batch_size_test"):
                optimal_bs.append(metrics["batch_size_test"].get("optimal_batch_size", 0))
            else:
                optimal_bs.append(0)
        
        label = data_type.replace("spatial_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, optimal_bs, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Optimal Batch Size")
    ax.set_title("Spatial VAE - Optimal Batch Size")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Peak Memory Usage - Regular VAE
    ax = axes[1, 0]
    for i, data_type in enumerate(regular_data_types):
        if data_type not in all_results:
            continue
        
        peak_mem = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                peak_mem.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("batch_size_test"):
                peak_mem.append(metrics["batch_size_test"].get("max_successful_memory_gb", 0))
            else:
                peak_mem.append(0)
        
        label = data_type.replace("regular_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, peak_mem, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("Regular VAE - Peak Memory Usage")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Peak Memory Usage - Spatial VAE
    ax = axes[1, 1]
    for i, data_type in enumerate(spatial_data_types):
        if data_type not in all_results:
            continue
        
        peak_mem = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                peak_mem.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("batch_size_test"):
                peak_mem.append(metrics["batch_size_test"].get("max_successful_memory_gb", 0))
            else:
                peak_mem.append(0)
        
        label = data_type.replace("spatial_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, peak_mem, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("Spatial VAE - Peak Memory Usage")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "summary_batch_size_memory.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved batch size and memory plots to: {output_path}")


def plot_metrics_comparison(all_results, output_dir):
    """Plot comparison of key metrics across architectures."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Group by VAE type
    regular_data_types = ["regular", "regular_rooms", "regular_scenes"]
    spatial_data_types = ["spatial", "spatial_rooms", "spatial_scenes"]
    attention_types = ["down", "bottleneck", "up", "all"]
    
    x_pos = np.arange(len(attention_types))
    width = 0.25
    
    # 1. VAE Reconstruction MSE - Regular VAE
    ax = axes[0, 0]
    for i, data_type in enumerate(regular_data_types):
        if data_type not in all_results:
            continue
        
        vae_mses = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                vae_mses.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("vae_round_trip"):
                vae_mses.append(metrics["vae_round_trip"].get("vae_reconstruction_mse", 0))
            else:
                vae_mses.append(0)
        
        label = data_type.replace("regular_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, vae_mses, width, 
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("VAE Reconstruction MSE")
    ax.set_title("Regular VAE - Reconstruction Quality")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1b. VAE Reconstruction MSE - Spatial VAE
    ax = axes[0, 1]
    for i, data_type in enumerate(spatial_data_types):
        if data_type not in all_results:
            continue
        
        vae_mses = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                vae_mses.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("vae_round_trip"):
                vae_mses.append(metrics["vae_round_trip"].get("vae_reconstruction_mse", 0))
            else:
                vae_mses.append(0)
        
        label = data_type.replace("spatial_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, vae_mses, width, 
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("VAE Reconstruction MSE")
    ax.set_title("Spatial VAE - Reconstruction Quality")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Initial Loss - Regular VAE
    ax = axes[0, 2]
    for i, data_type in enumerate(regular_data_types):
        if data_type not in all_results:
            continue
        
        init_losses = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                init_losses.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("overfit_test"):
                init_losses.append(metrics["overfit_test"].get("initial_loss", 0))
            else:
                init_losses.append(0)
        
        label = data_type.replace("regular_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, init_losses, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Initial Loss")
    ax.set_title("Regular VAE - Initial Loss")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Final Loss - Regular VAE
    ax = axes[1, 0]
    for i, data_type in enumerate(regular_data_types):
        if data_type not in all_results:
            continue
        
        final_losses = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                final_losses.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("overfit_test"):
                final_losses.append(metrics["overfit_test"].get("final_loss", 0))
            else:
                final_losses.append(0)
        
        label = data_type.replace("regular_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, final_losses, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Final Loss")
    ax.set_title("Regular VAE - Final Loss")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # 4. Loss Reduction Percentage - Regular VAE
    ax = axes[1, 1]
    for i, data_type in enumerate(regular_data_types):
        if data_type not in all_results:
            continue
        
        reductions = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                reductions.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("overfit_test"):
                reductions.append(metrics["overfit_test"].get("loss_reduction_percent", 0))
            else:
                reductions.append(0)
        
        label = data_type.replace("regular_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, reductions, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Loss Reduction (%)")
    ax.set_title("Regular VAE - Loss Reduction")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    
    # 5. Peak Memory from Overfit Test - Regular VAE
    ax = axes[1, 2]
    for i, data_type in enumerate(regular_data_types):
        if data_type not in all_results:
            continue
        
        peak_mem = []
        for attn_type in attention_types:
            if attn_type not in all_results[data_type]:
                peak_mem.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("overfit_test"):
                peak_mem.append(metrics["overfit_test"].get("peak_memory_gb", 0))
            else:
                peak_mem.append(0)
        
        label = data_type.replace("regular_", "").capitalize() or "All"
        ax.bar(x_pos + i * width, peak_mem, width,
              label=label, alpha=0.8)
    
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("Regular VAE - Peak Memory")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(attention_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "summary_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to: {output_path}")


def plot_heatmap(all_results, output_dir):
    """Create heatmap of loss reduction percentages."""
    # Include both regular and spatial VAE types
    all_data_types = ["regular", "regular_rooms", "regular_scenes", 
                     "spatial", "spatial_rooms", "spatial_scenes"]
    attention_types = ["down", "bottleneck", "up", "all"]
    
    # Build matrix
    matrix = []
    labels = []
    
    for data_type in all_data_types:
        row = []
        for attn_type in attention_types:
            if (data_type not in all_results or 
                attn_type not in all_results[data_type]):
                row.append(0)
                continue
            
            metrics = all_results[data_type][attn_type]
            if metrics and metrics.get("overfit_test"):
                row.append(metrics["overfit_test"].get("loss_reduction_percent", 0))
            else:
                row.append(0)
        matrix.append(row)
        labels.append(f"{data_type.capitalize()}")
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(len(all_data_types)):
        for j in range(len(attention_types)):
            text = ax.text(j, i, f"{matrix[i, j]:.1f}%",
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(np.arange(len(attention_types)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(attention_types)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Data Type")
    ax.set_title("Loss Reduction Percentage Heatmap")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Loss Reduction (%)", rotation=270, labelpad=20)
    
    plt.tight_layout()
    output_path = output_dir / "summary_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot summary of debug results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("debug_results"),
        help="Directory containing debug results"
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Path to summary JSON file (default: results_dir/summary_all_results.json)"
    )
    args = parser.parse_args()
    
    # Determine summary file path
    if args.summary_file:
        summary_path = args.summary_file
    else:
        summary_path = args.results_dir / "summary_all_results.json"
    
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        print("Please run run_debug_all_architectures.py first.")
        return
    
    # Load results
    print(f"Loading results from: {summary_path}")
    all_results = load_results(summary_path)
    
    # Create plots
    print("\nGenerating plots...")
    plot_loss_curves(all_results, args.results_dir)
    plot_metrics_comparison(all_results, args.results_dir)
    plot_heatmap(all_results, args.results_dir)
    plot_batch_size_results(all_results, args.results_dir)
    
    print(f"\n{'='*80}")
    print("All plots generated successfully!")
    print(f"Results directory: {args.results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

