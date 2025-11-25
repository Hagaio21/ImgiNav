#!/usr/bin/env python3
"""
Compare metrics across all diffusion experiments.

Loads metrics CSV files from all experiments and creates comparison plots
focusing on train_loss and val_loss.
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def find_experiment_metrics(base_dir):
    """
    Find all experiment metrics CSV files.
    
    Args:
        base_dir: Base directory containing experiment folders
        
    Returns:
        List of tuples: (exp_name, exp_path, metrics_path)
    """
    base_dir = Path(base_dir)
    experiments = []
    
    # Look for all directories that might contain experiments
    if not base_dir.exists():
        print(f"Warning: Base directory does not exist: {base_dir}")
        return experiments
    
    # Search for metrics CSV files
    for metrics_file in base_dir.rglob("*_metrics.csv"):
        # Extract experiment name from filename
        exp_name = metrics_file.stem.replace("_metrics", "")
        exp_dir = metrics_file.parent
        
        experiments.append((exp_name, exp_dir, metrics_file))
    
    return experiments


def load_experiment_metrics(metrics_path):
    """
    Load metrics from CSV file.
    
    Args:
        metrics_path: Path to metrics CSV file
        
    Returns:
        DataFrame with metrics, or None if loading fails
    """
    try:
        df = pd.read_csv(metrics_path)
        if len(df) == 0:
            return None
        return df
    except Exception as e:
        print(f"Warning: Could not load {metrics_path}: {e}")
        return None


def extract_experiment_info(exp_name):
    """
    Extract structured information from experiment name.
    
    Args:
        exp_name: Experiment name like "diff_clip_regular_small_all"
        
    Returns:
        dict with parsed components
    """
    parts = exp_name.split("_")
    info = {
        "full_name": exp_name,
        "vae_type": "unknown",
        "data_type": "unknown",
        "size": "unknown",
        "attention": "unknown"
    }
    
    # Parse common patterns
    if "regular" in parts:
        info["vae_type"] = "regular"
        idx = parts.index("regular")
        if idx + 1 < len(parts):
            if parts[idx + 1] in ["rooms", "scenes"]:
                info["data_type"] = parts[idx + 1]
            else:
                info["data_type"] = "both"
        else:
            info["data_type"] = "both"
    elif "spatial" in parts:
        info["vae_type"] = "spatial"
        idx = parts.index("spatial")
        if idx + 1 < len(parts):
            if parts[idx + 1] in ["rooms", "scenes"]:
                info["data_type"] = parts[idx + 1]
            else:
                info["data_type"] = "both"
        else:
            info["data_type"] = "both"
    
    # Find size
    for size in ["small", "medium", "large"]:
        if size in parts:
            info["size"] = size
            break
    
    # Find attention type
    for attn in ["down", "bottleneck", "up", "all"]:
        if attn in parts:
            info["attention"] = attn
            break
    
    return info


def create_comparison_plots(all_experiments, output_dir):
    """
    Create comparison plots for train_loss and val_loss.
    
    Args:
        all_experiments: List of (exp_name, exp_path, metrics_path, df, info) tuples
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter experiments with valid data
    valid_experiments = [(name, path, metrics_path, df, info) 
                         for name, path, metrics_path, df, info in all_experiments 
                         if df is not None and len(df) > 0]
    
    if len(valid_experiments) == 0:
        print("No valid experiments found for plotting")
        return
    
    print(f"Creating plots for {len(valid_experiments)} experiments...")
    
    # Create main comparison plot: Train and Val Loss
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Train Loss
    ax = axes[0]
    for exp_name, exp_path, metrics_path, df, info in valid_experiments:
        if "epoch" not in df.columns or "train_loss" not in df.columns:
            continue
        
        label = f"{info['size']}_{info['attention']}_{info['data_type']}"
        ax.plot(df["epoch"], df["train_loss"], 
               label=label, linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Train Loss Comparison Across All Experiments", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Val Loss
    ax = axes[1]
    for exp_name, exp_path, metrics_path, df, info in valid_experiments:
        if "epoch" not in df.columns or "val_loss" not in df.columns:
            continue
        
        label = f"{info['size']}_{info['attention']}_{info['data_type']}"
        ax.plot(df["epoch"], df["val_loss"], 
               label=label, linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val Loss", fontsize=12)
    ax.set_title("Validation Loss Comparison Across All Experiments", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_dir / "all_experiments_loss_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {plot_path}")
    
    # Create grouped plots by size
    sizes = ["small", "medium", "large"]
    fig, axes = plt.subplots(len(sizes), 2, figsize=(16, 6 * len(sizes)))
    if len(sizes) == 1:
        axes = axes.reshape(1, -1)
    
    for size_idx, size in enumerate(sizes):
        size_experiments = [(name, path, metrics_path, df, info) 
                           for name, path, metrics_path, df, info in valid_experiments 
                           if info["size"] == size]
        
        if len(size_experiments) == 0:
            continue
        
        # Train loss
        ax = axes[size_idx, 0]
        for exp_name, exp_path, metrics_path, df, info in size_experiments:
            if "epoch" not in df.columns or "train_loss" not in df.columns:
                continue
            
            label = f"{info['attention']}_{info['data_type']}"
            ax.plot(df["epoch"], df["train_loss"], 
                   label=label, linewidth=2, alpha=0.7)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Train Loss", fontsize=12)
        ax.set_title(f"Train Loss - {size.capitalize()} Models", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Val loss
        ax = axes[size_idx, 1]
        for exp_name, exp_path, metrics_path, df, info in size_experiments:
            if "epoch" not in df.columns or "val_loss" not in df.columns:
                continue
            
            label = f"{info['attention']}_{info['data_type']}"
            ax.plot(df["epoch"], df["val_loss"], 
                   label=label, linewidth=2, alpha=0.7)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Val Loss", fontsize=12)
        ax.set_title(f"Val Loss - {size.capitalize()} Models", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_dir / "all_experiments_loss_by_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved size-grouped plot: {plot_path}")


def create_summary_table(all_experiments, output_dir):
    """
    Create a summary table with key metrics.
    
    Args:
        all_experiments: List of (exp_name, exp_path, metrics_path, df, info) tuples
        output_dir: Directory to save summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    
    for exp_name, exp_path, metrics_path, df, info in all_experiments:
        if df is None or len(df) == 0:
            continue
        
        # Get best and final metrics
        row = {
            "experiment": exp_name,
            "vae_type": info["vae_type"],
            "data_type": info["data_type"],
            "size": info["size"],
            "attention": info["attention"],
            "num_epochs": len(df),
            "final_train_loss": df["train_loss"].iloc[-1] if "train_loss" in df.columns else None,
            "final_val_loss": df["val_loss"].iloc[-1] if "val_loss" in df.columns else None,
            "best_val_loss": df["val_loss"].min() if "val_loss" in df.columns else None,
            "best_val_epoch": df.loc[df["val_loss"].idxmin(), "epoch"] if "val_loss" in df.columns else None,
        }
        
        summary_data.append(row)
    
    if len(summary_data) == 0:
        print("No data for summary table")
        return
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(["size", "attention", "data_type"])
    
    # Save as CSV
    csv_path = output_dir / "experiments_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary table: {csv_path}")
    
    # Save as JSON for easier reading
    json_path = output_dir / "experiments_summary.json"
    summary_df.to_json(json_path, orient='records', indent=2)
    print(f"Saved summary JSON: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare metrics across all experiments")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default="/work3/s233249/ImgiNav/experiments/clip",
        help="Base directory containing experiment folders"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/work3/s233249/ImgiNav/experiments/clip/comparison_summary",
        help="Output directory for summary and plots"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("Comparing All Experiments")
    print("="*80)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Find all experiments
    print("Searching for experiment metrics...")
    experiments = find_experiment_metrics(args.base_dir)
    print(f"Found {len(experiments)} experiment metrics files")
    
    if len(experiments) == 0:
        print("No experiments found. Exiting.")
        return
    
    # Load all metrics
    print("\nLoading metrics...")
    all_experiments = []
    for exp_name, exp_path, metrics_path in experiments:
        df = load_experiment_metrics(metrics_path)
        info = extract_experiment_info(exp_name)
        all_experiments.append((exp_name, exp_path, metrics_path, df, info))
        if df is not None:
            print(f"  ✓ {exp_name}: {len(df)} epochs")
        else:
            print(f"  ✗ {exp_name}: Failed to load")
    
    # Create plots
    print("\nCreating comparison plots...")
    create_comparison_plots(all_experiments, args.output_dir)
    
    # Create summary
    print("\nCreating summary table...")
    create_summary_table(all_experiments, args.output_dir)
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

