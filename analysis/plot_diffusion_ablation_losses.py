#!/usr/bin/env python3
"""
Plot loss curves for diffusion ablation experiments.
Loads metrics CSV files from all capacity/depth ablation experiments and creates comparison plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import argparse
import sys
import re

# Set seaborn style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

# Color palette for experiments
import matplotlib
_tab20_colors = matplotlib.colormaps['tab20'].colors
EXPERIMENT_COLORS = [
    _tab20_colors[0],   # Blue
    _tab20_colors[1],   # Light Blue
    _tab20_colors[2],   # Red
    _tab20_colors[3],   # Light Red
    _tab20_colors[4],   # Green
    _tab20_colors[5],   # Light Green
    _tab20_colors[6],   # Orange
    _tab20_colors[7],   # Light Orange
    _tab20_colors[8],   # Purple
    _tab20_colors[9],   # Light Purple
    _tab20_colors[10],  # Brown
    _tab20_colors[11],  # Pink
]


def parse_experiment_name(dir_name):
    """Extract capacity, depth, and resblocks from directory name like 'capacity_unet64_d4' or 'capacity_unet256_d4_rb3'."""
    match = re.match(r'capacity_unet(\d+)_d(\d+)(?:_rb(\d+))?', dir_name)
    if match:
        capacity = int(match.group(1))
        depth = int(match.group(2))
        resblocks = int(match.group(3)) if match.group(3) else None
        if resblocks:
            return capacity, depth, resblocks, f"{capacity}_d{depth}_rb{resblocks}"
        else:
            return capacity, depth, None, f"{capacity}_d{depth}"
    return None, None, None, dir_name


def load_all_metrics(ablation_dir):
    """Load all metrics CSV files from ablation experiment directories."""
    ablation_path = Path(ablation_dir)
    if not ablation_path.exists():
        raise FileNotFoundError(f"Ablation directory not found: {ablation_dir}")
    
    all_data = {}
    experiment_info = {}
    
    # Find all experiment subdirectories
    for exp_dir in sorted(ablation_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Look for metrics CSV file
        csv_files = list(exp_dir.glob("*_metrics.csv"))
        if not csv_files:
            print(f"  Warning: No metrics CSV found in {exp_dir.name}")
            continue
        
        # Use the first CSV file found (should be only one)
        csv_file = csv_files[0]
        
        # Parse experiment name
        capacity, depth, resblocks, short_name = parse_experiment_name(exp_dir.name)
        
        # Load CSV
        try:
            df = pd.read_csv(csv_file)
            if 'epoch' not in df.columns:
                print(f"  Warning: CSV {csv_file} missing 'epoch' column")
                continue
            
            # Store with short name for display
            if capacity is not None:
                display_name = short_name
            else:
                display_name = exp_dir.name
            all_data[display_name] = df
            experiment_info[display_name] = {
                'capacity': capacity,
                'depth': depth,
                'resblocks': resblocks,
                'full_name': exp_dir.name,
                'csv_file': csv_file
            }
            print(f"  Loaded: {display_name} ({len(df)} epochs)")
        except Exception as e:
            print(f"  Warning: Failed to load {csv_file}: {e}")
            continue
    
    return all_data, experiment_info


def create_loss_curves(all_data, experiment_info, output_dir):
    """Create loss curve plots comparing all experiments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Diffusion Ablation: Loss Curves Comparison', fontsize=16, fontweight='bold')
    
    # Sort experiments by capacity, then depth, then resblocks for consistent ordering
    # Handle None values by treating them as 0 for sorting
    def sort_key(item):
        exp_name = item[0]
        info = experiment_info[exp_name]
        capacity = info.get('capacity')
        depth = info.get('depth')
        resblocks = info.get('resblocks')
        return (capacity if capacity is not None else 0, 
                depth if depth is not None else 0,
                resblocks if resblocks is not None else 0)
    
    sorted_exps = sorted(all_data.items(), key=sort_key)
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    for i, (exp_name, df) in enumerate(sorted_exps):
        if 'train_loss' in df.columns:
            color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            ax1.plot(df['epoch'], df['train_loss'], 
                    label=exp_name, color=color, linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale often better for loss curves
    
    # Plot 2: Validation Loss
    # Note: Validation loss is computed every 5 epochs (eval_interval), so filter empty values
    ax2 = axes[1]
    for i, (exp_name, df) in enumerate(sorted_exps):
        if 'val_loss' in df.columns:
            color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            # Filter out empty/NaN values (validation only computed every 5 epochs)
            # Empty cells in CSV are read as empty strings, so filter those out too
            val_mask = df['val_loss'].notna() & (df['val_loss'].astype(str).str.strip() != '')
            val_df = df.loc[val_mask, ['epoch', 'val_loss']]
            if len(val_df) > 0:
                ax2.plot(val_df['epoch'], val_df['val_loss'], 
                        label=exp_name, color=color, linewidth=2, alpha=0.8,
                        marker='o', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss (every 5 epochs)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale often better for loss curves
    
    plt.tight_layout()
    output_file = output_path / "diffusion_ablation_loss_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Also create separate plots for train and val
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for i, (exp_name, df) in enumerate(sorted_exps):
        if 'train_loss' in df.columns:
            color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            ax.plot(df['epoch'], df['train_loss'], 
                   label=exp_name, color=color, linewidth=2.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Diffusion Ablation: Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    output_file = output_path / "diffusion_ablation_train_loss.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for i, (exp_name, df) in enumerate(sorted_exps):
        if 'val_loss' in df.columns:
            color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            # Filter out empty/NaN values (validation only computed every 5 epochs)
            # Empty cells in CSV are read as empty strings, so filter those out too
            val_mask = df['val_loss'].notna() & (df['val_loss'].astype(str).str.strip() != '')
            val_df = df.loc[val_mask, ['epoch', 'val_loss']]
            if len(val_df) > 0:
                ax.plot(val_df['epoch'], val_df['val_loss'], 
                       label=exp_name, color=color, linewidth=2.5, alpha=0.8,
                       marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Diffusion Ablation: Validation Loss Comparison (every 5 epochs)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    output_file = output_path / "diffusion_ablation_val_loss.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def create_capacity_depth_heatmap(all_data, experiment_info, output_dir):
    """Create a heatmap showing final loss by capacity and depth."""
    output_path = Path(output_dir)
    
    # Collect final losses
    # Note: Validation loss is computed every 5 epochs, so we need to find the last non-NaN value
    capacity_depth_loss = {}
    for exp_name, df in all_data.items():
        info = experiment_info[exp_name]
        capacity = info.get('capacity')
        depth = info.get('depth')
        
        if capacity is None or depth is None:
            continue
        
        if 'val_loss' in df.columns:
            # Get the last non-empty validation loss value
            # Empty cells in CSV are read as empty strings, so filter those out too
            val_mask = df['val_loss'].notna() & (df['val_loss'].astype(str).str.strip() != '')
            val_losses = df.loc[val_mask, 'val_loss']
            if len(val_losses) > 0:
                final_loss = val_losses.iloc[-1]
                if (capacity, depth) not in capacity_depth_loss:
                    capacity_depth_loss[(capacity, depth)] = final_loss
    
    if not capacity_depth_loss:
        print("  Warning: No capacity/depth data for heatmap")
        return
    
    # Create DataFrame for heatmap
    capacities = sorted(set(c for c, d in capacity_depth_loss.keys()))
    depths = sorted(set(d for c, d in capacity_depth_loss.keys()))
    
    heatmap_data = np.full((len(capacities), len(depths)), np.nan)
    for (c, d), loss in capacity_depth_loss.items():
        i = capacities.index(c)
        j = depths.index(d)
        heatmap_data[i, j] = loss
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(heatmap_data, cmap='viridis_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f'd{d}' for d in depths])
    ax.set_yticks(range(len(capacities)))
    ax.set_yticklabels([f'{c}' for c in capacities])
    
    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Capacity (base_channels)', fontsize=12)
    ax.set_title('Final Validation Loss by Capacity × Depth', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(capacities)):
        for j in range(len(depths)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                             ha="center", va="center", color="white", fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation Loss', fontsize=11)
    
    plt.tight_layout()
    output_file = output_path / "diffusion_ablation_capacity_depth_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves for diffusion ablation experiments")
    parser.add_argument(
        "--ablation-dir",
        type=str,
        default="/work3/s233249/ImgiNav/experiments/diffusion/ablation",
        help="Path to ablation experiments directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: ablation-dir/analysis)"
    )
    
    args = parser.parse_args()
    
    ablation_dir = Path(args.ablation_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ablation_dir / "analysis"
    
    print("=" * 80)
    print("Diffusion Ablation: Loss Curve Analysis")
    print("=" * 80)
    print(f"Ablation directory: {ablation_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load all metrics
    print("Loading metrics from experiment directories...")
    all_data, experiment_info = load_all_metrics(ablation_dir)
    
    if not all_data:
        print("ERROR: No metrics files found!")
        return 1
    
    print(f"\nLoaded {len(all_data)} experiments\n")
    
    # Create visualizations
    print("Creating loss curve plots...")
    create_loss_curves(all_data, experiment_info, output_dir)
    
    print("\nCreating capacity × depth heatmap...")
    create_capacity_depth_heatmap(all_data, experiment_info, output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

