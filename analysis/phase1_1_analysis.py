#!/usr/bin/env python3
"""
Analysis script for Phase 1.1: Latent Channel Sweep
Loads metrics from all experiments and creates comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set seaborn style with dark grid and pastel colors
sns.set_style("darkgrid")
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Color palette for experiments
PASTEL_COLORS = [
    '#FFB3BA',  # Pastel red
    '#BAFFC9',  # Pastel green
    '#BAE1FF',  # Pastel blue
    '#FFFFBA',  # Pastel yellow
    '#FFDFBA',  # Pastel orange
]


def load_metrics(phase_dir):
    """Load all metrics CSV files from phase directory."""
    phase_path = Path(phase_dir)
    if not phase_path.exists():
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")
    
    metrics_files = list(phase_path.glob("*_metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found in {phase_dir}")
    
    print(f"Found {len(metrics_files)} metrics files:")
    all_data = {}
    
    for metrics_file in metrics_files:
        exp_name = metrics_file.stem.replace("_metrics", "")
        print(f"  - {exp_name}")
        
        df = pd.read_csv(metrics_file)
        df['experiment'] = exp_name
        all_data[exp_name] = df
    
    return all_data


def create_loss_curves(all_data, output_dir):
    """Create loss curve plots comparing all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1.1: Loss Curves Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for i, (exp_name, df) in enumerate(all_data.items()):
        if 'train_loss' in df.columns:
            ax1.plot(df['epoch'], df['train_loss'], 
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for i, (exp_name, df) in enumerate(all_data.items()):
        if 'val_loss' in df.columns:
            ax2.plot(df['epoch'], df['val_loss'], 
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training MSE (RGB)
    ax3 = axes[1, 0]
    for i, (exp_name, df) in enumerate(all_data.items()):
        mse_col = [c for c in df.columns if 'MSE' in c.upper() and 'train' in c.lower()]
        if mse_col:
            ax3.plot(df['epoch'], df[mse_col[0]], 
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training MSE (RGB)', fontsize=12)
    ax3.set_title('Training RGB MSE', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation MSE (RGB)
    ax4 = axes[1, 1]
    for i, (exp_name, df) in enumerate(all_data.items()):
        mse_col = [c for c in df.columns if 'MSE' in c.upper() and 'val' in c.lower()]
        if mse_col:
            ax4.plot(df['epoch'], df[mse_col[0]], 
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation MSE (RGB)', fontsize=12)
    ax4.set_title('Validation RGB MSE', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_final_metrics_comparison(all_data, output_dir):
    """Create bar plot comparing final metrics across experiments."""
    # Extract final epoch metrics
    final_metrics = []
    
    for exp_name, df in all_data.items():
        final_row = df.iloc[-1]
        metrics = {
            'experiment': exp_name,
            'train_loss': final_row.get('train_loss', np.nan),
            'val_loss': final_row.get('val_loss', np.nan),
        }
        
        # Extract MSE if available
        mse_cols = [c for c in df.columns if 'MSE' in c.upper()]
        if mse_cols:
            for col in mse_cols:
                if 'train' in col.lower():
                    metrics['train_MSE'] = final_row.get(col, np.nan)
                elif 'val' in col.lower():
                    metrics['val_MSE'] = final_row.get(col, np.nan)
        
        # Extract segmentation accuracy if available
        seg_cols = [c for c in df.columns if 'CE_segmentation' in c]
        if seg_cols:
            for col in seg_cols:
                if 'train' in col.lower():
                    metrics['train_seg_loss'] = final_row.get(col, np.nan)
                elif 'val' in col.lower():
                    metrics['val_seg_loss'] = final_row.get(col, np.nan)
        
        final_metrics.append(metrics)
    
    final_df = pd.DataFrame(final_metrics)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1.1: Final Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Final Validation Loss
    ax1 = axes[0, 0]
    sns.barplot(data=final_df, x='experiment', y='val_loss', ax=ax1, palette=PASTEL_COLORS)
    ax1.set_xlabel('Experiment', fontsize=12)
    ax1.set_ylabel('Final Validation Loss', fontsize=12)
    ax1.set_title('Final Validation Loss', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Final Validation MSE
    ax2 = axes[0, 1]
    if 'val_MSE' in final_df.columns:
        sns.barplot(data=final_df, x='experiment', y='val_MSE', ax=ax2, palette=PASTEL_COLORS)
        ax2.set_xlabel('Experiment', fontsize=12)
        ax2.set_ylabel('Final Validation MSE', fontsize=12)
        ax2.set_title('Final Validation MSE (RGB)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'MSE data not available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Final Validation MSE (RGB)', fontsize=14, fontweight='bold')
    
    # Plot 3: Final Training Loss
    ax3 = axes[1, 0]
    sns.barplot(data=final_df, x='experiment', y='train_loss', ax=ax3, palette=PASTEL_COLORS)
    ax3.set_xlabel('Experiment', fontsize=12)
    ax3.set_ylabel('Final Training Loss', fontsize=12)
    ax3.set_title('Final Training Loss', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Final Validation Segmentation Loss
    ax4 = axes[1, 1]
    if 'val_seg_loss' in final_df.columns:
        sns.barplot(data=final_df, x='experiment', y='val_seg_loss', ax=ax4, palette=PASTEL_COLORS)
        ax4.set_xlabel('Experiment', fontsize=12)
        ax4.set_ylabel('Final Validation Seg Loss', fontsize=12)
        ax4.set_title('Final Validation Segmentation Loss', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Segmentation loss data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Final Validation Segmentation Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "final_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save final metrics table
    table_path = output_dir / "final_metrics_table.csv"
    final_df.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")


def create_convergence_analysis(all_data, output_dir):
    """Analyze convergence speed and stability."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Phase 1.1: Convergence Analysis', fontsize=16, fontweight='bold')
    
    convergence_data = []
    
    for exp_name, df in all_data.items():
        if 'val_loss' not in df.columns:
            continue
        
        val_loss = df['val_loss'].values
        epochs = df['epoch'].values
        
        # Find epoch where loss stabilizes (within 5% of final loss)
        final_loss = val_loss[-1]
        threshold = final_loss * 1.05
        stable_epochs = np.where(val_loss <= threshold)[0]
        convergence_epoch = stable_epochs[0] + 1 if len(stable_epochs) > 0 else len(epochs)
        
        # Calculate improvement (first to last)
        initial_loss = val_loss[0]
        improvement = initial_loss - final_loss
        improvement_pct = (improvement / initial_loss) * 100
        
        convergence_data.append({
            'experiment': exp_name,
            'convergence_epoch': convergence_epoch,
            'improvement_pct': improvement_pct,
            'final_loss': final_loss
        })
        
        # Plot loss reduction over time
        axes[0].plot(epochs, val_loss, 
                    label=exp_name, 
                    color=PASTEL_COLORS[len(convergence_data) % len(PASTEL_COLORS)], 
                    linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Loss Reduction Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot: Convergence speed
    conv_df = pd.DataFrame(convergence_data)
    if len(conv_df) > 0:
        sns.barplot(data=conv_df, x='experiment', y='convergence_epoch', 
                   ax=axes[1], palette=PASTEL_COLORS)
        axes[1].set_xlabel('Experiment', fontsize=12)
        axes[1].set_ylabel('Convergence Epoch', fontsize=12)
        axes[1].set_title('Convergence Speed', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "convergence_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save convergence data
    if len(conv_df) > 0:
        conv_path = output_dir / "convergence_analysis.csv"
        conv_df.to_csv(conv_path, index=False)
        print(f"Saved: {conv_path}")


def create_summary_report(all_data, output_dir):
    """Create a text summary report."""
    report_path = output_dir / "analysis_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 1.1: Latent Channel Sweep - Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Best experiment by validation loss
        best_val_loss = float('inf')
        best_exp = None
        
        for exp_name, df in all_data.items():
            if 'val_loss' in df.columns:
                final_val_loss = df['val_loss'].iloc[-1]
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_exp = exp_name
        
        if best_exp:
            f.write(f"Best Experiment (by Final Validation Loss): {best_exp}\n")
            f.write(f"  Final Validation Loss: {best_val_loss:.6f}\n\n")
        
        # Summary for each experiment
        f.write("-" * 80 + "\n")
        f.write("Experiment Summaries:\n")
        f.write("-" * 80 + "\n\n")
        
        for exp_name, df in all_data.items():
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"  Total Epochs: {len(df)}\n")
            
            if 'train_loss' in df.columns:
                initial_train = df['train_loss'].iloc[0]
                final_train = df['train_loss'].iloc[-1]
                f.write(f"  Training Loss: {initial_train:.6f} → {final_train:.6f}\n")
            
            if 'val_loss' in df.columns:
                initial_val = df['val_loss'].iloc[0]
                final_val = df['val_loss'].iloc[-1]
                f.write(f"  Validation Loss: {initial_val:.6f} → {final_val:.6f}\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("Recommendations:\n")
        f.write("-" * 80 + "\n\n")
        f.write("1. Select top 2 experiments based on:\n")
        f.write("   - Lowest final validation loss\n")
        f.write("   - Stable convergence\n")
        f.write("   - Reasonable model size\n\n")
        f.write("2. Update Phase 1.2 configs with:\n")
        f.write("   - Best latent_channels\n")
        f.write("   - Best base_channels\n\n")
    
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1.1 experiments")
    parser.add_argument(
        "--phase-dir",
        type=str,
        default="outputs/phase1_1_latent_channels",
        help="Path to phase directory containing metrics CSVs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis plots (default: same as phase-dir/analysis)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    phase_dir = Path(args.phase_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = phase_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 1.1: Latent Channel Sweep Analysis")
    print("=" * 80)
    print(f"Loading metrics from: {phase_dir}")
    print(f"Saving plots to: {output_dir}")
    print()
    
    # Load all metrics
    all_data = load_metrics(phase_dir)
    print(f"\nLoaded {len(all_data)} experiments\n")
    
    # Create visualizations
    print("Creating visualizations...")
    create_loss_curves(all_data, output_dir)
    create_final_metrics_comparison(all_data, output_dir)
    create_convergence_analysis(all_data, output_dir)
    create_summary_report(all_data, output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

