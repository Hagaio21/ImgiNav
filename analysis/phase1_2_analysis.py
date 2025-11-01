#!/usr/bin/env python3
"""
Analysis script for Phase 1.2: VAE Test
Loads metrics from all experiments and creates comparison visualizations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import from phase1_1_analysis (reuse functions)
sys.path.insert(0, str(Path(__file__).parent))
from phase1_1_analysis import (
    load_metrics, create_loss_curves, create_final_metrics_comparison,
    create_convergence_analysis, create_summary_report,
    load_autoencoder_checkpoints, get_test_samples, create_visual_comparison
)

# Set seaborn style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Phase 1.2 experiments")
    parser.add_argument(
        "--phase-dir",
        type=str,
        default="outputs/phase1_2_vae_test",
        help="Path to phase directory containing metrics CSVs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis plots (default: same as phase-dir/analysis)"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="/work3/s233249/ImgiNav/experiments/phase1",
        help="Base directory containing individual experiment folders"
    )
    parser.add_argument(
        "--dataset-manifest",
        type=str,
        default="/work3/s233249/ImgiNav/datasets/layouts.csv",
        help="Path to dataset manifest for getting test samples"
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip visual comparison (faster if only metrics needed)"
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
    print("Phase 1.2: VAE Test Analysis")
    print("=" * 80)
    print(f"Loading metrics from: {phase_dir}")
    print(f"Saving plots to: {output_dir}")
    print()
    
    # Load all metrics
    all_data = load_metrics(phase_dir)
    exp_names = list(all_data.keys())
    print(f"\nLoaded {len(exp_names)} experiments\n")
    
    # Create visualizations from metrics
    print("Creating metric visualizations...")
    create_loss_curves(all_data, output_dir)
    create_final_metrics_comparison(all_data, output_dir)
    create_convergence_analysis(all_data, output_dir)
    
    # Phase 1.2 specific: VAE analysis
    print("Creating VAE-specific analysis...")
    create_vae_comparison(all_data, output_dir)
    create_kld_analysis(all_data, output_dir)
    create_side_by_side_metrics(all_data, output_dir)
    
    create_summary_report(all_data, output_dir)
    
    # Visual comparison with actual models
    if not args.skip_visual:
        print("\n" + "-" * 80)
        print("Loading autoencoder checkpoints for visual comparison...")
        print("-" * 80)
        checkpoints = load_autoencoder_checkpoints(exp_names, args.experiments_dir)
        
        if checkpoints:
            print("\nLoading test samples from dataset...")
            test_scenes, test_rooms = get_test_samples(args.dataset_manifest)
            
            if test_scenes and test_rooms:
                print("\nCreating visual comparisons...")
                create_visual_comparison(checkpoints, test_scenes, test_rooms, output_dir)
            else:
                print("  Could not find test samples, skipping visual comparison")
        else:
            print("  Could not load any checkpoints, skipping visual comparison")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


def create_vae_comparison(all_data, output_dir):
    """Create direct VAE vs AE comparison plots."""
    if len(all_data) != 2:
        print("  Warning: Expected 2 experiments (V1 and V2), skipping VAE comparison")
        return
    
    # Identify which is VAE and which is deterministic
    exp_names = sorted(all_data.keys())
    vae_exp = None
    ae_exp = None
    
    for exp_name in exp_names:
        if 'vae' in exp_name.lower():
            vae_exp = exp_name
        elif 'deterministic' in exp_name.lower() or 'V1' in exp_name:
            ae_exp = exp_name
    
    if not vae_exp or not ae_exp:
        # Fallback: assume first is AE, second is VAE
        ae_exp = exp_names[0]
        vae_exp = exp_names[1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 1.2: VAE vs Deterministic Encoder Comparison', fontsize=16, fontweight='bold')
    
    # Get dataframes
    df_ae = all_data[ae_exp]
    df_vae = all_data[vae_exp]
    
    # Plot 1: MSE Comparison
    ax = axes[0, 0]
    if 'val_MSE_rgb' in df_ae.columns and 'val_MSE_rgb' in df_vae.columns:
        ax.plot(df_ae['epoch'], df_ae['val_MSE_rgb'], label='Deterministic (AE)', linewidth=2.5, color='#2E86AB')
        ax.plot(df_vae['epoch'], df_vae['val_MSE_rgb'], label='VAE', linewidth=2.5, color='#A23B72')
    elif 'train_MSE_rgb' in df_ae.columns:
        ax.plot(df_ae['epoch'], df_ae.get('val_MSE_rgb', df_ae['train_MSE_rgb']), label='Deterministic (AE)', linewidth=2.5, color='#2E86AB')
        ax.plot(df_vae['epoch'], df_vae.get('val_MSE_rgb', df_vae['train_MSE_rgb']), label='VAE', linewidth=2.5, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation MSE (RGB)', fontsize=12)
    ax.set_title('RGB Reconstruction Quality', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Total Loss Comparison
    ax = axes[0, 1]
    if 'val_loss' in df_ae.columns:
        ax.plot(df_ae['epoch'], df_ae['val_loss'], label='Deterministic (AE)', linewidth=2.5, color='#2E86AB')
        ax.plot(df_vae['epoch'], df_vae['val_loss'], label='VAE', linewidth=2.5, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Total Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Metrics Bar Comparison
    ax = axes[0, 2]
    final_ae = df_ae.iloc[-1]
    final_vae = df_vae.iloc[-1]
    
    metrics_to_compare = []
    ae_values = []
    vae_values = []
    
    for metric in ['val_MSE_rgb', 'val_loss', 'train_MSE_rgb', 'train_loss']:
        if metric in final_ae.index and metric in final_vae.index:
            metrics_to_compare.append(metric.replace('_', ' ').title())
            ae_values.append(final_ae[metric])
            vae_values.append(final_vae[metric])
    
    if metrics_to_compare:
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        ax.bar(x - width/2, ae_values, width, label='Deterministic (AE)', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, vae_values, width, label='VAE', color='#A23B72', alpha=0.8)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_compare, rotation=45, ha='right')
        ax.legend(frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: KLD Loss (VAE only)
    ax = axes[1, 0]
    kld_cols = [c for c in df_vae.columns if 'KLD' in c.upper()]
    if kld_cols:
        kld_col = kld_cols[0]  # Prefer val_KLD if available
        ax.plot(df_vae['epoch'], df_vae[kld_col], label='KLD Loss', linewidth=2.5, color='#A23B72')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('KLD Loss', fontsize=12)
        ax.set_title('VAE: KLD Loss Trend', fontsize=14, fontweight='bold')
        ax.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5, label='Low threshold')
        ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='High threshold')
        ax.legend(frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3)
        
        # Add final KLD value annotation
        final_kld = df_vae[kld_col].iloc[-1]
        ax.text(0.98, 0.02, f'Final KLD: {final_kld:.6f}', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: Training Stability (Loss Curve Smoothness)
    ax = axes[1, 1]
    if 'val_loss' in df_ae.columns and 'val_loss' in df_vae.columns:
        # Calculate rolling std as smoothness metric
        window = min(5, len(df_ae) // 3)
        if window > 1:
            ae_smooth = df_ae['val_loss'].rolling(window=window, center=True).std()
            vae_smooth = df_vae['val_loss'].rolling(window=window, center=True).std()
            ax.plot(df_ae['epoch'], ae_smooth, label='Deterministic (AE)', linewidth=2.5, color='#2E86AB')
            ax.plot(df_vae['epoch'], vae_smooth, label='VAE', linewidth=2.5, color='#A23B72')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss Std Dev (Rolling)', fontsize=12)
            ax.set_title('Training Stability', fontsize=14, fontweight='bold')
            ax.legend(frameon=True, fancybox=False, shadow=False)
            ax.grid(True, alpha=0.3)
    
    # Plot 6: Metrics Difference (VAE - AE)
    ax = axes[1, 2]
    if 'val_MSE_rgb' in df_ae.columns and 'val_MSE_rgb' in df_vae.columns:
        # Align by epoch
        common_epochs = set(df_ae['epoch']) & set(df_vae['epoch'])
        if common_epochs:
            ae_aligned = df_ae[df_ae['epoch'].isin(common_epochs)].set_index('epoch').sort_index()
            vae_aligned = df_vae[df_vae['epoch'].isin(common_epochs)].set_index('epoch').sort_index()
            diff = vae_aligned['val_MSE_rgb'] - ae_aligned['val_MSE_rgb']
            diff_pct = (diff / ae_aligned['val_MSE_rgb']) * 100
            
            ax.plot(diff_pct.index, diff_pct.values, linewidth=2.5, color='#F18F01')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.fill_between(diff_pct.index, 0, diff_pct.values, alpha=0.3, color='#F18F01')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('MSE Difference (%)', fontsize=12)
            ax.set_title('VAE vs AE: MSE Difference', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add final difference annotation
            final_diff = diff_pct.iloc[-1]
            ax.text(0.98, 0.98, f'Final diff: {final_diff:+.2f}%', 
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "vae_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_kld_analysis(all_data, output_dir):
    """Analyze KLD loss specifically (VAE experiment only)."""
    # Find VAE experiment
    vae_exp = None
    for exp_name, df in all_data.items():
        kld_cols = [c for c in df.columns if 'KLD' in c.upper()]
        if kld_cols:
            vae_exp = exp_name
            break
    
    if not vae_exp:
        print("  No KLD loss found in any experiment, skipping KLD analysis")
        return
    
    df_vae = all_data[vae_exp]
    kld_cols = [c for c in df_vae.columns if 'KLD' in c.upper()]
    if not kld_cols:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Phase 1.2: KLD Loss Analysis ({vae_exp})', fontsize=16, fontweight='bold')
    
    kld_col = kld_cols[0]  # Prefer val_KLD
    
    # Plot 1: KLD Loss Curve
    ax = axes[0]
    ax.plot(df_vae['epoch'], df_vae[kld_col], linewidth=2.5, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KLD Loss', fontsize=12)
    ax.set_title('KLD Loss Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference zones
    final_kld = df_vae[kld_col].iloc[-1]
    if final_kld < 0.0001:
        zone = "Very Low (Potential Collapse)"
        zone_color = 'red'
    elif final_kld < 0.001:
        zone = "Low (Good)"
        zone_color = 'orange'
    elif final_kld < 0.01:
        zone = "Medium (Good)"
        zone_color = 'green'
    elif final_kld < 0.1:
        zone = "High (Acceptable)"
        zone_color = 'orange'
    else:
        zone = "Very High (May Degrade Quality)"
        zone_color = 'red'
    
    ax.axhspan(0, 0.001, alpha=0.1, color='green', label='Good Range')
    ax.axhspan(0.001, 0.01, alpha=0.1, color='orange', label='Acceptable Range')
    ax.axhspan(0.01, 0.1, alpha=0.1, color='red', label='High (Warning)')
    
    ax.text(0.02, 0.98, f'Final KLD: {final_kld:.6f}\nZone: {zone}', 
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
    
    # Plot 2: KLD Statistics
    ax = axes[1]
    stats = {
        'Initial': df_vae[kld_col].iloc[0],
        'Final': df_vae[kld_col].iloc[-1],
        'Mean': df_vae[kld_col].mean(),
        'Min': df_vae[kld_col].min(),
        'Max': df_vae[kld_col].max(),
        'Std': df_vae[kld_col].std()
    }
    
    bars = ax.bar(stats.keys(), stats.values(), color='#A23B72', alpha=0.7)
    ax.set_ylabel('KLD Loss Value', fontsize=12)
    ax.set_title('KLD Loss Statistics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for bar, (key, val) in zip(bars, stats.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / "kld_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_side_by_side_metrics(all_data, output_dir):
    """Create side-by-side comparison table of key metrics."""
    if len(all_data) != 2:
        return
    
    exp_names = sorted(all_data.keys())
    metrics_table = []
    
    for exp_name in exp_names:
        df = all_data[exp_name]
        final = df.iloc[-1]
        
        metrics_table.append({
            'Experiment': exp_name.replace('phase1_2_AE_', ''),
            'Final Val Loss': final.get('val_loss', 'N/A'),
            'Final Val MSE': final.get('val_MSE_rgb', 'N/A'),
            'Final Train Loss': final.get('train_loss', 'N/A'),
            'Final Train MSE': final.get('train_MSE_rgb', 'N/A'),
            'KLD Loss': final.get('val_KLD', final.get('train_KLD', 'N/A')),
            'Total Epochs': len(df),
            'Best Epoch': df.loc[df['val_loss'].idxmin(), 'epoch'] if 'val_loss' in df.columns else 'N/A'
        })
    
    df_table = pd.DataFrame(metrics_table)
    
    # Save as CSV
    csv_path = output_dir / "metrics_comparison_table.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(df_table.columns)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Phase 1.2: Metrics Comparison Table', fontsize=14, fontweight='bold', pad=20)
    output_path = output_dir / "metrics_comparison_table.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()

