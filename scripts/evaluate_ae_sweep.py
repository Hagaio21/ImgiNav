#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Skipping reconstruction analysis.")

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300


def parse_config_name(config_name):
    pattern = r'(?:ae_)?diff_(\d+)ch_(\d+)x(\d+)_(vanilla|skip|medium|deep)'
    match = re.search(pattern, config_name)
    
    if match:
        channels, base_w, base_h, arch = match.groups()
        if arch == 'skip':
            arch = 'medium'
        return {
            'latent_channels': int(channels),
            'latent_base': int(base_w),
            'latent_spatial': base_w + 'x' + base_h,
            'latent_dim': int(channels) * int(base_w) * int(base_h),
            'architecture': arch
        }
    return None


def load_experiment_results(experiments_dir):
    experiments_dir = Path(experiments_dir)
    all_results = []
    
    print("Scanning directory: " + str(experiments_dir))
    print("=" * 80)
    
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    print("Found " + str(len(exp_dirs)) + " experiment directories")
    
    for exp_dir in exp_dirs:
        metrics_file = exp_dir / "metrics.csv"
        config_file = exp_dir / "config_used.yml"
        
        if not metrics_file.exists():
            print("Skipping " + exp_dir.name + " (no metrics.csv)")
            continue
        
        metrics_df = pd.read_csv(metrics_file)
        
        config_params = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if 'model_cfg' in config and 'encoder' in config['model_cfg']:
                enc_cfg = config['model_cfg']['encoder']
                config_params = {
                    'latent_dim': enc_cfg.get('latent_dim'),
                    'latent_channels': enc_cfg.get('latent_channels'),
                    'latent_base': enc_cfg.get('latent_base'),
                    'image_size': enc_cfg.get('image_size'),
                    'num_layers': len(enc_cfg.get('layers', []))
                }
                
                if config_params['latent_base']:
                    base_str = str(config_params['latent_base'])
                    config_params['latent_spatial'] = base_str + 'x' + base_str
                
                dir_name = exp_dir.name.lower()
                if 'vanilla' in dir_name:
                    config_params['architecture'] = 'vanilla'
                elif 'skip' in dir_name or 'medium' in dir_name:
                    config_params['architecture'] = 'medium'
                elif 'deep' in dir_name:
                    config_params['architecture'] = 'deep'
                else:
                    if config_params['num_layers'] == 3:
                        config_params['architecture'] = 'vanilla'
                    elif config_params['num_layers'] == 4:
                        config_params['architecture'] = 'medium'
                    elif config_params['num_layers'] == 5:
                        config_params['architecture'] = 'deep'
        
        if not config_params or 'architecture' not in config_params:
            parsed = parse_config_name(exp_dir.name)
            if parsed:
                config_params.update(parsed)
        
        for col, val in config_params.items():
            metrics_df[col] = val
        
        metrics_df['experiment_name'] = exp_dir.name
        metrics_df['experiment_dir'] = str(exp_dir)
        
        all_results.append(metrics_df)
        print("Loaded " + exp_dir.name + ": " + str(len(metrics_df)) + " epochs")
    
    if not all_results:
        raise ValueError("No valid experiment results found!")
    
    df = pd.concat(all_results, ignore_index=True)
    
    print("=" * 80)
    print("Total experiments loaded: " + str(df['experiment_name'].nunique()))
    print("Total epochs: " + str(len(df)))
    print()
    
    return df


def plot_training_curves(df, output_dir):
    print("Generating training curves...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Training Loss Curves - Overview', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    for exp_name in df['experiment_name'].unique():
        exp_data = df[df['experiment_name'] == exp_name]
        ax.plot(exp_data['epoch'], exp_data['loss'], alpha=0.6, linewidth=1.5, label=exp_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('All Experiments')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    if 'architecture' in df.columns:
        for arch in df['architecture'].dropna().unique():
            arch_data = df[df['architecture'] == arch]
            for exp_name in arch_data['experiment_name'].unique():
                exp_data = arch_data[arch_data['experiment_name'] == exp_name]
                label_text = arch + ": " + exp_name
                ax.plot(exp_data['epoch'], exp_data['loss'], label=label_text, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('By Architecture')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file_png = output_dir / "training_curves_overview.png"
    output_file_svg = output_dir / "training_curves_overview.svg"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("  Saved: " + str(output_file_png))
    print("  Saved: " + str(output_file_svg))
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Training Loss Curves - Hyperparameters', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    if 'latent_channels' in df.columns:
        for channels in sorted(df['latent_channels'].dropna().unique()):
            channel_data = df[df['latent_channels'] == channels]
            mean_loss = channel_data.groupby('epoch')['loss'].mean()
            std_loss = channel_data.groupby('epoch')['loss'].std()
            epochs = mean_loss.index
            label_text = str(int(channels)) + ' channels'
            ax.plot(epochs, mean_loss, label=label_text, linewidth=2)
            ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('By Latent Channels (Mean +/- Std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    if 'latent_spatial' in df.columns:
        for spatial in sorted(df['latent_spatial'].dropna().unique()):
            spatial_data = df[df['latent_spatial'] == spatial]
            mean_loss = spatial_data.groupby('epoch')['loss'].mean()
            std_loss = spatial_data.groupby('epoch')['loss'].std()
            epochs = mean_loss.index
            ax.plot(epochs, mean_loss, label=str(spatial), linewidth=2)
            ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('By Latent Spatial Size (Mean +/- Std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file_png = output_dir / "training_curves_hyperparams.png"
    output_file_svg = output_dir / "training_curves_hyperparams.svg"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("  Saved: " + str(output_file_png))
    print("  Saved: " + str(output_file_svg))
    plt.close()


def plot_final_performance(df, output_dir):
    print("Generating final performance comparisons...")
    
    final_results = df.loc[df.groupby('experiment_name')['epoch'].idxmax()]
    best_results = df.loc[df.groupby('experiment_name')['loss'].idxmin()]
    best_results = best_results.rename(columns={'loss': 'best_loss', 'epoch': 'best_epoch'})
    
    comparison = final_results.merge(
        best_results[['experiment_name', 'best_loss', 'best_epoch']], 
        on='experiment_name', 
        suffixes=('', '_best')
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Final Performance - Best Loss Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    if 'architecture' in comparison.columns:
        arch_order = comparison.groupby('architecture')['best_loss'].median().sort_values().index
        sns.boxplot(data=comparison, x='architecture', y='best_loss', order=arch_order, ax=ax)
        sns.swarmplot(data=comparison, x='architecture', y='best_loss', order=arch_order, 
                     color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title('Best Loss by Architecture')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Best Loss')
    
    ax = axes[1]
    if 'latent_channels' in comparison.columns:
        comparison_sorted = comparison.sort_values('latent_channels')
        sns.boxplot(data=comparison_sorted, x='latent_channels', y='best_loss', ax=ax)
        sns.swarmplot(data=comparison_sorted, x='latent_channels', y='best_loss', 
                     color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title('Best Loss by Latent Channels')
        ax.set_xlabel('Latent Channels')
        ax.set_ylabel('Best Loss')
    
    plt.tight_layout()
    output_file_png = output_dir / "final_performance_loss.png"
    output_file_svg = output_dir / "final_performance_loss.svg"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("  Saved: " + str(output_file_png))
    print("  Saved: " + str(output_file_svg))
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Final Performance - Spatial Size & Convergence', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    if 'latent_spatial' in comparison.columns:
        spatial_order = comparison.groupby('latent_spatial')['best_loss'].median().sort_values().index
        sns.boxplot(data=comparison, x='latent_spatial', y='best_loss', order=spatial_order, ax=ax)
        sns.swarmplot(data=comparison, x='latent_spatial', y='best_loss', order=spatial_order,
                     color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title('Best Loss by Latent Spatial Size')
        ax.set_xlabel('Latent Spatial Size')
        ax.set_ylabel('Best Loss')
    
    ax = axes[1]
    if 'architecture' in comparison.columns:
        sns.boxplot(data=comparison, x='architecture', y='best_epoch', ax=ax)
        sns.swarmplot(data=comparison, x='architecture', y='best_epoch', 
                     color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title('Epoch of Best Performance')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Best Epoch')
    
    plt.tight_layout()
    output_file_png = output_dir / "final_performance_convergence.png"
    output_file_svg = output_dir / "final_performance_convergence.svg"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("  Saved: " + str(output_file_png))
    print("  Saved: " + str(output_file_svg))
    plt.close()


def plot_heatmaps(df, output_dir):
    print("Generating heatmaps...")
    
    best_results = df.loc[df.groupby('experiment_name')['loss'].idxmin()]
    
    if 'architecture' not in best_results.columns or 'latent_channels' not in best_results.columns:
        print("  Skipping heatmaps: missing required columns")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Performance Heatmaps (Best Loss)', fontsize=16, fontweight='bold')
    
    architectures = sorted(best_results['architecture'].unique())
    
    for idx, arch in enumerate(architectures):
        arch_data = best_results[best_results['architecture'] == arch]
        
        if 'latent_channels' in arch_data.columns and 'latent_base' in arch_data.columns:
            pivot = arch_data.pivot_table(
                values='loss',
                index='latent_base',
                columns='latent_channels',
                aggfunc='mean'
            )
            
            ax = axes[idx]
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd_r', 
                       cbar_kws={'label': 'Best Loss'}, ax=ax)
            ax.set_title('Architecture: ' + arch.capitalize())
            ax.set_xlabel('Latent Channels')
            ax.set_ylabel('Latent Base Size')
    
    plt.tight_layout()
    output_file_png = output_dir / "performance_heatmaps.png"
    output_file_svg = output_dir / "performance_heatmaps.svg"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("  Saved: " + str(output_file_png))
    print("  Saved: " + str(output_file_svg))
    plt.close()


def plot_convergence_analysis(df, output_dir):
    print("Generating convergence analysis...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Convergence Analysis - Variance & Stability', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    if 'architecture' in df.columns:
        for arch in df['architecture'].dropna().unique():
            arch_data = df[df['architecture'] == arch]
            variance = arch_data.groupby('epoch')['loss'].var()
            ax.plot(variance.index, variance.values, label=arch.capitalize(), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Variance')
    ax.set_title('Loss Variance Across Configurations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    if 'std_loss' in df.columns:
        final_stability = df.loc[df.groupby('experiment_name')['epoch'].idxmax()]
        if 'architecture' in final_stability.columns:
            sns.boxplot(data=final_stability, x='architecture', y='std_loss', ax=ax)
            ax.set_title('Training Stability (Final Epoch Std)')
            ax.set_xlabel('Architecture')
            ax.set_ylabel('Batch Loss Std Dev')
    
    plt.tight_layout()
    output_file_png = output_dir / "convergence_variance.png"
    output_file_svg = output_dir / "convergence_variance.svg"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("  Saved: " + str(output_file_png))
    print("  Saved: " + str(output_file_svg))
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Convergence Analysis - Speed & Improvement', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    convergence_epochs = []
    for exp_name in df['experiment_name'].unique():
        exp_data = df[df['experiment_name'] == exp_name].sort_values('epoch')
        best_loss = exp_data['loss'].min()
        threshold = best_loss * 1.05
        
        converged = exp_data[exp_data['loss'] <= threshold]
        if not converged.empty:
            conv_epoch = converged.iloc[0]['epoch']
            convergence_epochs.append({
                'experiment_name': exp_name,
                'convergence_epoch': conv_epoch,
                'architecture': exp_data.iloc[0].get('architecture'),
                'latent_channels': exp_data.iloc[0].get('latent_channels')
            })
    
    if convergence_epochs:
        conv_df = pd.DataFrame(convergence_epochs)
        if 'architecture' in conv_df.columns:
            sns.boxplot(data=conv_df, x='architecture', y='convergence_epoch', ax=ax)
            ax.set_title('Convergence Speed (Epochs to 95% of Best)')
            ax.set_xlabel('Architecture')
            ax.set_ylabel('Epochs')
    
    ax = axes[1]
    improvement_rates = []
    for exp_name in df['experiment_name'].unique():
        exp_data = df[df['experiment_name'] == exp_name].sort_values('epoch')
        if len(exp_data) > 1:
            initial_loss = exp_data.iloc[0]['loss']
            final_loss = exp_data.iloc[-1]['loss']
            epochs = exp_data.iloc[-1]['epoch']
            rate = (initial_loss - final_loss) / epochs
            improvement_rates.append({
                'experiment_name': exp_name,
                'improvement_rate': rate,
                'architecture': exp_data.iloc[0].get('architecture')
            })
    
    if improvement_rates:
        imp_df = pd.DataFrame(improvement_rates)
        if 'architecture' in imp_df.columns:
            sns.boxplot(data=imp_df, x='architecture', y='improvement_rate', ax=ax)
            ax.set_title('Average Improvement Rate')
            ax.set_xlabel('Architecture')
            ax.set_ylabel('Loss Decrease per Epoch')
    
    plt.tight_layout()
    output_file_png = output_dir / "convergence_speed.png"
    output_file_svg = output_dir / "convergence_speed.svg"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("  Saved: " + str(output_file_png))
    print("  Saved: " + str(output_file_svg))
    plt.close()


def generate_summary_table(df, output_dir):
    print("Generating summary table...")
    
    best_results = df.loc[df.groupby('experiment_name')['loss'].idxmin()]
    final_results = df.loc[df.groupby('experiment_name')['epoch'].idxmax()]
    
    summary_data = []
    for exp_name in df['experiment_name'].unique():
        exp_data = df[df['experiment_name'] == exp_name]
        best_row = best_results[best_results['experiment_name'] == exp_name].iloc[0]
        final_row = final_results[final_results['experiment_name'] == exp_name].iloc[0]
        
        summary_data.append({
            'Experiment': exp_name,
            'Architecture': best_row.get('architecture', 'N/A'),
            'Latent Channels': best_row.get('latent_channels', 'N/A'),
            'Latent Spatial': best_row.get('latent_spatial', 'N/A'),
            'Latent Dim': best_row.get('latent_dim', 'N/A'),
            'Best Loss': best_row['loss'],
            'Best Epoch': best_row['epoch'],
            'Final Loss': final_row['loss'],
            'Total Epochs': final_row['epoch'],
            'Improvement': best_row['loss'] - final_row['loss'],
            'Experiment Dir': best_row.get('experiment_dir', '')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Best Loss')
    
    output_file = output_dir / "summary_table.csv"
    summary_df.to_csv(output_file, index=False)
    print("  Saved: " + str(output_file))
    
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (by Best Loss):")
    print("="*80)
    print(summary_df.head(10).to_string(index=False))
    print()
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate autoencoder sweep experiments")
    parser.add_argument("--experiments_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--test_image", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("AUTOENCODER SWEEP EVALUATION")
    print("="*80)
    print("Experiments directory: " + args.experiments_dir)
    print("Output directory: " + str(output_dir))
    if args.test_image:
        print("Test image: " + args.test_image)
        print("Device: " + args.device)
    print()
    
    df = load_experiment_results(args.experiments_dir)
    
    print("\nGenerating visualizations...")
    print("="*80)
    
    plot_training_curves(df, output_dir)
    plot_final_performance(df, output_dir)
    plot_heatmaps(df, output_dir)
    plot_convergence_analysis(df, output_dir)
    summary_df = generate_summary_table(df, output_dir)
    
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nAll results saved to: " + str(output_dir))
    print("\nGenerated files:")
    print("  training_curves_overview.png / .svg")
    print("  training_curves_hyperparams.png / .svg")
    print("  final_performance_loss.png / .svg")
    print("  final_performance_convergence.png / .svg")
    print("  performance_heatmaps.png / .svg")
    print("  convergence_variance.png / .svg")
    print("  convergence_speed.png / .svg")
    print("  summary_table.csv")
    print()


if __name__ == "__main__":
    main()