#!/usr/bin/env python3
"""
Multi-Model Autoencoder Reconstruction Analysis
Compare reconstruction quality across multiple trained models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300


def parse_model_name(exp_dir_name):
    """Extract model info from directory name."""
    import re
    pattern = r'(?:ae_)?diff_(\d+)ch_(\d+)x(\d+)_(vanilla|skip|medium|deep)'
    match = re.search(pattern, exp_dir_name.lower())
    
    if match:
        channels, base, _, arch = match.groups()
        if arch == 'skip':
            arch = 'medium'
        return {
            'channels': int(channels),
            'base': int(base),
            'arch': arch,
            'name': str(channels) + 'ch_' + str(base) + 'x' + str(base) + '_' + arch
        }
    return None


def load_autoencoder_model(config_path, checkpoint_path, device='cpu'):
    """Load an autoencoder model."""
    sys.path.append(str(Path(__file__).parent.parent / "modules"))
    from autoencoder import AutoEncoder
    
    try:
        # Load config file
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Debug: print config structure
        print("      Config keys: " + str(list(full_config.keys())))
        
        # Extract model config if wrapped
        if 'model_cfg' in full_config:
            model_config = full_config['model_cfg']
            print("      Extracted model_cfg, keys: " + str(list(model_config.keys())))
        else:
            model_config = full_config
            print("      Using full config directly, keys: " + str(list(model_config.keys())))
        
        # Verify structure
        if 'encoder' not in model_config:
            print("      ERROR: No 'encoder' key found!")
            print("      Available keys: " + str(list(model_config.keys())))
            return None
        
        if 'decoder' not in model_config:
            print("      ERROR: No 'decoder' key found!")
            print("      Available keys: " + str(list(model_config.keys())))
            return None
        
        print("      Config structure verified, creating model...")
        
        # Pass the extracted config dict directly
        ae = AutoEncoder.from_config(model_config)
        ae.load_state_dict(torch.load(checkpoint_path, map_location=device))
        ae.to(device)
        ae.eval()
        return ae
    except Exception as e:
        print("      ERROR: " + str(e))
        import traceback
        traceback.print_exc()
        return None


def load_test_images(image_paths, image_size=512):
    """Load test images."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    images = []
    names = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
        names.append(Path(img_path).stem)
    
    return torch.stack(images), names


def load_taxonomy_colors(taxonomy_path=None):
    """Load category colors from taxonomy file."""
    import json
    
    if taxonomy_path and Path(taxonomy_path).exists():
        with open(taxonomy_path, 'r') as f:
            taxonomy = json.load(f)
    else:
        # Fallback: define key colors manually
        taxonomy = {
            'id2color': {
                '2053': [200, 200, 200],  # wall
                '1001': [228, 26, 28],     # Bed
                '1002': [55, 126, 184],    # Cabinet/Shelf/Desk
                '1003': [77, 175, 74],     # Chair
                '1004': [152, 78, 163],    # Lighting
                '1007': [255, 255, 51],    # Sofa
                '1009': [166, 86, 40],     # Table
            },
            'id2category': {
                '2053': 'wall',
                '1001': 'Bed',
                '1002': 'Cabinet/Shelf/Desk',
                '1003': 'Chair',
                '1004': 'Lighting',
                '1007': 'Sofa',
                '1009': 'Table',
            }
        }
    
    # Convert to numpy arrays and normalize to 0-1
    colors_dict = {}
    for cat_id, rgb in taxonomy['id2color'].items():
        if cat_id in taxonomy.get('id2category', {}) or cat_id in taxonomy.get('id2super', {}):
            cat_name = taxonomy.get('id2category', {}).get(cat_id) or taxonomy.get('id2super', {}).get(cat_id)
            if cat_name:
                colors_dict[cat_name] = np.array(rgb) / 255.0
    
    return colors_dict


def compute_color_class_preservation(original, reconstruction, target_colors):
    """
    Compute how well specific class colors are preserved.
    Returns per-class color preservation metrics.
    """
    orig_np = original.numpy().transpose(1, 2, 0)  # H x W x 3
    recon_np = reconstruction.numpy().transpose(1, 2, 0)
    
    metrics = {}
    
    for class_name, target_rgb in target_colors.items():
        # Find pixels close to this target color
        color_diff = np.linalg.norm(orig_np - target_rgb, axis=2)
        mask = color_diff < 0.15  # Pixels within threshold of target color
        
        if mask.sum() == 0:
            continue
        
        # Get original and reconstructed pixels for this class
        orig_pixels = orig_np[mask]
        recon_pixels = recon_np[mask]
        
        # Compute metrics for this class
        color_error = np.mean(np.abs(orig_pixels - recon_pixels), axis=0)
        color_mse = np.mean((orig_pixels - recon_pixels) ** 2, axis=0)
        
        # How well is the target color maintained?
        orig_distance = np.mean(np.linalg.norm(orig_pixels - target_rgb, axis=1))
        recon_distance = np.mean(np.linalg.norm(recon_pixels - target_rgb, axis=1))
        
        metrics[class_name] = {
            'pixel_count': int(mask.sum()),
            'mae_r': float(color_error[0]),
            'mae_g': float(color_error[1]),
            'mae_b': float(color_error[2]),
            'mae_overall': float(color_error.mean()),
            'orig_color_distance': float(orig_distance),
            'recon_color_distance': float(recon_distance),
            'color_drift': float(recon_distance - orig_distance)
        }
    
    return metrics


def compute_reconstruction_metrics(original, reconstruction):
    """Compute comprehensive reconstruction metrics."""
    diff = torch.abs(original - reconstruction)
    
    metrics = {
        'overall_mae': diff.mean().item(),
        'overall_mse': ((original - reconstruction) ** 2).mean().item(),
    }
    
    # Per-channel metrics
    for i, channel in enumerate(['red', 'green', 'blue']):
        metrics[channel + '_mae'] = diff[:, i].mean().item()
        metrics[channel + '_mse'] = ((original[:, i] - reconstruction[:, i]) ** 2).mean().item()
        
        # Correlation
        orig_flat = original[:, i].flatten()
        recon_flat = reconstruction[:, i].flatten()
        orig_mean = orig_flat.mean()
        recon_mean = recon_flat.mean()
        
        numerator = ((orig_flat - orig_mean) * (recon_flat - recon_mean)).sum()
        denominator = torch.sqrt(((orig_flat - orig_mean) ** 2).sum() * 
                                 ((recon_flat - recon_mean) ** 2).sum())
        
        metrics[channel + '_corr'] = (numerator / denominator).item()
    
    return metrics
    """Compute comprehensive reconstruction metrics."""
    diff = torch.abs(original - reconstruction)
    
    metrics = {
        'overall_mae': diff.mean().item(),
        'overall_mse': ((original - reconstruction) ** 2).mean().item(),
    }
    
    # Per-channel metrics
    for i, channel in enumerate(['red', 'green', 'blue']):
        metrics[channel + '_mae'] = diff[:, i].mean().item()
        metrics[channel + '_mse'] = ((original[:, i] - reconstruction[:, i]) ** 2).mean().item()
        
        # Correlation
        orig_flat = original[:, i].flatten()
        recon_flat = reconstruction[:, i].flatten()
        orig_mean = orig_flat.mean()
        recon_mean = recon_flat.mean()
        
        numerator = ((orig_flat - orig_mean) * (recon_flat - recon_mean)).sum()
        denominator = torch.sqrt(((orig_flat - orig_mean) ** 2).sum() * 
                                 ((recon_flat - recon_mean) ** 2).sum())
        
        metrics[channel + '_corr'] = (numerator / denominator).item()
    
    return metrics


def find_models_to_compare(experiments_dir, num_models=10):
    """Find top N models based on training loss."""
    experiments_dir = Path(experiments_dir)
    
    model_info = []
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        config_file = exp_dir / "config_used.yml"
        checkpoint_file = exp_dir / "best.pt"
        metrics_file = exp_dir / "metrics.csv"
        
        if not (config_file.exists() and checkpoint_file.exists() and metrics_file.exists()):
            continue
        
        # Get best loss
        metrics_df = pd.read_csv(metrics_file)
        best_loss = metrics_df['loss'].min()
        
        # Parse model name
        model_params = parse_model_name(exp_dir.name)
        if model_params is None:
            continue
        
        model_info.append({
            'exp_dir': exp_dir,
            'config': config_file,
            'checkpoint': checkpoint_file,
            'best_loss': best_loss,
            'name': model_params['name'],
            'arch': model_params['arch'],
            'channels': model_params['channels'],
            'base': model_params['base']
        })
    
    # Sort by best loss and take top N
    model_info.sort(key=lambda x: x['best_loss'])
    return model_info[:num_models]


def plot_single_model_reconstruction(model_name, original, reconstruction, img_name, output_dir):
    """Plot input, output, and difference heatmap for a single model."""
    
    orig_np = original.permute(1, 2, 0).numpy()
    recon_np = reconstruction.permute(1, 2, 0).numpy()
    diff = np.abs(orig_np - recon_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model: ' + model_name + ' | Image: ' + img_name, fontsize=14, fontweight='bold')
    
    # Input
    axes[0].imshow(orig_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Output
    axes[1].imshow(recon_np)
    mae = np.mean(diff)
    axes[1].set_title('Reconstruction (MAE: ' + str(round(mae, 4)) + ')')
    axes[1].axis('off')
    
    # Difference heatmap
    overall_diff = diff.mean(axis=2)
    im = axes[2].imshow(overall_diff, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Error Heatmap')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save with model name and image name
    safe_model_name = model_name.replace('/', '_')
    safe_img_name = img_name.replace('/', '_')
    output_file_png = output_dir / ('recon_' + safe_model_name + '_' + safe_img_name + '.png')
    output_file_svg = output_dir / ('recon_' + safe_model_name + '_' + safe_img_name + '.svg')
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    plt.close()


def plot_class_color_preservation(all_models_data, taxonomy_colors, output_dir):
    """Plot class color preservation: one plot for latent sizes, one for architectures."""
    print("  Creating class color preservation analysis...")
    
    # Aggregate metrics across all models and images
    class_metrics = {}
    
    for model_data in all_models_data:
        model_name = model_data['name']
        arch = model_data['arch']
        channels = model_data['channels']
        base = model_data['base']
        
        for img_name, result in model_data['images'].items():
            if 'class_metrics' not in result:
                continue
            
            for class_name, metrics in result['class_metrics'].items():
                if class_name not in class_metrics:
                    class_metrics[class_name] = []
                
                class_metrics[class_name].append({
                    'model': model_name,
                    'arch': arch,
                    'channels': channels,
                    'base': base,
                    'mae': metrics['mae_overall'],
                    'drift': metrics['color_drift']
                })
    
    if not class_metrics:
        print("    No class metrics available")
        return
    
    # Convert to DataFrame for easier plotting
    all_metrics = []
    for class_name, metrics_list in class_metrics.items():
        for m in metrics_list:
            all_metrics.append({
                'class': class_name,
                'model': m['model'],
                'arch': m['arch'],
                'channels': m['channels'],
                'base': m['base'],
                'mae': m['mae'],
                'drift': m['drift']
            })
    
    df = pd.DataFrame(all_metrics)
    
    if df.empty:
        print("    No data for class color preservation")
        return
    
    class_names = sorted(df['class'].unique())
    
    # Plot 1: Comparison by architecture (8 channels - middle ground)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Class Color Preservation by Architecture (8 Channels)', fontsize=14, fontweight='bold')
    
    # Focus on 8 channel models for fair comparison
    ch8_df = df[df['channels'] == 8]
    
    if not ch8_df.empty:
        ax = axes[0]
        arch_list = sorted(ch8_df['arch'].unique())
        x = np.arange(len(class_names))
        width = 0.8 / max(len(arch_list), 1)
        
        for i, arch in enumerate(arch_list):
            arch_data = ch8_df[ch8_df['arch'] == arch]
            mae_by_class = [arch_data[arch_data['class'] == c]['mae'].mean() 
                           if not arch_data[arch_data['class'] == c].empty else 0 
                           for c in class_names]
            ax.bar(x + i * width, mae_by_class, width, 
                  label=arch.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Color Error by Architecture (8 Channels)')
        ax.set_xticks(x + width * (len(arch_list) - 1) / 2)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[1]
        for i, arch in enumerate(arch_list):
            arch_data = ch8_df[ch8_df['arch'] == arch]
            drift_by_class = [arch_data[arch_data['class'] == c]['drift'].mean() 
                             if not arch_data[arch_data['class'] == c].empty else 0 
                             for c in class_names]
            ax.plot(x, drift_by_class, marker='o', 
                   label=arch.capitalize(), linewidth=2)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Color Drift')
        ax.set_title('Color Drift by Architecture')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'No 8-channel data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_file_png = output_dir / 'class_preservation_by_architecture.png'
    output_file_svg = output_dir / 'class_preservation_by_architecture.svg'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("    Saved: " + str(output_file_png))
    plt.close()
    
    # Plot 2: Comparison by latent channels (base=32)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Class Color Preservation by Latent Channels (Base=32)', fontsize=14, fontweight='bold')
    
    # Focus on base=32 models for fair comparison
    base32_df = df[df['base'] == 32]
    
    if not base32_df.empty:
        ax = axes[0]
        channels_list = sorted(base32_df['channels'].unique())
        x = np.arange(len(class_names))
        width = 0.8 / max(len(channels_list), 1)
        
        for i, ch in enumerate(channels_list):
            ch_data = base32_df[base32_df['channels'] == ch]
            mae_by_class = [ch_data[ch_data['class'] == c]['mae'].mean() 
                           if not ch_data[ch_data['class'] == c].empty else 0 
                           for c in class_names]
            ax.bar(x + i * width, mae_by_class, width, 
                  label=str(ch) + ' channels', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Color Error by Latent Channels (Base=32)')
        ax.set_xticks(x + width * (len(channels_list) - 1) / 2)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[1]
        for i, ch in enumerate(channels_list):
            ch_data = base32_df[base32_df['channels'] == ch]
            drift_by_class = [ch_data[ch_data['class'] == c]['drift'].mean() 
                             if not ch_data[ch_data['class'] == c].empty else 0 
                             for c in class_names]
            ax.plot(x, drift_by_class, marker='o', 
                   label=str(ch) + ' channels', linewidth=2)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Color Drift')
        ax.set_title('Color Drift by Latent Channels')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'No base=32 data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_file_png = output_dir / 'class_preservation_by_channels.png'
    output_file_svg = output_dir / 'class_preservation_by_channels.svg'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print("    Saved: " + str(output_file_png))
    plt.close()


def plot_aggregate_metrics(all_models_data, output_dir):
    """Plot aggregate metrics across all models and images."""
    print("  Creating aggregate comparison plots...")
    
    # Prepare data
    rows = []
    for model_data in all_models_data:
        for img_name, result in model_data['images'].items():
            row = {
                'model': model_data['name'],
                'architecture': model_data['arch'],
                'channels': model_data['channels'],
                'base': model_data['base'],
                'image': img_name
            }
            row.update(result['metrics'])
            rows.append(row)
    
    if not rows:
        print("  WARNING: No data to plot")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Check if we have the required columns
    if 'model' not in df.columns or 'overall_mae' not in df.columns:
        print("  ERROR: Missing required columns in dataframe")
        print("  Available columns: " + str(df.columns.tolist()))
        return df
    
    # Plot 1: Overall MAE and Architecture comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    
    ax = axes[0]
    model_mae = df.groupby('model')['overall_mae'].mean().sort_values()
    colors = ['red' if 'vanilla' in m else 'green' if 'medium' in m else 'blue' 
              for m in model_mae.index]
    model_mae.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
    ax.set_xlabel('Mean Absolute Error')
    ax.set_title('Overall Reconstruction Error by Model')
    ax.grid(True, alpha=0.3, axis='x')
    
    ax = axes[1]
    if 'architecture' in df.columns and df['architecture'].notna().any():
        arch_mae = df.groupby('architecture')['overall_mae'].agg(['mean', 'std'])
        arch_mae['mean'].plot(kind='bar', ax=ax, yerr=arch_mae['std'], 
                              capsize=5, alpha=0.7)
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Error by Architecture Type')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, 'Architecture data not available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_file = output_dir / 'aggregate_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("    Saved: " + str(output_file))
    plt.close()
    
    # Plot 2: Color channel comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Color Channel Analysis', fontsize=14, fontweight='bold')
    
    ax = axes[0]
    channel_cols = ['red_mae', 'green_mae', 'blue_mae']
    if all(col in df.columns for col in channel_cols):
        channel_means = df[channel_cols].mean()
        bars = ax.bar(['Red', 'Green', 'Blue'], channel_means, 
                      color=['red', 'green', 'blue'], alpha=0.7)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Average Error by Color Channel (all models)')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, channel_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    str(round(val, 5)), ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Channel data not available', 
                ha='center', va='center', transform=ax.transAxes)
    
    ax = axes[1]
    corr_cols = ['red_corr', 'green_corr', 'blue_corr']
    if all(col in df.columns for col in corr_cols):
        corr_means = df[corr_cols].mean()
        bars = ax.bar(['Red', 'Green', 'Blue'], corr_means,
                      color=['red', 'green', 'blue'], alpha=0.7)
        ax.set_ylabel('Correlation')
        ax.set_title('Color Preservation by Channel (all models)')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, corr_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    str(round(val, 4)), ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Correlation data not available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_file = output_dir / 'color_channel_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("    Saved: " + str(output_file))
    plt.close()
    
    # Plot 3: Detailed model ranking
    fig, ax = plt.subplots(1, 1, figsize=(14, max(8, len(df['model'].unique()) * 0.4)))
    
    # Calculate composite score (lower is better)
    model_scores = df.groupby('model').agg({
        'overall_mae': 'mean',
        'red_corr': 'mean',
        'green_corr': 'mean',
        'blue_corr': 'mean'
    })
    model_scores['avg_corr'] = model_scores[['red_corr', 'green_corr', 'blue_corr']].mean(axis=1)
    model_scores['composite_score'] = model_scores['overall_mae'] * (2 - model_scores['avg_corr'])
    model_scores = model_scores.sort_values('composite_score')
    
    y_pos = np.arange(len(model_scores))
    bars = ax.barh(y_pos, model_scores['composite_score'], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_scores.index)
    ax.set_xlabel('Composite Score (lower is better)')
    ax.set_title('Model Ranking by Reconstruction Quality')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Color bars by rank
    cmap = plt.cm.RdYlGn_r
    for i, bar in enumerate(bars):
        bar.set_color(cmap(i / len(bars)))
    
    plt.tight_layout()
    output_file = output_dir / 'model_ranking.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("    Saved: " + str(output_file))
    plt.close()
    
    return df
    
    # Plot 1: Overall MAE by model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    model_mae = df.groupby('model')['overall_mae'].mean().sort_values()
    colors = ['red' if 'vanilla' in m else 'green' if 'medium' in m else 'blue' 
              for m in model_mae.index]
    model_mae.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
    ax.set_xlabel('Mean Absolute Error')
    ax.set_title('Overall Reconstruction Error by Model')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: MAE by architecture
    ax = axes[0, 1]
    if 'architecture' in df.columns:
        arch_mae = df.groupby('architecture')['overall_mae'].agg(['mean', 'std'])
        arch_mae['mean'].plot(kind='bar', ax=ax, yerr=arch_mae['std'], 
                              capsize=5, alpha=0.7)
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Error by Architecture Type')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Plot 3: Color channel comparison
    ax = axes[1, 0]
    channel_cols = ['red_mae', 'green_mae', 'blue_mae']
    channel_means = df[channel_cols].mean()
    bars = ax.bar(['Red', 'Green', 'Blue'], channel_means, 
                  color=['red', 'green', 'blue'], alpha=0.7)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Average Error by Color Channel (all models)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, channel_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                str(round(val, 5)), ha='center', va='bottom')
    
    # Plot 4: Correlation by channel
    ax = axes[1, 1]
    corr_cols = ['red_corr', 'green_corr', 'blue_corr']
    corr_means = df[corr_cols].mean()
    bars = ax.bar(['Red', 'Green', 'Blue'], corr_means,
                  color=['red', 'green', 'blue'], alpha=0.7)
    ax.set_ylabel('Correlation')
    ax.set_title('Color Preservation by Channel (all models)')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, corr_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                str(round(val, 4)), ha='center', va='bottom')
    
    plt.tight_layout()
    output_file = output_dir / 'aggregate_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("    Saved: " + str(output_file))
    plt.close()
    
    # Plot 5: Detailed model ranking
    fig, ax = plt.subplots(1, 1, figsize=(14, max(8, len(df['model'].unique()) * 0.4)))
    
    # Calculate composite score (lower is better)
    model_scores = df.groupby('model').agg({
        'overall_mae': 'mean',
        'red_corr': 'mean',
        'green_corr': 'mean',
        'blue_corr': 'mean'
    })
    model_scores['avg_corr'] = model_scores[['red_corr', 'green_corr', 'blue_corr']].mean(axis=1)
    model_scores['composite_score'] = model_scores['overall_mae'] * (2 - model_scores['avg_corr'])
    model_scores = model_scores.sort_values('composite_score')
    
    y_pos = np.arange(len(model_scores))
    bars = ax.barh(y_pos, model_scores['composite_score'], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_scores.index)
    ax.set_xlabel('Composite Score (lower is better)')
    ax.set_title('Model Ranking by Reconstruction Quality')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Color bars by rank
    cmap = plt.cm.RdYlGn_r
    for i, bar in enumerate(bars):
        bar.set_color(cmap(i / len(bars)))
    
    plt.tight_layout()
    output_file = output_dir / 'model_ranking.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("    Saved: " + str(output_file))
    plt.close()
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare autoencoder models")
    parser.add_argument("--experiments_dir", type=str, required=True)
    parser.add_argument("--test_images", type=str, nargs='+', required=True)
    parser.add_argument("--output_dir", type=str, default="model_comparison")
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--taxonomy_path", type=str, default=None, help="Path to taxonomy.json file")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("MULTI-MODEL RECONSTRUCTION COMPARISON")
    print("="*80)
    print("Experiments directory: " + args.experiments_dir)
    print("Test images: " + str(len(args.test_images)))
    print("Number of models to compare: " + str(args.num_models))
    print("Device: " + args.device)
    print("="*80)
    print()
    
    # Load taxonomy colors
    print("[0/5] Loading taxonomy colors...")
    taxonomy_colors = load_taxonomy_colors(args.taxonomy_path)
    print("  Loaded " + str(len(taxonomy_colors)) + " class colors")
    print()
    
    # Find models
    print("[1/5] Finding models to compare...")
    models_to_compare = find_models_to_compare(args.experiments_dir, args.num_models)
    print("  Found " + str(len(models_to_compare)) + " models")
    for m in models_to_compare:
        print("    - " + m['name'] + " (loss: " + str(round(m['best_loss'], 6)) + ")")
    print()
    
    # Load test images
    print("[2/5] Loading test images...")
    test_images, image_names = load_test_images(args.test_images)
    print("  Loaded " + str(len(test_images)) + " images")
    print()
    
    # Process each model
    print("[3/5] Processing models...")
    all_models_data = []
    
    for model_info in models_to_compare:
        print("\n  Processing model: " + model_info['name'])
        
        try:
            model = load_autoencoder_model(
                str(model_info['config']),
                str(model_info['checkpoint']),
                args.device
            )
            
            if model is None:
                continue
            
            model_results = {
                'name': model_info['name'],
                'arch': model_info['arch'],
                'channels': model_info['channels'],
                'base': model_info['base'],
                'images': {}
            }
            
            # Process each test image
            for i, img_name in enumerate(image_names):
                img = test_images[i:i+1].to(args.device)
                
                with torch.no_grad():
                    recon = model(img)
                
                # Overall metrics
                metrics = compute_reconstruction_metrics(
                    img.cpu(), recon.cpu()
                )
                
                # Class-specific color metrics
                class_metrics = compute_color_class_preservation(
                    test_images[i],
                    recon.cpu().squeeze(0),
                    taxonomy_colors
                )
                
                model_results['images'][img_name] = {
                    'original': test_images[i],
                    'reconstruction': recon.cpu().squeeze(0),
                    'metrics': metrics,
                    'class_metrics': class_metrics
                }
            
            all_models_data.append(model_results)
            print("    Completed")
            
        except Exception as e:
            print("    ERROR: " + str(e))
            continue
    
    print("\n  Successfully processed " + str(len(all_models_data)) + " models")
    print()
    
    # Generate comparison plots
    print("[4/5] Generating comparison visualizations...")
    
    # Per-model, per-image reconstructions
    for model_data in all_models_data:
        for img_name, result in model_data['images'].items():
            plot_single_model_reconstruction(
                model_data['name'],
                result['original'],
                result['reconstruction'],
                img_name,
                output_dir
            )
    
    # Aggregate plots
    df = plot_aggregate_metrics(all_models_data, output_dir)
    
    # Class color preservation analysis
    plot_class_color_preservation(all_models_data, taxonomy_colors, output_dir)
    
    # Save metrics to CSV
    csv_file = output_dir / 'comparison_metrics.csv'
    df.to_csv(csv_file, index=False)
    print("  Saved metrics CSV: " + str(csv_file))
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nResults saved to: " + str(output_dir))
    print("\nGenerated files:")
    print("  comparison_*.png - Per-image model comparisons")
    print("  aggregate_comparison.png - Overall metrics comparison")
    print("  model_ranking.png - Ranked models by quality")
    print("  class_color_preservation.png - Class-specific color analysis")
    print("  comparison_metrics.csv - All metrics data")
    print()


if __name__ == "__main__":
    main()