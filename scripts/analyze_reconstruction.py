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
    """Load super-category colors from taxonomy file."""
    import json
    
    if not taxonomy_path or not Path(taxonomy_path).exists():
        print("WARNING: No taxonomy file provided or file not found")
        return {}
    
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)
    
    # Only use super-categories (id2super), not fine-grained categories
    colors_dict = {}
    id2super = taxonomy.get('id2super', {})
    id2color = taxonomy.get('id2color', {})
    
    for cat_id in id2super.keys():
        if cat_id in id2color:
            cat_name = id2super[cat_id]
            colors_dict[cat_name] = np.array(id2color[cat_id]) / 255.0
    
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


def plot_architecture_family_comparison(family_name, models_data, sample_image_name, output_dir):
    """
    Create a single comprehensive figure comparing all models in an architecture family.
    Shows: 1 input + N reconstructions + N difference heatmaps for one test image.
    
    Args:
        family_name: 'vanilla', 'medium', or 'deep'
        models_data: list of model data dicts filtered to this family
        sample_image_name: name of the test image to visualize
        output_dir: where to save the figure
    """
    if not models_data:
        print(f"    No models found for family: {family_name}")
        return
    
    # Get the input image (same for all models)
    input_img = None
    reconstructions = []
    model_names = []
    
    for model_data in models_data:
        if sample_image_name in model_data['images']:
            result = model_data['images'][sample_image_name]
            if input_img is None:
                input_img = result['original']
            reconstructions.append(result['reconstruction'])
            model_names.append(model_data['name'])
    
    if input_img is None or len(reconstructions) == 0:
        print(f"    No data available for {family_name} / {sample_image_name}")
        return
    
    n_models = len(reconstructions)
    
    # Create figure: 1 input + n_models outputs + n_models diffs
    # Layout: top row = input (spans multiple columns) + outputs, bottom row = diffs
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, n_models + 1, hspace=0.3, wspace=0.2)
    
    fig.suptitle(f'Architecture Family: {family_name.upper()} | Image: {sample_image_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Convert input to numpy
    orig_np = input_img.permute(1, 2, 0).numpy()
    
    # Top-left: Input image (spans first column, both rows)
    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(orig_np)
    ax_input.set_title('INPUT IMAGE', fontsize=12, fontweight='bold', pad=10)
    ax_input.axis('off')
    
    # Top row: Reconstructions
    for i, (recon, model_name) in enumerate(zip(reconstructions, model_names)):
        ax = fig.add_subplot(gs[0, i + 1])
        recon_np = recon.permute(1, 2, 0).numpy()
        ax.imshow(recon_np)
        
        # Calculate MAE for this reconstruction
        diff = np.abs(orig_np - recon_np)
        mae = np.mean(diff)
        
        # Shortened model name for display
        short_name = model_name.replace(f'_{family_name}', '').replace('ch_', 'ch\n')
        ax.set_title(f'{short_name}\nMAE: {mae:.4f}', fontsize=9)
        ax.axis('off')
    
    # Bottom row: Difference heatmaps
    for i, (recon, model_name) in enumerate(zip(reconstructions, model_names)):
        ax = fig.add_subplot(gs[1, i + 1])
        recon_np = recon.permute(1, 2, 0).numpy()
        diff = np.abs(orig_np - recon_np)
        overall_diff = diff.mean(axis=2)
        
        im = ax.imshow(overall_diff, cmap='hot', vmin=0, vmax=0.5)
        ax.set_title('Error Heatmap', fontsize=9)
        ax.axis('off')
        
        # Add colorbar to the rightmost heatmap
        if i == len(reconstructions) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Absolute Error', rotation=270, labelpad=15, fontsize=9)
    
    plt.tight_layout()
    
    # Save
    safe_family = family_name.replace('/', '_')
    safe_img = sample_image_name.replace('/', '_')
    output_file_png = output_dir / f'family_comparison_{safe_family}_{safe_img}.png'
    output_file_svg = output_dir / f'family_comparison_{safe_family}_{safe_img}.svg'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print(f"    Saved: {output_file_png}")
    plt.close()


def generate_family_comparison_plots(all_models_data, output_dir):
    """
    Generate one comprehensive plot per architecture family.
    Uses the first test image that all models have processed.
    """
    print("  Creating architecture family comparison plots...")
    
    # Group models by architecture family
    families = {}
    for model_data in all_models_data:
        arch = model_data['arch']
        if arch not in families:
            families[arch] = []
        families[arch].append(model_data)
    
    # Find a common test image that all models have
    all_image_names = set()
    for model_data in all_models_data:
        if all_image_names:
            all_image_names &= set(model_data['images'].keys())
        else:
            all_image_names = set(model_data['images'].keys())
    
    if not all_image_names:
        print("    WARNING: No common test images across all models")
        return
    
    # Use the first common image
    sample_image = sorted(list(all_image_names))[0]
    print(f"    Using test image: {sample_image}")
    
    # Generate one plot per family
    for family_name in ['vanilla', 'medium', 'deep']:
        if family_name in families:
            # Sort models by name for consistent ordering
            family_models = sorted(families[family_name], key=lambda x: x['name'])
            plot_architecture_family_comparison(
                family_name, 
                family_models, 
                sample_image, 
                output_dir
            )
        else:
            print(f"    No models found for family: {family_name}")


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


def main():
    parser = argparse.ArgumentParser(description="Compare autoencoder models")
    parser.add_argument("--experiments_dir", type=str, required=True)
    parser.add_argument("--test_images", type=str, nargs='+', required=True)
    parser.add_argument("--output_dir", type=str, default="model_comparison")
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--taxonomy_path", type=str, default=None, help="Path to taxonomy.json file")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of images to process at once")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80, flush=True)
    print("MULTI-MODEL RECONSTRUCTION COMPARISON", flush=True)
    print("="*80, flush=True)
    print("Experiments directory: " + args.experiments_dir, flush=True)
    print("Test images: " + str(len(args.test_images)), flush=True)
    print("Batch size: " + str(args.batch_size), flush=True)
    print("Number of models to compare: " + str(args.num_models), flush=True)
    print("Device: " + args.device, flush=True)
    print("="*80, flush=True)
    print()
    
    # Load taxonomy colors
    print("[0/6] Loading taxonomy colors...", flush=True)
    taxonomy_colors = load_taxonomy_colors(args.taxonomy_path)
    print("  Loaded " + str(len(taxonomy_colors)) + " class colors", flush=True)
    print()
    
    # Find models
    print("[1/6] Finding models to compare...", flush=True)
    models_to_compare = find_models_to_compare(args.experiments_dir, args.num_models)
    print("  Found " + str(len(models_to_compare)) + " models", flush=True)
    for m in models_to_compare:
        print("    - " + m['name'] + " (loss: " + str(round(m['best_loss'], 6)) + ")", flush=True)
    print()
    
    # Get image paths
    print("[2/6] Preparing test images...", flush=True)
    image_paths = args.test_images
    print("  Total images to process: " + str(len(image_paths)), flush=True)
    visualization_image_path = image_paths[0]  # First image for visualization
    print()
    
    # Process each model for STATISTICS ONLY
    print("[3/6] Processing models (computing statistics)...", flush=True)
    all_models_data = []
    
    for model_idx, model_info in enumerate(models_to_compare):
        print(f"\n  [{model_idx+1}/{len(models_to_compare)}] Processing model: {model_info['name']}", flush=True)
        
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
                'model': model,  # Keep model for later visualization
                'config': model_info['config'],
                'checkpoint': model_info['checkpoint'],
                'images': {}
            }
            
            # Process images in batches - STATISTICS ONLY
            num_batches = (len(image_paths) - 1) // args.batch_size + 1
            
            for batch_idx in range(0, len(image_paths), args.batch_size):
                batch_end = min(batch_idx + args.batch_size, len(image_paths))
                batch_paths = image_paths[batch_idx:batch_end]
                batch_num = batch_idx // args.batch_size + 1
                
                print(f"    Batch {batch_num}/{num_batches} ({len(batch_paths)} images)...", end=' ', flush=True)
                
                # Load batch
                test_images, image_names = load_test_images(batch_paths)
                
                # Process each image in batch
                for i, img_name in enumerate(image_names):
                    img = test_images[i:i+1].to(args.device)
                    
                    with torch.no_grad():
                        recon = model(img)
                    
                    # Compute metrics
                    metrics = compute_reconstruction_metrics(
                        img.cpu(), recon.cpu()
                    )
                    
                    # Class-specific color metrics
                    class_metrics = compute_color_class_preservation(
                        test_images[i],
                        recon.cpu().squeeze(0),
                        taxonomy_colors
                    )
                    
                    # Store ONLY metrics (no tensors!)
                    model_results['images'][img_name] = {
                        'metrics': metrics,
                        'class_metrics': class_metrics
                    }
                
                print("Done", flush=True)
                
                # Free memory
                del test_images
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
            
            all_models_data.append(model_results)
            print("    Model completed", flush=True)
            
        except Exception as e:
            print(f"    ERROR: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    print("\n  Successfully processed " + str(len(all_models_data)) + " models", flush=True)
    print()
    
    # Now generate visualizations by running inference on ONE image
    print("[4/6] Generating visualization data (running inference on sample image)...", flush=True)
    
    # Load the visualization image once
    viz_images, viz_names = load_test_images([visualization_image_path])
    viz_img_tensor = viz_images[0]
    viz_img_name = viz_names[0]
    
    print(f"  Using image: {viz_img_name}", flush=True)
    
    # Run inference on this one image for each model
    for model_data in all_models_data:
        print(f"    Processing {model_data['name']}...", end=' ', flush=True)
        
        model = model_data['model']
        img = viz_img_tensor.unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            recon = model(img)
        
        # Store visualization data
        model_data['visualization'] = {
            'image_name': viz_img_name,
            'original': viz_img_tensor.clone(),
            'reconstruction': recon.cpu().squeeze(0).clone()
        }
        
        # Clean up model reference to free memory
        del model_data['model']
        
        print("Done", flush=True)
    
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    print()
    
    # Generate comparison plots
    print("[5/6] Generating comparison visualizations...", flush=True)
    
    # Architecture family comparisons (3 comprehensive plots)
    generate_family_comparison_plots_from_viz(all_models_data, output_dir)
    
    # Aggregate plots
    df = plot_aggregate_metrics(all_models_data, output_dir)
    
    # Class color preservation analysis
    plot_class_color_preservation(all_models_data, taxonomy_colors, output_dir)
    
    # Save metrics to CSV
    print("[6/6] Saving results...", flush=True)
    csv_file = output_dir / 'comparison_metrics.csv'
    df.to_csv(csv_file, index=False)
    print("  Saved metrics CSV: " + str(csv_file), flush=True)
    
    print("\n" + "="*80, flush=True)
    print("COMPARISON COMPLETE", flush=True)
    print("="*80, flush=True)
    print("\nResults saved to: " + str(output_dir), flush=True)
    print("\nGenerated files:", flush=True)
    print("  family_comparison_*.png - Per-family comprehensive comparisons (3 files)", flush=True)
    print("  aggregate_comparison.png - Overall metrics comparison", flush=True)
    print("  color_channel_analysis.png - Channel-specific analysis", flush=True)
    print("  model_ranking.png - Ranked models by quality", flush=True)
    print("  class_preservation_by_architecture.png - Class-specific color analysis by arch", flush=True)
    print("  class_preservation_by_channels.png - Class-specific color analysis by channels", flush=True)
    print("  comparison_metrics.csv - All metrics data", flush=True)
    print()


def generate_family_comparison_plots_from_viz(all_models_data, output_dir):
    """
    Generate one comprehensive plot per architecture family using pre-computed visualizations.
    """
    print("  Creating architecture family comparison plots...", flush=True)
    
    # Group models by architecture family
    families = {}
    for model_data in all_models_data:
        arch = model_data['arch']
        if arch not in families:
            families[arch] = []
        families[arch].append(model_data)
    
    # Get the visualization image name
    viz_img_name = all_models_data[0]['visualization']['image_name'] if all_models_data else None
    if not viz_img_name:
        print("    WARNING: No visualization data available")
        return
    
    print(f"    Using test image: {viz_img_name}", flush=True)
    
    # Generate one plot per family
    for family_name in ['vanilla', 'medium', 'deep']:
        if family_name in families:
            # Sort models by name for consistent ordering
            family_models = sorted(families[family_name], key=lambda x: x['name'])
            plot_architecture_family_comparison_from_viz(
                family_name, 
                family_models, 
                output_dir
            )
        else:
            print(f"    No models found for family: {family_name}", flush=True)


def plot_architecture_family_comparison_from_viz(family_name, models_data, output_dir):
    """
    Create a single comprehensive figure comparing all models in an architecture family.
    Uses pre-computed visualization data.
    """
    if not models_data:
        print(f"    No models found for family: {family_name}")
        return
    
    # Get the input image and reconstructions from visualization data
    input_img = models_data[0]['visualization']['original']
    img_name = models_data[0]['visualization']['image_name']
    
    reconstructions = []
    model_names = []
    
    for model_data in models_data:
        reconstructions.append(model_data['visualization']['reconstruction'])
        model_names.append(model_data['name'])
    
    n_models = len(reconstructions)
    
    # Create figure
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, n_models + 1, hspace=0.3, wspace=0.2)
    
    fig.suptitle(f'Architecture Family: {family_name.upper()} | Image: {img_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Convert input to numpy
    orig_np = input_img.permute(1, 2, 0).numpy()
    
    # Top-left: Input image (spans first column, both rows)
    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(orig_np)
    ax_input.set_title('INPUT IMAGE', fontsize=12, fontweight='bold', pad=10)
    ax_input.axis('off')
    
    # Top row: Reconstructions
    for i, (recon, model_name) in enumerate(zip(reconstructions, model_names)):
        ax = fig.add_subplot(gs[0, i + 1])
        recon_np = recon.permute(1, 2, 0).numpy()
        ax.imshow(recon_np)
        
        # Calculate MAE
        diff = np.abs(orig_np - recon_np)
        mae = np.mean(diff)
        
        # Shortened model name
        short_name = model_name.replace(f'_{family_name}', '').replace('ch_', 'ch\n')
        ax.set_title(f'{short_name}\nMAE: {mae:.4f}', fontsize=9)
        ax.axis('off')
    
    # Bottom row: Difference heatmaps
    for i, (recon, model_name) in enumerate(zip(reconstructions, model_names)):
        ax = fig.add_subplot(gs[1, i + 1])
        recon_np = recon.permute(1, 2, 0).numpy()
        diff = np.abs(orig_np - recon_np)
        overall_diff = diff.mean(axis=2)
        
        im = ax.imshow(overall_diff, cmap='hot', vmin=0, vmax=0.5)
        ax.set_title('Error Heatmap', fontsize=9)
        ax.axis('off')
        
        # Add colorbar to rightmost heatmap
        if i == len(reconstructions) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Absolute Error', rotation=270, labelpad=15, fontsize=9)
    
    plt.tight_layout()
    
    # Save
    safe_family = family_name.replace('/', '_')
    safe_img = img_name.replace('/', '_')
    output_file_png = output_dir / f'family_comparison_{safe_family}_{safe_img}.png'
    output_file_svg = output_dir / f'family_comparison_{safe_family}_{safe_img}.svg'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print(f"    Saved: {output_file_png}", flush=True)
    plt.close()

if __name__ == "__main__":
    main()