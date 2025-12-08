"""
Plotting utilities for training metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import warnings


def plot_loss_curves(history_df, output_dir, exp_name="experiment"):
    """
    Simple unified function to plot train/val loss curves.
    Works for both VAE and diffusion training.
    Automatically detects and plots all loss component pairs (train_X, val_X) as individual plots.
    
    Args:
        history_df: DataFrame with training metrics (must have 'train_loss' and optionally 'val_loss')
        output_dir: Directory to save plots
        exp_name: Experiment name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine x-axis column
    x_col = "step" if "step" in history_df.columns else "epoch"
    
    # Exclude non-loss columns
    exclude_cols = {'epoch', 'step', 'cfg_dropout_rate', 'learning_rate', 'kid', 'lpips'}
    
    # Image quality metrics to plot separately
    image_metrics = {
        'kid': {'color': 'orange', 'label': 'KID (lower is better)', 'title': 'KID (Kernel Inception Distance)'},
        'lpips': {'color': 'green', 'label': 'LPIPS (lower is better)', 'title': 'LPIPS (Learned Perceptual Image Patch Similarity)'},
    }
    
    # Plot image quality metrics separately
    for metric, config in image_metrics.items():
        col_name = f'val_{metric}'
        if col_name in history_df.columns:
            try:
                metric_data = history_df[[x_col, col_name]].dropna()
                if len(metric_data) > 0:
                    metric_data = metric_data[metric_data[col_name] != float('inf')]
                    metric_data = metric_data[np.isfinite(metric_data[col_name])]
                    if len(metric_data) > 0:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                        fig.suptitle(f'{config["title"]} - {exp_name}', fontsize=16, fontweight='bold')
                        
                        ax.plot(metric_data[x_col], metric_data[col_name], 
                               linewidth=2, marker='s', markersize=4, 
                               color=config['color'], linestyle='-', alpha=0.8)
                        ax.set_xlabel(x_col.capitalize(), fontsize=12)
                        ax.set_ylabel(config['label'], fontsize=12)
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        metric_plot_path = output_dir / f'{exp_name}_{metric}_curve.png'
                        plt.savefig(metric_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
                        plt.close()
                        print(f"  Saved {metric.upper()} plot: {metric_plot_path}")
            except Exception as e:
                warnings.warn(f"Failed to plot {metric}: {e}")
    
    # Find all loss component pairs
    loss_components = []
    seen_components = set()
    
    for col in history_df.columns:
        if col.startswith('train_'):
            component_name = col.replace('train_', '')
            if component_name not in exclude_cols and component_name not in seen_components:
                seen_components.add(component_name)
                val_col = f'val_{component_name}'
                has_train = col in history_df.columns
                has_val = val_col in history_df.columns
                if has_train or has_val:
                    loss_components.append({
                        'name': component_name,
                        'train_col': col if has_train else None,
                        'val_col': val_col if has_val else None
                    })
    
    if not loss_components:
        return
    
    # Plot each component individually
    for component in loss_components:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        component_display_name = component['name'].replace('_', ' ').title()
        fig.suptitle(f'{component_display_name} - {exp_name}', fontsize=16, fontweight='bold')
        
        has_data = False
        
        # Plot train component
        if component['train_col']:
            try:
                train_data = history_df[[x_col, component['train_col']]].dropna()
                if len(train_data) > 0:
                    train_data = train_data[train_data[component['train_col']] != float('inf')]
                    train_data = train_data[np.isfinite(train_data[component['train_col']])]
                    if len(train_data) > 0:
                        ax.plot(train_data[x_col], train_data[component['train_col']], 
                               label='Train', linewidth=2, marker='o', markersize=3, color='blue', alpha=0.8)
                        has_data = True
            except Exception:
                pass
        
        # Plot val component
        if component['val_col']:
            try:
                val_data = history_df[[x_col, component['val_col']]].dropna()
                if len(val_data) > 0:
                    val_data = val_data[val_data[component['val_col']] != float('inf')]
                    val_data = val_data[np.isfinite(val_data[component['val_col']])]
                    if len(val_data) > 0:
                        ax.plot(val_data[x_col], val_data[component['val_col']], 
                               label='Val', linewidth=2, marker='s', markersize=3, color='red', linestyle='--', alpha=0.8)
                        has_data = True
            except Exception:
                pass
        
        if not has_data:
            plt.close()
            continue
        
        ax.set_xlabel(x_col.capitalize(), fontsize=12)
        ax.set_ylabel(component_display_name, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if values span multiple orders of magnitude
        if component['train_col'] and component['train_col'] in history_df.columns:
            try:
                train_values = pd.to_numeric(history_df[component['train_col']], errors='coerce').dropna()
                if len(train_values) > 1:
                    max_val = train_values.max()
                    min_val = train_values.min()
                    if isinstance(max_val, (int, float)) and isinstance(min_val, (int, float)):
                        if max_val > 0 and min_val > 0 and max_val / min_val > 10:
                            ax.set_yscale('log')
            except (TypeError, ValueError):
                pass
        
        plt.tight_layout()
        
        safe_name = component['name'].replace('/', '_').replace('\\', '_').replace(' ', '_')
        component_plot_path = output_dir / f'{exp_name}_{safe_name}_curves.png'
        plt.savefig(component_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()