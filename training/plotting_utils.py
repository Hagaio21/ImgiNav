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
    
    # Find all loss component pairs (train_X, val_X)
    # Exclude non-loss columns
    exclude_cols = {'epoch', 'step', 'cfg_dropout_rate', 'learning_rate'}
    
    # Special handling for image quality metrics (plot together)
    image_metrics = ['fid', 'kid', 'lpips', 'clip_score']
    available_metrics = [m for m in image_metrics if f'val_{m}' in history_df.columns]
    
    if available_metrics:
        try:
            # Create a combined plot for all image quality metrics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Image Quality Metrics - {exp_name}', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            colors = {'fid': 'purple', 'kid': 'orange', 'lpips': 'green', 'clip_score': 'blue'}
            labels = {
                'fid': 'FID (lower is better)',
                'kid': 'KID (lower is better)',
                'lpips': 'LPIPS (lower is better)',
                'clip_score': 'CLIP Score (higher is better)'
            }
            
            for idx, metric in enumerate(image_metrics):
                if metric in available_metrics:
                    ax = axes[idx]
                    col_name = f'val_{metric}'
                    metric_data = history_df[[x_col, col_name]].dropna()
                    if len(metric_data) > 0:
                        metric_data = metric_data[metric_data[col_name] != float('inf')]
                        metric_data = metric_data[np.isfinite(metric_data[col_name])]
                        if len(metric_data) > 0:
                            ax.plot(metric_data[x_col], metric_data[col_name], 
                                   linewidth=2, marker='s', markersize=3, 
                                   color=colors.get(metric, 'black'), linestyle='-', alpha=0.8)
                            ax.set_xlabel(x_col.capitalize(), fontsize=10)
                            ax.set_ylabel(labels[metric], fontsize=10)
                            ax.grid(True, alpha=0.3)
                            ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
            
            # Hide unused subplots
            for idx in range(len(available_metrics), 4):
                axes[idx].axis('off')
            
            plt.tight_layout()
            metrics_plot_path = output_dir / f'{exp_name}_image_quality_metrics.png'
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        except Exception as e:
            warnings.warn(f"Failed to plot image quality metrics: {e}")
    
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
        return  # No loss data to plot
    
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
                    # Filter out inf and non-finite values
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
                    # Filter out inf and non-finite values
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
        
        # Save individual component plot
        safe_name = component['name'].replace('/', '_').replace('\\', '_').replace(' ', '_')
        component_plot_path = output_dir / f'{exp_name}_{safe_name}_curves.png'
        plt.savefig(component_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()



