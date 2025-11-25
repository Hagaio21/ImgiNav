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
    
    Args:
        history_df: DataFrame with training metrics (must have 'train_loss' and optionally 'val_loss')
        output_dir: Directory to save plots
        exp_name: Experiment name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine x-axis column
    x_col = "step" if "step" in history_df.columns else "epoch"
    
    # Check if loss columns exist
    has_train_loss = 'train_loss' in history_df.columns
    has_val_loss = 'val_loss' in history_df.columns
    
    if not has_train_loss:
        return  # No loss data to plot
    
    epochs = history_df[x_col].values
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Training and Validation Loss - {exp_name}', fontsize=16, fontweight='bold')
    
    # Plot training loss
    if has_train_loss:
        ax.plot(epochs, history_df['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3, color='blue', alpha=0.8)
    
    # Plot validation loss if available
    if has_val_loss:
        # Filter out inf values
        val_data = history_df[history_df['val_loss'] != float('inf')]
        if len(val_data) > 0:
            ax.plot(val_data[x_col], val_data['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3, color='red', linestyle='--', alpha=0.8)
    
    ax.set_xlabel(x_col.capitalize(), fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if loss values span multiple orders of magnitude
    if has_train_loss and len(history_df) > 1:
        try:
            max_loss = history_df['train_loss'].max()
            min_loss = history_df['train_loss'].min()
            if isinstance(max_loss, (int, float)) and isinstance(min_loss, (int, float)):
                if max_loss > 0 and min_loss > 0 and max_loss / min_loss > 10:
                    ax.set_yscale('log')
        except (TypeError, ValueError):
            pass  # Skip log scale if values are not numeric
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'{exp_name}_loss_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()



