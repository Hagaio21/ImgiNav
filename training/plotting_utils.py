"""
Plotting utilities for training metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np


def plot_discriminator_metrics(history_df, output_dir, iteration, exp_name="discriminator"):
    """
    Plot discriminator training metrics.
    
    Args:
        history_df: DataFrame with columns: epoch/step, train_loss, train_acc, val_loss, val_acc
        output_dir: Directory to save plots
        iteration: Iteration number
        exp_name: Experiment name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine x-axis column
    x_col = "step" if "step" in history_df.columns else "epoch"
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Discriminator Training Metrics - Iteration {iteration}', fontsize=16)
    
    # Loss plot
    ax = axes[0, 0]
    ax.plot(history_df[x_col], history_df['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax.plot(history_df[x_col], history_df['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax = axes[0, 1]
    ax.plot(history_df[x_col], history_df['train_acc'], label='Train Acc', marker='o', markersize=3)
    ax.plot(history_df[x_col], history_df['val_acc'], label='Val Acc', marker='s', markersize=3)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Accuracy')
    ax.set_title('Discriminator Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Loss difference (overfitting indicator)
    ax = axes[1, 0]
    if 'val_loss' in history_df.columns and 'train_loss' in history_df.columns:
        loss_diff = history_df['val_loss'] - history_df['train_loss']
        ax.plot(history_df[x_col], loss_diff, label='Val - Train Loss', color='red', marker='o', markersize=3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss Difference')
        ax.set_title('Overfitting Indicator (Val - Train Loss)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Accuracy difference
    ax = axes[1, 1]
    if 'val_acc' in history_df.columns and 'train_acc' in history_df.columns:
        acc_diff = history_df['train_acc'] - history_df['val_acc']
        ax.plot(history_df[x_col], acc_diff, label='Train - Val Acc', color='purple', marker='o', markersize=3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Accuracy Difference')
        ax.set_title('Overfitting Indicator (Train - Val Acc)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'learning_rate' in history_df.columns:
        # Add learning rate to a subplot if we have space, or create additional figure
        pass  # Will be handled in a separate plot if needed
    
    plt.tight_layout()
    plot_path = output_dir / f"{exp_name}_iter_{iteration}_discriminator_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved discriminator metrics plot to: {plot_path}")


def plot_diffusion_metrics(history_df, output_dir, iteration, exp_name="diffusion"):
    """
    Plot diffusion model training metrics.
    
    Args:
        history_df: DataFrame with training metrics
        output_dir: Directory to save plots
        iteration: Iteration number
        exp_name: Experiment name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine x-axis column
    x_col = "step" if "step" in history_df.columns else "epoch"
    
    # Get available metrics
    train_loss_cols = [col for col in history_df.columns if col.startswith('train_') and col != 'train_loss']
    val_loss_cols = [col for col in history_df.columns if col.startswith('val_') and col != 'val_loss']
    
    # Create subplots based on available metrics
    num_plots = 2 + len([c for c in train_loss_cols if 'noise' in c or 'discriminator' in c])
    ncols = 2
    nrows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Diffusion Model Training Metrics - Iteration {iteration}', fontsize=16)
    
    plot_idx = 0
    
    # Total loss
    ax = axes[plot_idx // ncols, plot_idx % ncols]
    if 'train_loss' in history_df.columns:
        ax.plot(history_df[x_col], history_df['train_loss'], label='Train Loss', marker='o', markersize=2)
    if 'val_loss' in history_df.columns:
        val_data = history_df[history_df['val_loss'] != float('inf')]
        if len(val_data) > 0:
            ax.plot(val_data[x_col], val_data['val_loss'], label='Val Loss', marker='s', markersize=2)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # Noise loss
    noise_train_col = next((c for c in train_loss_cols if 'noise' in c.lower()), None)
    noise_val_col = next((c for c in val_loss_cols if 'noise' in c.lower()), None)
    if noise_train_col or noise_val_col:
        ax = axes[plot_idx // ncols, plot_idx % ncols]
        if noise_train_col:
            ax.plot(history_df[x_col], history_df[noise_train_col], label='Train Noise Loss', marker='o', markersize=2)
        if noise_val_col:
            val_data = history_df[history_df[noise_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[noise_val_col], label='Val Noise Loss', marker='s', markersize=2)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Noise Prediction Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Discriminator loss
    disc_train_col = next((c for c in train_loss_cols if 'discriminator' in c.lower()), None)
    disc_val_col = next((c for c in val_loss_cols if 'discriminator' in c.lower()), None)
    if disc_train_col or disc_val_col:
        ax = axes[plot_idx // ncols, plot_idx % ncols]
        if disc_train_col:
            ax.plot(history_df[x_col], history_df[disc_train_col], label='Train Discriminator Loss', marker='o', markersize=2, color='orange')
        if disc_val_col:
            val_data = history_df[history_df[disc_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[disc_val_col], label='Val Discriminator Loss', marker='s', markersize=2, color='red')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Discriminator Adversarial Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Viability score
    viability_train_col = next((c for c in train_loss_cols if 'viability' in c.lower()), None)
    viability_val_col = next((c for c in val_loss_cols if 'viability' in c.lower()), None)
    if viability_train_col or viability_val_col:
        ax = axes[plot_idx // ncols, plot_idx % ncols]
        if viability_train_col:
            ax.plot(history_df[x_col], history_df[viability_train_col], label='Train Viability Score', marker='o', markersize=2, color='green')
        if viability_val_col:
            val_data = history_df[history_df[viability_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[viability_val_col], label='Val Viability Score', marker='s', markersize=2, color='darkgreen')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Score')
        ax.set_title('Viability Score (from Discriminator)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plot_idx += 1
    
    # Sample discriminator scores (if available)
    if 'mean_discriminator_score' in history_df.columns:
        ax = axes[plot_idx // ncols, plot_idx % ncols]
        sample_data = history_df[history_df['mean_discriminator_score'].notna()]
        if len(sample_data) > 0:
            ax.plot(sample_data[x_col], sample_data['mean_discriminator_score'], label='Mean Sample Score', marker='o', markersize=3, color='purple')
            if 'best_mean_score' in sample_data.columns:
                ax.plot(sample_data[x_col], sample_data['best_mean_score'], label='Best Samples Mean', marker='^', markersize=3, color='blue')
            if 'worst_mean_score' in sample_data.columns:
                ax.plot(sample_data[x_col], sample_data['worst_mean_score'], label='Worst Samples Mean', marker='v', markersize=3, color='red')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Discriminator Score')
        ax.set_title('Generated Sample Discriminator Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plot_idx += 1
    
    # Learning rate
    if 'learning_rate' in history_df.columns:
        ax = axes[plot_idx // ncols, plot_idx % ncols]
        ax.plot(history_df[x_col], history_df['learning_rate'], label='Learning Rate', marker='o', markersize=2, color='brown')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, nrows * ncols):
        axes[i // ncols, i % ncols].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / f"{exp_name}_iter_{iteration}_diffusion_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved diffusion metrics plot to: {plot_path}")

