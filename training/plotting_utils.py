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
    
    # Use tight_layout with error handling (some axes may not be compatible)
    try:
        plt.tight_layout()
    except Exception:
        pass  # bbox_inches='tight' in savefig will handle layout
    
    # Overwrite same file each time (no iteration in filename for latest)
    plot_path = output_dir / f"{exp_name}_iter_{iteration}_discriminator_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved discriminator metrics plot to: {plot_path}")


def plot_diffusion_metrics(history_df, output_dir, iteration, exp_name="diffusion"):
    """
    Plot diffusion model training metrics with comprehensive loss breakdown.
    
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
    
    # Create comprehensive loss breakdown plot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Diffusion Model Training Metrics - Iteration {iteration}', fontsize=16, y=0.995)
    
    plot_idx = 0
    
    # Plot 1: Total loss
    ax = fig.add_subplot(gs[0, 0])
    if 'train_loss' in history_df.columns:
        ax.plot(history_df[x_col], history_df['train_loss'], label='Train Loss', marker='o', markersize=2, linewidth=1.5)
    if 'val_loss' in history_df.columns:
        val_data = history_df[history_df['val_loss'] != float('inf')]
        if len(val_data) > 0:
            ax.plot(val_data[x_col], val_data['val_loss'], label='Val Loss', marker='s', markersize=2, linewidth=1.5)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss Breakdown (Stacked Area or Multiple Lines)
    ax = fig.add_subplot(gs[0, 1])
    loss_components = []
    component_names = []
    
    # Find all loss components
    noise_train_col = next((c for c in train_loss_cols if 'noise' in c.lower()), None)
    structural_train_col = next((c for c in train_loss_cols if 'structural' in c.lower()), None)
    semantic_train_col = next((c for c in train_loss_cols if 'semantic' in c.lower()), None)
    disc_train_col = next((c for c in train_loss_cols if 'discriminator' in c.lower() and 'viability' not in c.lower()), None)
    
    if noise_train_col:
        ax.plot(history_df[x_col], history_df[noise_train_col], label='Noise Loss', marker='o', markersize=2, linewidth=1.5, alpha=0.8)
    if structural_train_col:
        ax.plot(history_df[x_col], history_df[structural_train_col], label='Structural Loss', marker='^', markersize=2, linewidth=1.5, alpha=0.8)
    if semantic_train_col:
        ax.plot(history_df[x_col], history_df[semantic_train_col], label='Semantic Loss', marker='v', markersize=2, linewidth=1.5, alpha=0.8)
    if disc_train_col:
        ax.plot(history_df[x_col], history_df[disc_train_col], label='Discriminator Loss', marker='s', markersize=2, linewidth=1.5, alpha=0.8)
    
    # Plot any other loss components
    other_loss_cols = [c for c in train_loss_cols if c not in [noise_train_col, structural_train_col, semantic_train_col, disc_train_col] 
                       and 'viability' not in c.lower()]
    for col in other_loss_cols:
        ax.plot(history_df[x_col], history_df[col], label=col.replace('train_', '').replace('_', ' ').title(), 
                marker='.', markersize=1.5, linewidth=1, alpha=0.7)
    
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components Breakdown', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss Components Comparison (Bar chart of latest values)
    ax = fig.add_subplot(gs[0, 2])
    if len(history_df) > 0:
        latest_idx = len(history_df) - 1
        component_values = []
        component_labels = []
        
        if noise_train_col and noise_train_col in history_df.columns:
            component_values.append(history_df[noise_train_col].iloc[latest_idx])
            component_labels.append('Noise')
        if structural_train_col and structural_train_col in history_df.columns:
            component_values.append(history_df[structural_train_col].iloc[latest_idx])
            component_labels.append('Structural')
        if semantic_train_col and semantic_train_col in history_df.columns:
            component_values.append(history_df[semantic_train_col].iloc[latest_idx])
            component_labels.append('Semantic')
        if disc_train_col and disc_train_col in history_df.columns:
            component_values.append(history_df[disc_train_col].iloc[latest_idx])
            component_labels.append('Discriminator')
        
        if component_values:
            colors = plt.cm.Set3(np.linspace(0, 1, len(component_values)))
            bars = ax.bar(component_labels, component_values, color=colors, alpha=0.7)
            ax.set_ylabel('Loss Value')
            ax.set_title(f'Latest Loss Components (Step {history_df[x_col].iloc[latest_idx]})', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Noise Loss (detailed)
    noise_train_col = next((c for c in train_loss_cols if 'noise' in c.lower()), None)
    noise_val_col = next((c for c in val_loss_cols if 'noise' in c.lower()), None)
    if noise_train_col or noise_val_col:
        ax = fig.add_subplot(gs[1, 0])
        if noise_train_col:
            ax.plot(history_df[x_col], history_df[noise_train_col], label='Train Noise Loss', marker='o', markersize=2, linewidth=1.5)
        if noise_val_col:
            val_data = history_df[history_df[noise_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[noise_val_col], label='Val Noise Loss', marker='s', markersize=2, linewidth=1.5)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Noise Prediction Loss (MSE)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Structural Loss (if available)
    structural_train_col = next((c for c in train_loss_cols if 'structural' in c.lower()), None)
    structural_val_col = next((c for c in val_loss_cols if 'structural' in c.lower()), None)
    if structural_train_col or structural_val_col:
        ax = fig.add_subplot(gs[1, 1])
        if structural_train_col:
            ax.plot(history_df[x_col], history_df[structural_train_col], label='Train Structural Loss', marker='o', markersize=2, linewidth=1.5, color='purple')
        if structural_val_col:
            val_data = history_df[history_df[structural_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[structural_val_col], label='Val Structural Loss', marker='s', markersize=2, linewidth=1.5, color='darkviolet')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Latent Structural Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Discriminator loss
    disc_train_col = next((c for c in train_loss_cols if 'discriminator' in c.lower() and 'viability' not in c.lower()), None)
    disc_val_col = next((c for c in val_loss_cols if 'discriminator' in c.lower() and 'viability' not in c.lower()), None)
    if disc_train_col or disc_val_col:
        ax = fig.add_subplot(gs[1, 2])
        if disc_train_col:
            ax.plot(history_df[x_col], history_df[disc_train_col], label='Train Discriminator Loss', marker='o', markersize=2, linewidth=1.5, color='orange')
        if disc_val_col:
            val_data = history_df[history_df[disc_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[disc_val_col], label='Val Discriminator Loss', marker='s', markersize=2, linewidth=1.5, color='red')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Discriminator Adversarial Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Viability score
    viability_train_col = next((c for c in train_loss_cols if 'viability' in c.lower()), None)
    viability_val_col = next((c for c in val_loss_cols if 'viability' in c.lower()), None)
    if viability_train_col or viability_val_col:
        ax = fig.add_subplot(gs[2, 0])
        if viability_train_col:
            ax.plot(history_df[x_col], history_df[viability_train_col], label='Train Viability Score', marker='o', markersize=2, linewidth=1.5, color='green')
        if viability_val_col:
            val_data = history_df[history_df[viability_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[viability_val_col], label='Val Viability Score', marker='s', markersize=2, linewidth=1.5, color='darkgreen')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Score')
        ax.set_title('Viability Score (from Discriminator)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Plot 8: Sample discriminator scores (if available)
    if 'mean_discriminator_score' in history_df.columns:
        ax = fig.add_subplot(gs[2, 1])
        sample_data = history_df[history_df['mean_discriminator_score'].notna()]
        if len(sample_data) > 0:
            ax.plot(sample_data[x_col], sample_data['mean_discriminator_score'], label='Mean Sample Score', marker='o', markersize=3, linewidth=1.5, color='purple')
            if 'best_mean_score' in sample_data.columns:
                ax.plot(sample_data[x_col], sample_data['best_mean_score'], label='Best Samples Mean', marker='^', markersize=3, linewidth=1.5, color='blue')
            if 'worst_mean_score' in sample_data.columns:
                ax.plot(sample_data[x_col], sample_data['worst_mean_score'], label='Worst Samples Mean', marker='v', markersize=3, linewidth=1.5, color='red')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Discriminator Score')
        ax.set_title('Generated Sample Discriminator Scores', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Plot 9: Learning rate
    if 'learning_rate' in history_df.columns:
        ax = fig.add_subplot(gs[2, 2])
        ax.plot(history_df[x_col], history_df['learning_rate'], label='Learning Rate', marker='o', markersize=2, linewidth=1.5, color='brown')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Use tight_layout with error handling (some axes may not be compatible)
    try:
        plt.tight_layout()
    except Exception:
        pass  # bbox_inches='tight' in savefig will handle layout
    
    # Overwrite same file each time (same filename per iteration)
    plot_path = output_dir / f"{exp_name}_iter_{iteration}_diffusion_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved diffusion metrics plot with loss breakdown to: {plot_path}")


def plot_diffusion_metrics_epochs(history_df, output_dir, exp_name="diffusion"):
    """
    Plot diffusion model training metrics for epoch-based training (regular diffusion training).
    Similar to plot_diffusion_metrics but works with epochs instead of iterations.
    
    Args:
        history_df: DataFrame with training metrics
        output_dir: Directory to save plots
        exp_name: Experiment name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine x-axis column
    x_col = "step" if "step" in history_df.columns else "epoch"
    
    # Get available metrics
    train_loss_cols = [col for col in history_df.columns if col.startswith('train_') and col != 'train_loss']
    val_loss_cols = [col for col in history_df.columns if col.startswith('val_') and col != 'val_loss']
    
    # Create comprehensive loss breakdown plot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Diffusion Model Training Metrics - {exp_name}', fontsize=16, y=0.995)
    
    # Plot 1: Total loss
    ax = fig.add_subplot(gs[0, 0])
    if 'train_loss' in history_df.columns:
        ax.plot(history_df[x_col], history_df['train_loss'], label='Train Loss', marker='o', markersize=2, linewidth=1.5)
    if 'val_loss' in history_df.columns:
        val_data = history_df[history_df['val_loss'] != float('inf')]
        if len(val_data) > 0:
            ax.plot(val_data[x_col], val_data['val_loss'], label='Val Loss', marker='s', markersize=2, linewidth=1.5)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss Breakdown
    ax = fig.add_subplot(gs[0, 1])
    
    # Find all loss components
    noise_train_col = next((c for c in train_loss_cols if 'noise' in c.lower()), None)
    structural_train_col = next((c for c in train_loss_cols if 'structural' in c.lower()), None)
    semantic_train_col = next((c for c in train_loss_cols if 'semantic' in c.lower()), None)
    disc_train_col = next((c for c in train_loss_cols if 'discriminator' in c.lower() and 'viability' not in c.lower()), None)
    
    if noise_train_col:
        ax.plot(history_df[x_col], history_df[noise_train_col], label='Noise Loss', marker='o', markersize=2, linewidth=1.5, alpha=0.8)
    if structural_train_col:
        ax.plot(history_df[x_col], history_df[structural_train_col], label='Structural Loss', marker='^', markersize=2, linewidth=1.5, alpha=0.8)
    if semantic_train_col:
        ax.plot(history_df[x_col], history_df[semantic_train_col], label='Semantic Loss', marker='v', markersize=2, linewidth=1.5, alpha=0.8)
    if disc_train_col:
        ax.plot(history_df[x_col], history_df[disc_train_col], label='Discriminator Loss', marker='s', markersize=2, linewidth=1.5, alpha=0.8)
    
    # Plot any other loss components
    other_loss_cols = [c for c in train_loss_cols if c not in [noise_train_col, structural_train_col, semantic_train_col, disc_train_col] 
                       and 'viability' not in c.lower()]
    for col in other_loss_cols:
        ax.plot(history_df[x_col], history_df[col], label=col.replace('train_', '').replace('_', ' ').title(), 
                marker='.', markersize=1.5, linewidth=1, alpha=0.7)
    
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components Breakdown', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss Components Comparison (Bar chart of latest values)
    ax = fig.add_subplot(gs[0, 2])
    if len(history_df) > 0:
        latest_idx = len(history_df) - 1
        component_values = []
        component_labels = []
        
        if noise_train_col and noise_train_col in history_df.columns:
            component_values.append(history_df[noise_train_col].iloc[latest_idx])
            component_labels.append('Noise')
        if structural_train_col and structural_train_col in history_df.columns:
            component_values.append(history_df[structural_train_col].iloc[latest_idx])
            component_labels.append('Structural')
        if semantic_train_col and semantic_train_col in history_df.columns:
            component_values.append(history_df[semantic_train_col].iloc[latest_idx])
            component_labels.append('Semantic')
        if disc_train_col and disc_train_col in history_df.columns:
            component_values.append(history_df[disc_train_col].iloc[latest_idx])
            component_labels.append('Discriminator')
        
        if component_values:
            colors = plt.cm.Set3(np.linspace(0, 1, len(component_values)))
            bars = ax.bar(component_labels, component_values, color=colors, alpha=0.7)
            ax.set_ylabel('Loss Value')
            ax.set_title(f'Latest Loss Components ({x_col.capitalize()} {history_df[x_col].iloc[latest_idx]})', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Noise Loss (detailed)
    noise_train_col = next((c for c in train_loss_cols if 'noise' in c.lower()), None)
    noise_val_col = next((c for c in val_loss_cols if 'noise' in c.lower()), None)
    if noise_train_col or noise_val_col:
        ax = fig.add_subplot(gs[1, 0])
        if noise_train_col:
            ax.plot(history_df[x_col], history_df[noise_train_col], label='Train Noise Loss', marker='o', markersize=2, linewidth=1.5)
        if noise_val_col:
            val_data = history_df[history_df[noise_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[noise_val_col], label='Val Noise Loss', marker='s', markersize=2, linewidth=1.5)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Noise Prediction Loss (MSE)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Structural Loss (if available)
    structural_train_col = next((c for c in train_loss_cols if 'structural' in c.lower()), None)
    structural_val_col = next((c for c in val_loss_cols if 'structural' in c.lower()), None)
    if structural_train_col or structural_val_col:
        ax = fig.add_subplot(gs[1, 1])
        if structural_train_col:
            ax.plot(history_df[x_col], history_df[structural_train_col], label='Train Structural Loss', marker='o', markersize=2, linewidth=1.5, color='purple')
        if structural_val_col:
            val_data = history_df[history_df[structural_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[structural_val_col], label='Val Structural Loss', marker='s', markersize=2, linewidth=1.5, color='darkviolet')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Latent Structural Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Discriminator loss
    disc_train_col = next((c for c in train_loss_cols if 'discriminator' in c.lower() and 'viability' not in c.lower()), None)
    disc_val_col = next((c for c in val_loss_cols if 'discriminator' in c.lower() and 'viability' not in c.lower()), None)
    if disc_train_col or disc_val_col:
        ax = fig.add_subplot(gs[1, 2])
        if disc_train_col:
            ax.plot(history_df[x_col], history_df[disc_train_col], label='Train Discriminator Loss', marker='o', markersize=2, linewidth=1.5, color='orange')
        if disc_val_col:
            val_data = history_df[history_df[disc_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[disc_val_col], label='Val Discriminator Loss', marker='s', markersize=2, linewidth=1.5, color='red')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title('Discriminator Adversarial Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Viability score
    viability_train_col = next((c for c in train_loss_cols if 'viability' in c.lower()), None)
    viability_val_col = next((c for c in val_loss_cols if 'viability' in c.lower()), None)
    if viability_train_col or viability_val_col:
        ax = fig.add_subplot(gs[2, 0])
        if viability_train_col:
            ax.plot(history_df[x_col], history_df[viability_train_col], label='Train Viability Score', marker='o', markersize=2, linewidth=1.5, color='green')
        if viability_val_col:
            val_data = history_df[history_df[viability_val_col] != float('inf')]
            if len(val_data) > 0:
                ax.plot(val_data[x_col], val_data[viability_val_col], label='Val Viability Score', marker='s', markersize=2, linewidth=1.5, color='darkgreen')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Score')
        ax.set_title('Viability Score (from Discriminator)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Plot 8: Learning rate (if available)
    if 'learning_rate' in history_df.columns:
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(history_df[x_col], history_df['learning_rate'], label='Learning Rate', marker='o', markersize=2, linewidth=1.5, color='brown')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Use tight_layout with error handling (some axes may not be compatible)
    try:
        plt.tight_layout()
    except Exception:
        pass  # bbox_inches='tight' in savefig will handle layout
    
    plot_path = output_dir / f"{exp_name}_diffusion_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved diffusion metrics plot with loss breakdown to: {plot_path}")


def plot_overall_iteration_metrics(output_dir, exp_name="diffusion"):
    """
    Create an overall plot showing performance across all iterations.
    Aggregates metrics from all iteration CSV files.
    
    Args:
        output_dir: Output directory containing iteration subdirectories
        exp_name: Experiment name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Find all iteration directories
    iteration_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("discriminator_iter_")])
    
    if len(iteration_dirs) == 0:
        print("  No iteration directories found for overall plot")
        return
    
    # Collect metrics from each iteration
    all_iteration_data = []
    
    for iter_dir in iteration_dirs:
        iteration_num = int(iter_dir.name.split("_")[-1])
        
        # Load discriminator metrics
        disc_csv = iter_dir / "discriminator_history.csv"
        if disc_csv.exists():
            disc_df = pd.read_csv(disc_csv)
            disc_df['iteration'] = iteration_num
            disc_df['phase'] = 'discriminator'
            all_iteration_data.append(disc_df)
        
        # Load diffusion metrics
        # Try different possible locations
        metrics_csv = output_dir / f"{exp_name}_metrics_iter_{iteration_num}.csv"
        if not metrics_csv.exists():
            metrics_csv = output_dir / "checkpoints" / f"{exp_name}_iter_{iteration_num}_metrics.csv"
        if not metrics_csv.exists():
            metrics_csv = output_dir / f"{exp_name}_iter_{iteration_num}_metrics.csv"
        
        if metrics_csv.exists():
            diff_df = pd.read_csv(metrics_csv)
            diff_df['iteration'] = iteration_num
            diff_df['phase'] = 'diffusion'
            all_iteration_data.append(diff_df)
    
    if len(all_iteration_data) == 0:
        print("  No metrics found for overall plot")
        return
    
    # Combine all data
    combined_df = pd.concat(all_iteration_data, ignore_index=True)
    
    # Create overall plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Overall Training Metrics Across All Iterations - {exp_name}', fontsize=16)
    
    # Plot 1: Best validation loss per iteration (discriminator)
    ax = axes[0, 0]
    disc_iter_data = combined_df[combined_df['phase'] == 'discriminator']
    if len(disc_iter_data) > 0 and 'val_loss' in disc_iter_data.columns:
        best_val_loss_per_iter = disc_iter_data.groupby('iteration')['val_loss'].min()
        if len(best_val_loss_per_iter) > 0:
            ax.plot(best_val_loss_per_iter.index, best_val_loss_per_iter.values, marker='o', markersize=8, linewidth=2, label='Best Val Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Validation Loss')
            ax.set_title('Discriminator: Best Val Loss per Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No discriminator val_loss data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No discriminator data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 2: Best validation loss per iteration (diffusion)
    ax = axes[0, 1]
    diff_iter_data = combined_df[combined_df['phase'] == 'diffusion']
    if len(diff_iter_data) > 0 and 'val_loss' in diff_iter_data.columns:
        best_val_loss_per_iter = diff_iter_data.groupby('iteration')['val_loss'].min()
        if len(best_val_loss_per_iter) > 0:
            ax.plot(best_val_loss_per_iter.index, best_val_loss_per_iter.values, marker='s', markersize=8, linewidth=2, label='Best Val Loss', color='orange')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Validation Loss')
            ax.set_title('Diffusion: Best Val Loss per Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No diffusion val_loss data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No diffusion data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 3: Discriminator accuracy per iteration
    ax = axes[1, 0]
    if len(disc_iter_data) > 0 and 'val_acc' in disc_iter_data.columns:
        best_val_acc_per_iter = disc_iter_data.groupby('iteration')['val_acc'].max()
        if len(best_val_acc_per_iter) > 0:
            ax.plot(best_val_acc_per_iter.index, best_val_acc_per_iter.values, marker='o', markersize=8, linewidth=2, label='Best Val Accuracy', color='green')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Validation Accuracy')
            ax.set_title('Discriminator: Best Val Accuracy per Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No discriminator val_acc data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No discriminator accuracy data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Sample discriminator scores per iteration (if available)
    ax = axes[1, 1]
    if 'mean_discriminator_score' in diff_iter_data.columns:
        sample_scores_per_iter = diff_iter_data.groupby('iteration')['mean_discriminator_score'].mean()
        ax.plot(sample_scores_per_iter.index, sample_scores_per_iter.values, marker='^', markersize=8, linewidth=2, label='Mean Sample Score', color='purple')
        if 'best_mean_score' in diff_iter_data.columns:
            best_scores_per_iter = diff_iter_data.groupby('iteration')['best_mean_score'].mean()
            ax.plot(best_scores_per_iter.index, best_scores_per_iter.values, marker='s', markersize=6, linewidth=1.5, label='Best Samples Mean', color='blue', linestyle='--')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discriminator Score')
        ax.set_title('Generated Sample Quality (Discriminator Scores)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No sample metrics available', ha='center', va='center', transform=ax.transAxes)
    
    # Use tight_layout with error handling (some axes may not be compatible)
    try:
        plt.tight_layout()
    except Exception:
        pass  # bbox_inches='tight' in savefig will handle layout
    
    plot_path = output_dir / f"{exp_name}_overall_iteration_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved overall iteration metrics plot to: {plot_path}")


def plot_iterative_refinement_metrics(output_dir, exp_name="diffusion"):
    """
    Simple plot to verify iterative refinement is working.
    Shows key metrics: viability scores, sample scores, discriminator adversarial loss.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Find all iteration directories
    iteration_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("discriminator_iter_")])
    
    if len(iteration_dirs) == 0:
        print("  No iteration directories found")
        return
    
    # Collect diffusion metrics from each iteration
    all_metrics = []
    
    for iter_dir in iteration_dirs:
        iteration_num = int(iter_dir.name.split("_")[-1])
        
        # Load diffusion metrics
        metrics_csv = output_dir / f"{exp_name}_metrics_iter_{iteration_num}.csv"
        if not metrics_csv.exists():
            metrics_csv = output_dir / "checkpoints" / f"{exp_name}_iter_{iteration_num}_metrics.csv"
        if not metrics_csv.exists():
            metrics_csv = output_dir / f"{exp_name}_iter_{iteration_num}_metrics.csv"
        
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            # Get best row (lowest val_loss) or last row
            if 'val_loss' in df.columns:
                best_row = df.loc[df['val_loss'].idxmin()]
            else:
                best_row = df.iloc[-1]
            best_row['iteration'] = iteration_num
            all_metrics.append(best_row)
    
    if len(all_metrics) == 0:
        print("  No diffusion metrics found")
        return
    
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('iteration')
    
    # Create simple plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Iterative Refinement Metrics - {exp_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Viability Scores (should INCREASE)
    ax = axes[0, 0]
    viability_cols = [c for c in df.columns if 'viability' in c.lower() and 'train' in c.lower()]
    if viability_cols:
        viability_col = viability_cols[0]
        ax.plot(df['iteration'], df[viability_col], marker='o', markersize=8, linewidth=2, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Viability Score')
        ax.set_title('Viability Score (Should INCREASE)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No viability score data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 2: Sample Discriminator Scores (should INCREASE)
    ax = axes[0, 1]
    if 'mean_discriminator_score' in df.columns:
        ax.plot(df['iteration'], df['mean_discriminator_score'], marker='o', markersize=8, linewidth=2, color='purple')
        if 'best_mean_score' in df.columns:
            ax.plot(df['iteration'], df['best_mean_score'], marker='^', markersize=6, linewidth=1.5, 
                   color='blue', linestyle='--', label='Best samples')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discriminator Score')
        ax.set_title('Sample Discriminator Scores (Should INCREASE)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No sample score data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 3: Discriminator Adversarial Loss (should DECREASE)
    ax = axes[1, 0]
    disc_loss_cols = [c for c in df.columns if 'discriminator' in c.lower() and 'loss' in c.lower() and 'train' in c.lower()]
    if disc_loss_cols:
        disc_loss_col = disc_loss_cols[0]
        ax.plot(df['iteration'], df[disc_loss_col], marker='s', markersize=8, linewidth=2, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Discriminator Adversarial Loss (Should DECREASE)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No discriminator loss data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = "KEY INDICATORS:\n" + "="*30 + "\n\n"
    
    if viability_cols:
        if len(df) > 1:
            first = df[viability_col].iloc[0]
            last = df[viability_col].iloc[-1]
            change = last - first
            status = "✓ INCREASING" if change > 0 else "✗ NOT IMPROVING"
            summary += f"Viability Score: {status}\n"
            summary += f"  {first:.3f} → {last:.3f} ({change:+.3f})\n\n"
    
    if 'mean_discriminator_score' in df.columns:
        if len(df) > 1:
            first = df['mean_discriminator_score'].iloc[0]
            last = df['mean_discriminator_score'].iloc[-1]
            change = last - first
            status = "✓ INCREASING" if change > 0 else "✗ NOT IMPROVING"
            summary += f"Sample Scores: {status}\n"
            summary += f"  {first:.3f} → {last:.3f} ({change:+.3f})\n\n"
    
    if disc_loss_cols:
        if len(df) > 1:
            first = df[disc_loss_col].iloc[0]
            last = df[disc_loss_col].iloc[-1]
            change = last - first
            status = "✓ DECREASING" if change < 0 else "✗ NOT IMPROVING"
            summary += f"Adv. Loss: {status}\n"
            summary += f"  {first:.3f} → {last:.3f} ({change:+.3f})\n"
    
    summary += "\n" + "="*30 + "\n"
    summary += "If viability/scores increase\nand loss decreases → WORKING!"
    
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Use tight_layout with error handling (some axes may not be compatible)
    try:
        plt.tight_layout()
    except Exception:
        pass  # bbox_inches='tight' in savefig will handle layout
    
    plot_path = output_dir / f"{exp_name}_iterative_refinement_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved iterative refinement metrics to: {plot_path}")

