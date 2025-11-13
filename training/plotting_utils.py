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
    # Overwrite same file each time (no iteration in filename for latest)
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
    
    # Count all possible plots
    num_plots = 1  # Total loss (always)
    if any('noise' in c.lower() for c in train_loss_cols + val_loss_cols):
        num_plots += 1
    if any('discriminator' in c.lower() for c in train_loss_cols + val_loss_cols):
        num_plots += 1
    if any('viability' in c.lower() for c in train_loss_cols + val_loss_cols):
        num_plots += 1
    if 'mean_discriminator_score' in history_df.columns:
        num_plots += 1
    if 'learning_rate' in history_df.columns:
        num_plots += 1
    
    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols  # Ceiling division
    
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
    if 'mean_discriminator_score' in history_df.columns and plot_idx < nrows * ncols:
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
    if 'learning_rate' in history_df.columns and plot_idx < nrows * ncols:
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
    # Overwrite same file each time (same filename per iteration)
    plot_path = output_dir / f"{exp_name}_iter_{iteration}_diffusion_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved diffusion metrics plot to: {plot_path}")


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
    
    plt.tight_layout()
    plot_path = output_dir / f"{exp_name}_overall_iteration_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved overall iteration metrics plot to: {plot_path}")

