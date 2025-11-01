#!/usr/bin/env python3
"""
Analysis script for Phase 1.1: Channel × Spatial Resolution Sweep
Loads metrics from all experiments and creates comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import yaml
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.utils import load_config

# Set seaborn style with dark grid
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Color palette for experiments - distinct colors from tab20 colormap (12 colors for 12 experiments)
import matplotlib.cm as cm
_tab20_colors = cm.get_cmap('tab20').colors
# Select distinct colors from tab20 for 12 experiments
EXPERIMENT_COLORS = [
    _tab20_colors[0],   # Blue
    _tab20_colors[1],   # Light Blue
    _tab20_colors[2],   # Red
    _tab20_colors[3],   # Light Red
    _tab20_colors[4],   # Green
    _tab20_colors[5],   # Light Green
    _tab20_colors[6],   # Orange
    _tab20_colors[7],   # Light Orange
    _tab20_colors[8],   # Purple
    _tab20_colors[9],   # Light Purple
    _tab20_colors[10],  # Brown
    _tab20_colors[11],  # Pink
]
# Convert to hex for compatibility
EXPERIMENT_COLORS = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in EXPERIMENT_COLORS]


def parse_latent_dimensions(exp_name):
    """
    Parse latent dimensions from experiment name.
    Example: 'phase1_1_AE_S1_ch16_ds4' -> channels=16, ds=4, spatial='32×32', dims=16384
    
    Returns dict with: channels, downsampling_steps, spatial_str, total_dims
    """
    # Extract channels and downsampling steps from name
    ch_match = re.search(r'ch(\d+)', exp_name)
    ds_match = re.search(r'ds(\d+)', exp_name)
    
    if not ch_match or not ds_match:
        return {'channels': None, 'downsampling_steps': None, 'spatial_str': '?×?', 'total_dims': None}
    
    channels = int(ch_match.group(1))
    downsampling_steps = int(ds_match.group(1))
    
    # Calculate spatial resolution (512 / 2^downsampling_steps)
    spatial_res = 512 // (2 ** downsampling_steps)
    spatial_str = f'{spatial_res}×{spatial_res}'
    total_dims = spatial_res * spatial_res * channels
    
    return {
        'channels': channels,
        'downsampling_steps': downsampling_steps,
        'spatial_str': spatial_str,
        'total_dims': total_dims
    }


def load_metrics(phase_dir):
    """Load all metrics CSV files from phase directory."""
    phase_path = Path(phase_dir)
    if not phase_path.exists():
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")
    
    metrics_files = list(phase_path.glob("*_metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found in {phase_dir}")
    
    print(f"Found {len(metrics_files)} metrics files:")
    all_data = {}
    
    for metrics_file in metrics_files:
        exp_name = metrics_file.stem.replace("_metrics", "")
        print(f"  - {exp_name}")
        
        df = pd.read_csv(metrics_file)
        df['experiment'] = exp_name
        all_data[exp_name] = df
    
    return all_data


def create_loss_curves(all_data, output_dir):
    """Create loss curve plots comparing all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1.1: Loss Curves Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for i, (exp_name, df) in enumerate(all_data.items()):
        if 'train_loss' in df.columns:
            ax1.plot(df['epoch'], df['train_loss'], 
                    label=exp_name, color=EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='none')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for i, (exp_name, df) in enumerate(all_data.items()):
        if 'val_loss' in df.columns:
            ax2.plot(df['epoch'], df['val_loss'], 
                    label=exp_name, color=EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='none')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training MSE (RGB)
    ax3 = axes[1, 0]
    for i, (exp_name, df) in enumerate(all_data.items()):
        mse_col = [c for c in df.columns if 'MSE' in c.upper() and 'train' in c.lower()]
        if mse_col:
            ax3.plot(df['epoch'], df[mse_col[0]], 
                    label=exp_name, color=EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)], linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training MSE (RGB)', fontsize=12)
    ax3.set_title('Training RGB MSE', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='none')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation MSE (RGB)
    ax4 = axes[1, 1]
    for i, (exp_name, df) in enumerate(all_data.items()):
        mse_col = [c for c in df.columns if 'MSE' in c.upper() and 'val' in c.lower()]
        if mse_col:
            ax4.plot(df['epoch'], df[mse_col[0]], 
                    label=exp_name, color=EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)], linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation MSE (RGB)', fontsize=12)
    ax4.set_title('Validation RGB MSE', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='none')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_final_metrics_comparison(all_data, output_dir):
    """Create bar plot comparing final metrics across experiments."""
    # Extract final epoch metrics
    final_metrics = []
    
    for exp_name, df in all_data.items():
        final_row = df.iloc[-1]
        metrics = {
            'experiment': exp_name,
            'train_loss': final_row.get('train_loss', np.nan),
            'val_loss': final_row.get('val_loss', np.nan),
        }
        
        # Extract MSE if available
        mse_cols = [c for c in df.columns if 'MSE' in c.upper()]
        if mse_cols:
            for col in mse_cols:
                if 'train' in col.lower():
                    metrics['train_MSE'] = final_row.get(col, np.nan)
                elif 'val' in col.lower():
                    metrics['val_MSE'] = final_row.get(col, np.nan)
        
        # Extract segmentation accuracy if available
        seg_cols = [c for c in df.columns if 'CE_segmentation' in c]
        if seg_cols:
            for col in seg_cols:
                if 'train' in col.lower():
                    metrics['train_seg_loss'] = final_row.get(col, np.nan)
                elif 'val' in col.lower():
                    metrics['val_seg_loss'] = final_row.get(col, np.nan)
        
        final_metrics.append(metrics)
    
    final_df = pd.DataFrame(final_metrics)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1.1: Final Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Final Validation Loss
    ax1 = axes[0, 0]
    bars1 = sns.barplot(data=final_df, x='experiment', y='val_loss', ax=ax1, palette=EXPERIMENT_COLORS)
    ax1.set_xlabel('Experiment', fontsize=12)
    ax1.set_ylabel('Final Validation Loss', fontsize=12)
    ax1.set_title('Final Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for container in bars1.containers:
        ax1.bar_label(container, fmt='%.4f', rotation=90, padding=3, fontsize=8)
    
    # Add latent dimension info to dataframe
    final_df['latent_dims'] = final_df['experiment'].apply(
        lambda x: parse_latent_dimensions(x).get('total_dims', None)
    )
    final_df['latent_shape'] = final_df['experiment'].apply(
        lambda x: parse_latent_dimensions(x).get('spatial_str', '?×?')
    )
    final_df['latent_channels'] = final_df['experiment'].apply(
        lambda x: parse_latent_dimensions(x).get('channels', None)
    )
    
    # Plot 2: Final Validation MSE (zoomed to highlight differences)
    ax2 = axes[0, 1]
    if 'val_MSE' in final_df.columns:
        bars2 = sns.barplot(data=final_df, x='experiment', y='val_MSE', ax=ax2, palette=EXPERIMENT_COLORS)
        ax2.set_xlabel('Experiment', fontsize=12)
        ax2.set_ylabel('Final Validation MSE', fontsize=12)
        # Zoom y-axis to highlight differences
        mse_min = final_df['val_MSE'].min()
        mse_max = final_df['val_MSE'].max()
        mse_range = mse_max - mse_min
        ax2.set_ylim(max(0, mse_min - mse_range*0.15), mse_max + mse_range*0.15)
        ax2.set_title(f'Final Validation MSE (RGB)\nRange: {mse_min:.4f} - {mse_max:.4f}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for container in bars2.containers:
            ax2.bar_label(container, fmt='%.4f', rotation=90, padding=3, fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'MSE data not available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Final Validation MSE (RGB)', fontsize=14, fontweight='bold')
    
    # Plot 3: Final Training Loss
    ax3 = axes[1, 0]
    bars3 = sns.barplot(data=final_df, x='experiment', y='train_loss', ax=ax3, palette=EXPERIMENT_COLORS)
    ax3.set_xlabel('Experiment', fontsize=12)
    ax3.set_ylabel('Final Training Loss', fontsize=12)
    ax3.set_title('Final Training Loss', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for container in bars3.containers:
        ax3.bar_label(container, fmt='%.4f', rotation=90, padding=3, fontsize=8)
    
    # Plot 4: Final Validation Segmentation Loss
    ax4 = axes[1, 1]
    if 'val_seg_loss' in final_df.columns:
        bars4 = sns.barplot(data=final_df, x='experiment', y='val_seg_loss', ax=ax4, palette=EXPERIMENT_COLORS)
        ax4.set_xlabel('Experiment', fontsize=12)
        ax4.set_ylabel('Final Validation Seg Loss', fontsize=12)
        ax4.set_title('Final Validation Segmentation Loss', fontsize=14, fontweight='bold')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for container in bars4.containers:
            ax4.bar_label(container, fmt='%.4f', rotation=90, padding=3, fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'Segmentation loss data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Final Validation Segmentation Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "final_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Create relative performance plot (difference from best)
    if 'val_MSE' in final_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        best_mse = final_df['val_MSE'].min()
        final_df_sorted = final_df.sort_values('val_MSE')
        relative_mse = ((final_df_sorted['val_MSE'] - best_mse) / best_mse) * 100
        
        # Map experiments to colors (use experiment index in original sorted list)
        exp_order = sorted(final_df['experiment'].tolist())
        colors_list = [EXPERIMENT_COLORS[exp_order.index(exp) % len(EXPERIMENT_COLORS)] 
                      for exp in final_df_sorted['experiment']]
        
        bars = ax.barh(range(len(final_df_sorted)), relative_mse.values, color=colors_list)
        ax.set_yticks(range(len(final_df_sorted)))
        ax.set_yticklabels([name.replace('phase1_1_AE_', '') for name in final_df_sorted['experiment']])
        ax.set_xlabel('% Difference from Best MSE', fontsize=12)
        ax.set_title('Relative Performance: Validation MSE vs Best', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, val) in enumerate(relative_mse.items()):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        rel_path = output_dir / "relative_performance_mse.png"
        plt.savefig(rel_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {rel_path}")
        plt.close()
    
    # Save final metrics table
    table_path = output_dir / "final_metrics_table.csv"
    final_df.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")


def create_convergence_analysis(all_data, output_dir):
    """Analyze convergence speed and stability."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Phase 1.1: Convergence Analysis', fontsize=16, fontweight='bold')
    
    convergence_data = []
    
    for exp_name, df in all_data.items():
        if 'val_loss' not in df.columns:
            continue
        
        val_loss = df['val_loss'].values
        epochs = df['epoch'].values
        
        # Find epoch where loss stabilizes (within 5% of final loss)
        final_loss = val_loss[-1]
        threshold = final_loss * 1.05
        stable_epochs = np.where(val_loss <= threshold)[0]
        convergence_epoch = stable_epochs[0] + 1 if len(stable_epochs) > 0 else len(epochs)
        
        # Calculate improvement (first to last)
        initial_loss = val_loss[0]
        improvement = initial_loss - final_loss
        improvement_pct = (improvement / initial_loss) * 100
        
        convergence_data.append({
            'experiment': exp_name,
            'convergence_epoch': convergence_epoch,
            'improvement_pct': improvement_pct,
            'final_loss': final_loss
        })
        
        # Plot loss reduction over time
        axes[0].plot(epochs, val_loss, 
                    label=exp_name, 
                    color=EXPERIMENT_COLORS[len(convergence_data) % len(EXPERIMENT_COLORS)], 
                    linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Loss Reduction Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(frameon=True, fancybox=False, shadow=False, edgecolor='none')
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot: Convergence speed
    conv_df = pd.DataFrame(convergence_data)
    if len(conv_df) > 0:
        sns.barplot(data=conv_df, x='experiment', y='convergence_epoch', 
                   ax=axes[1], palette=EXPERIMENT_COLORS)
        axes[1].set_xlabel('Experiment', fontsize=12)
        axes[1].set_ylabel('Convergence Epoch', fontsize=12)
        axes[1].set_title('Convergence Speed', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "convergence_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save convergence data
    if len(conv_df) > 0:
        conv_path = output_dir / "convergence_analysis.csv"
        conv_df.to_csv(conv_path, index=False)
        print(f"Saved: {conv_path}")


def load_autoencoder_checkpoints(exp_names, base_experiments_dir):
    """Load autoencoder checkpoints for all experiments."""
    checkpoints = {}
    base_dir = Path(base_experiments_dir)
    
    for exp_name in exp_names:
        # Try to find best checkpoint first, then latest
        exp_dir = base_dir / exp_name
        best_ckpt = exp_dir / f"{exp_name}_checkpoint_best.pt"
        latest_ckpt = exp_dir / f"{exp_name}_checkpoint_latest.pt"
        
        checkpoint_path = None
        if best_ckpt.exists():
            checkpoint_path = best_ckpt
        elif latest_ckpt.exists():
            checkpoint_path = latest_ckpt
        else:
            print(f"  Warning: No checkpoint found for {exp_name}, skipping visual comparison")
            continue
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Autoencoder.load_checkpoint(checkpoint_path, map_location=device)
            model = model.to(device)
            model.eval()
            checkpoints[exp_name] = {
                'model': model,
                'device': device,
                'path': checkpoint_path
            }
            print(f"  Loaded: {exp_name} from {checkpoint_path.name}")
        except Exception as e:
            print(f"  Error loading {exp_name}: {e}")
            continue
    
    return checkpoints


def get_test_samples(dataset_manifest, test_samples_per_type=1):
    """Get test samples from dataset: 1 scene and 1 room."""
    dataset = ManifestDataset(
        manifest=dataset_manifest,
        outputs={
            'rgb': 'layout_path',
            'segmentation': 'layout_path',
            'label': 'type'
        },
        filters={'is_empty': [False]},
        return_path=False
    )
    
    # Find scene and room samples
    scenes = []
    rooms = []
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            label = sample.get('label', None)
            
            if isinstance(label, str):
                label_lower = label.lower().strip()
                if label_lower == 'scene' and len(scenes) < test_samples_per_type:
                    scenes.append((i, sample))
                elif label_lower == 'room' and len(rooms) < test_samples_per_type:
                    rooms.append((i, sample))
            
            if len(scenes) >= test_samples_per_type and len(rooms) >= test_samples_per_type:
                break
        except Exception as e:
            continue
    
    print(f"  Found {len(scenes)} scene(s) and {len(rooms)} room(s) for testing")
    return scenes, rooms


def create_visual_comparison(checkpoints, test_scenes, test_rooms, output_dir):
    """Create visual comparison of all experiments on the same test samples."""
    if not checkpoints:
        print("  No checkpoints loaded, skipping visual comparison")
        return
    
    if not test_scenes or not test_rooms:
        print("  No test samples found, skipping visual comparison")
        return
    
    device = list(checkpoints.values())[0]['device']
    
    # Get one scene and one room
    scene_idx, scene_sample = test_scenes[0]
    room_idx, room_sample = test_rooms[0]
    
    # Prepare input tensors
    scene_rgb = scene_sample['rgb'].unsqueeze(0).to(device)
    room_rgb = room_sample['rgb'].unsqueeze(0).to(device)
    
    # Run inference through all models
    scene_reconstructions = {}
    room_reconstructions = {}
    
    with torch.no_grad():
        for exp_name, checkpoint_info in checkpoints.items():
            model = checkpoint_info['model']
            
            # Scene reconstruction
            scene_out = model(scene_rgb)
            scene_reconstructions[exp_name] = scene_out['rgb'].cpu()
            
            # Room reconstruction
            room_out = model(room_rgb)
            room_reconstructions[exp_name] = room_out['rgb'].cpu()
    
    # Create comparison grids
    # Scene comparison
    scene_fig = create_comparison_grid(
        scene_rgb.cpu(),
        scene_reconstructions,
        title="Scene Reconstruction Comparison"
    )
    scene_path = output_dir / "visual_comparison_scene.png"
    scene_fig.savefig(scene_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(scene_fig)
    print(f"  Saved: {scene_path}")
    
    # Room comparison
    room_fig = create_comparison_grid(
        room_rgb.cpu(),
        room_reconstructions,
        title="Room Reconstruction Comparison"
    )
    room_path = output_dir / "visual_comparison_room.png"
    room_fig.savefig(room_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(room_fig)
    print(f"  Saved: {room_path}")


def create_comparison_grid(original, reconstructions, title=""):
    """
    Create a grid showing original and all reconstructions with labels.
    Uses matplotlib to add text labels for each image.
    
    Args:
        original: [1, 3, H, W] tensor
        reconstructions: dict of {exp_name: [1, 3, H, W] tensor}
        title: Optional title
    
    Returns:
        PIL Image or matplotlib figure (ready for saving)
    """
    import matplotlib.patches as mpatches
    
    # Denormalize from [-1, 1] to [0, 1]
    original = (original + 1) / 2.0
    reconstructions = {k: (v + 1) / 2.0 for k, v in reconstructions.items()}
    
    # Resize to 256 for visualization
    target_size = 256
    if original.shape[-1] != target_size:
        original = F.interpolate(original, size=(target_size, target_size), 
                                mode='bilinear', align_corners=False)
        reconstructions = {
            k: F.interpolate(v, size=(target_size, target_size), 
                            mode='bilinear', align_corners=False)
            for k, v in reconstructions.items()
        }
    
    # Convert to numpy for matplotlib
    original_np = original.squeeze(0).cpu().permute(1, 2, 0).numpy()
    reconstructions_np = {
        k: v.squeeze(0).cpu().permute(1, 2, 0).numpy()
        for k, v in reconstructions.items()
    }
    
    # Create figure with subplots - use 2 rows if more than 6 experiments
    exp_names = sorted(reconstructions.keys())
    n_images = 1 + len(exp_names)  # Original + reconstructions
    
    # Determine grid layout: 2 rows if more than 7 total images, otherwise 1 row
    if n_images > 7:
        ncols = 7  # 1 original + 6 per row max
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3))
        axes = axes.flatten()
    else:
        ncols = n_images
        nrows = 1
        fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2.5, 3))
        if n_images == 1:
            axes = [axes]  # Make iterable if single subplot
    
    # Plot original
    axes[0].imshow(original_np)
    axes[0].set_title('Original', fontsize=11, fontweight='bold', pad=8)
    axes[0].axis('off')
    
    # Plot reconstructions with experiment names and dimensions
    for i, exp_name in enumerate(exp_names, start=1):
        if i >= len(axes):
            break  # Safety check
        axes[i].imshow(reconstructions_np[exp_name])
        # Clean up experiment name for display
        display_name = exp_name.replace('phase1_1_AE_', '').replace('phase1_1_', '')
        # Add latent dimensions to title
        dims_info = parse_latent_dimensions(exp_name)
        if dims_info['total_dims']:
            dims_str = f"{dims_info['spatial_str']}×{dims_info['channels']} = {dims_info['total_dims']:,}"
            title = f"{display_name}\n({dims_str})"
        else:
            title = display_name
        axes[i].set_title(title, fontsize=9, fontweight='bold', pad=6)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(exp_names) + 1, len(axes)):
        axes[i].axis('off')
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def create_efficiency_analysis(all_data, output_dir):
    """
    Analyze quality vs efficiency trade-off for Phase 1.1.
    Focus on whether improvement is worth the increased latent dimensions.
    """
    # Calculate latent dimensions from experiment names
    final_data = []
    for exp_name, df in all_data.items():
        if 'val_MSE' not in df.columns or len(df) == 0:
            continue
        
        final_mse = df['val_MSE'].iloc[-1]
        dims_info = parse_latent_dimensions(exp_name)
        
        if dims_info['total_dims']:
            mse_per_dim = final_mse / dims_info['total_dims']
            final_data.append({
                'experiment': exp_name,
                'val_MSE': final_mse,
                'latent_dims': dims_info['total_dims'],
                'spatial': dims_info['spatial_str'],
                'channels': dims_info['channels'],
                'MSE_per_dim': mse_per_dim,
            })
    
    if not final_data:
        print("  No efficiency data available")
        return
    
    eff_df = pd.DataFrame(final_data)
    eff_df = eff_df.sort_values('latent_dims')
    
    # Create efficiency comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1.1: Efficiency Analysis (Quality vs Latent Dimensions)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: MSE vs Latent Dimensions (scatter)
    ax1 = axes[0, 0]
    for i, row in eff_df.iterrows():
        color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
        ax1.scatter(row['latent_dims'], row['val_MSE'], 
                   s=200, color=color, alpha=0.7)
        # Add label
        display_name = row['experiment'].replace('phase1_1_AE_', '').replace('phase1_1_', '')
        ax1.text(row['latent_dims'], row['val_MSE'], 
                f"  {display_name}", fontsize=8, va='bottom')
    ax1.set_xlabel('Total Latent Dimensions (H×W×C)', fontsize=12)
    ax1.set_ylabel('Validation MSE', fontsize=12)
    ax1.set_title('Quality vs Capacity Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MSE per Dimension (efficiency metric)
    ax2 = axes[0, 1]
    bars2 = sns.barplot(data=eff_df, x='experiment', y='MSE_per_dim', 
                       ax=ax2, palette=EXPERIMENT_COLORS)
    ax2.set_xlabel('Experiment', fontsize=12)
    ax2.set_ylabel('MSE per Latent Dimension (×10⁻⁸)', fontsize=12)
    ax2.set_title('Efficiency: Lower is Better', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    # Scale y-axis to show differences
    y_min, y_max = eff_df['MSE_per_dim'].min(), eff_df['MSE_per_dim'].max()
    y_range = y_max - y_min
    ax2.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    for container in bars2.containers:
        ax2.bar_label(container, fmt='%.2e', rotation=90, padding=3, fontsize=7)
    
    # Plot 3: Total Latent Dimensions
    ax3 = axes[1, 0]
    bars3 = sns.barplot(data=eff_df, x='experiment', y='latent_dims', 
                       ax=ax3, palette=EXPERIMENT_COLORS)
    ax3.set_xlabel('Experiment', fontsize=12)
    ax3.set_ylabel('Total Latent Dimensions', fontsize=12)
    ax3.set_title('Latent Space Size', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    for container in bars3.containers:
        ax3.bar_label(container, fmt='%d', rotation=90, padding=3, fontsize=7)
    
    # Plot 4: Improvement vs Cost (relative to smallest)
    ax4 = axes[1, 1]
    smallest = eff_df.loc[eff_df['latent_dims'].idxmin()]
    largest = eff_df.loc[eff_df['latent_dims'].idxmax()]
    
    eff_df['mse_improvement'] = ((smallest['val_MSE'] - eff_df['val_MSE']) / smallest['val_MSE']) * 100
    eff_df['dim_increase'] = ((eff_df['latent_dims'] - smallest['latent_dims']) / smallest['latent_dims']) * 100
    
    # Scatter: improvement % vs dimension increase %
    for i, row in eff_df.iterrows():
        color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
        ax4.scatter(row['dim_increase'], row['mse_improvement'], 
                   s=200, color=color, alpha=0.7)
        display_name = row['experiment'].replace('phase1_1_AE_', '').replace('phase1_1_', '')
        ax4.text(row['dim_increase'], row['mse_improvement'], 
                f"  {display_name}", fontsize=8, va='bottom')
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('% Increase in Latent Dimensions (vs Smallest)', fontsize=12)
    ax4.set_ylabel('% Improvement in MSE (vs Smallest)', fontsize=12)
    ax4.set_title('Cost-Benefit Analysis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "efficiency_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save efficiency table
    table_path = output_dir / "efficiency_analysis.csv"
    eff_df.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")


def create_summary_report(all_data, output_dir):
    """Create a text summary report with actual results."""
    report_path = output_dir / "analysis_summary.txt"
    
    # Collect all final metrics
    final_metrics = []
    for exp_name, df in all_data.items():
        dims_info = parse_latent_dimensions(exp_name)
        final_row = df.iloc[-1]
        metrics = {
            'experiment': exp_name,
            'train_loss': final_row.get('train_loss', np.nan),
            'val_loss': final_row.get('val_loss', np.nan),
        }
        
        # Extract MSE if available
        mse_cols = [c for c in df.columns if 'MSE' in c.upper()]
        if mse_cols:
            for col in mse_cols:
                if 'train' in col.lower():
                    metrics['train_MSE'] = final_row.get(col, np.nan)
                elif 'val' in col.lower():
                    metrics['val_MSE'] = final_row.get(col, np.nan)
        
        # Extract segmentation loss if available
        seg_cols = [c for c in df.columns if 'CE_segmentation' in c]
        if seg_cols:
            for col in seg_cols:
                if 'train' in col.lower():
                    metrics['train_seg_loss'] = final_row.get(col, np.nan)
                elif 'val' in col.lower():
                    metrics['val_seg_loss'] = final_row.get(col, np.nan)
        
        final_metrics.append(metrics)
    
    final_df = pd.DataFrame(final_metrics)
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 1.1: Latent Channel Sweep - Results Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Rank experiments by key metrics
        if 'val_MSE' in final_df.columns:
            final_df_sorted = final_df.sort_values('val_MSE')
            f.write("Ranking by Validation MSE (RGB Reconstruction Quality):\n")
            f.write("-" * 80 + "\n")
            for rank, (idx, row) in enumerate(final_df_sorted.iterrows(), 1):
                exp_clean = row['experiment'].replace('phase1_1_AE_', '')
                f.write(f"{rank}. {exp_clean}\n")
                f.write(f"   Validation MSE: {row['val_MSE']:.6f}\n")
                if 'val_loss' in row:
                    f.write(f"   Validation Loss: {row['val_loss']:.6f}\n")
                if rank == 1:
                    best_mse = row['val_MSE']
                    best_exp = row['experiment']
                f.write("\n")
        
        # Best experiment summary
        if 'val_MSE' in final_df.columns:
            f.write("=" * 80 + "\n")
            f.write("BEST EXPERIMENT:\n")
            f.write("=" * 80 + "\n")
            best_row = final_df_sorted.iloc[0]
            best_exp_clean = best_row['experiment'].replace('phase1_1_AE_', '')
            f.write(f"Experiment: {best_exp_clean}\n")
            f.write(f"Validation MSE: {best_row['val_MSE']:.6f}\n")
            if 'val_loss' in best_row:
                f.write(f"Validation Loss: {best_row['val_loss']:.6f}\n")
            if 'train_MSE' in best_row:
                f.write(f"Training MSE: {best_row['train_MSE']:.6f}\n")
            if 'val_seg_loss' in best_row:
                f.write(f"Segmentation Loss: {best_row['val_seg_loss']:.6f}\n")
            f.write("\n")
            
            # Performance difference from best
            f.write("Performance vs Best (Validation MSE):\n")
            f.write("-" * 80 + "\n")
            for idx, row in final_df.iterrows():
                if row['experiment'] != best_exp:
                    exp_clean = row['experiment'].replace('phase1_1_AE_', '')
                    diff_pct = ((row['val_MSE'] - best_mse) / best_mse) * 100
                    f.write(f"{exp_clean}: {diff_pct:+.1f}% vs best\n")
            f.write("\n")
        
        # Detailed metrics for all experiments
        f.write("=" * 80 + "\n")
        f.write("Detailed Metrics by Experiment:\n")
        f.write("=" * 80 + "\n\n")
        
        for exp_name, df in all_data.items():
            exp_clean = exp_name.replace('phase1_1_AE_', '')
            f.write(f"Experiment: {exp_clean}\n")
            f.write(f"  Total Epochs: {len(df)}\n")
            
            # Get final metrics from final_df
            exp_row = final_df[final_df['experiment'] == exp_name].iloc[0]
            
            if 'train_loss' in exp_row:
                f.write(f"  Final Training Loss: {exp_row['train_loss']:.6f}\n")
            if 'val_loss' in exp_row:
                f.write(f"  Final Validation Loss: {exp_row['val_loss']:.6f}\n")
            if 'train_MSE' in exp_row:
                f.write(f"  Final Training MSE: {exp_row['train_MSE']:.6f}\n")
            if 'val_MSE' in exp_row:
                f.write(f"  Final Validation MSE: {exp_row['val_MSE']:.6f}\n")
            if 'train_seg_loss' in exp_row:
                f.write(f"  Final Training Seg Loss: {exp_row['train_seg_loss']:.6f}\n")
            if 'val_seg_loss' in exp_row:
                f.write(f"  Final Validation Seg Loss: {exp_row['val_seg_loss']:.6f}\n")
            
            # Training progress
            if 'val_loss' in df.columns:
                initial_val = df['val_loss'].iloc[0]
                final_val = df['val_loss'].iloc[-1]
                improvement = initial_val - final_val
                f.write(f"  Validation Loss Improvement: {improvement:.6f} ({initial_val:.6f} → {final_val:.6f})\n")
            
            f.write("\n")
    
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1.1 experiments")
    parser.add_argument(
        "--phase-dir",
        type=str,
        default="outputs/phase1_1_latent_channels",
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
    print("Phase 1.1: Latent Channel Sweep Analysis")
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


if __name__ == "__main__":
    main()

