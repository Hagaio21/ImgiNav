#!/usr/bin/env python3
"""
Analysis script for Phase 1.1: Latent Channel Sweep
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.utils import load_config

# Set seaborn style with dark grid and pastel colors
sns.set_style("darkgrid")
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Color palette for experiments
PASTEL_COLORS = [
    '#FFB3BA',  # Pastel red
    '#BAFFC9',  # Pastel green
    '#BAE1FF',  # Pastel blue
    '#FFFFBA',  # Pastel yellow
    '#FFDFBA',  # Pastel orange
]


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
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for i, (exp_name, df) in enumerate(all_data.items()):
        if 'val_loss' in df.columns:
            ax2.plot(df['epoch'], df['val_loss'], 
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training MSE (RGB)
    ax3 = axes[1, 0]
    for i, (exp_name, df) in enumerate(all_data.items()):
        mse_col = [c for c in df.columns if 'MSE' in c.upper() and 'train' in c.lower()]
        if mse_col:
            ax3.plot(df['epoch'], df[mse_col[0]], 
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training MSE (RGB)', fontsize=12)
    ax3.set_title('Training RGB MSE', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation MSE (RGB)
    ax4 = axes[1, 1]
    for i, (exp_name, df) in enumerate(all_data.items()):
        mse_col = [c for c in df.columns if 'MSE' in c.upper() and 'val' in c.lower()]
        if mse_col:
            ax4.plot(df['epoch'], df[mse_col[0]], 
                    label=exp_name, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation MSE (RGB)', fontsize=12)
    ax4.set_title('Validation RGB MSE', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
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
    sns.barplot(data=final_df, x='experiment', y='val_loss', ax=ax1, palette=PASTEL_COLORS)
    ax1.set_xlabel('Experiment', fontsize=12)
    ax1.set_ylabel('Final Validation Loss', fontsize=12)
    ax1.set_title('Final Validation Loss', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Final Validation MSE
    ax2 = axes[0, 1]
    if 'val_MSE' in final_df.columns:
        sns.barplot(data=final_df, x='experiment', y='val_MSE', ax=ax2, palette=PASTEL_COLORS)
        ax2.set_xlabel('Experiment', fontsize=12)
        ax2.set_ylabel('Final Validation MSE', fontsize=12)
        ax2.set_title('Final Validation MSE (RGB)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'MSE data not available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Final Validation MSE (RGB)', fontsize=14, fontweight='bold')
    
    # Plot 3: Final Training Loss
    ax3 = axes[1, 0]
    sns.barplot(data=final_df, x='experiment', y='train_loss', ax=ax3, palette=PASTEL_COLORS)
    ax3.set_xlabel('Experiment', fontsize=12)
    ax3.set_ylabel('Final Training Loss', fontsize=12)
    ax3.set_title('Final Training Loss', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Final Validation Segmentation Loss
    ax4 = axes[1, 1]
    if 'val_seg_loss' in final_df.columns:
        sns.barplot(data=final_df, x='experiment', y='val_seg_loss', ax=ax4, palette=PASTEL_COLORS)
        ax4.set_xlabel('Experiment', fontsize=12)
        ax4.set_ylabel('Final Validation Seg Loss', fontsize=12)
        ax4.set_title('Final Validation Segmentation Loss', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Segmentation loss data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Final Validation Segmentation Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "final_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
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
                    color=PASTEL_COLORS[len(convergence_data) % len(PASTEL_COLORS)], 
                    linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Loss Reduction Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot: Convergence speed
    conv_df = pd.DataFrame(convergence_data)
    if len(conv_df) > 0:
        sns.barplot(data=conv_df, x='experiment', y='convergence_epoch', 
                   ax=axes[1], palette=PASTEL_COLORS)
        axes[1].set_xlabel('Experiment', fontsize=12)
        axes[1].set_ylabel('Convergence Epoch', fontsize=12)
        axes[1].set_title('Convergence Speed', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, ha='right')
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
    scene_comparison = create_comparison_grid(
        scene_rgb.cpu(),
        scene_reconstructions,
        title="Scene Reconstruction Comparison"
    )
    scene_path = output_dir / "visual_comparison_scene.png"
    save_image(scene_comparison, scene_path, normalize=False)
    print(f"  Saved: {scene_path}")
    
    # Room comparison
    room_comparison = create_comparison_grid(
        room_rgb.cpu(),
        room_reconstructions,
        title="Room Reconstruction Comparison"
    )
    room_path = output_dir / "visual_comparison_room.png"
    save_image(room_comparison, room_path, normalize=False)
    print(f"  Saved: {room_path}")


def create_comparison_grid(original, reconstructions, title=""):
    """
    Create a grid showing original and all reconstructions.
    
    Args:
        original: [1, 3, H, W] tensor
        reconstructions: dict of {exp_name: [1, 3, H, W] tensor}
        title: Optional title
    
    Returns:
        Grid tensor ready for saving
    """
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
    
    # Create row: [Original | Exp1 | Exp2 | ... | ExpN]
    exp_names = sorted(reconstructions.keys())
    images_to_grid = [original.squeeze(0)]  # Original first
    images_to_grid.extend([reconstructions[name].squeeze(0) for name in exp_names])
    
    # Stack horizontally (all in one row)
    grid = torch.stack(images_to_grid)
    grid_image = make_grid(grid, nrow=len(images_to_grid), padding=4, normalize=False)
    
    return grid_image


def create_summary_report(all_data, output_dir):
    """Create a text summary report."""
    report_path = output_dir / "analysis_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 1.1: Latent Channel Sweep - Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Best experiment by validation loss
        best_val_loss = float('inf')
        best_exp = None
        
        for exp_name, df in all_data.items():
            if 'val_loss' in df.columns:
                final_val_loss = df['val_loss'].iloc[-1]
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_exp = exp_name
        
        if best_exp:
            f.write(f"Best Experiment (by Final Validation Loss): {best_exp}\n")
            f.write(f"  Final Validation Loss: {best_val_loss:.6f}\n\n")
        
        # Summary for each experiment
        f.write("-" * 80 + "\n")
        f.write("Experiment Summaries:\n")
        f.write("-" * 80 + "\n\n")
        
        for exp_name, df in all_data.items():
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"  Total Epochs: {len(df)}\n")
            
            if 'train_loss' in df.columns:
                initial_train = df['train_loss'].iloc[0]
                final_train = df['train_loss'].iloc[-1]
                f.write(f"  Training Loss: {initial_train:.6f} → {final_train:.6f}\n")
            
            if 'val_loss' in df.columns:
                initial_val = df['val_loss'].iloc[0]
                final_val = df['val_loss'].iloc[-1]
                f.write(f"  Validation Loss: {initial_val:.6f} → {final_val:.6f}\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("Recommendations:\n")
        f.write("-" * 80 + "\n\n")
        f.write("1. Select top 2 experiments based on:\n")
        f.write("   - Lowest final validation loss\n")
        f.write("   - Stable convergence\n")
        f.write("   - Reasonable model size\n\n")
        f.write("2. Update Phase 1.2 configs with:\n")
        f.write("   - Best latent_channels\n")
        f.write("   - Best base_channels\n\n")
    
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

