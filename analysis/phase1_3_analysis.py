#!/usr/bin/env python3
"""
Analysis script for Phase 1.3: Loss Tuning
Loads metrics from all experiments and creates comparison visualizations.
Includes UMAP visualization for latent space structure analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

# Import from phase1_1_analysis (reuse functions)
sys.path.insert(0, str(Path(__file__).parent))
from phase1_1_analysis import (
    load_metrics, create_loss_curves, create_final_metrics_comparison,
    create_convergence_analysis, create_summary_report,
    load_autoencoder_checkpoints, get_test_samples, create_visual_comparison
)

# Import for config loading
from training.utils import load_config
import yaml

# Set seaborn style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Try to import UMAP (optional dependency)
UMAP_AVAILABLE = False
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap
        UMAP = umap.UMAP
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("Warning: UMAP not available. Install with: pip install umap-learn")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Phase 1.3 experiments")
    parser.add_argument(
        "--phase-dir",
        type=str,
        default="outputs/phase1_3_loss_tuning",
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
    
    # Determine config directory
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        # Try to infer from experiments_dir
        base_dir = Path(args.experiments_dir).resolve()
        # Go up to find experiments/autoencoders/phase1
        config_dir = base_dir.parent / "experiments" / "autoencoders" / "phase1"
        if not config_dir.exists():
            config_dir = base_dir.parent.parent / "experiments" / "autoencoders" / "phase1"
    
    # Determine output directory
    phase_dir = Path(args.phase_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = phase_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 1.3: Loss Tuning Analysis")
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
    
    # Phase 1.3 specific: Loss component analysis
    print("Creating loss component analysis...")
    create_loss_component_breakdown(all_data, output_dir, config_dir)
    create_multitask_analysis(all_data, output_dir, config_dir)
    create_loss_interaction_analysis(all_data, output_dir, config_dir)
    
    # Phase 1.3 specific: Two-dimensional analysis (Loss Config × Encoder Type)
    print("Creating two-dimensional analysis (Loss Config × Encoder Type)...")
    create_encoder_type_comparison(all_data, output_dir, config_dir)
    create_loss_vs_encoder_grid(all_data, output_dir, config_dir)
    
    
    create_summary_report(all_data, output_dir)
    
    # Visual comparison with actual models
    if not args.skip_visual:
        print("\n" + "-" * 80)
        print("Loading autoencoder checkpoints for visual comparison...")
        print("-" * 80)
        checkpoints = load_autoencoder_checkpoints(exp_names, args.experiments_dir)
        
        if checkpoints:
            print("\nLoading test samples from dataset...")
            test_scenes, test_rooms = get_test_samples(args.dataset_manifest, test_samples_per_type=10)
            
            if test_scenes and test_rooms:
                print("\nCreating visual comparisons...")
                create_visual_comparison(checkpoints, test_scenes, test_rooms, output_dir)
                
                # Phase 1.3 specific: UMAP latent space visualization
                if UMAP_AVAILABLE:
                    print("\nCreating UMAP latent space visualizations...")
                    create_umap_visualization(checkpoints, args.dataset_manifest, output_dir, 
                                              n_samples_per_type=50)
                else:
                    print("  Skipping UMAP visualization (not installed)")
            else:
                print("  Could not find test samples, skipping visual comparison")
        else:
            print("  Could not load any checkpoints, skipping visual comparison")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


def create_loss_component_breakdown(all_data, output_dir, config_dir=None):
    """Break down total loss into individual components (MSE, Seg, Cls)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 1.3: Loss Component Breakdown', fontsize=16, fontweight='bold', y=0.995)
    
    exp_names = sorted(all_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Plot 1: Final Loss Components Comparison
    ax = axes[0, 0]
    final_mse = []
    final_seg = []
    final_cls = []
    exp_labels = []
    
    for exp_name in exp_names:
        df = all_data[exp_name]
        final = df.iloc[-1]
        # Use meaningful display name
        exp_labels.append(get_display_name(exp_name, config_dir))
        
        final_mse.append(final.get('val_MSE_rgb', final.get('train_MSE_rgb', 0)))
        
        seg_cols = [c for c in final.index if 'CE_segmentation' in c or 'seg' in c.lower()]
        if seg_cols:
            final_seg.append(final[seg_cols[0]])
        else:
            final_seg.append(0)
        
        cls_cols = [c for c in final.index if 'CE_classification' in c or ('cls' in c.lower() and 'seg' not in c.lower())]
        if cls_cols:
            final_cls.append(final[cls_cols[0]])
        else:
            final_cls.append(0)
    
    x = np.arange(len(exp_names))
    width = 0.25
    
    ax.bar(x - width, final_mse, width, label='MSE (RGB)', color=colors[0], alpha=0.8)
    ax.bar(x, final_seg, width, label='Segmentation', color=colors[1], alpha=0.8)
    ax.bar(x + width, final_cls, width, label='Classification', color=colors[2], alpha=0.8)
    
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title('Final Loss Components', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='x', pad=10)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Loss Component Trends (Training)
    ax = axes[0, 1]
    for i, exp_name in enumerate(exp_names):
        df = all_data[exp_name]
        mse_cols = [c for c in df.columns if 'MSE' in c.upper() and 'train' in c.lower()]
        if mse_cols:
            ax.plot(df['epoch'], df[mse_cols[0]], 
                   label=f"{exp_labels[i]} - MSE", linewidth=2, color=colors[0], alpha=0.7)
        
        seg_cols = [c for c in df.columns if ('CE_segmentation' in c or 'seg' in c.lower()) and 'train' in c.lower()]
        if seg_cols:
            ax.plot(df['epoch'], df[seg_cols[0]], 
                   label=f"{exp_labels[i]} - Seg", linewidth=2, color=colors[1], alpha=0.7, linestyle='--')
        
        cls_cols = [c for c in df.columns if ('CE_classification' in c or ('cls' in c.lower() and 'seg' not in c.lower())) and 'train' in c.lower()]
        if cls_cols:
            ax.plot(df['epoch'], df[cls_cols[0]], 
                   label=f"{exp_labels[i]} - Cls", linewidth=2, color=colors[2], alpha=0.7, linestyle=':')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title('Loss Component Trends (Training)', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=False, shadow=False, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss Component Ratios
    ax = axes[1, 0]
    ratios_data = []
    for exp_name in exp_names:
        df = all_data[exp_name]
        final = df.iloc[-1]
        
        mse_val = final.get('val_MSE_rgb', final.get('train_MSE_rgb', 1))
        seg_cols = [c for c in final.index if 'CE_segmentation' in c or 'seg' in c.lower()]
        seg_val = final[seg_cols[0]] if seg_cols else 0
        cls_cols = [c for c in final.index if 'CE_classification' in c or ('cls' in c.lower() and 'seg' not in c.lower())]
        cls_val = final[cls_cols[0]] if cls_cols else 0
        
        total = mse_val + seg_val + cls_val
        if total > 0:
            ratios_data.append({
                'Experiment': exp_labels[exp_names.index(exp_name)],
                'MSE %': (mse_val / total) * 100,
                'Seg %': (seg_val / total) * 100,
                'Cls %': (cls_val / total) * 100
            })
    
    if ratios_data:
        df_ratios = pd.DataFrame(ratios_data)
        x = np.arange(len(df_ratios))
        width = 0.6
        
        bottom_mse = np.zeros(len(df_ratios))
        bottom_seg = df_ratios['MSE %'].values
        bottom_cls = bottom_seg + df_ratios['Seg %'].values
        
        ax.bar(x, df_ratios['MSE %'], width, label='MSE', color=colors[0], alpha=0.8)
        ax.bar(x, df_ratios['Seg %'], width, bottom=bottom_seg, label='Seg', color=colors[1], alpha=0.8)
        ax.bar(x, df_ratios['Cls %'], width, bottom=bottom_cls, label='Cls', color=colors[2], alpha=0.8)
        
    ax.set_ylabel('Percentage of Total Loss', fontsize=12)
    ax.set_title('Loss Component Ratios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_ratios['Experiment'], rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='x', pad=10)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Task Performance Summary
    ax = axes[1, 1]
    task_perf = []
    for exp_name in exp_names:
        df = all_data[exp_name]
        final = df.iloc[-1]
        label = exp_labels[exp_names.index(exp_name)]
        
        # RGB quality (MSE - lower is better, so invert)
        mse_val = final.get('val_MSE_rgb', final.get('train_MSE_rgb', 1))
        rgb_score = 1 / (1 + mse_val * 100)  # Normalize to [0, 1]
        
        # Segmentation performance (if available)
        seg_cols = [c for c in df.columns if ('CE_segmentation' in c or 'seg' in c.lower()) and 'val' in c.lower()]
        if seg_cols:
            seg_val = final[seg_cols[0]]
            seg_score = 1 / (1 + seg_val * 10)
        else:
            seg_score = 0
        
        # Classification performance (if available)
        cls_cols = [c for c in df.columns if ('CE_classification' in c or ('cls' in c.lower() and 'seg' not in c.lower())) and 'val' in c.lower()]
        if cls_cols:
            cls_val = final[cls_cols[0]]
            cls_score = 1 / (1 + cls_val * 10)
        else:
            cls_score = 0
        
        task_perf.append({
            'Experiment': label,
            'RGB Quality': rgb_score,
            'Segmentation': seg_score,
            'Classification': cls_score
        })
    
    if task_perf:
        df_task = pd.DataFrame(task_perf)
        x = np.arange(len(df_task))
        width = 0.25
        
        ax.bar(x - width, df_task['RGB Quality'], width, label='RGB Quality', color=colors[0], alpha=0.8)
        ax.bar(x, df_task['Segmentation'], width, label='Segmentation', color=colors[1], alpha=0.8)
        ax.bar(x + width, df_task['Classification'], width, label='Classification', color=colors[2], alpha=0.8)
        
        ax.set_ylabel('Normalized Performance', fontsize=12)
        ax.set_title('Task Performance Summary', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_task['Experiment'], rotation=45, ha='right', fontsize=9)
        ax.tick_params(axis='x', pad=10)
        ax.legend(frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
    output_path = output_dir / "loss_component_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_multitask_analysis(all_data, output_dir, config_dir=None):
    """Analyze multi-task interactions and trade-offs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Phase 1.3: Multi-Task Interaction Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    exp_names = sorted(all_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Extract final metrics
    metrics_data = []
    for exp_name in exp_names:
        df = all_data[exp_name]
        final = df.iloc[-1]
        
        mse_val = final.get('val_MSE_rgb', final.get('train_MSE_rgb', 0))
        seg_cols = [c for c in final.index if 'CE_segmentation' in c or 'seg' in c.lower()]
        seg_val = final[seg_cols[0]] if seg_cols else None
        cls_cols = [c for c in final.index if 'CE_classification' in c or ('cls' in c.lower() and 'seg' not in c.lower())]
        cls_val = final[cls_cols[0]] if cls_cols else None
        
        # Use meaningful display name
        display_name = get_display_name(exp_name, config_dir)
        metrics_data.append({
            'exp': display_name,
            'mse': mse_val,
            'seg': seg_val,
            'cls': cls_val
        })
    
    # Plot 1: MSE vs Auxiliary Tasks
    ax = axes[0]
    for i, m in enumerate(metrics_data):
        label = m['exp']  # Already formatted display name
        has_seg = m['seg'] is not None
        has_cls = m['cls'] is not None
        
        if has_cls:
            marker = 'o'
            color = colors[2]
        elif has_seg:
            marker = 's'
            color = colors[1]
        else:
            marker = '^'
            color = colors[0]
        
        ax.scatter(i, m['mse'], s=200, marker=marker, color=color, alpha=0.7, edgecolors='black', linewidth=2)
    
    ax.set_ylabel('Validation MSE (RGB)', fontsize=12)
    ax.set_title('RGB Quality by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(metrics_data)))
    ax.set_xticklabels([m['exp'] for m in metrics_data], rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='x', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=colors[0], markersize=10, label='RGB Only'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markersize=10, label='RGB + Seg'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=10, label='RGB + Seg + Cls')
    ]
    ax.legend(handles=legend_elements, frameon=True, fancybox=False, shadow=False)
    
    # Plot 2: Auxiliary Task Performance
    ax = axes[1]
    seg_values = [m['seg'] for m in metrics_data if m['seg'] is not None]
    cls_values = [m['cls'] for m in metrics_data if m['cls'] is not None]
    
    if seg_values:
        seg_indices = [i for i, m in enumerate(metrics_data) if m['seg'] is not None]
        ax.scatter(seg_indices, seg_values, s=200, marker='s', color=colors[1], alpha=0.7, 
                  edgecolors='black', linewidth=2, label='Segmentation Loss')
    
    if cls_values:
        cls_indices = [i for i, m in enumerate(metrics_data) if m['cls'] is not None]
        ax.scatter(cls_indices, cls_values, s=200, marker='o', color=colors[2], alpha=0.7,
                  edgecolors='black', linewidth=2, label='Classification Loss')
    
    ax.set_ylabel('Auxiliary Task Loss', fontsize=12)
    ax.set_title('Auxiliary Task Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(metrics_data)))
    ax.set_xticklabels([m['exp'] for m in metrics_data], rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='x', pad=10)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Trade-off Analysis (MSE change vs auxiliary tasks)
    ax = axes[2]
    baseline_mse = metrics_data[0]['mse']  # Assume F1 (RGB only) is first
    
    mse_changes = []
    task_counts = []
    labels = []
    
    for m in metrics_data:
        mse_change = ((m['mse'] - baseline_mse) / baseline_mse) * 100 if baseline_mse > 0 else 0
        num_tasks = (1 if m['seg'] is not None else 0) + (1 if m['cls'] is not None else 0)
        
        mse_changes.append(mse_change)
        task_counts.append(num_tasks)
        labels.append(m['exp'].replace('_', ' ').title())
    
    scatter = ax.scatter(task_counts, mse_changes, s=300, c=range(len(task_counts)), 
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, (x, y, label) in enumerate(zip(task_counts, mse_changes, labels)):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Auxiliary Tasks', fontsize=12)
    ax.set_ylabel('MSE Change vs Baseline (%)', fontsize=12)
    ax.set_title('Multi-Task Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
    output_path = output_dir / "multitask_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_loss_interaction_analysis(all_data, output_dir, config_dir=None):
    """Analyze if auxiliary tasks help or hurt RGB reconstruction."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    exp_names = sorted(all_data.keys())
    
    # Get baseline (RGB only)
    baseline_exp = None
    for exp_name in exp_names:
        loss_config, _, _ = parse_experiment_name(exp_name, config_dir)
        if 'F1' in loss_config or 'RGB only' in loss_config:
            baseline_exp = exp_name
            break
    
    if not baseline_exp:
        baseline_exp = exp_names[0]  # Assume first is baseline
    
    baseline_df = all_data[baseline_exp]
    baseline_final_mse = baseline_df.iloc[-1].get('val_MSE_rgb', baseline_df.iloc[-1].get('train_MSE_rgb', 0))
    
    comparisons = []
    for exp_name in exp_names:
        if exp_name == baseline_exp:
            continue
        
        df = all_data[exp_name]
        final_mse = df.iloc[-1].get('val_MSE_rgb', df.iloc[-1].get('train_MSE_rgb', 0))
        
        mse_diff = ((final_mse - baseline_final_mse) / baseline_final_mse) * 100
        # Use meaningful display name
        exp_label = get_display_name(exp_name, config_dir)
        
        # Determine auxiliary tasks from parsed config
        _, _, loss_weights = parse_experiment_name(exp_name, config_dir)
        if loss_weights.get('cls', 0) > 0:
            aux_tasks = 'Seg+Cls'
        elif loss_weights.get('seg', 0) > 0:
            aux_tasks = 'Seg'
        else:
            aux_tasks = 'None'
        
        comparisons.append({
            'Experiment': exp_label,
            'MSE Change (%)': mse_diff,
            'Auxiliary Tasks': aux_tasks
        })
    
    if comparisons:
        df_comp = pd.DataFrame(comparisons)
        colors_map = {'Seg': '#A23B72', 'Seg+Cls': '#F18F01', 'None': '#2E86AB'}
        bar_colors = [colors_map.get(comp['Auxiliary Tasks'], '#666666') for comp in comparisons]
        
        bars = ax.barh(df_comp['Experiment'], df_comp['MSE Change (%)'], color=bar_colors, alpha=0.7, edgecolor='black')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('MSE Change vs RGB-Only Baseline (%)', fontsize=12)
        ax.set_title('Impact of Auxiliary Tasks on RGB Quality', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars, df_comp['MSE Change (%)'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{val:+.2f}%',
                   ha='left' if width > 0 else 'right', va='center', fontsize=10, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors_map[k], label=k) for k in colors_map.keys()]
        ax.legend(handles=legend_elements, frameon=True, fancybox=False, shadow=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
    output_path = output_dir / "loss_interaction_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_umap_visualization(checkpoints, dataset_manifest, output_dir, n_samples_per_type=50):
    """Create UMAP visualization of latent space for each experiment."""
    if not UMAP_AVAILABLE:
        print("  UMAP not available, skipping latent space visualization")
        return
    
    if not checkpoints:
        print("  No checkpoints available for UMAP visualization")
        return
    
    # Load dataset
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.datasets.datasets import ManifestDataset
    
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
    
    # Sample balanced dataset
    scenes = []
    rooms = []
    room_types = {}  # Track room types if available
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            label = sample.get('label', None)
            
            if isinstance(label, str):
                label_lower = label.lower().strip()
                if label_lower == 'scene' and len(scenes) < n_samples_per_type:
                    scenes.append((i, sample))
                elif label_lower == 'room' and len(rooms) < n_samples_per_type:
                    rooms.append((i, sample))
                    # Try to get room type if available
                    # (You might need to adjust this based on your dataset structure)
            
            if len(scenes) >= n_samples_per_type and len(rooms) >= n_samples_per_type:
                break
        except Exception as e:
            continue
    
    all_samples = scenes + rooms
    if len(all_samples) < 10:
        print(f"  Only found {len(all_samples)} samples, need at least 10 for UMAP")
        return
    
    print(f"  Using {len(all_samples)} samples for UMAP visualization ({len(scenes)} scenes, {len(rooms)} rooms)")
    
    # Create UMAP visualization for each experiment
    device = list(checkpoints.values())[0]['device']
    
    for exp_name, checkpoint_info in checkpoints.items():
        model = checkpoint_info['model']
        model.eval()
        
        # Extract latents
        latents_list = []
        labels_list = []
        sample_types = []
        
        with torch.no_grad():
            for idx, sample in all_samples:
                rgb_input = sample['rgb'].unsqueeze(0).to(device)
                
                # Encode to latent
                encoder_out = model.encode(rgb_input)
                if 'latent' in encoder_out:
                    latent = encoder_out['latent']
                elif 'mu' in encoder_out:
                    latent = encoder_out['mu']  # Use mu for VAE
                else:
                    continue
                
                # Flatten latent
                latent_flat = latent.cpu().view(-1).numpy()
                latents_list.append(latent_flat)
                
                # Store label
                label = sample.get('label', 'unknown')
                labels_list.append(label)
                
                if isinstance(label, str):
                    sample_types.append(0 if label.lower() == 'room' else 1)  # 0=room, 1=scene
                else:
                    sample_types.append(0)
        
        if len(latents_list) < 10:
            print(f"  Skipping {exp_name}: not enough valid latents")
            continue
        
        latents_array = np.array(latents_list)
        print(f"  Computing UMAP for {exp_name} (latent shape: {latents_array.shape})...")
        
        # Fit UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(latents_array)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Phase 1.3: UMAP Latent Space - {exp_name.replace("phase1_3_AE_", "")}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Room vs Scene separation
        ax = axes[0]
        room_mask = np.array(sample_types) == 0
        scene_mask = np.array(sample_types) == 1
        
        ax.scatter(embedding[room_mask, 0], embedding[room_mask, 1], 
                  c='#2E86AB', label='Room', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.scatter(embedding[scene_mask, 0], embedding[scene_mask, 1], 
                  c='#A23B72', label='Scene', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title('Room vs Scene Separation', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: All samples (colored by type)
        ax = axes[1]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=sample_types, cmap='coolwarm', alpha=0.6, s=50,
                           edgecolors='black', linewidth=0.5)
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title('Latent Space Structure', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Type (0=Room, 1=Scene)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Calculate separation metric (simple distance-based)
        room_center = embedding[room_mask].mean(axis=0) if room_mask.sum() > 0 else None
        scene_center = embedding[scene_mask].mean(axis=0) if scene_mask.sum() > 0 else None
        
        if room_center is not None and scene_center is not None:
            separation = np.linalg.norm(room_center - scene_center)
            ax.text(0.02, 0.98, f'Center Separation: {separation:.3f}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
        output_path = output_dir / f"umap_{exp_name.replace('phase1_3_AE_', '')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_path}")
    
    # Create comparison plot (all experiments side-by-side)
    if len(checkpoints) > 1:
        fig, axes = plt.subplots(1, len(checkpoints), figsize=(6 * len(checkpoints), 6))
        if len(checkpoints) == 1:
            axes = [axes]
        
        fig.suptitle('Phase 1.3: UMAP Comparison - Room vs Scene Separation', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for ax_idx, (exp_name, checkpoint_info) in enumerate(sorted(checkpoints.items())):
            model = checkpoint_info['model']
            model.eval()
            
            latents_list = []
            sample_types = []
            
            with torch.no_grad():
                for idx, sample in all_samples:
                    rgb_input = sample['rgb'].unsqueeze(0).to(device)
                    encoder_out = model.encode(rgb_input)
                    if 'latent' in encoder_out:
                        latent = encoder_out['latent']
                    elif 'mu' in encoder_out:
                        latent = encoder_out['mu']
                    else:
                        continue
                    latent_flat = latent.cpu().view(-1).numpy()
                    latents_list.append(latent_flat)
                    
                    label = sample.get('label', 'unknown')
                    if isinstance(label, str):
                        sample_types.append(0 if label.lower() == 'room' else 1)
                    else:
                        sample_types.append(0)
            
            if len(latents_list) < 10:
                continue
            
            latents_array = np.array(latents_list)
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embedding = reducer.fit_transform(latents_array)
            
            room_mask = np.array(sample_types) == 0
            scene_mask = np.array(sample_types) == 1
            
            ax = axes[ax_idx]
            ax.scatter(embedding[room_mask, 0], embedding[room_mask, 1], 
                      c='#2E86AB', label='Room', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
            ax.scatter(embedding[scene_mask, 0], embedding[scene_mask, 1], 
                      c='#A23B72', label='Scene', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
            
            ax.set_title(exp_name.replace('phase1_3_AE_', '').replace('_', ' '), 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=10)
            ax.set_ylabel('UMAP 2', fontsize=10)
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(frameon=True, fancybox=False, shadow=False, fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
        output_path = output_dir / "umap_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_path}")


def load_experiment_config(exp_name, config_dir):
    """Load config file for an experiment to extract meaningful metadata."""
    if config_dir is None:
        return None
        
    # Try to find config file
    config_paths = [
        Path(config_dir) / f"{exp_name}.yaml",
        Path(config_dir).parent / "phase1" / f"{exp_name}.yaml",
    ]
    
    config = None
    for config_path in config_paths:
        if config_path.exists():
            try:
                config = load_config(config_path)
                break
            except Exception as e:
                continue
    
    return config


def parse_experiment_name(exp_name, config_dir=None):
    """Parse experiment name to extract loss config and encoder type from config if available, otherwise from filename."""
    # Try to load config first
    config = None
    if config_dir:
        config = load_experiment_config(exp_name, config_dir)
    
    encoder_type = 'Deterministic'
    loss_config = 'Unknown'
    loss_weights = {}
    
    if config:
        # Extract encoder type
        variational = config.get('autoencoder', {}).get('encoder', {}).get('variational', False)
        encoder_type = 'VAE' if variational else 'Deterministic'
        
        # Extract loss weights
        loss_config_obj = config.get('training', {}).get('loss', {})
        losses = loss_config_obj.get('losses', [])
        
        mse_weight = 0.0
        seg_weight = 0.0
        cls_weight = 0.0
        kld_weight = 0.0
        
        for loss in losses:
            loss_type = loss.get('type', '').lower()
            weight = loss.get('weight', 0.0)
            
            if 'mse' in loss_type:
                mse_weight = weight
            elif 'kld' in loss_type or 'kl' in loss_type:
                kld_weight = weight
            elif 'crossentropy' in loss_type or 'ce' in loss_type:
                if 'segmentation' in loss.get('key', '').lower():
                    seg_weight = weight
                elif 'classification' in loss.get('key', '').lower():
                    cls_weight = weight
        
        loss_weights = {
            'mse': mse_weight,
            'seg': seg_weight,
            'cls': cls_weight,
            'kld': kld_weight
        }
        
        # Create meaningful loss config name
        if seg_weight == 0 and cls_weight == 0:
            loss_config = 'F1 (RGB only)'
        elif cls_weight == 0:
            loss_config = f'F2 (RGB+Seg, w={seg_weight:.2f})'
        else:
            loss_config = f'F3 (RGB+Seg+Cls, w_seg={seg_weight:.2f}, w_cls={cls_weight:.2f})'
    else:
        # Fallback to filename parsing
        exp_name_clean = exp_name.replace('phase1_3_AE_', '').lower()
        
        # Determine encoder type
        is_vae = 'vae' in exp_name_clean
        encoder_type = 'VAE' if is_vae else 'Deterministic'
        
        # Determine loss config
        if 'f1' in exp_name_clean or 'rgb_only' in exp_name_clean:
            loss_config = 'F1 (RGB only)'
        elif 'f2' in exp_name_clean or ('rgb' in exp_name_clean and 'seg' in exp_name_clean and 'cls' not in exp_name_clean):
            loss_config = 'F2 (RGB+Seg)'
        elif 'f3' in exp_name_clean or ('rgb' in exp_name_clean and 'seg' in exp_name_clean and 'cls' in exp_name_clean):
            loss_config = 'F3 (RGB+Seg+Cls)'
    
    return loss_config, encoder_type, loss_weights


def get_display_name(exp_name, config_dir=None):
    """Get a short, meaningful display name for an experiment."""
    loss_config, encoder_type, loss_weights = parse_experiment_name(exp_name, config_dir)
    
    # Extract F1, F2, or F3 from loss_config
    if 'F1' in loss_config or 'RGB only' in loss_config:
        config_short = 'F1'
    elif 'F2' in loss_config:
        config_short = 'F2'
    elif 'F3' in loss_config:
        config_short = 'F3'
    else:
        config_short = loss_config.split()[0] if loss_config else 'Unknown'
    
    # Use short labels for encoder type: "AE" for deterministic, "V" for VAE
    encoder_short = 'V' if encoder_type == 'VAE' else 'AE'
    
    # Short format: "F1-AE", "F2-V", etc.
    return f"{config_short}-{encoder_short}"


def create_encoder_type_comparison(all_data, output_dir, config_dir=None):
    """Compare Deterministic vs VAE for each loss configuration."""
    # Group experiments by loss config and encoder type
    grouped = {}
    for exp_name, df in all_data.items():
        loss_config, encoder_type, _ = parse_experiment_name(exp_name, config_dir)
        # Normalize encoder type: use "AE" instead of "Deterministic"
        if encoder_type == 'Deterministic':
            encoder_type = 'AE'
        key = loss_config
        if key not in grouped:
            grouped[key] = {}
        grouped[key][encoder_type] = {'name': exp_name, 'df': df, 'display_name': get_display_name(exp_name, config_dir)}
    
    if not grouped:
        print("  Could not parse experiment names, skipping encoder type comparison")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Phase 1.3: Encoder Type Comparison (AE vs VAE)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    loss_configs = ['F1 (RGB only)', 'F2 (RGB+Seg)', 'F3 (RGB+Seg+Cls)']
    
    for row_idx, loss_config in enumerate(loss_configs):
        if loss_config not in grouped:
            continue
        
        group = grouped[loss_config]
        det_data = group.get('AE') or group.get('Deterministic')  # Support both for backward compatibility
        vae_data = group.get('VAE')
        
        # Row 1: Loss curves comparison
        ax = axes[0, row_idx]
        if det_data and vae_data:
            det_df = det_data['df']
            vae_df = vae_data['df']
            
            # Plot MSE
            if 'val_MSE_rgb' in det_df.columns and 'val_MSE_rgb' in vae_df.columns:
                ax.plot(det_df['epoch'], det_df['val_MSE_rgb'], 
                       label='AE', linewidth=2.5, color='#2E86AB', linestyle='-')
                ax.plot(vae_df['epoch'], vae_df['val_MSE_rgb'], 
                       label='VAE', linewidth=2.5, color='#A23B72', linestyle='--')
            elif 'train_MSE_rgb' in det_df.columns:
                ax.plot(det_df['epoch'], det_df.get('val_MSE_rgb', det_df['train_MSE_rgb']), 
                       label='AE', linewidth=2.5, color='#2E86AB', linestyle='-')
                ax.plot(vae_df['epoch'], vae_df.get('val_MSE_rgb', vae_df['train_MSE_rgb']), 
                       label='VAE', linewidth=2.5, color='#A23B72', linestyle='--')
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('MSE (RGB)', fontsize=11)
            ax.set_title(f'{loss_config}', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=False, shadow=False)
            ax.grid(True, alpha=0.3)
        
        # Row 2: Final metrics comparison
        ax = axes[1, row_idx]
        if det_data and vae_data:
            det_final = det_data['df'].iloc[-1]
            vae_final = vae_data['df'].iloc[-1]
            
            metrics = ['MSE', 'Total Loss']
            det_values = [
                det_final.get('val_MSE_rgb', det_final.get('train_MSE_rgb', 0)),
                det_final.get('val_loss', det_final.get('train_loss', 0))
            ]
            vae_values = [
                vae_final.get('val_MSE_rgb', vae_final.get('train_MSE_rgb', 0)),
                vae_final.get('val_loss', vae_final.get('train_loss', 0))
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            ax.bar(x - width/2, det_values, width, label='AE', 
                  color='#2E86AB', alpha=0.8)
            ax.bar(x + width/2, vae_values, width, label='VAE', 
                  color='#A23B72', alpha=0.8)
            
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.set_title(f'{loss_config} - Final Metrics', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=10)
            ax.legend(frameon=True, fancybox=False, shadow=False)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
    output_path = output_dir / "encoder_type_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_loss_vs_encoder_grid(all_data, output_dir, config_dir=None):
    """Create a grid visualization showing both dimensions: Loss Config × Encoder Type."""
    # Organize data into grid - collect all unique loss configs from parsed names
    loss_configs_set = set()
    grid_data = {}
    
    for exp_name, df in all_data.items():
        loss_config, encoder_type, _ = parse_experiment_name(exp_name, config_dir)
        # Normalize encoder type: use "AE" instead of "Deterministic"
        if encoder_type == 'Deterministic':
            encoder_type = 'AE'
        loss_configs_set.add(loss_config)
        
        final = df.iloc[-1]
        
        key = (loss_config, encoder_type)
        grid_data[key] = {
            'exp_name': exp_name,
            'display_name': get_display_name(exp_name, config_dir),
            'mse': final.get('val_MSE_rgb', final.get('train_MSE_rgb', 0)),
            'total_loss': final.get('val_loss', final.get('train_loss', 0))
        }
    
    # Sort loss configs in logical order
    loss_configs = []
    for cfg in ['F1 (RGB only)', 'F2 (RGB+Seg', 'F3 (RGB+Seg+Cls']:
        matching = [c for c in loss_configs_set if c.startswith(cfg)]
        loss_configs.extend(sorted(matching))
    loss_configs.extend([c for c in loss_configs_set if c not in loss_configs])
    
    encoder_types = ['AE', 'VAE']
    
    # Create heatmap-style grid
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Phase 1.3: Two-Dimensional Analysis (Loss Config × Encoder Type)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: MSE heatmap
    ax = axes[0]
    mse_grid = np.zeros((len(encoder_types), len(loss_configs)))
    for i, encoder_type in enumerate(encoder_types):
        for j, loss_config in enumerate(loss_configs):
            key = (loss_config, encoder_type)
            if key in grid_data:
                mse_grid[i, j] = grid_data[key]['mse']
            else:
                mse_grid[i, j] = np.nan
    
    im = ax.imshow(mse_grid, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(loss_configs)))
    ax.set_yticks(np.arange(len(encoder_types)))
    ax.set_xticklabels([lc.replace(' (', '\n(') for lc in loss_configs], fontsize=10)
    ax.set_yticklabels(encoder_types, fontsize=11)
    ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Encoder Type', fontsize=12, fontweight='bold')
    ax.set_title('Final MSE (RGB) Heatmap', fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(len(encoder_types)):
        for j in range(len(loss_configs)):
            key = (loss_configs[j], encoder_types[i])
            if key in grid_data:
                text = ax.text(j, i, f'{mse_grid[i, j]:.4f}',
                            ha="center", va="center", color="white", fontweight='bold', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='MSE Value')
    
    # Plot 2: Bar chart comparing across dimensions
    ax = axes[1]
    x_pos = np.arange(len(loss_configs))
    width = 0.35
    
    det_values = []
    vae_values = []
    for loss_config in loss_configs:
        # Try both 'AE' and 'Deterministic' for backward compatibility
        det_key = (loss_config, 'AE')
        det_key_alt = (loss_config, 'Deterministic')
        vae_key = (loss_config, 'VAE')
        det_values.append(grid_data.get(det_key, grid_data.get(det_key_alt, {})).get('mse', 0))
        vae_values.append(grid_data.get(vae_key, {}).get('mse', 0))
    
    bars1 = ax.bar(x_pos - width/2, det_values, width, label='AE', 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, vae_values, width, label='VAE', 
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Final MSE (RGB)', fontsize=12)
    ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Encoder Type Comparison Across Loss Configs', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([lc.replace(' (', '\n(') for lc in loss_configs], fontsize=10)
    ax.tick_params(axis='x', pad=8)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
    output_path = output_dir / "loss_vs_encoder_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()

