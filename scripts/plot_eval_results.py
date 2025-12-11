#!/usr/bin/env python3
"""
Plot and compare evaluation results from JSON files.

Works with the updated evaluation_metrics.py output format.

Usage:
    python plot_eval_results.py /path/to/evaluation_results/
    python plot_eval_results.py /path/to/evaluation_results/ --output comparison.png
    python plot_eval_results.py /path/to/evaluation_results/ --plot-type all
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict


# =============================================================================
# Data Loading
# =============================================================================

def load_results(results_dir: Path) -> dict:
    """Load all JSON result files from directory."""
    results = {}
    
    # Try summary files first
    for json_file in sorted(results_dir.glob("*_summary_*.json")):
        name = json_file.stem
        parts = name.rsplit("_summary_", 1)
        exp_name = parts[0] if len(parts) > 1 else name
        exp_name = clean_experiment_name(exp_name)
        
        with open(json_file, "r") as f:
            data = json.load(f)
            results[exp_name] = data
    
    if not results:
        # Try full results files
        for json_file in sorted(results_dir.glob("*_results_*.json")):
            name = json_file.stem
            parts = name.rsplit("_results_", 1)
            exp_name = parts[0] if len(parts) > 1 else name
            exp_name = clean_experiment_name(exp_name)
            
            with open(json_file, "r") as f:
                data = json.load(f)
                if "aggregated" in data:
                    results[exp_name] = data["aggregated"]
                else:
                    results[exp_name] = data
    
    return results


def clean_experiment_name(name: str) -> str:
    """Shorten experiment name for display."""
    name = name.replace("diff_clip_v2_seg_rooms_", "")
    name = name.replace("diff_clip_seg_rooms_", "")
    name = name.replace("_down_bottleneck", "")
    name = name.replace("_v2", "")
    return name


def get_conditioning_type(exp_name: str) -> str:
    """Extract conditioning type from experiment name."""
    exp_lower = exp_name.lower()
    if "both" in exp_lower:
        return "both"
    elif "graph" in exp_lower:
        return "graph"
    elif "pov" in exp_lower:
        return "pov"
    return "other"


def get_model_size(exp_name: str) -> int:
    """Extract model size for sorting (0=small, 1=medium, 2=large)."""
    exp_lower = exp_name.lower()
    if "small" in exp_lower:
        return 0
    elif "medium" in exp_lower:
        return 1
    elif "large" in exp_lower:
        return 2
    return 3


def sort_experiments(experiments: list) -> list:
    """Sort experiments by conditioning type then size."""
    cond_order = {"pov": 0, "graph": 1, "both": 2, "other": 3}
    return sorted(experiments, key=lambda x: (cond_order.get(get_conditioning_type(x), 3), get_model_size(x)))


# =============================================================================
# Metric Definitions
# =============================================================================

METRIC_GROUPS = {
    "Class Presence": ["class_f1", "class_recall"],
    "Object Counts": ["count_exact_match_rate", "count_accuracy"],
    "Detection": ["detection_f1", "detection_recall", "mean_bbox_iou"],
    "Pixel Accuracy": ["weighted_pixel_accuracy", "object_pixel_accuracy", "mean_iou"],
    "Structural": ["wall_iou", "floor_iou"],
    "Navigation": ["reachability_preservation", "path_overlap", "path_length_ratio"],
}

METRIC_DISPLAY_NAMES = {
    "class_f1": "Class F1",
    "class_recall": "Class Recall",
    "count_exact_match_rate": "Count Exact Match",
    "count_accuracy": "Count Accuracy",
    "detection_f1": "Detection F1",
    "detection_recall": "Detection Recall",
    "mean_bbox_iou": "Mean BBox IoU",
    "weighted_pixel_accuracy": "Weighted Px Acc",
    "object_pixel_accuracy": "Object Px Acc",
    "mean_iou": "Mean IoU",
    "wall_iou": "Wall IoU",
    "floor_iou": "Floor IoU",
    "reachability_preservation": "Reachability",
    "path_overlap": "Path Overlap",
    "path_length_ratio": "Path Length Ratio",
}

# Metrics where lower is better
LOWER_IS_BETTER = {"path_length_ratio"}  # Ideal is 1.0, but >1 means longer paths


def get_metric_value(results: dict, metric: str) -> tuple:
    """Get metric mean and std from results."""
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    
    mean_val = results.get(mean_key, results.get(metric, 0))
    std_val = results.get(std_key, 0)
    
    return float(mean_val), float(std_val)


def get_available_metrics(results: dict) -> list:
    """Get list of metrics available across all experiments."""
    all_metrics = set()
    for exp_data in results.values():
        for key in exp_data.keys():
            if key.endswith("_mean"):
                metric_name = key.rsplit("_mean", 1)[0]
                all_metrics.add(metric_name)
            elif key in METRIC_DISPLAY_NAMES:
                all_metrics.add(key)
    return sorted(all_metrics)


# =============================================================================
# Plotting Functions
# =============================================================================

def get_conditioning_colors():
    """Get colors for each conditioning type."""
    return {
        "pov": "#2ecc71",      # Green
        "graph": "#3498db",    # Blue
        "both": "#9b59b6",     # Purple
        "other": "#95a5a6",    # Gray
    }


def plot_grouped_comparison(results: dict, output_path: Path = None):
    """
    Create grouped bar chart comparing key metrics across experiments.
    Groups experiments by conditioning type with consistent colors.
    """
    if not results:
        print("No results to plot!")
        return
    
    experiments = sort_experiments(list(results.keys()))
    
    # Select key metrics for comparison
    key_metrics = [
        "class_f1", "count_accuracy", "detection_f1", 
        "mean_bbox_iou", "weighted_pixel_accuracy", "mean_iou"
    ]
    
    available = get_available_metrics(results)
    metrics = [m for m in key_metrics if m in available]
    
    if not metrics:
        metrics = available[:6]
    
    n_experiments = len(experiments)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(max(14, n_experiments * 1.5), 8))
    
    x = np.arange(n_experiments)
    width = 0.8 / n_metrics
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    cond_colors = get_conditioning_colors()
    
    for i, metric in enumerate(metrics):
        means = []
        stds = []
        for exp in experiments:
            mean, std = get_metric_value(results[exp], metric)
            means.append(mean)
            stds.append(std)
        
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=2,
                      label=METRIC_DISPLAY_NAMES.get(metric, metric), 
                      color=colors[i], alpha=0.85)
    
    # Add conditioning type indicators below x-axis
    for i, exp in enumerate(experiments):
        cond_type = get_conditioning_type(exp)
        color = cond_colors.get(cond_type, "#95a5a6")
        ax.axvspan(i - 0.4, i + 0.4, ymin=0, ymax=0.02, color=color, alpha=0.8)
    
    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Evaluation Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add legend for conditioning types
    cond_patches = [mpatches.Patch(color=c, label=t.upper()) for t, c in cond_colors.items() if t != "other"]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + cond_patches, 
              loc="upper right", fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metric_groups(results: dict, output_path: Path = None):
    """
    Create multi-panel figure with one subplot per metric group.
    """
    if not results:
        print("No results to plot!")
        return
    
    experiments = sort_experiments(list(results.keys()))
    available = get_available_metrics(results)
    
    # Filter metric groups to available metrics
    active_groups = {}
    for group_name, metrics in METRIC_GROUPS.items():
        group_metrics = [m for m in metrics if m in available]
        if group_metrics:
            active_groups[group_name] = group_metrics
    
    if not active_groups:
        print("No metric groups available")
        return
    
    n_groups = len(active_groups)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    cond_colors = get_conditioning_colors()
    
    for idx, (group_name, metrics) in enumerate(active_groups.items()):
        ax = axes[idx]
        
        n_experiments = len(experiments)
        n_metrics = len(metrics)
        
        x = np.arange(n_experiments)
        width = 0.8 / n_metrics
        colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))
        
        for i, metric in enumerate(metrics):
            means = []
            stds = []
            for exp in experiments:
                mean, std = get_metric_value(results[exp], metric)
                means.append(mean)
                stds.append(std)
            
            offset = (i - n_metrics / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=2,
                   label=METRIC_DISPLAY_NAMES.get(metric, metric),
                   color=colors[i], alpha=0.85)
        
        # Color bars by conditioning type
        for i, exp in enumerate(experiments):
            cond_type = get_conditioning_type(exp)
            color = cond_colors.get(cond_type, "#95a5a6")
            ax.axvspan(i - 0.4, i + 0.4, ymin=0, ymax=0.015, color=color, alpha=0.8)
        
        ax.set_title(group_name, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # Hide unused axes
    for idx in range(len(active_groups), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Metrics by Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_radar(results: dict, output_path: Path = None):
    """
    Create radar chart comparing experiments across key metrics.
    """
    if not results:
        print("No results to plot!")
        return
    
    experiments = sort_experiments(list(results.keys()))
    available = get_available_metrics(results)
    
    # Select metrics for radar (need at least 3)
    radar_metrics = ["class_f1", "count_accuracy", "detection_f1", 
                     "mean_bbox_iou", "weighted_pixel_accuracy", "mean_iou"]
    metrics = [m for m in radar_metrics if m in available]
    
    if len(metrics) < 3:
        print("Need at least 3 metrics for radar chart")
        return
    
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    cond_colors = get_conditioning_colors()
    
    for exp in experiments:
        values = []
        for metric in metrics:
            mean, _ = get_metric_value(results[exp], metric)
            values.append(mean)
        values += values[:1]  # Complete the circle
        
        cond_type = get_conditioning_type(exp)
        color = cond_colors.get(cond_type, "#95a5a6")
        
        ax.plot(angles, values, "o-", linewidth=2, label=exp, color=color, alpha=0.7)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_DISPLAY_NAMES.get(m, m) for m in metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.set_title("Experiment Comparison (Radar)", fontsize=14, fontweight="bold", pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class(results: dict, output_path: Path = None):
    """
    Plot per-class metrics if available.
    Shows detection F1 and IoU per object class.
    """
    if not results:
        print("No results to plot!")
        return
    
    # Check if per-class data exists
    has_per_class = False
    for exp_data in results.values():
        if "per_class" in exp_data:
            has_per_class = True
            break
    
    if not has_per_class:
        print("No per-class data available (need full results files, not summaries)")
        return
    
    experiments = sort_experiments(list(results.keys()))
    
    # Collect all class names
    all_classes = set()
    for exp_data in results.values():
        if "per_class" in exp_data:
            all_classes.update(exp_data["per_class"].keys())
    
    all_classes = sorted(all_classes)
    
    if not all_classes:
        print("No per-class data found")
        return
    
    # Plot detection F1 per class
    fig, ax = plt.subplots(figsize=(max(12, len(all_classes) * 0.8), 8))
    
    x = np.arange(len(all_classes))
    width = 0.8 / len(experiments)
    cond_colors = get_conditioning_colors()
    
    for i, exp in enumerate(experiments):
        f1_values = []
        for cls in all_classes:
            per_class = results[exp].get("per_class", {})
            cls_data = per_class.get(cls, {})
            f1 = cls_data.get("detection_f1_mean", cls_data.get("detection_f1", 0))
            f1_values.append(f1)
        
        cond_type = get_conditioning_type(exp)
        color = cond_colors.get(cond_type, "#95a5a6")
        
        offset = (i - len(experiments) / 2 + 0.5) * width
        ax.bar(x + offset, f1_values, width, label=exp, color=color, alpha=0.8)
    
    ax.set_xlabel("Object Class", fontsize=12)
    ax.set_ylabel("Detection F1", fontsize=12)
    ax.set_title("Per-Class Detection F1", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_conditioning_comparison(results: dict, output_path: Path = None):
    """
    Compare conditioning types (POV vs Graph vs Both) aggregated across model sizes.
    """
    if not results:
        print("No results to plot!")
        return
    
    # Group by conditioning type
    cond_groups = defaultdict(list)
    for exp, data in results.items():
        cond_type = get_conditioning_type(exp)
        cond_groups[cond_type].append(data)
    
    if len(cond_groups) < 2:
        print("Need at least 2 conditioning types to compare")
        return
    
    # Aggregate metrics per conditioning type
    available = get_available_metrics(results)
    key_metrics = ["class_f1", "count_accuracy", "detection_f1", 
                   "mean_bbox_iou", "weighted_pixel_accuracy", "mean_iou"]
    metrics = [m for m in key_metrics if m in available]
    
    cond_types = ["pov", "graph", "both"]
    cond_types = [c for c in cond_types if c in cond_groups]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(metrics))
    width = 0.8 / len(cond_types)
    cond_colors = get_conditioning_colors()
    
    for i, cond_type in enumerate(cond_types):
        means = []
        stds = []
        
        for metric in metrics:
            values = []
            for data in cond_groups[cond_type]:
                mean, _ = get_metric_value(data, metric)
                values.append(mean)
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        color = cond_colors.get(cond_type, "#95a5a6")
        offset = (i - len(cond_types) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=cond_type.upper(), color=color, alpha=0.85)
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Conditioning Type Comparison (Averaged Across Sizes)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY_NAMES.get(m, m) for m in metrics], rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_size_comparison(results: dict, output_path: Path = None):
    """
    Compare model sizes (Small vs Medium vs Large) aggregated across conditioning types.
    """
    if not results:
        print("No results to plot!")
        return
    
    # Group by model size
    size_groups = defaultdict(list)
    size_names = {0: "small", 1: "medium", 2: "large"}
    
    for exp, data in results.items():
        size_idx = get_model_size(exp)
        if size_idx <= 2:
            size_groups[size_names[size_idx]].append(data)
    
    if len(size_groups) < 2:
        print("Need at least 2 model sizes to compare")
        return
    
    available = get_available_metrics(results)
    key_metrics = ["class_f1", "count_accuracy", "detection_f1", 
                   "mean_bbox_iou", "weighted_pixel_accuracy", "mean_iou"]
    metrics = [m for m in key_metrics if m in available]
    
    sizes = ["small", "medium", "large"]
    sizes = [s for s in sizes if s in size_groups]
    
    size_colors = {"small": "#e74c3c", "medium": "#f39c12", "large": "#27ae60"}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(metrics))
    width = 0.8 / len(sizes)
    
    for i, size in enumerate(sizes):
        means = []
        stds = []
        
        for metric in metrics:
            values = []
            for data in size_groups[size]:
                mean, _ = get_metric_value(data, metric)
                values.append(mean)
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        color = size_colors.get(size, "#95a5a6")
        offset = (i - len(sizes) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=size.capitalize(), color=color, alpha=0.85)
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Size Comparison (Averaged Across Conditioning Types)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY_NAMES.get(m, m) for m in metrics], rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_heatmap(results: dict, output_path: Path = None):
    """
    Create a heatmap of all metrics across experiments.
    Great for seeing patterns at a glance.
    """
    if not results:
        print("No results to plot!")
        return
    
    experiments = sort_experiments(list(results.keys()))
    available = get_available_metrics(results)
    
    # Select metrics to show
    key_metrics = [
        "class_f1", "class_recall", "count_exact_match_rate", "count_accuracy",
        "detection_f1", "detection_recall", "mean_bbox_iou",
        "weighted_pixel_accuracy", "object_pixel_accuracy", "mean_iou",
        "wall_iou", "floor_iou", "reachability_preservation", "path_overlap"
    ]
    metrics = [m for m in key_metrics if m in available]
    
    if not metrics:
        metrics = available
    
    # Build data matrix
    data = np.zeros((len(experiments), len(metrics)))
    for i, exp in enumerate(experiments):
        for j, metric in enumerate(metrics):
            mean, _ = get_metric_value(results[exp], metric)
            data[i, j] = mean
    
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 0.8), max(6, len(experiments) * 0.5)))
    
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom", fontsize=10)
    
    # Labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels([METRIC_DISPLAY_NAMES.get(m, m) for m in metrics], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(experiments, fontsize=9)
    
    # Add text annotations
    for i in range(len(experiments)):
        for j in range(len(metrics)):
            val = data[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    
    ax.set_title("Metrics Heatmap", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Summary Table
# =============================================================================

def print_summary_table(results: dict):
    """Print a comprehensive text summary table."""
    if not results:
        print("No results!")
        return
    
    experiments = sort_experiments(list(results.keys()))
    available = get_available_metrics(results)
    
    # Key metrics to display
    key_metrics = ["class_f1", "count_accuracy", "detection_f1", 
                   "mean_bbox_iou", "weighted_pixel_accuracy", "mean_iou", 
                   "wall_iou", "reachability_preservation"]
    metrics = [m for m in key_metrics if m in available]
    
    # Header
    header = f"{'Experiment':<25}"
    for m in metrics:
        name = METRIC_DISPLAY_NAMES.get(m, m)[:12]
        header += f"{name:<14}"
    
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    
    # Rows
    for exp in experiments:
        row = f"{exp:<25}"
        for metric in metrics:
            mean, std = get_metric_value(results[exp], metric)
            row += f"{mean:.3f}±{std:.2f}   "
        print(row)
    
    print("=" * len(header))
    
    # Best per metric
    print("\nBest per metric:")
    for metric in metrics:
        best_exp = None
        best_val = -1
        for exp in experiments:
            mean, _ = get_metric_value(results[exp], metric)
            if mean > best_val:
                best_val = mean
                best_exp = exp
        name = METRIC_DISPLAY_NAMES.get(metric, metric)
        print(f"  {name}: {best_exp} ({best_val:.3f})")


def print_conditioning_summary(results: dict):
    """Print summary grouped by conditioning type."""
    if not results:
        return
    
    cond_groups = defaultdict(list)
    for exp, data in results.items():
        cond_type = get_conditioning_type(exp)
        cond_groups[cond_type].append((exp, data))
    
    print("\n" + "=" * 60)
    print("CONDITIONING TYPE SUMMARY")
    print("=" * 60)
    
    key_metrics = ["detection_f1", "mean_bbox_iou", "weighted_pixel_accuracy"]
    
    for cond_type in ["pov", "graph", "both"]:
        if cond_type not in cond_groups:
            continue
        
        print(f"\n[{cond_type.upper()}]")
        for exp, data in cond_groups[cond_type]:
            metrics_str = ""
            for m in key_metrics:
                mean, _ = get_metric_value(data, m)
                name = METRIC_DISPLAY_NAMES.get(m, m)[:10]
                metrics_str += f"{name}={mean:.3f}  "
            print(f"  {exp}: {metrics_str}")


def export_latex_table(results: dict, output_path: Path = None):
    """Export results as a LaTeX table for thesis."""
    if not results:
        print("No results!")
        return
    
    experiments = sort_experiments(list(results.keys()))
    available = get_available_metrics(results)
    
    # Select metrics for table
    key_metrics = ["class_f1", "detection_f1", "mean_bbox_iou", 
                   "weighted_pixel_accuracy", "mean_iou", "wall_iou"]
    metrics = [m for m in key_metrics if m in available]
    
    # Find best values for bolding
    best_values = {}
    for metric in metrics:
        best_val = -1
        for exp in experiments:
            mean, _ = get_metric_value(results[exp], metric)
            if mean > best_val:
                best_val = mean
        best_values[metric] = best_val
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Evaluation Results}")
    lines.append("\\label{tab:eval_results}")
    
    col_spec = "l" + "c" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header
    header_cols = ["Model"] + [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    lines.append(" & ".join(header_cols) + " \\\\")
    lines.append("\\midrule")
    
    # Data rows
    for exp in experiments:
        row = [exp.replace("_", "\\_")]
        for metric in metrics:
            mean, std = get_metric_value(results[exp], metric)
            val_str = f"{mean:.3f}"
            if abs(mean - best_values[metric]) < 0.001:
                val_str = f"\\textbf{{{val_str}}}"
            row.append(val_str)
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    latex_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table: {output_path}")
    else:
        print(latex_str)
    
    return latex_str


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results comparison")
    parser.add_argument("results_dir", type=Path, help="Directory containing result JSON files")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output image path/prefix")
    parser.add_argument("--plot-type", "-t", 
                        choices=["grouped", "groups", "radar", "conditioning", "size", "per_class", "heatmap", "all"], 
                        default="grouped", help="Type of plot")
    parser.add_argument("--table", action="store_true", help="Print summary table")
    parser.add_argument("--latex", type=Path, default=None, help="Export LaTeX table to file")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots (only save)")
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Directory not found: {args.results_dir}")
        return
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No result files found in: {args.results_dir}")
        print("Looking for files matching: *_summary_*.json or *_results_*.json")
        return
    
    print(f"Loaded {len(results)} experiments:")
    for exp in sort_experiments(list(results.keys())):
        cond = get_conditioning_type(exp)
        print(f"  - {exp} [{cond}]")
    print()
    
    # Print tables
    if args.table:
        print_summary_table(results)
        print_conditioning_summary(results)
        print()
    
    # Export LaTeX
    if args.latex:
        export_latex_table(results, args.latex)
    
    # Generate plots
    if args.output:
        base = args.output.stem
        out_dir = args.output.parent
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        base = "comparison"
        out_dir = Path(".")
    
    if args.plot_type == "grouped" or args.plot_type == "all":
        out_path = out_dir / f"{base}_grouped.png" if args.output or args.plot_type == "all" else None
        plot_grouped_comparison(results, out_path)
    
    if args.plot_type == "groups" or args.plot_type == "all":
        out_path = out_dir / f"{base}_by_category.png" if args.output or args.plot_type == "all" else None
        plot_metric_groups(results, out_path)
    
    if args.plot_type == "radar" or args.plot_type == "all":
        out_path = out_dir / f"{base}_radar.png" if args.output or args.plot_type == "all" else None
        plot_radar(results, out_path)
    
    if args.plot_type == "conditioning" or args.plot_type == "all":
        out_path = out_dir / f"{base}_conditioning.png" if args.output or args.plot_type == "all" else None
        plot_conditioning_comparison(results, out_path)
    
    if args.plot_type == "size" or args.plot_type == "all":
        out_path = out_dir / f"{base}_size.png" if args.output or args.plot_type == "all" else None
        plot_size_comparison(results, out_path)
    
    if args.plot_type == "per_class" or args.plot_type == "all":
        out_path = out_dir / f"{base}_per_class.png" if args.output or args.plot_type == "all" else None
        plot_per_class(results, out_path)
    
    if args.plot_type == "heatmap" or args.plot_type == "all":
        out_path = out_dir / f"{base}_heatmap.png" if args.output or args.plot_type == "all" else None
        plot_heatmap(results, out_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()