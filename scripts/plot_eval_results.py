#!/usr/bin/env python3
"""
Plot and compare evaluation results from JSON files.

Usage:
    python plot_eval_results.py /path/to/evaluation_results/
    python plot_eval_results.py /path/to/evaluation_results/ --output comparison.png
    python plot_eval_results.py /path/to/evaluation_results/ --metrics pixel_accuracy bbox_iou
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_results(results_dir: Path) -> dict:
    """Load all JSON result files from directory."""
    results = {}
    
    for json_file in sorted(results_dir.glob("*_summary_*.json")):
        # Extract experiment name from filename
        name = json_file.stem
        # Clean up name: remove _summary_timestamp
        parts = name.rsplit("_summary_", 1)
        exp_name = parts[0] if len(parts) > 1 else name
        
        # Shorten experiment name for display
        exp_name = exp_name.replace("diff_clip_v2_seg_rooms_", "")
        exp_name = exp_name.replace("_down_bottleneck", "")
        
        with open(json_file, "r") as f:
            data = json.load(f)
            results[exp_name] = data
    
    if not results:
        # Try loading full results files instead
        for json_file in sorted(results_dir.glob("*_results_*.json")):
            name = json_file.stem
            parts = name.rsplit("_results_", 1)
            exp_name = parts[0] if len(parts) > 1 else name
            exp_name = exp_name.replace("diff_clip_v2_seg_rooms_", "")
            exp_name = exp_name.replace("_down_bottleneck", "")
            
            with open(json_file, "r") as f:
                data = json.load(f)
                if "aggregated" in data:
                    results[exp_name] = data["aggregated"]
                else:
                    results[exp_name] = data
    
    return results


def get_available_metrics(results: dict) -> list:
    """Get list of metrics available across all experiments."""
    all_metrics = set()
    for exp_data in results.values():
        for key in exp_data.keys():
            if key.endswith("_mean"):
                metric_name = key.rsplit("_mean", 1)[0]
                all_metrics.add(metric_name)
    return sorted(all_metrics)


def plot_metric_comparison(results: dict, metrics: list = None, output_path: Path = None):
    """Create bar chart comparing metrics across experiments."""
    if not results:
        print("No results to plot!")
        return
    
    available_metrics = get_available_metrics(results)
    
    if metrics is None:
        # Default metrics to show
        default_metrics = [
            "pixel_accuracy", "bbox_iou", "object_recall", "object_precision",
            "object_f1", "centroid_distance", "count_accuracy"
        ]
        metrics = [m for m in default_metrics if m in available_metrics]
        if not metrics:
            metrics = available_metrics[:8]  # Show first 8 available
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in available_metrics]
    
    if not metrics:
        print(f"No matching metrics found. Available: {available_metrics}")
        return
    
    experiments = list(results.keys())
    n_metrics = len(metrics)
    n_experiments = len(experiments)
    
    # Create figure
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_experiments))
    
    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        
        for exp in experiments:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            
            mean_val = results[exp].get(mean_key, 0)
            std_val = results[exp].get(std_key, 0)
            
            means.append(mean_val)
            stds.append(std_val)
        
        x = np.arange(n_experiments)
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
        
        ax.set_xlabel("Experiment")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f"{mean:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_radar_chart(results: dict, metrics: list = None, output_path: Path = None):
    """Create radar chart comparing experiments across metrics."""
    if not results:
        print("No results to plot!")
        return
    
    available_metrics = get_available_metrics(results)
    
    if metrics is None:
        default_metrics = [
            "pixel_accuracy", "bbox_iou", "object_recall", "object_precision",
            "object_f1", "count_accuracy"
        ]
        metrics = [m for m in default_metrics if m in available_metrics]
        if not metrics:
            metrics = available_metrics[:6]
    
    metrics = [m for m in metrics if m in available_metrics]
    
    if len(metrics) < 3:
        print("Need at least 3 metrics for radar chart")
        return
    
    experiments = list(results.keys())
    n_metrics = len(metrics)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for exp, color in zip(experiments, colors):
        values = []
        for metric in metrics:
            mean_key = f"{metric}_mean"
            values.append(results[exp].get(mean_key, 0))
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, "o-", linewidth=2, label=exp, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Experiment Comparison", fontsize=14, fontweight="bold", pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_grouped_bars(results: dict, metrics: list = None, output_path: Path = None):
    """Create grouped bar chart - experiments grouped by conditioning type."""
    if not results:
        print("No results to plot!")
        return
    
    available_metrics = get_available_metrics(results)
    
    if metrics is None:
        default_metrics = ["pixel_accuracy", "bbox_iou", "object_f1", "count_accuracy"]
        metrics = [m for m in default_metrics if m in available_metrics]
        if not metrics:
            metrics = available_metrics[:4]
    
    metrics = [m for m in metrics if m in available_metrics]
    
    # Group experiments by conditioning type
    groups = defaultdict(list)
    for exp in results.keys():
        if "both" in exp.lower():
            groups["both"].append(exp)
        elif "pov" in exp.lower():
            groups["pov"].append(exp)
        elif "graph" in exp.lower():
            groups["graph"].append(exp)
        else:
            groups["other"].append(exp)
    
    # Sort within groups by size (small, medium, large)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: (
            0 if "small" in x.lower() else 
            1 if "medium" in x.lower() else 
            2 if "large" in x.lower() else 3
        ))
    
    experiments = []
    for cond_type in ["pov", "graph", "both", "other"]:
        experiments.extend(groups.get(cond_type, []))
    
    if not experiments:
        experiments = list(results.keys())
    
    n_metrics = len(metrics)
    n_experiments = len(experiments)
    
    fig, ax = plt.subplots(figsize=(max(12, n_experiments * 1.5), 7))
    
    x = np.arange(n_experiments)
    width = 0.8 / n_metrics
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, metric in enumerate(metrics):
        means = []
        stds = []
        for exp in experiments:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            means.append(results[exp].get(mean_key, 0))
            stds.append(results[exp].get(std_key, 0))
        
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=2,
                      label=metric.replace("_", " ").title(), color=colors[i], alpha=0.85)
    
    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Evaluation Metrics by Experiment", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add vertical lines to separate conditioning types
    cumsum = 0
    for cond_type in ["pov", "graph", "both"]:
        if cond_type in groups and groups[cond_type]:
            cumsum += len(groups[cond_type])
            if cumsum < n_experiments:
                ax.axvline(x=cumsum - 0.5, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def print_summary_table(results: dict, metrics: list = None):
    """Print a text summary table of results."""
    if not results:
        print("No results!")
        return
    
    available_metrics = get_available_metrics(results)
    
    if metrics is None:
        metrics = available_metrics[:6]
    
    metrics = [m for m in metrics if m in available_metrics]
    
    # Header
    header = f"{'Experiment':<35}" + "".join(f"{m[:12]:<14}" for m in metrics)
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    
    # Rows
    for exp in sorted(results.keys()):
        row = f"{exp:<35}"
        for metric in metrics:
            mean_val = results[exp].get(f"{metric}_mean", 0)
            std_val = results[exp].get(f"{metric}_std", 0)
            row += f"{mean_val:.3f}±{std_val:.2f}  "
        print(row)
    
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results comparison")
    parser.add_argument("results_dir", type=Path, help="Directory containing result JSON files")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output image path")
    parser.add_argument("--metrics", "-m", nargs="+", default=None, help="Metrics to plot")
    parser.add_argument("--plot-type", "-t", choices=["bar", "radar", "grouped", "all"], 
                        default="grouped", help="Type of plot")
    parser.add_argument("--table", action="store_true", help="Print summary table")
    
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
    for exp in results.keys():
        print(f"  - {exp}")
    print()
    
    if args.table:
        print_summary_table(results, args.metrics)
        print()
    
    if args.plot_type == "bar":
        plot_metric_comparison(results, args.metrics, args.output)
    elif args.plot_type == "radar":
        plot_radar_chart(results, args.metrics, args.output)
    elif args.plot_type == "grouped":
        plot_grouped_bars(results, args.metrics, args.output)
    elif args.plot_type == "all":
        base_output = args.output.stem if args.output else "comparison"
        output_dir = args.output.parent if args.output else Path(".")
        
        plot_grouped_bars(results, args.metrics, output_dir / f"{base_output}_grouped.png")
        plot_metric_comparison(results, args.metrics, output_dir / f"{base_output}_bars.png")
        plot_radar_chart(results, args.metrics, output_dir / f"{base_output}_radar.png")


if __name__ == "__main__":
    main()