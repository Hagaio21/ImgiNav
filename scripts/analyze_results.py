#!/usr/bin/env python3
"""
Advanced analysis script for baseline evaluation results using per-sample data.

Performs detailed statistical analysis using the metrics.csv files:
- Per-sample metric distributions
- Empty vs furnished room comparisons
- Correlation analysis
- Ranking and comparative analysis
- Experiment-to-experiment comparisons

Usage:
    python analyze_results.py /path/to/evaluation_results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
sns.set_palette("husl")

METRIC_COLORS = {
    "Structure": "#FF6B6B",
    "Furniture": "#4ECDC4",
    "Empty": "#A8DADC",
    "Furnished": "#F1FAEE"
}


def format_experiment_name(exp_name: str) -> str:
    """Convert experiment name to nice format."""
    exp_lower = exp_name.lower()
    
    depth_map = {
        ("DN", "Small"): 3,
        ("DN", "Medium"): 4,
        ("DN", "Large"): 4,
        ("WS", "Small"): 2,
        ("WS", "Medium"): 3,
        ("WS", "Large"): 3,
    }
    
    size = None
    arch = None
    conditioning = ""
    
    if "small" in exp_lower:
        size = "Small"
    elif "medium" in exp_lower:
        size = "Medium"
    elif "large" in exp_lower:
        size = "Large"
    
    if "down_bottleneck" in exp_lower:
        arch = "DN"
    elif "wide_shallow" in exp_lower:
        arch = "WS"
    
    if "graph" in exp_lower:
        conditioning = "Graph"
    elif "povs" in exp_lower:
        conditioning = "POV"
    
    if size and arch:
        depth = depth_map.get((arch, size), "")
        if depth:
            name = f"{arch}-{size} (d={depth})"
        else:
            name = f"{arch}-{size}"
        
        if conditioning:
            name = f"{name} - {conditioning}"
        
        return name
    
    return exp_name


def load_all_metrics_from_csv(results_dir: Path) -> pd.DataFrame:
    """Load all metrics.csv files from each experiment."""
    results_dir = Path(results_dir)
    all_data = []
    
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        csv_file = exp_dir / "metrics.csv"
        if not csv_file.exists():
            continue
        
        try:
            df = pd.read_csv(csv_file)
            exp_name = exp_dir.name
            exp_short = format_experiment_name(exp_name)
            df["experiment"] = exp_name
            df["experiment_short"] = exp_short
            all_data.append(df)
            print(f"✓ Loaded: {exp_short} ({len(df)} samples)")
        except Exception as e:
            print(f"✗ Error loading {csv_file}: {e}")
    
    if not all_data:
        raise ValueError(f"No metrics.csv files found in {results_dir}")
    
    return pd.concat(all_data, ignore_index=True)


def plot_metric_distributions(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Plot distributions of key metrics across all experiments."""
    metrics = [
        ("floor_iou", "Floor IoU"),
        ("presence_accuracy", "Presence Accuracy"),
        ("detection_f1", "Detection F1"),
        ("unified_score", "Unified Score"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metric Distributions Across All Experiments", fontsize=14, fontweight="bold")
    
    for ax, (metric_col, label) in zip(axes.flat, metrics):
        data_by_exp = [group[metric_col].values for name, group in df.groupby("experiment_short")]
        exp_names = [name for name, group in df.groupby("experiment_short")]
        
        bp = ax.boxplot(data_by_exp, labels=exp_names, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor(METRIC_COLORS["Furniture"])
        
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"01_metric_distributions.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_empty_vs_furnished(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Compare metrics for empty vs furnished rooms."""
    # Convert is_empty to boolean if needed
    if df["is_empty"].dtype == object:
        df["is_empty"] = df["is_empty"].str.upper() == "TRUE"
    
    metrics = [
        ("floor_iou", "Floor IoU"),
        ("detection_f1", "Detection F1"),
        ("unified_score", "Unified Score"),
    ]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    fig.suptitle("Empty vs Furnished Rooms Comparison", fontsize=14, fontweight="bold")
    
    for ax, (metric_col, label) in zip(axes, metrics):
        empty_data = df[df["is_empty"] == True][metric_col].dropna()
        furnished_data = df[df["is_empty"] == False][metric_col].dropna()
        
        parts = ax.violinplot([empty_data.values, furnished_data.values], 
                             positions=[0, 1], showmeans=True, showmedians=True)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Empty", "Furnished"])
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        
        # Add statistics
        t_stat, p_val = stats.ttest_ind(empty_data, furnished_data)
        
        ax.text(0.5, 0.95, f"p-value: {p_val:.4f}", transform=ax.transAxes,
               ha="center", va="top", fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f"02_empty_vs_furnished.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metric_correlation(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Plot correlation matrix between metrics."""
    metric_cols = [
        "floor_iou", "wall_iou", "openings_iou",
        "presence_accuracy", "count_accuracy", "detection_f1",
        "mean_bbox_iou", "unified_score"
    ]
    
    # Filter to available columns
    available_cols = [col for col in metric_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()
    
    # Rename for display
    col_names = {
        "floor_iou": "Floor IoU",
        "wall_iou": "Wall IoU",
        "openings_iou": "Openings IoU",
        "presence_accuracy": "Presence Acc",
        "count_accuracy": "Count Acc",
        "detection_f1": "Detection F1",
        "mean_bbox_iou": "BBox IoU",
        "unified_score": "Unified Score"
    }
    
    corr_matrix = corr_matrix.rename(columns=col_names, index=col_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
               vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Correlation"})
    
    ax.set_title("Metric Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = output_dir / f"03_metric_correlations.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_experiment_comparison(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Compare experiments across key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment Comparison", fontsize=14, fontweight="bold")
    
    metrics = [
        ("floor_iou", "Floor IoU"),
        ("presence_accuracy", "Presence Accuracy"),
        ("detection_f1", "Detection F1"),
        ("unified_score", "Unified Score"),
    ]
    
    for ax, (metric_col, label) in zip(axes.flat, metrics):
        summary = df.groupby("experiment_short")[metric_col].agg(["mean", "std"]).reset_index()
        
        x = np.arange(len(summary))
        ax.bar(x, summary["mean"], yerr=summary["std"], capsize=5, 
              color=METRIC_COLORS["Furniture"], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(summary["experiment_short"], rotation=45, ha="right", fontsize=7)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"04_experiment_comparison.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_statistical_summary(df: pd.DataFrame, output_dir: Path):
    """Generate detailed statistical summary."""
    report_lines = [
        "=" * 100,
        "DETAILED STATISTICAL ANALYSIS",
        "=" * 100,
        "",
    ]
    
    # Overall statistics
    is_empty_bool = df["is_empty"].str.upper() == "TRUE" if df["is_empty"].dtype == object else df["is_empty"] == True
    
    report_lines.extend([
        "OVERALL STATISTICS",
        "-" * 100,
        f"Total samples: {len(df)}",
        f"Empty rooms: {is_empty_bool.sum()}",
        f"Furnished rooms: {(~is_empty_bool).sum()}",
        f"Unique experiments: {df['experiment_short'].nunique()}",
        "",
    ])
    
    # Per-experiment statistics
    report_lines.extend([
        "PER-EXPERIMENT STATISTICS",
        "-" * 100,
    ])
    
    metrics_to_analyze = [
        ("floor_iou", "Floor IoU"),
        ("wall_iou", "Wall IoU"),
        ("openings_iou", "Openings IoU"),
        ("presence_accuracy", "Presence Accuracy"),
        ("detection_f1", "Detection F1"),
        ("unified_score", "Unified Score"),
    ]
    
    for exp_name in sorted(df["experiment_short"].unique()):
        exp_df = df[df["experiment_short"] == exp_name]
        report_lines.append(f"\n{exp_name} ({len(exp_df)} samples):")
        report_lines.append(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        report_lines.append("-" * 73)
        
        for metric_col, metric_label in metrics_to_analyze:
            data = exp_df[metric_col].dropna()
            if len(data) > 0:
                report_lines.append(
                    f"{metric_label:<25} {data.mean():<12.4f} {data.std():<12.4f} "
                    f"{data.min():<12.4f} {data.max():<12.4f}"
                )
    
    # Ranking by unified score
    report_lines.extend([
        "",
        "=" * 100,
        "RANKING BY UNIFIED SCORE",
        "-" * 100,
    ])
    
    ranking = df.groupby("experiment_short")["unified_score"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    report_lines.append(f"{'Rank':<6} {'Configuration':<30} {'Mean':<12} {'Std':<12} {'Samples':<8}")
    report_lines.append("-" * 100)
    
    for rank, (exp_name, row) in enumerate(ranking.iterrows(), 1):
        report_lines.append(
            f"{rank:<6} {exp_name:<30} {row['mean']:<12.4f} {row['std']:<12.4f} {int(row['count']):<8}"
        )
    
    report_lines.append("=" * 100)
    
    # Write to file
    report_path = output_dir / "STATISTICAL_SUMMARY.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print("\n" + "\n".join(report_lines))
    print(f"\n✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced analysis of baseline evaluation results"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to evaluation_results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis (default: results_dir/analysis)"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "jpg"],
        default="png",
        help="Output format for plots"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir / "analysis"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("ADVANCED BASELINE EVALUATION ANALYSIS")
    print("=" * 100)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    # Load data
    print("Loading per-sample metrics from CSV files...")
    df = load_all_metrics_from_csv(args.results_dir)
    print(f"✓ Loaded {len(df)} total samples\n")
    
    # Generate plots
    print("Generating analysis plots...\n")
    
    plot_metric_distributions(df, args.output_dir, args.format)
    plot_empty_vs_furnished(df, args.output_dir, args.format)
    plot_metric_correlation(df, args.output_dir, args.format)
    plot_experiment_comparison(df, args.output_dir, args.format)
    
    # Generate summary
    print("\nGenerating statistical summary...")
    generate_statistical_summary(df, args.output_dir)
    
    # Save CSV
    csv_path = args.output_dir / "all_samples.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ All samples CSV saved to: {csv_path}")
    
    print("\n" + "=" * 100)
    print("✓ Analysis complete!")
    print(f"✓ All outputs saved to: {args.output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()