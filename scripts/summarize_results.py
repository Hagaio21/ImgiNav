#!/usr/bin/env python3
"""
Visualization and summarization script for baseline evaluation results.

This script reads evaluation results from the output folder and creates
comprehensive plots and summaries using seaborn and matplotlib.

Usage:
    python summarize_results.py /path/to/evaluation_results
    python summarize_results.py /path/to/evaluation_results --output-dir ./plots
    python summarize_results.py /path/to/evaluation_results --format pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Colors for different metric categories
METRIC_COLORS = {
    "Structure": "#FF6B6B",
    "Furniture": "#4ECDC4",
    "Unified": "#95E1D3",
    "Empty": "#A8DADC",
    "Furnished": "#F1FAEE"
}

def format_experiment_name(exp_name: str) -> str:
    """
    Convert experiment name to nice format matching LaTeX table.
    
    Examples:
        diff_clip_v2_seg_rooms_small_down_bottleneck_both_checkpoint_best 
        -> DN-Small (d=3)
        
        diff_clip_v2_seg_rooms_large_wide_shallow_graph_checkpoint_best_kid
        -> WS-Large (Graph)
    """
    exp_lower = exp_name.lower()
    
    # Mapping of architecture+size to depth and nice name
    depth_map = {
        ("DN", "Small"): 3,
        ("DN", "Medium"): 4,
        ("DN", "Large"): 4,
        ("WS", "Small"): 2,
        ("WS", "Medium"): 3,
        ("WS", "Large"): 3,
    }
    
    # Extract size, architecture, conditioning
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
    
    # Extract conditioning
    if "graph" in exp_lower:
        conditioning = "Graph"
    elif "povs" in exp_lower:
        conditioning = "POV"
    # "both" gets empty string
    
    # Build name
    if size and arch:
        depth = depth_map.get((arch, size), "")
        if depth:
            name = f"{arch}-{size} (d={depth})"
        else:
            name = f"{arch}-{size}"
        
        # Add conditioning if not empty
        if conditioning:
            name = f"{name} - {conditioning}"
        
        return name
    
    return exp_name


# =============================================================================
# Data Loading
# =============================================================================

def load_results(results_dir: Path) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Load all evaluation results from the results directory.
    
    Args:
        results_dir: Path to evaluation_results directory
        
    Returns:
        Tuple of (list of result dicts, aggregated dataframe)
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    all_results = []
    
    # Find all results_*.json files
    for json_file in sorted(results_dir.rglob("results_*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Extract experiment name from folder structure
            exp_name = json_file.parent.name
            data["experiment"] = exp_name
            data["result_file"] = str(json_file)
            
            all_results.append(data)
            print(f"✓ Loaded: {exp_name}")
        except Exception as e:
            print(f"✗ Error loading {json_file}: {e}")
    
    if not all_results:
        raise ValueError(f"No results_*.json files found in {results_dir}")
    
    # Convert to DataFrame
    df = flatten_results(all_results)
    
    return all_results, df


def flatten_results(results: List[Dict]) -> pd.DataFrame:
    """Flatten nested result dictionaries into a pandas DataFrame."""
    rows = []
    
    for result in results:
        exp_name = result.get("experiment", "Unknown")
        row = {
            "experiment": exp_name,
            "experiment_short": format_experiment_name(exp_name)
        }
        
        # Metadata
        if "metadata" in result:
            meta = result["metadata"]
            row["num_samples"] = meta.get("max_samples", "Unknown")
            row["guidance_scale"] = meta.get("guidance_scale", "Unknown")
        
        # Use aggregated section (main metrics)
        if "aggregated" in result:
            agg = result["aggregated"]
            
            # Counts
            row["total_samples"] = agg.get("num_samples", 0)
            row["empty_samples"] = agg.get("num_empty", 0)
            row["furnished_samples"] = agg.get("num_furnished", 0)
            
            # Structure metrics - get from top level first, then from structure dict
            row["floor_iou"] = agg.get("floor_iou_mean", agg.get("structure", {}).get("floor_iou", {}).get("mean", 0))
            row["wall_iou"] = agg.get("wall_iou_mean", agg.get("structure", {}).get("wall_iou", {}).get("mean", 0))
            row["openings_iou"] = agg.get("openings_iou_mean", agg.get("structure", {}).get("openings_iou", {}).get("mean", 0))
            
            # Furniture metrics
            row["presence_accuracy"] = agg.get("presence_accuracy_mean", 0)
            row["count_accuracy"] = agg.get("count_accuracy_mean", 0)
            row["detection_f1"] = agg.get("detection_f1_mean", 0)
            row["mean_l1_distance"] = agg.get("mean_l1_distance_mean", 0)
            row["mean_bbox_iou"] = agg.get("mean_bbox_iou_mean", 0)
            row["detection_recall"] = agg.get("detection_recall_mean", 0)
            row["detection_precision"] = agg.get("detection_precision_mean", 0)
            
            # Unified scores
            row["unified_score_all"] = agg.get("unified_score_mean", 0)
            
            # Empty/furnished specific scores
            if "empty" in agg:
                row["unified_score_empty"] = agg["empty"].get("unified_score", {}).get("mean", 0)
            if "furnished" in agg:
                row["unified_score_furnished"] = agg["furnished"].get("unified_score", {}).get("mean", 0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_structure_metrics(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Plot structure metrics (Floor, Wall, Openings IoU)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Structure Metrics Comparison", fontsize=16, fontweight="bold")
    
    metrics = [
        ("floor_iou", "Floor IoU", 0),
        ("wall_iou", "Wall IoU", 1),
        ("openings_iou", "Openings IoU", 2)
    ]
    
    for metric_col, metric_label, ax_idx in metrics:
        ax = axes[ax_idx]
        
        # Filter out zero values (missing data)
        data = df[df[metric_col] > 0].copy()
        
        if len(data) > 0:
            sns.barplot(
                data=data,
                x="experiment_short",
                y=metric_col,
                ax=ax,
                color=METRIC_COLORS["Structure"],
                errorbar="sd"
            )
            ax.set_title(metric_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Configuration", fontsize=10)
            ax.set_ylabel("IoU", fontsize=10)
            ax.set_ylim([0, 1.0])
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
            
            # Add value labels on bars
            for i, v in enumerate(data[metric_col]):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / f"01_structure_metrics.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_furniture_metrics(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Plot furniture metrics (presence, count, detection, L1, IoU)."""
    furniture_cols = [
        "presence_accuracy",
        "count_accuracy",
        "detection_f1",
        "mean_bbox_iou"
    ]
    
    # Filter experiments with furniture data
    furn_df = df[df["presence_accuracy"] > 0].copy()
    
    if len(furn_df) == 0:
        print("⚠ No furniture metrics found (likely empty-only evaluation)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Furniture Metrics Comparison", fontsize=16, fontweight="bold")
    
    labels = [
        "Presence Accuracy",
        "Count Accuracy",
        "Detection F1",
        "Bounding Box IoU"
    ]
    
    for idx, (metric_col, label) in enumerate(zip(furniture_cols, labels)):
        ax = axes[idx // 2, idx % 2]
        
        data = furn_df[furn_df[metric_col] > 0].copy()
        
        sns.barplot(
            data=data,
            x="experiment_short",
            y=metric_col,
            ax=ax,
            color=METRIC_COLORS["Furniture"],
            errorbar="sd"
        )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Configuration", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
        
        # Add value labels
        for i, v in enumerate(data[metric_col]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / f"02_furniture_metrics.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_unified_scores(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Plot unified scores across experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Unified Score Comparison", fontsize=16, fontweight="bold")
    
    score_cols = [
        ("unified_score_all", "All Samples", 0),
        ("unified_score_empty", "Empty Rooms", 1),
        ("unified_score_furnished", "Furnished Rooms", 2)
    ]
    
    for score_col, label, ax_idx in score_cols:
        ax = axes[ax_idx]
        
        # Filter out null values
        data = df[df[score_col].notna() & (df[score_col] > 0)].copy()
        
        if len(data) > 0:
            sns.barplot(
                data=data,
                x="experiment_short",
                y=score_col,
                ax=ax,
                color=METRIC_COLORS["Unified"],
                errorbar="sd"
            )
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Configuration", fontsize=10)
            ax.set_ylabel("Unified Score", fontsize=10)
            ax.set_ylim([0, 1.0])
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
            
            # Add value labels
            for i, v in enumerate(data[score_col]):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    output_path = output_dir / f"03_unified_scores.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_all_metrics_heatmap(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Create a heatmap of all metrics for easy comparison."""
    # Select numeric columns
    metric_cols = [
        "floor_iou", "wall_iou", "openings_iou",
        "presence_accuracy", "count_accuracy", "detection_f1",
        "mean_bbox_iou", "unified_score_all"
    ]
    
    # Filter to available columns
    available_cols = [col for col in metric_cols if col in df.columns]
    
    # Create a subset with available data
    heatmap_df = df[["experiment_short"] + available_cols].copy()
    heatmap_df = heatmap_df.set_index("experiment_short")
    
    # Remove rows that are all zeros
    heatmap_df = heatmap_df[(heatmap_df != 0).any(axis=1)]
    
    if len(heatmap_df) == 0 or len(heatmap_df.columns) == 0:
        print("⚠ No metrics available for heatmap")
        return
    
    # Rename columns for display
    col_names = {
        "floor_iou": "Floor IoU",
        "wall_iou": "Wall IoU",
        "openings_iou": "Openings IoU",
        "presence_accuracy": "Presence Acc",
        "count_accuracy": "Count Acc",
        "detection_f1": "Detection F1",
        "mean_bbox_iou": "BBox IoU",
        "unified_score_all": "Unified Score"
    }
    
    heatmap_df = heatmap_df.rename(columns=col_names)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Score", "shrink": 0.8},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title("All Metrics Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metrics", fontsize=10)
    ax.set_ylabel("Configuration", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f"04_metrics_heatmap.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metric_distributions(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Plot distributions of key metrics across experiments."""
    metric_groups = {
        "Structure Metrics": ["floor_iou", "wall_iou", "openings_iou"],
        "Furniture Metrics": ["presence_accuracy", "count_accuracy", "detection_f1", "mean_bbox_iou"],
    }
    
    for group_name, metrics in metric_groups.items():
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns and (df[m] > 0).any()]
        
        if not available_metrics:
            continue
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 5))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        fig.suptitle(f"{group_name} - Distribution across Experiments", 
                    fontsize=14, fontweight="bold")
        
        for ax, metric in zip(axes, available_metrics):
            data = df[df[metric] > 0][[metric]]
            
            sns.histplot(data=data, x=metric, bins=10, ax=ax, kde=True, color="#4ECDC4")
            ax.set_title(metric.replace("_", " ").title(), fontsize=11)
            ax.set_xlabel("Score", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
        
        plt.tight_layout()
        filename = group_name.lower().replace(" ", "_")
        output_path = output_dir / f"05_{filename}_distribution.{format_ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()


def plot_sample_counts(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Plot sample counts breakdown by empty/furnished."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Sample Distribution", fontsize=14, fontweight="bold")
    
    # Total samples per experiment
    ax = axes[0]
    sns.barplot(data=df, x="experiment_short", y="total_samples", ax=ax, color="#A8DADC")
    ax.set_title("Total Samples per Configuration", fontsize=12, fontweight="bold")
    ax.set_xlabel("Configuration", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    
    # Stacked bar for empty/furnished breakdown
    ax = axes[1]
    breakdown_df = df[["experiment_short", "empty_samples", "furnished_samples", "total_samples"]].copy()
    breakdown_df = breakdown_df[breakdown_df["total_samples"] > 0]
    
    if len(breakdown_df) > 0:
        x = np.arange(len(breakdown_df))
        width = 0.6
        
        ax.bar(x, breakdown_df["empty_samples"], width, label="Empty", color="#A8DADC")
        ax.bar(x, breakdown_df["furnished_samples"], width, 
               bottom=breakdown_df["empty_samples"], label="Furnished", color="#F1FAEE")
        
        ax.set_xlabel("Configuration", fontsize=10)
        ax.set_ylabel("Number of Samples", fontsize=10)
        ax.set_title("Empty vs Furnished Breakdown", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(breakdown_df["experiment_short"], rotation=45, ha="right", fontsize=9)
        ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / f"06_sample_counts.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metric_radar(df: pd.DataFrame, output_dir: Path, format_ext: str = "png"):
    """Create radar charts for metric comparison."""
    # Normalize all metrics to 0-1 range
    metrics_to_plot = [
        "floor_iou", "wall_iou", "openings_iou",
        "presence_accuracy", "count_accuracy", "detection_f1",
        "mean_bbox_iou", "unified_score_all"
    ]
    
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    df_plot = df[["experiment_short"] + available_metrics].copy()
    df_plot = df_plot[(df_plot[available_metrics] != 0).any(axis=1)]
    
    if len(df_plot) == 0:
        return
    
    # Limit to 5 experiments for readability
    if len(df_plot) > 5:
        df_plot = df_plot.nlargest(5, "unified_score_all") if "unified_score_all" in df_plot.columns else df_plot.head(5)
    
    num_vars = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))
    
    colors = sns.color_palette("husl", len(df_plot))
    
    for idx, (_, row) in enumerate(df_plot.iterrows()):
        values = [row[m] for m in available_metrics]
        values += values[:1]
        
        ax.plot(angles, values, "o-", linewidth=2, label=row["experiment_short"], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in available_metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    ax.set_title("Metric Comparison Radar Chart", fontsize=14, fontweight="bold", pad=20)
    
    plt.tight_layout()
    output_path = output_dir / f"07_metrics_radar.{format_ext}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# Summary Statistics
# =============================================================================

def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate a text summary report of the results."""
    report_lines = [
        "=" * 80,
        "BASELINE EVALUATION RESULTS SUMMARY",
        "=" * 80,
        "",
        f"Number of Experiments: {len(df)}",
        f"Total Samples Evaluated: {int(df['total_samples'].sum())}",
        f"  - Empty Rooms: {int(df['empty_samples'].sum())}",
        f"  - Furnished Rooms: {int(df['furnished_samples'].sum())}",
        "",
    ]
    
    # Structure metrics
    report_lines.extend([
        "STRUCTURE METRICS (averaged across all experiments):",
        "-" * 80,
    ])
    
    for metric, label in [
        ("floor_iou", "Floor IoU"),
        ("wall_iou", "Wall IoU"),
        ("openings_iou", "Openings IoU"),
    ]:
        if metric in df.columns:
            data = df[df[metric] > 0][metric]
            if len(data) > 0:
                report_lines.append(
                    f"  {label:20s}: {data.mean():.4f} ± {data.std():.4f} "
                    f"(min: {data.min():.4f}, max: {data.max():.4f})"
                )
    
    report_lines.append("")
    
    # Furniture metrics
    furn_data = df[df["presence_accuracy"] > 0]
    if len(furn_data) > 0:
        report_lines.extend([
            "FURNITURE METRICS (for furnished rooms):",
            "-" * 80,
        ])
        
        for metric, label in [
            ("presence_accuracy", "Presence Accuracy"),
            ("count_accuracy", "Count Accuracy"),
            ("detection_f1", "Detection F1"),
            ("mean_l1_distance", "Mean L1 Distance"),
            ("mean_bbox_iou", "Mean BBox IoU"),
        ]:
            if metric in df.columns:
                data = furn_data[furn_data[metric] > 0][metric]
                if len(data) > 0:
                    report_lines.append(
                        f"  {label:20s}: {data.mean():.4f} ± {data.std():.4f} "
                        f"(min: {data.min():.4f}, max: {data.max():.4f})"
                    )
        
        report_lines.append("")
    
    # Unified scores
    report_lines.extend([
        "UNIFIED SCORES:",
        "-" * 80,
    ])
    
    for score_col, label in [
        ("unified_score_all", "All Samples"),
        ("unified_score_empty", "Empty Rooms"),
        ("unified_score_furnished", "Furnished Rooms"),
    ]:
        if score_col in df.columns:
            data = df[df[score_col].notna() & (df[score_col] > 0)][score_col]
            if len(data) > 0:
                report_lines.append(
                    f"  {label:20s}: {data.mean():.4f} ± {data.std():.4f} "
                    f"(min: {data.min():.4f}, max: {data.max():.4f})"
                )
    
    report_lines.extend([
        "",
        "PER-EXPERIMENT SUMMARY:",
        "-" * 80,
    ])
    
    # Create a simplified summary table
    summary_cols = [
        "experiment", "floor_iou", "presence_accuracy", "unified_score_all", "total_samples"
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    
    report_lines.append(f"{'Configuration':<35} {'Floor IoU':<12} {'Presence':<12} {'Unified':<12} {'Samples':<8}")
    report_lines.append("-" * 80)
    
    for _, row in df.iterrows():
        exp_name = str(row["experiment_short"])[:33]
        floor_iou = f"{row.get('floor_iou', 0):.4f}" if "floor_iou" in row and row["floor_iou"] > 0 else "N/A"
        presence = f"{row.get('presence_accuracy', 0):.4f}" if "presence_accuracy" in row and row["presence_accuracy"] > 0 else "N/A"
        unified = f"{row.get('unified_score_all', 0):.4f}" if "unified_score_all" in row and row["unified_score_all"] > 0 else "N/A"
        samples = int(row.get("total_samples", 0))
        
        report_lines.append(
            f"{exp_name:<35} {floor_iou:<12} {presence:<12} {unified:<12} {samples:<8}"
        )
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    # Write to file
    report_path = output_dir / "RESULTS_SUMMARY.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    # Print to console
    print("\n" + "\n".join(report_lines))
    print(f"\n✓ Report saved to: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize and summarize baseline evaluation results"
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
        help="Output directory for plots (default: results_dir/plots)"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "jpg"],
        default="png",
        help="Output format for plots"
    )
    parser.add_argument(
        "--no-radar",
        action="store_true",
        help="Skip radar chart generation"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir / "plots"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BASELINE EVALUATION RESULTS VISUALIZATION")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    # Load results
    print("Loading results...")
    all_results, df = load_results(args.results_dir)
    print(f"✓ Loaded {len(all_results)} experiment results\n")
    
    # Generate plots
    print("Generating plots...\n")
    
    plot_structure_metrics(df, args.output_dir, args.format)
    plot_furniture_metrics(df, args.output_dir, args.format)
    plot_unified_scores(df, args.output_dir, args.format)
    plot_all_metrics_heatmap(df, args.output_dir, args.format)
    plot_metric_distributions(df, args.output_dir, args.format)
    plot_sample_counts(df, args.output_dir, args.format)
    
    if not args.no_radar:
        plot_metric_radar(df, args.output_dir, args.format)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, args.output_dir)
    
    # Save processed data
    csv_path = args.output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV summary saved to: {csv_path}")
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print(f"✓ All plots saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()