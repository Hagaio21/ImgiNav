#!/usr/bin/env python3
"""
Generic column distribution analysis for any manifest column.

This script analyzes the distribution of values in any column of a manifest CSV
and generates statistics, visualizations, and weighting information.

Usage:
    python analysis/analyze_column_distribution.py \
        --manifest datasets/augmented/manifest.csv \
        --column content_category \
        --output_dir analysis/content_category_distribution
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
from pathlib import Path
import json
import argparse
import sys
from collections import defaultdict, Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import safe_mkdir, write_json


def compute_weights_from_counts(counts_dict, method="inverse_frequency", max_weight=None, min_weight=1.0):
    """
    Compute weights from class counts using various methods.
    
    Args:
        counts_dict: Dict mapping class_id to count
        method: Weighting method ("inverse_frequency", "sqrt", "log", "balanced")
        max_weight: Maximum weight to cap (None = no capping)
        min_weight: Minimum weight (default: 1.0)
    
    Returns:
        Dict mapping class_id to weight
    """
    if not counts_dict:
        return {}
    
    counts = np.array(list(counts_dict.values()))
    max_count = max(counts)
    total_samples = sum(counts)
    num_classes = len(counts_dict)
    
    weights = {}
    
    if method == "inverse_frequency":
        # Standard inverse frequency: weight = total_samples / (num_classes * count)
        for class_id, count in counts_dict.items():
            weight = total_samples / (num_classes * count)
            weights[class_id] = weight
    
    elif method == "sqrt":
        # Square root weighting: weight = sqrt(max_count / count)
        for class_id, count in counts_dict.items():
            weight = np.sqrt(max_count / count)
            weights[class_id] = weight
    
    elif method == "log":
        # Logarithmic weighting: weight = log(max_count / count + 1)
        for class_id, count in counts_dict.items():
            weight = np.log(max_count / count + 1)
            weights[class_id] = weight
    
    elif method == "balanced":
        # Balanced: weight = max_count / count
        for class_id, count in counts_dict.items():
            weight = max_count / count
            weights[class_id] = weight
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Apply min/max constraints
    if min_weight is not None:
        weights = {k: max(v, min_weight) for k, v in weights.items()}
    
    if max_weight is not None:
        weights = {k: min(v, max_weight) for k, v in weights.items()}
    
    return weights


def analyze_column_distribution(
    manifest_path: Path,
    column_name: str,
    output_dir: Path,
    rare_threshold_percentile: float = 10.0,
    min_samples_threshold: int = 50,
    weighting_method: str = "inverse_frequency",
    max_weight: float = None,
    min_weight: float = 1.0
):
    """
    Analyze distribution of values in a manifest column.
    
    Args:
        manifest_path: Path to CSV manifest
        column_name: Name of column to analyze
        output_dir: Output directory for results
        rare_threshold_percentile: Percentile below which classes are considered rare
        min_samples_threshold: Minimum samples to avoid memorization
        weighting_method: Method for computing weights ("inverse_frequency", "sqrt", "log", "balanced")
        max_weight: Maximum weight to cap (None = no capping)
        min_weight: Minimum weight
    """
    output_dir = Path(output_dir)
    safe_mkdir(output_dir)
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} samples")
    
    # Check if column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in manifest. Available columns: {list(df.columns)}")
    
    # Filter out NaN values
    valid_df = df[df[column_name].notna()].copy()
    invalid_count = len(df) - len(valid_df)
    
    if invalid_count > 0:
        print(f"Warning: {invalid_count} rows have missing values in '{column_name}', will be excluded from analysis")
    
    if len(valid_df) == 0:
        raise ValueError(f"No valid values found in column '{column_name}'")
    
    # ============================================================
    # 1. Overall distribution
    # ============================================================
    print("\n" + "="*60)
    print(f"1. Overall {column_name} Distribution")
    print("="*60)
    
    value_counts = valid_df[column_name].value_counts().sort_values(ascending=False)
    total_samples = len(valid_df)
    
    class_stats = []
    for class_id, count in value_counts.items():
        percentage = (count / total_samples) * 100
        class_stats.append({
            "class_id": str(class_id),
            "count": int(count),
            "percentage": float(percentage)
        })
        print(f"  {str(class_id):30s}: {count:6d} samples ({percentage:5.2f}%)")
    
    # ============================================================
    # 2. Identify rare classes
    # ============================================================
    print("\n" + "="*60)
    print("2. Rare Class Identification")
    print("="*60)
    
    counts_array = np.array([stat["count"] for stat in class_stats])
    threshold_count = np.percentile(counts_array, rare_threshold_percentile)
    
    print(f"  Rare class threshold (count < {threshold_count:.0f}, {rare_threshold_percentile}th percentile)")
    print(f"  Minimum samples for training: {min_samples_threshold}")
    print(f"  Total classes: {len(class_stats)}")
    
    rare_classes = []
    common_classes = []
    extremely_rare_classes = []
    
    for stat in class_stats:
        if stat["count"] < min_samples_threshold:
            extremely_rare_classes.append(stat)
            rare_classes.append(stat)
        elif stat["count"] < threshold_count:
            rare_classes.append(stat)
        else:
            common_classes.append(stat)
    
    print(f"  Rare classes: {len(rare_classes)}")
    print(f"  Common classes: {len(common_classes)}")
    print(f"  ⚠️  EXTREMELY RARE (may cause memorization): {len(extremely_rare_classes)}")
    
    if rare_classes:
        print("\n  Rare classes list:")
        for stat in sorted(rare_classes, key=lambda x: x["count"]):
            warning = " ⚠️ MEMORIZATION RISK" if stat["count"] < min_samples_threshold else ""
            print(f"    {stat['class_id']:30s}: {stat['count']:6d} samples{warning}")
    
    # ============================================================
    # 3. Compute weights
    # ============================================================
    print("\n" + "="*60)
    print(f"3. Weighting Strategy ({weighting_method})")
    print("="*60)
    
    # Create counts dict
    counts_dict = {stat["class_id"]: stat["count"] for stat in class_stats}
    
    # Compute weights
    weights = compute_weights_from_counts(
        counts_dict,
        method=weighting_method,
        max_weight=max_weight,
        min_weight=min_weight
    )
    
    # Normalize weights (optional - for display)
    max_weight_value = max(weights.values()) if weights else 1.0
    normalized_weights = {k: v / max_weight_value for k, v in weights.items()}
    
    print(f"  Weighting method: {weighting_method}")
    if max_weight:
        print(f"  Max weight cap: {max_weight}")
    print(f"  Min weight: {min_weight}")
    print(f"\n  Class weights (top 20):")
    sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
    for class_id, weight in sorted_weights[:20]:
        count = counts_dict[class_id]
        norm_weight = normalized_weights[class_id]
        print(f"    {class_id:30s}: count={count:6d}, weight={weight:8.3f} (normalized: {norm_weight:.3f})")
    
    # ============================================================
    # 4. Group rare classes (optional)
    # ============================================================
    print("\n" + "="*60)
    print("4. Rare Class Grouping")
    print("="*60)
    
    rare_class_ids = [stat["class_id"] for stat in rare_classes]
    
    # Create grouping: rare classes -> "rare", others -> themselves
    class_grouping = {}
    for stat in class_stats:
        if stat["class_id"] in rare_class_ids:
            class_grouping[stat["class_id"]] = "rare"
        else:
            class_grouping[stat["class_id"]] = stat["class_id"]
    
    # Compute grouped counts
    grouped_counts = defaultdict(int)
    for class_id, grouped_id in class_grouping.items():
        count = counts_dict[class_id]
        grouped_counts[grouped_id] += count
    
    # Compute grouped weights
    grouped_weights = compute_weights_from_counts(
        dict(grouped_counts),
        method=weighting_method,
        max_weight=max_weight,
        min_weight=min_weight
    )
    
    print(f"  Grouped classes into {len(grouped_counts)} categories:")
    for grouped_id, count in sorted(grouped_counts.items(), key=lambda x: -x[1]):
        weight = grouped_weights[grouped_id]
        if grouped_id == "rare":
            print(f"    {grouped_id:30s}: {count:6d} samples, weight={weight:.3f} (includes {len(rare_classes)} rare classes)")
        else:
            print(f"    {grouped_id:30s}: {count:6d} samples, weight={weight:.3f}")
    
    # ============================================================
    # Save results
    # ============================================================
    results = {
        "column_name": column_name,
        "total_samples": int(total_samples),
        "total_classes": len(class_stats),
        "rare_threshold_count": float(threshold_count),
        "rare_threshold_percentile": rare_threshold_percentile,
        "rare_classes_count": len(rare_classes),
        "common_classes_count": len(common_classes),
        "extremely_rare_classes_count": len(extremely_rare_classes),
        "min_samples_threshold": min_samples_threshold,
        "weighting_method": weighting_method,
        "max_weight": max_weight,
        "min_weight": min_weight,
        "class_statistics": class_stats,
        "rare_classes": [{"class_id": s["class_id"], "count": s["count"]} for s in rare_classes],
        "extremely_rare_classes": [{"class_id": s["class_id"], "count": s["count"]} for s in extremely_rare_classes],
        "weights": {k: float(v) for k, v in weights.items()},
        "normalized_weights": {k: float(v) for k, v in normalized_weights.items()},
        "class_grouping": class_grouping,
        "grouped_counts": {k: int(v) for k, v in grouped_counts.items()},
        "grouped_weights": {k: float(v) for k, v in grouped_weights.items()}
    }
    
    write_json(results, output_dir / f"{column_name}_distribution_stats.json")
    
    # ============================================================
    # Create visualizations
    # ============================================================
    print("\n" + "="*60)
    print("5. Creating Visualizations")
    print("="*60)
    
    # Plot 1: Top 20 classes
    fig, ax = plt.subplots(figsize=(14, 8))
    top_classes = class_stats[:20]
    class_names = [s["class_id"] for s in top_classes]
    counts = [s["count"] for s in top_classes]
    
    df_top20 = pd.DataFrame({"Class": class_names, "Count": counts})
    sns.barplot(data=df_top20, y="Class", x="Count", ax=ax, palette="viridis")
    ax.set_xlabel("Sample Count", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(f"Top 20 {column_name} Distribution (Total: {total_samples:,} samples)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{column_name}_distribution_top20.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: All classes
    fig, ax = plt.subplots(figsize=(14, max(8, len(class_stats) * 0.3)))
    all_class_names = [s["class_id"] for s in class_stats]
    all_counts = [s["count"] for s in class_stats]
    is_rare = [stat["class_id"] in rare_class_ids for stat in class_stats]
    
    df_all = pd.DataFrame({
        "Class": all_class_names,
        "Count": all_counts,
        "IsRare": is_rare
    })
    
    # Create color palette: red for rare, blue for common
    colors = ['#d62728' if rare else '#2ca02c' for rare in is_rare]
    
    sns.barplot(data=df_all, y="Class", x="Count", ax=ax, palette=colors)
    ax.axvline(x=threshold_count, color='orange', linestyle='--', linewidth=2, 
               label=f'Rare threshold ({threshold_count:.0f})', zorder=10)
    ax.set_xlabel("Sample Count", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(f"All {column_name} Distribution (Red = Rare, Green = Common)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"{column_name}_distribution_all.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Pie chart
    plt.figure(figsize=(14, 10))
    
    # Group small classes into "Others"
    min_percentage_for_main = 2.0
    min_count_for_main = int(total_samples * min_percentage_for_main / 100)
    max_main_classes = 12
    
    main_classes = []
    other_classes = []
    for stat in class_stats:
        if stat["count"] >= min_count_for_main:
            main_classes.append(stat)
        else:
            other_classes.append(stat)
    
    if len(main_classes) > max_main_classes:
        top_main = main_classes[:max_main_classes - 1]
        remaining_main = main_classes[max_main_classes - 1:]
        other_classes = remaining_main + other_classes
        main_classes = top_main
    
    if other_classes:
        other_total = sum(s["count"] for s in other_classes)
        pie_labels = [s["class_id"] for s in main_classes] + [f"Others ({len(other_classes)} classes)"]
        pie_counts = [s["count"] for s in main_classes] + [other_total]
        pie_colors = ['red' if stat["class_id"] in rare_class_ids else 'steelblue' for stat in main_classes] + ['lightgray']
    else:
        pie_labels = [s["class_id"] for s in main_classes]
        pie_counts = [s["count"] for s in main_classes]
        pie_colors = ['red' if stat["class_id"] in rare_class_ids else 'steelblue' for stat in main_classes]
    
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(pie_labels)) if len(pie_labels) <= 10 else sns.color_palette("Set3", len(pie_labels))
    if len(main_classes) <= max_main_classes:
        colors = pie_colors
    
    wedges, texts, autotexts = plt.pie(
        pie_counts,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 9, 'color': 'white', 'fontweight': 'bold'}
    )
    
    legend_labels = []
    for i, (label, count) in enumerate(zip(pie_labels, pie_counts)):
        percentage = (count / total_samples) * 100
        legend_labels.append(f"{label}: {count:,} ({percentage:.1f}%)")
    
    plt.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    plt.title(f"{column_name} Distribution\n(Total: {total_samples:,} samples, {len(class_stats)} classes)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"{column_name}_distribution_pie.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Weight visualization
    plt.figure(figsize=(14, max(8, len(sorted_weights) * 0.3)))
    top_weights = sorted_weights[:30]  # Top 30 by weight
    weight_class_names = [w[0] for w in top_weights]
    weight_values = [w[1] for w in top_weights]
    
    plt.barh(range(len(weight_class_names)), weight_values)
    plt.yticks(range(len(weight_class_names)), weight_class_names)
    plt.xlabel("Weight")
    plt.title(f"Class Weights ({weighting_method} method)")
    if max_weight:
        plt.axvline(x=max_weight, color='red', linestyle='--', label=f'Max weight cap ({max_weight})')
        plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f"{column_name}_weights.png", dpi=200)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - {column_name}_distribution_stats.json")
    print(f"  - {column_name}_distribution_top20.png")
    print(f"  - {column_name}_distribution_all.png")
    print(f"  - {column_name}_distribution_pie.png")
    print(f"  - {column_name}_weights.png")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze distribution of any column in manifest CSV")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to dataset CSV manifest")
    parser.add_argument("--column", type=str, required=True, help="Column name to analyze")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for analysis results")
    parser.add_argument("--rare_threshold_percentile", type=float, default=10.0,
                        help="Percentile below which classes are considered rare (default: 10.0)")
    parser.add_argument("--min_samples_threshold", type=int, default=50,
                        help="Minimum samples to avoid memorization (default: 50)")
    parser.add_argument("--weighting_method", type=str, default="inverse_frequency",
                        choices=["inverse_frequency", "sqrt", "log", "balanced"],
                        help="Weighting method (default: inverse_frequency)")
    parser.add_argument("--max_weight", type=float, default=None,
                        help="Maximum weight cap (default: None)")
    parser.add_argument("--min_weight", type=float, default=1.0,
                        help="Minimum weight (default: 1.0)")
    
    args = parser.parse_args()
    
    analyze_column_distribution(
        manifest_path=args.manifest,
        column_name=args.column,
        output_dir=args.output_dir,
        rare_threshold_percentile=args.rare_threshold_percentile,
        min_samples_threshold=args.min_samples_threshold,
        weighting_method=args.weighting_method,
        max_weight=args.max_weight,
        min_weight=args.min_weight
    )

