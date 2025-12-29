#!/usr/bin/env python3
"""
Aggregate refinement CSV and compare per POV.

Takes refinement CSV files where POV 0 = baseline reference.
Creates aggregations and comparison plots showing progression from baseline.

Usage:
    python aggregate_and_compare.py /path/to/refinement_results
    python aggregate_and_compare.py /path/to/refinement_results --output ./analysis
"""

import argparse
from pathlib import Path
from typing import Dict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
sns.set_palette("husl")


def format_experiment_name(exp_name: str) -> str:
    """Convert long experiment name to short format."""
    exp_lower = exp_name.lower()
    
    size = None
    if "small" in exp_lower:
        size = "Small"
    elif "medium" in exp_lower:
        size = "Medium"
    elif "large" in exp_lower:
        size = "Large"
    
    arch = None
    if "down_bottleneck" in exp_lower:
        arch = "DN"
    elif "wide_shallow" in exp_lower:
        arch = "WS"
    
    conditioning = ""
    if "graph" in exp_lower:
        conditioning = "Graph"
    elif "povs" in exp_lower:
        conditioning = "POV"
    
    if size and arch:
        name = f"{arch}-{size}"
        if conditioning:
            name = f"{name} ({conditioning})"
        return name
    
    return exp_name


def load_and_aggregate(results_dir: Path) -> pd.DataFrame:
    """Load CSV files and aggregate, treating POV 0 as baseline."""
    
    results_dir = Path(results_dir)
    all_rows = []
    
    print("Loading refinement CSV files...\n")
    
    csv_files = sorted(results_dir.glob("*_results_*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {results_dir}")
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    for csv_file in csv_files:
        try:
            # Extract experiment name
            filename = csv_file.stem
            exp_name = filename.rsplit('_results', 1)[0]
            short_name = format_experiment_name(exp_name)
            
            # Read CSV
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded: {short_name}")
            
            # Aggregate by method and n_povs (and noise_strength for refinement)
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                
                for n_povs in sorted(method_df['n_povs'].unique()):
                    pov_df = method_df[method_df['n_povs'] == n_povs]
                    
                    if method == 'refinement':
                        # Separate by noise strength
                        for ns in pov_df['noise_strength'].unique():
                            if pd.isna(ns) or ns == "N/A":
                                continue
                            
                            ns_data = pov_df[pov_df['noise_strength'] == ns]
                            
                            row = {
                                'experiment': short_name,
                                'method': method,
                                'n_povs': int(n_povs),
                                'noise_strength': float(ns),
                                'n_samples': len(ns_data),
                                'unified_score_mean': float(ns_data['unified_score'].mean()),
                                'unified_score_std': float(ns_data['unified_score'].std()),
                                'mean_bbox_iou_mean': float(ns_data['mean_bbox_iou'].mean()),
                                'mean_bbox_iou_std': float(ns_data['mean_bbox_iou'].std()),
                                'detection_f1_mean': float(ns_data['detection_f1'].mean()),
                                'detection_f1_std': float(ns_data['detection_f1'].std()),
                                'count_accuracy_mean': float(ns_data['count_accuracy'].mean()),
                                'count_accuracy_std': float(ns_data['count_accuracy'].std()),
                            }
                            all_rows.append(row)
                    else:  # accumulation
                        row = {
                            'experiment': short_name,
                            'method': method,
                            'n_povs': int(n_povs),
                            'noise_strength': None,
                            'n_samples': len(pov_df),
                            'unified_score_mean': float(pov_df['unified_score'].mean()),
                            'unified_score_std': float(pov_df['unified_score'].std()),
                            'mean_bbox_iou_mean': float(pov_df['mean_bbox_iou'].mean()),
                            'mean_bbox_iou_std': float(pov_df['mean_bbox_iou'].std()),
                            'detection_f1_mean': float(pov_df['detection_f1'].mean()),
                            'detection_f1_std': float(pov_df['detection_f1'].std()),
                            'count_accuracy_mean': float(pov_df['count_accuracy'].mean()),
                            'count_accuracy_std': float(pov_df['count_accuracy'].std()),
                        }
                        all_rows.append(row)
        
        except Exception as e:
            print(f"✗ Error loading {csv_file}: {e}")
    
    return pd.DataFrame(all_rows)


def plot_progression_per_experiment(df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """Create progression plots for each experiment (POV 0 = baseline reference)."""
    
    experiments = sorted(df['experiment'].unique())
    
    for exp in experiments:
        exp_data = df[df['experiment'] == exp]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        fig.suptitle(f'{exp}: Baseline (POV 0) → Accumulation & Refinement', 
                    fontsize=14, fontweight='bold')
        
        metrics = [
            ('unified_score_mean', 'Unified Score'),
            ('mean_bbox_iou_mean', 'Mean BBox IoU'),
            ('detection_f1_mean', 'Detection F1'),
            ('count_accuracy_mean', 'Count Accuracy'),
        ]
        
        for idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[idx]
            
            all_values = []
            
            # Baseline (POV 0)
            baseline_acc = exp_data[(exp_data['method'] == 'accumulation') & (exp_data['n_povs'] == 0)]
            if len(baseline_acc) > 0:
                baseline_val = baseline_acc[metric_col].values[0]
                ax.axhline(y=baseline_val, color='#E74C3C', linestyle='--', linewidth=3.5, 
                          label='Baseline (POV 0)', alpha=0.95, zorder=10)
                all_values.append(baseline_val)
            
            # Accumulation (1-7 POVs)
            acc_data = exp_data[(exp_data['method'] == 'accumulation') & (exp_data['n_povs'] > 0)].sort_values('n_povs')
            if len(acc_data) > 0:
                ax.plot(acc_data['n_povs'], acc_data[metric_col], 
                       marker='o', linewidth=3.5, markersize=11, label='Accumulation', 
                       color='#3498DB', zorder=5)
                all_values.extend(acc_data[metric_col].tolist())
            
            # Refinement (1-7 POVs, by noise strength) - separate lines
            ref_data = exp_data[(exp_data['method'] == 'refinement') & (exp_data['n_povs'] > 0)].sort_values('n_povs')
            ns_list = sorted(ref_data['noise_strength'].dropna().unique())
            colors_ref = ['#2ECC71', '#F39C12', '#9B59B6']  # Green, Orange, Purple
            
            for color_idx, ns in enumerate(ns_list):
                ns_data = ref_data[ref_data['noise_strength'] == ns].sort_values('n_povs')
                if len(ns_data) > 0:
                    ax.plot(ns_data['n_povs'], ns_data[metric_col], 
                           marker='s', linewidth=3.5, markersize=10, 
                           label=f'Refinement (ns={ns})', 
                           color=colors_ref[color_idx % len(colors_ref)], alpha=0.95, zorder=4)
                    all_values.extend(ns_data[metric_col].tolist())
            
            # Smart y-axis scaling based on actual data
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                margin = (max_val - min_val) * 0.15  # 15% margin
                y_min = max(0, min_val - margin)
                y_max = min(1, max_val + margin)
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(0, 1)
            
            ax.set_xlabel('Number of POVs', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
            ax.set_title(metric_label, fontsize=13, fontweight='bold')
            ax.legend(fontsize=11, loc='best', framealpha=0.95)
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.set_xticks(range(0, 8))
            ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        exp_safe = exp.replace('(', '').replace(')', '').replace(' ', '_')
        output_path = output_dir / f"progression_{exp_safe}.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def plot_per_pov_comparison(df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """Create per-POV comparison plots per experiment (each experiment has its own baseline)."""
    
    experiments = sorted(df['experiment'].unique())
    
    for exp in experiments:
        exp_data = df[df['experiment'] == exp]
        n_povs_range = sorted(exp_data[exp_data['method'] == 'accumulation']['n_povs'].unique())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        fig.suptitle(f'{exp}: Baseline (POV 0) vs Accumulation vs Refinement - Per POV', 
                    fontsize=14, fontweight='bold')
        
        metrics = [
            ('unified_score_mean', 'Unified Score'),
            ('mean_bbox_iou_mean', 'Mean BBox IoU'),
            ('detection_f1_mean', 'Detection F1'),
            ('count_accuracy_mean', 'Count Accuracy'),
        ]
        
        x = np.arange(len(n_povs_range))
        width = 0.35
        
        for idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[idx]
            
            baseline_val = None
            acc_vals = []
            ref_vals = []
            
            for n_povs in n_povs_range:
                # Baseline (POV 0 only - for THIS experiment)
                if n_povs == 0:
                    baseline = exp_data[(exp_data['method'] == 'accumulation') & (exp_data['n_povs'] == 0)]
                    baseline_val = float(baseline[metric_col].values[0]) if len(baseline) > 0 else None
                
                # Accumulation at this POV
                acc = exp_data[(exp_data['method'] == 'accumulation') & (exp_data['n_povs'] == n_povs)]
                acc_vals.append(float(acc[metric_col].values[0]) if len(acc) > 0 else 0)
                
                # Refinement at this POV (average across noise strengths for THIS experiment)
                ref = exp_data[(exp_data['method'] == 'refinement') & (exp_data['n_povs'] == n_povs)]
                ref_vals.append(float(ref[metric_col].mean()) if len(ref) > 0 else 0)
            
            # Plot baseline as horizontal reference line
            if baseline_val is not None:
                ax.axhline(y=baseline_val, color='#E74C3C', linestyle='--', linewidth=3.5, 
                          label='Baseline (POV 0)', alpha=0.95, zorder=5)
            
            # Plot accumulation and refinement for all POVs
            bars1 = ax.bar(x - width/2, acc_vals, width, label='Accumulation', alpha=0.9, 
                          color='#3498DB', edgecolor='#2874A6', linewidth=2.5)
            bars2 = ax.bar(x + width/2, ref_vals, width, label='Refinement', alpha=0.9, 
                          color='#2ECC71', edgecolor='#27AE60', linewidth=2.5)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Smart y-axis scaling based on actual data
            all_vals = acc_vals + ref_vals
            if baseline_val is not None:
                all_vals.append(baseline_val)
            if all_vals:
                min_val = min(all_vals)
                max_val = max(all_vals)
                margin = (max_val - min_val) * 0.20  # 20% margin for labels
                y_min = max(0, min_val - margin)
                y_max = min(1, max_val + margin)
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(0, 1)
            
            ax.set_xlabel('Number of POVs', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
            ax.set_title(metric_label, fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{int(n)}' for n in n_povs_range], fontsize=11)
            ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
            ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        exp_safe = exp.replace('(', '').replace(')', '').replace(' ', '_')
        output_path = output_dir / f"per_pov_comparison_{exp_safe}.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive summary table showing per-POV comparison."""
    
    experiments = sorted(df['experiment'].unique())
    n_povs_range = sorted(df[df['method'] == 'accumulation']['n_povs'].unique())
    
    lines = []
    lines.append("=" * 150)
    lines.append("PER-POV COMPARISON: ACCUMULATION vs REFINEMENT")
    lines.append("=" * 150)
    lines.append("")
    lines.append("POV 0 = Baseline (starting point)")
    lines.append("POV 1-7 = Performance with additional POVs")
    lines.append("(Note: Higher at lower POV is better - shows efficient improvement)")
    lines.append("")
    
    for exp in experiments:
        exp_data = df[df['experiment'] == exp]
        
        lines.append(f"\n{exp}")
        lines.append("-" * 150)
        
        # Header
        header = "POV | Accumulation (unified_score) | Refinement (unified_score, by noise strength)"
        lines.append(header)
        lines.append("-" * 150)
        
        # Baseline (POV 0)
        baseline = exp_data[(exp_data['method'] == 'accumulation') & (exp_data['n_povs'] == 0)]
        if len(baseline) > 0:
            baseline_row = baseline.iloc[0]
            baseline_score = baseline_row['unified_score_mean']
            lines.append(f"  0 | {baseline_score:.4f} ± {baseline_row['unified_score_std']:.4f} (BASELINE) |")
        else:
            baseline_score = 0
        
        # Per-POV (1-7)
        for n_povs in n_povs_range:
            if n_povs == 0:
                continue
            
            acc_data = exp_data[(exp_data['method'] == 'accumulation') & (exp_data['n_povs'] == n_povs)]
            ref_data = exp_data[(exp_data['method'] == 'refinement') & (exp_data['n_povs'] == n_povs)]
            
            acc_str = ""
            if len(acc_data) > 0:
                acc_row = acc_data.iloc[0]
                acc_score = acc_row['unified_score_mean']
                acc_improve = (acc_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0
                acc_str = f"{acc_score:.4f} ± {acc_row['unified_score_std']:.4f} ({acc_improve:+.1f}%)"
            
            ref_str = ""
            if len(ref_data) > 0:
                ref_scores = []
                for _, ref_row in ref_data.iterrows():
                    ref_score = ref_row['unified_score_mean']
                    ref_improve = (ref_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0
                    ns = ref_row['noise_strength']
                    ref_scores.append(f"ns={ns}: {ref_score:.4f} ({ref_improve:+.1f}%)")
                ref_str = " | ".join(ref_scores)
            
            lines.append(f"  {int(n_povs)} | {acc_str:<30} | {ref_str}")
    
    lines.append("\n" + "=" * 150)
    
    # Write to file
    report_path = output_dir / "COMPARISON_SUMMARY.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print("\n" + "\n".join(lines))
    print(f"\n✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate refinement results per POV (POV 0 = baseline)"
    )
    parser.add_argument("results_dir", type=Path, help="Path to refinement_results directory")
    parser.add_argument("--output", type=Path, default=None, 
                       help="Output directory (default: refinement_results/comparison)")
    parser.add_argument("--format", choices=["png", "pdf"], default="png", 
                       help="Output format")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.results_dir / "comparison"
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AGGREGATE AND COMPARE: BASELINE (POV 0) → METHODS")
    print("=" * 80)
    print(f"Refinement results directory: {args.results_dir}")
    print(f"Output directory: {args.output}")
    print("")
    
    # Load and aggregate
    print("Loading and aggregating CSV files...\n")
    df = load_and_aggregate(args.results_dir)
    
    print(f"\n✓ Aggregated {len(df)} rows\n")
    
    # Save aggregated CSV
    csv_path = args.output / "aggregated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Aggregated CSV saved to: {csv_path}\n")
    
    # Create plots
    print("Generating plots...\n")
    plot_progression_per_experiment(df, args.output, args.format)
    plot_per_pov_comparison(df, args.output, args.format)
    
    # Generate summary
    print("\nGenerating summary report...")
    generate_summary_table(df, args.output)
    
    print("\n" + "=" * 80)
    print("✓ Aggregation and comparison complete!")
    print(f"✓ All files saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()