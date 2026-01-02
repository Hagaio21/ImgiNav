#!/usr/bin/env python3
"""
Complete Baseline Evaluation Analysis for Thesis Results Chapter

Generates:
  1. Statistical analysis plots (8 main figures for results chapter) - FIGURES 1-8
  2. Qualitative results (Target/POV on top, models below) - FIGURE 9
  3. Appendix figures (A1-A18: same samples across all models) - APPENDIX
  4. Overleaf-ready folder structure with LaTeX snippets

Output structure:
  results_figures/
  ├── results_chapter/           # Main results chapter figures (01-09)
  │   ├── 01_*.pdf              # Architecture comparisons
  │   ├── 02_*.pdf              # Capacity scaling
  │   ├── 03_*.pdf              # Conditioning modality
  │   └── ... 09_qualitative_*  # Qualitative results
  ├── appendix/                  # Detailed appendix figures (A1-A18)
  │   ├── A1_POV_best_empty.pdf
  │   ├── A2_POV_best_furnished.pdf
  │   └── ... A18_Both_worst_furnished.pdf
  └── latex_snippets/            # LaTeX code for inclusion
      └── figures.tex            # All figure references with captions

Usage:
    python analyze_baseline_complete.py /path/to/evaluation_results
    python analyze_baseline_complete.py C:\\Users\\...\\evaluation_results
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from PIL import Image

warnings.filterwarnings("ignore")

# Configure seaborn for publication quality
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

# VIBRANT COLOR PALETTE
COLORS = {
    "architecture": {
        "DN": "#0173B2",
        "WS": "#DE8F05",
    },
    "capacity": {
        "Small": "#CC78BC",
        "Medium": "#CA9161",
        "Large": "#56B4E9",
    },
    "conditioning": {
        "POV": "#0173B2",
        "Graph": "#029E73",
        "Both": "#DE8F05",
    },
    "room_type": {
        "Empty": "#D45113",
        "Furnished": "#0173B2",
    },
}


# =============================================================================
# DATA LOADING AND PARSING
# =============================================================================

def parse_experiment_name(exp_name: str) -> Dict[str, Optional[str]]:
    """Parse experiment name into components."""
    exp_lower = exp_name.lower()
    
    result = {
        "full_name": exp_name,
        "architecture": None,
        "capacity": None,
        "conditioning": None,
    }
    
    if "down_bottleneck" in exp_lower:
        result["architecture"] = "DN"
    elif "wide_shallow" in exp_lower:
        result["architecture"] = "WS"
    
    if "small" in exp_lower:
        result["capacity"] = "Small"
    elif "medium" in exp_lower:
        result["capacity"] = "Medium"
    elif "large" in exp_lower:
        result["capacity"] = "Large"
    
    if "graph" in exp_lower:
        result["conditioning"] = "Graph"
    elif "pov" in exp_lower or "povs" in exp_lower:
        result["conditioning"] = "POV"
    elif "both" in exp_lower:
        result["conditioning"] = "Both"
    
    return result


def format_experiment_label(exp_name: str) -> str:
    """Create readable label."""
    parsed = parse_experiment_name(exp_name)
    arch = parsed["architecture"] or "?"
    capacity = parsed["capacity"] or "?"
    cond = parsed["conditioning"] or "—"
    
    if cond == "—":
        return f"{arch}-{capacity}"
    else:
        return f"{arch}-{capacity}\n({cond})"


def find_metrics_csv(results_dir: Path, exp_name: str) -> Optional[Path]:
    """Find metrics.csv for an experiment."""
    exp_dir = results_dir / exp_name
    
    # Try direct path
    if (exp_dir / "metrics.csv").exists():
        return exp_dir / "metrics.csv"
    
    # Try analysis folder
    if (exp_dir / "analysis" / "metrics.csv").exists():
        return exp_dir / "analysis" / "metrics.csv"
    
    return None


def load_all_metrics_from_csv(results_dir: Path) -> pd.DataFrame:
    """Load all metrics.csv files from experiments."""
    results_dir = Path(results_dir)
    all_data = []
    
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        csv_file = find_metrics_csv(results_dir, exp_dir.name)
        if csv_file is None:
            print(f"⚠ Skipped: {exp_dir.name} (no metrics.csv found)")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            exp_name = exp_dir.name
            parsed = parse_experiment_name(exp_name)
            
            df["experiment"] = exp_name
            df["experiment_label"] = format_experiment_label(exp_name)
            df["architecture"] = parsed["architecture"]
            df["capacity"] = parsed["capacity"]
            df["conditioning"] = parsed["conditioning"]
            
            all_data.append(df)
            print(f"✓ Loaded: {format_experiment_label(exp_name):25s} ({len(df)} samples)")
        except Exception as e:
            print(f"✗ Error loading {csv_file}: {e}")
    
    if not all_data:
        raise ValueError(f"No metrics.csv files found in {results_dir}")
    
    return pd.concat(all_data, ignore_index=True)



# =============================================================================
# IMAGE LOADING UTILITIES
# =============================================================================

def load_generated_floorplan(exp_dir: Path, sample_idx: int) -> Optional[Image.Image]:
    """Load generated floorplan prediction for a sample (model output)."""
    images_dir = exp_dir / "images"
    if not images_dir.exists():
        return None
    
    # Look for {0000-9999}_*_pred.png (generated floorplan)
    pattern = f"{sample_idx:04d}_*_pred.png"
    matching = list(images_dir.glob(pattern))
    
    if matching:
        try:
            return Image.open(matching[0])
        except:
            pass
    return None


def load_ground_truth_floorplan(exp_dir: Path, sample_idx: int) -> Optional[Image.Image]:
    """Load ground truth floorplan for a sample (what we should generate)."""
    images_dir = exp_dir / "images"
    if not images_dir.exists():
        return None
    
    # Look for {0000-9999}_*_target.png (ground truth floorplan)
    pattern = f"{sample_idx:04d}_*_target.png"
    matching = list(images_dir.glob(pattern))
    
    if matching:
        try:
            return Image.open(matching[0])
        except:
            pass
    return None


def load_pov_conditioning_input(exp_dir: Path, sample_idx: int) -> Optional[Image.Image]:
    """Load POV (camera view) conditioning input for a sample."""
    cond_dir = exp_dir / "images" / "conditions"
    if not cond_dir.exists():
        return None
    
    # Look specifically for {0000-9999}_pov.png (camera viewpoint)
    pov_file = cond_dir / f"{sample_idx:04d}_pov.png"
    if pov_file.exists():
        try:
            return Image.open(pov_file)
        except:
            pass
    return None


def load_graph_conditioning_text(exp_dir: Path, sample_idx: int) -> Optional[str]:
    """Load graph/text conditioning input for a sample."""
    cond_dir = exp_dir / "images" / "conditions"
    if not cond_dir.exists():
        return None
    
    # Look for {0000-9999}_text.txt (graph or text conditioning)
    text_file = cond_dir / f"{sample_idx:04d}_text.txt"
    if text_file.exists():
        try:
            with open(text_file, 'r') as f:
                content = f.read().strip()
                # Limit text length for display
                if len(content) > 200:
                    content = content[:197] + "..."
                return content
        except:
            pass
    return None


def get_best_sample_for_conditioning_group(df: pd.DataFrame, exp_dirs: List[Path], 
                                          room_type: str = "furnished") -> Optional[int]:
    """
    Get the best sample index across all models in a conditioning group.
    Uses unified_score to rank samples.
    """
    exp_names = [d.name for d in exp_dirs]
    group_df = df[df["experiment"].isin(exp_names)].copy()
    
    if "is_empty" in group_df.columns and group_df["is_empty"].dtype == object:
        group_df["is_empty"] = group_df["is_empty"].astype(str).str.upper() == "TRUE"
    
    if room_type == "empty":
        group_df = group_df[group_df["is_empty"] == True]
    else:
        group_df = group_df[group_df["is_empty"] == False]
    
    if len(group_df) == 0:
        return None
    
    best_row = group_df.loc[group_df["unified_score"].idxmax()]
    return int(best_row["eval_idx"])


def get_multiple_samples_for_conditioning_group(df: pd.DataFrame, exp_dirs: List[Path], 
                                               room_type: str = "furnished",
                                               num_samples: int = 3) -> List[Tuple[int, str]]:
    """
    Get multiple sample indices (best, median, worst) for a conditioning group.
    Returns list of (sample_idx, rank_name) tuples.
    """
    exp_names = [d.name for d in exp_dirs]
    group_df = df[df["experiment"].isin(exp_names)].copy()
    
    if "is_empty" in group_df.columns and group_df["is_empty"].dtype == object:
        group_df["is_empty"] = group_df["is_empty"].astype(str).str.upper() == "TRUE"
    
    if room_type == "empty":
        group_df = group_df[group_df["is_empty"] == True]
    else:
        group_df = group_df[group_df["is_empty"] == False]
    
    if len(group_df) == 0:
        return []
    
    # Sort by unified_score
    sorted_df = group_df.sort_values("unified_score", ascending=False).reset_index(drop=True)
    samples = []
    
    # Best (highest unified_score)
    samples.append((int(sorted_df.iloc[0]["eval_idx"]), "best"))
    
    # Median (middle score)
    median_idx = len(sorted_df) // 2
    samples.append((int(sorted_df.iloc[median_idx]["eval_idx"]), "median"))
    
    # Worst (lowest unified_score)
    samples.append((int(sorted_df.iloc[-1]["eval_idx"]), "worst"))
    
    return samples


# =============================================================================
# STATISTICAL PLOTS (Results Chapter Figures 1-8)
# =============================================================================

def plot_architecture_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare architectures."""
    metrics = ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]
    metric_labels = ["Floor IoU", "Presence Accuracy", "Detection F1", "Unified Score"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Architecture Comparison: Deep-Narrow vs Wide-Shallow", 
                 fontsize=15, fontweight="bold", y=0.995)
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        arch_data = []
        arch_labels = []
        
        for arch in ["DN", "WS"]:
            arch_df = df[df["architecture"] == arch][metric].dropna()
            if len(arch_df) > 0:
                arch_data.append(arch_df.values)
                arch_labels.append(arch)
        
        if arch_data:
            bp = ax.boxplot(arch_data, labels=arch_labels, patch_artist=True,
                           widths=0.6, showmeans=True)
            
            for patch, arch in zip(bp['boxes'], arch_labels):
                patch.set_facecolor(COLORS["architecture"][arch])
                patch.set_alpha(0.7)
            
            ax.set_ylabel(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Architecture", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim([0, 1.0])
            
            for i, arch in enumerate(arch_labels):
                mean_val = np.mean(arch_data[i])
                ax.text(i+1, mean_val + 0.05, f"{mean_val:.3f}", 
                       ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "01_architecture_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_capacity_scaling(df: pd.DataFrame, output_dir: Path):
    """Show capacity scaling effects."""
    metrics = ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]
    metric_labels = ["Floor IoU", "Presence Accuracy", "Detection F1", "Unified Score"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Capacity Scaling: Small → Medium → Large", 
                 fontsize=15, fontweight="bold", y=0.995)
    
    capacity_order = {"Small": 0, "Medium": 1, "Large": 2}
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        capacity_data = defaultdict(list)
        
        for _, row in df.iterrows():
            if pd.notna(row[metric]) and row["capacity"]:
                capacity_data[row["capacity"]].append(row[metric])
        
        if capacity_data:
            caps = sorted(capacity_data.keys(), key=lambda x: capacity_order.get(x, 99))
            data = [capacity_data[cap] for cap in caps]
            
            bp = ax.boxplot(data, labels=caps, patch_artist=True,
                           widths=0.6, showmeans=True)
            
            for patch, cap in zip(bp['boxes'], caps):
                patch.set_facecolor(COLORS["capacity"][cap])
                patch.set_alpha(0.7)
            
            ax.set_ylabel(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Model Capacity", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim([0, 1.0])
            
            for i, cap in enumerate(caps):
                mean_val = np.mean(data[i])
                ax.text(i+1, mean_val + 0.05, f"{mean_val:.3f}", 
                       ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "02_capacity_scaling.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_conditioning_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare conditioning modalities."""
    metrics = ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]
    metric_labels = ["Floor IoU", "Presence Accuracy", "Detection F1", "Unified Score"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Conditioning Modality: POV vs Graph vs Both", 
                 fontsize=15, fontweight="bold", y=0.995)
    
    cond_order = {"POV": 0, "Graph": 1, "Both": 2}
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        cond_data = defaultdict(list)
        
        for _, row in df.iterrows():
            if pd.notna(row[metric]) and row["conditioning"]:
                cond_data[row["conditioning"]].append(row[metric])
        
        if cond_data:
            conds = sorted(cond_data.keys(), key=lambda x: cond_order.get(x, 99))
            data = [cond_data[cond] for cond in conds]
            
            bp = ax.boxplot(data, labels=conds, patch_artist=True,
                           widths=0.6, showmeans=True)
            
            for patch, cond in zip(bp['boxes'], conds):
                patch.set_facecolor(COLORS["conditioning"][cond])
                patch.set_alpha(0.7)
            
            ax.set_ylabel(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Conditioning Modality", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim([0, 1.0])
            
            for i, cond in enumerate(conds):
                mean_val = np.mean(data[i])
                ax.text(i+1, mean_val + 0.05, f"{mean_val:.3f}", 
                       ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "03_conditioning_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_empty_vs_furnished(df: pd.DataFrame, output_dir: Path):
    """Compare empty vs furnished rooms."""
    if "is_empty" not in df.columns:
        print("⚠ Skipping empty vs furnished comparison")
        return
    
    if df["is_empty"].dtype == object:
        df["is_empty"] = df["is_empty"].astype(str).str.upper() == "TRUE"
    
    metrics = ["floor_iou", "detection_f1", "unified_score"]
    metric_labels = ["Floor IoU", "Detection F1", "Unified Score"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Empty vs Furnished Rooms Comparison", 
                 fontsize=15, fontweight="bold", y=1.00)
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        empty_data = df[df["is_empty"] == True][metric].dropna()
        furnished_data = df[df["is_empty"] == False][metric].dropna()
        
        data = [empty_data.values, furnished_data.values]
        labels = ["Empty", "Furnished"]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       widths=0.6, showmeans=True)
        
        colors_list = [COLORS["room_type"]["Empty"], COLORS["room_type"]["Furnished"]]
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Room Type", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        if len(empty_data) > 1 and len(furnished_data) > 1:
            t_stat, p_val = stats.ttest_ind(empty_data, furnished_data)
            ax.text(0.5, 0.95, f"p={p_val:.4f}", transform=ax.transAxes,
                   ha="center", va="top", fontsize=9, 
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        for i, (data_arr, lbl) in enumerate(zip([empty_data.values, furnished_data.values], labels)):
            mean_val = np.mean(data_arr)
            ax.text(i+1, mean_val + 0.05, f"{mean_val:.3f}", 
                   ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "04_empty_vs_furnished.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metric_correlations(df: pd.DataFrame, output_dir: Path):
    """Plot correlation matrix."""
    metric_cols = [
        "floor_iou", "wall_iou", "openings_iou",
        "presence_accuracy", "count_accuracy", "detection_f1",
        "mean_bbox_iou", "unified_score"
    ]
    
    available_cols = [col for col in metric_cols if col in df.columns]
    corr_df = df[available_cols].corr()
    
    col_names = {
        "floor_iou": "Floor",
        "wall_iou": "Wall",
        "openings_iou": "Openings",
        "presence_accuracy": "Presence",
        "count_accuracy": "Count",
        "detection_f1": "Detection",
        "mean_bbox_iou": "BBox",
        "unified_score": "Unified"
    }
    
    corr_df = corr_df.rename(columns=col_names, index=col_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
               vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Correlation"},
               square=True, linewidths=0.5)
    
    ax.set_title("Metric Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = output_dir / "05_metric_correlations.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_experiment_comparison(df: pd.DataFrame, output_dir: Path):
    """Create per-experiment comparison."""
    metrics = ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]
    metric_labels = ["Floor IoU", "Presence Accuracy", "Detection F1", "Unified Score"]
    
    exp_summary = df.groupby("experiment_label")[metrics].mean().reset_index()
    exp_summary = exp_summary.sort_values("unified_score", ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Per-Experiment Performance Ranking", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        data = exp_summary.sort_values(metric, ascending=False)
        
        colors = []
        for _, row in data.iterrows():
            if "POV" in row["experiment_label"]:
                colors.append(COLORS["conditioning"]["POV"])
            elif "Graph" in row["experiment_label"]:
                colors.append(COLORS["conditioning"]["Graph"])
            elif "Both" in row["experiment_label"]:
                colors.append(COLORS["conditioning"]["Both"])
            else:
                colors.append("#888888")
        
        ax.barh(range(len(data)), data[metric], color=colors, alpha=0.8)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data["experiment_label"], fontsize=8)
        ax.set_xlabel(label, fontsize=11, fontweight="bold")
        ax.set_xlim([0, 1.0])
        ax.grid(axis="x", alpha=0.3)
        
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.text(row[metric] + 0.02, i, f"{row[metric]:.3f}", 
                   va="center", fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / "06_per_experiment_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metric_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot metric distributions separated by room type (empty vs furnished)."""
    if "is_empty" not in df.columns:
        print("⚠ Skipping metric distributions (no is_empty column)")
        return
    
    df_copy = df.copy()
    if df_copy["is_empty"].dtype == object:
        df_copy["is_empty"] = df_copy["is_empty"].astype(str).str.upper() == "TRUE"
    
    metrics = ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]
    metric_labels = ["Floor IoU", "Presence Accuracy", "Detection F1", "Unified Score"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metric Distributions: Empty vs Furnished Rooms", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        empty_data = df_copy[df_copy["is_empty"] == True][metric].dropna()
        furnished_data = df_copy[df_copy["is_empty"] == False][metric].dropna()
        
        # Plot histograms with transparency
        ax.hist(empty_data, bins=30, color=COLORS["room_type"]["Empty"], alpha=0.6, 
                label=f"Empty (n={len(empty_data)})", edgecolor="black")
        ax.hist(furnished_data, bins=30, color=COLORS["room_type"]["Furnished"], alpha=0.6, 
                label=f"Furnished (n={len(furnished_data)})", edgecolor="black")
        
        # Add vertical lines for means
        if len(empty_data) > 0:
            ax.axvline(empty_data.mean(), color=COLORS["room_type"]["Empty"], 
                      linestyle="--", linewidth=2.5, alpha=0.8)
        if len(furnished_data) > 0:
            ax.axvline(furnished_data.mean(), color=COLORS["room_type"]["Furnished"], 
                      linestyle="--", linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
    
    plt.tight_layout()
    output_path = output_dir / "07_metric_distributions.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# QUALITATIVE RESULTS (Figure 8: Results Chapter)
# =============================================================================

def create_qualitative_results(results_dir: Path, output_dir: Path, df: pd.DataFrame):
    """Create Figure 9: Qualitative results - same sample across all models per conditioning type."""
    
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and 
                      (d / "images").exists()])
    
    if not exp_dirs:
        print("⚠ No image directories found for qualitative results")
        return
    
    cond_groups = defaultdict(list)
    for exp_dir in exp_dirs:
        parsed = parse_experiment_name(exp_dir.name)
        if parsed["conditioning"]:
            cond_groups[parsed["conditioning"]].append(exp_dir)
    
    fig_counter = 8
    
    for cond in ["POV", "Graph", "Both"]:
        if cond not in cond_groups:
            continue
        
        exp_dirs_cond = sorted(cond_groups[cond], 
                              key=lambda x: (parse_experiment_name(x.name)["architecture"] or "", 
                                            parse_experiment_name(x.name)["capacity"] or ""))
        
        # Get a furnished sample with good unified_score for this conditioning group
        sample_idx = get_best_sample_for_conditioning_group(
            df, exp_dirs_cond, room_type="furnished"
        )
        
        if sample_idx is None:
            print(f"  ⚠ No suitable sample found for {cond} qualitative")
            continue
        
        # Load images from all models for this SAME sample
        model_images = []
        model_labels = []
        target_img = None
        pov_img = None
        graph_text = None
        
        for exp_dir in exp_dirs_cond:
            gen_img = load_generated_floorplan(exp_dir, sample_idx)
            if gen_img:
                model_images.append(gen_img)
                parsed = parse_experiment_name(exp_dir.name)
                label = f"{parsed['architecture']}-{parsed['capacity']}"
                model_labels.append(label)
                
                # Load target once (same for all models)
                if target_img is None:
                    target_img = load_ground_truth_floorplan(exp_dir, sample_idx)
                # Load POV for POV/Both conditioning
                if pov_img is None and cond in ["POV", "Both"]:
                    pov_img = load_pov_conditioning_input(exp_dir, sample_idx)
                # Load graph text for Graph/Both conditioning
                if graph_text is None and cond in ["Graph", "Both"]:
                    graph_text = load_graph_conditioning_text(exp_dir, sample_idx)
        
        if not model_images:
            print(f"  ⚠ Could not load images for {cond} sample {sample_idx}")
            continue
        
        # Create grid: top row [Target] [POV/Graph] [Graph Text if Both], bottom 2x3 grid of models
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.15)
        
        # Top row: Target and conditioning inputs
        ax_target = fig.add_subplot(gs[0, 0])
        if target_img:
            ax_target.imshow(target_img)
            ax_target.set_title("Target\n(Ground Truth)", fontsize=11, fontweight="bold")
        else:
            ax_target.text(0.5, 0.5, "Target N/A", ha="center", va="center")
            ax_target.set_title("Target", fontsize=11, fontweight="bold")
        ax_target.set_xticks([])
        ax_target.set_yticks([])
        
        ax_middle = fig.add_subplot(gs[0, 1])
        if cond == "POV":
            # POV only
            if pov_img:
                ax_middle.imshow(pov_img)
                ax_middle.set_title("POV\n(Input View)", fontsize=11, fontweight="bold")
            else:
                ax_middle.text(0.5, 0.5, "POV N/A", ha="center", va="center")
                ax_middle.set_title("POV", fontsize=11, fontweight="bold")
            ax_middle.set_xticks([])
            ax_middle.set_yticks([])
        elif cond == "Graph":
            # Graph text only
            ax_middle.axis('off')
            if graph_text:
                ax_middle.text(0.5, 0.5, graph_text, ha="center", va="center", 
                             fontsize=9, wrap=True, family="monospace",
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
                ax_middle.set_title("Graph\n(Text Input)", fontsize=11, fontweight="bold")
            else:
                ax_middle.text(0.5, 0.5, "Graph N/A", ha="center", va="center")
                ax_middle.set_title("Graph", fontsize=11, fontweight="bold")
        elif cond == "Both":
            # POV image
            if pov_img:
                ax_middle.imshow(pov_img)
                ax_middle.set_title("POV\n(Input View)", fontsize=11, fontweight="bold")
            else:
                ax_middle.text(0.5, 0.5, "POV N/A", ha="center", va="center")
                ax_middle.set_title("POV", fontsize=11, fontweight="bold")
            ax_middle.set_xticks([])
            ax_middle.set_yticks([])
        
        # Top-right: Graph text for "Both" conditioning
        ax_right = fig.add_subplot(gs[0, 2])
        if cond == "Both" and graph_text:
            ax_right.axis('off')
            ax_right.text(0.5, 0.5, graph_text, ha="center", va="center", 
                         fontsize=9, wrap=True, family="monospace",
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            ax_right.set_title("Graph\n(Text Input)", fontsize=11, fontweight="bold")
        else:
            ax_right.axis('off')
        
        # Bottom 2x3 grid: model outputs (show up to 6 models)
        for idx, (img, label) in enumerate(zip(model_images[:6], model_labels[:6])):
            row = 1 + (idx // 3)
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.suptitle(f"Qualitative Results ({cond} Conditioning) — Sample #{sample_idx:04d}", 
                    fontsize=14, fontweight="bold", y=0.995)
        
        output_path = output_dir / f"{fig_counter:02d}_qualitative_{cond}_sample{sample_idx:04d}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {fig_counter:02d}_qualitative_{cond}_sample{sample_idx:04d}.pdf")
        plt.close()
        
        fig_counter += 1


# =============================================================================
# SAME-SAMPLE COMPARISONS (Appendix A1-A18)
# =============================================================================

def create_appendix_same_samples_per_conditioning(results_dir: Path, output_dir: Path, df: pd.DataFrame):
    """Create appendix figures - multiple samples (best/median/worst) per conditioning type and room type."""
    
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and 
                      (d / "images").exists()])
    
    if len(exp_dirs) == 0:
        print("⚠ No image directories found for appendix")
        return
    
    # Group experiments by conditioning type
    cond_groups = defaultdict(list)
    for exp_dir in exp_dirs:
        parsed = parse_experiment_name(exp_dir.name)
        cond = parsed["conditioning"]
        if cond:
            cond_groups[cond].append(exp_dir)
    
    figure_counter = 1
    
    # For each conditioning type
    for cond in ["POV", "Graph", "Both"]:
        if cond not in cond_groups:
            continue
        
        exp_dirs_cond = sorted(cond_groups[cond], 
                              key=lambda x: (parse_experiment_name(x.name)["architecture"] or "", 
                                            parse_experiment_name(x.name)["capacity"] or ""))
        
        print(f"\n{cond} Conditioning ({len(exp_dirs_cond)} models):")
        
        # For each room type
        for room_type in ["empty", "furnished"]:
            # Get best/median/worst samples for this conditioning group and room type
            samples = get_multiple_samples_for_conditioning_group(
                df, exp_dirs_cond, room_type=room_type, num_samples=3
            )
            
            if not samples:
                print(f"  ⚠ No {room_type} samples found")
                continue
            
            # For each sample (best, median, worst)
            for sample_idx, rank in samples:
                # Load images from all models for this sample
                model_images = []
                model_labels = []
                target_img = None
                pov_img = None
                graph_text = None
                
                for exp_dir in exp_dirs_cond:
                    gen_img = load_generated_floorplan(exp_dir, sample_idx)
                    if gen_img:
                        model_images.append(gen_img)
                        parsed = parse_experiment_name(exp_dir.name)
                        label = f"{parsed['architecture']}-{parsed['capacity']}"
                        model_labels.append(label)
                        
                        # Load target once (same for all models)
                        if target_img is None:
                            target_img = load_ground_truth_floorplan(exp_dir, sample_idx)
                        # Load POV for POV/Both conditioning
                        if pov_img is None and cond in ["POV", "Both"]:
                            pov_img = load_pov_conditioning_input(exp_dir, sample_idx)
                        # Load graph text for Graph/Both conditioning
                        if graph_text is None and cond in ["Graph", "Both"]:
                            graph_text = load_graph_conditioning_text(exp_dir, sample_idx)
                
                if not model_images:
                    continue
                
                # Calculate number of columns
                n_cols = len(model_images) + 1  # +1 for target
                if pov_img is not None:
                    n_cols += 1
                if graph_text is not None:
                    n_cols += 1
                
                fig_width = max(16, n_cols * 2.5)
                fig_height = 3.5
                
                fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))
                if n_cols == 1:
                    axes = [axes]
                
                col_idx = 0
                
                # Target ground truth floorplan
                ax = axes[col_idx]
                if target_img:
                    ax.imshow(target_img)
                    ax.set_title("Target\n(Ground Truth)", fontsize=8, fontweight="bold", pad=5)
                else:
                    ax.text(0.5, 0.5, "Target N/A", ha="center", va="center", fontsize=7)
                    ax.set_title("Target", fontsize=8, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                col_idx += 1
                
                # POV input if applicable
                if pov_img is not None:
                    ax = axes[col_idx]
                    ax.imshow(pov_img)
                    ax.set_title("POV\nInput", fontsize=8, fontweight="bold", pad=5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    col_idx += 1
                
                # Graph text if applicable
                if graph_text is not None:
                    ax = axes[col_idx]
                    ax.axis('off')
                    ax.text(0.5, 0.5, graph_text, ha="center", va="center", 
                           fontsize=7, wrap=True, family="monospace",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                    ax.set_title("Graph\nText", fontsize=8, fontweight="bold", pad=5)
                    col_idx += 1
                
                # Model outputs
                for img, label in zip(model_images, model_labels):
                    ax = axes[col_idx]
                    ax.imshow(img)
                    ax.set_title(label, fontsize=9, fontweight="bold", pad=5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    col_idx += 1
                
                room_label = "Empty" if room_type == "empty" else "Furnished"
                fig.suptitle(
                    f"{cond} Conditioning — {room_label} ({rank.upper()}) — Sample #{sample_idx:04d}",
                    fontsize=12, fontweight="bold", y=0.98
                )
                
                plt.tight_layout()
                output_path = output_dir / f"A{figure_counter}_{cond}_{room_type}_{rank}_sample{sample_idx:04d}.pdf"
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"  ✓ A{figure_counter}_{cond}_{room_type}_{rank}_sample{sample_idx:04d}.pdf")
                plt.close()
                
                figure_counter += 1


# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================

def generate_comprehensive_summary(df: pd.DataFrame, output_dir: Path, results_dir: Path):
    """Generate comprehensive summary document describing all figures and findings."""
    
    summary_lines = [
        "=" * 100,
        "BASELINE EVALUATION - COMPREHENSIVE RESULTS SUMMARY",
        "=" * 100,
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total samples analyzed: {len(df)}",
        f"Unique experiments: {df['experiment'].nunique()}",
        "",
    ]
    
    # Dataset overview
    if "is_empty" in df.columns:
        df_copy = df.copy()
        if df_copy["is_empty"].dtype == object:
            df_copy["is_empty"] = df_copy["is_empty"].astype(str).str.upper() == "TRUE"
        n_empty = (df_copy["is_empty"] == True).sum()
        n_furnished = (df_copy["is_empty"] == False).sum()
        summary_lines.extend([
            "DATASET COMPOSITION",
            "-" * 100,
            f"  Empty rooms:      {n_empty:6d} ({100*n_empty/len(df):.1f}%)",
            f"  Furnished rooms:  {n_furnished:6d} ({100*n_furnished/len(df):.1f}%)",
            "",
        ])
    
    # Model configurations
    summary_lines.extend([
        "MODEL CONFIGURATIONS EVALUATED",
        "-" * 100,
    ])
    for exp_name in sorted(df["experiment"].unique()):
        parsed = parse_experiment_name(exp_name)
        n_samples = len(df[df["experiment"] == exp_name])
        summary_lines.append(
            f"  {parsed['architecture']}-{parsed['capacity']:6s} ({parsed['conditioning']:5s}): {n_samples:4d} samples"
        )
    summary_lines.append("")
    
    # Key metrics summary
    summary_lines.extend([
        "PERFORMANCE METRICS SUMMARY",
        "-" * 100,
        "",
    ])
    
    metrics = ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]
    metric_names = ["Floor IoU", "Presence Accuracy", "Detection F1", "Unified Score"]
    
    for metric, metric_name in zip(metrics, metric_names):
        data = df[metric].dropna()
        summary_lines.extend([
            f"{metric_name}:",
            f"  Mean:     {data.mean():.4f}",
            f"  Std:      {data.std():.4f}",
            f"  Min:      {data.min():.4f}",
            f"  Max:      {data.max():.4f}",
            f"  Median:   {data.median():.4f}",
            "",
        ])
    
    # Architecture comparison
    summary_lines.extend([
        "ARCHITECTURE COMPARISON (Deep-Narrow vs Wide-Shallow)",
        "-" * 100,
    ])
    for arch in ["DN", "WS"]:
        arch_df = df[df["architecture"] == arch]
        if len(arch_df) > 0:
            summary_lines.append(f"\n{arch} - {len(arch_df)} samples:")
            for metric, metric_name in zip(metrics, metric_names):
                data = arch_df[metric].dropna()
                if len(data) > 0:
                    summary_lines.append(f"  {metric_name:20s}: {data.mean():.4f} ± {data.std():.4f}")
    summary_lines.append("")
    
    # Capacity comparison
    summary_lines.extend([
        "",
        "CAPACITY SCALING (Small → Medium → Large)",
        "-" * 100,
    ])
    capacity_order = {"Small": 0, "Medium": 1, "Large": 2}
    for cap in sorted(df["capacity"].dropna().unique(), key=lambda x: capacity_order.get(x, 99)):
        cap_df = df[df["capacity"] == cap]
        if len(cap_df) > 0:
            summary_lines.append(f"\n{cap} - {len(cap_df)} samples:")
            for metric, metric_name in zip(metrics, metric_names):
                data = cap_df[metric].dropna()
                if len(data) > 0:
                    summary_lines.append(f"  {metric_name:20s}: {data.mean():.4f} ± {data.std():.4f}")
    summary_lines.append("")
    
    # Conditioning comparison
    summary_lines.extend([
        "",
        "CONDITIONING MODALITY (POV vs Graph vs Both)",
        "-" * 100,
    ])
    cond_order = {"POV": 0, "Graph": 1, "Both": 2}
    for cond in sorted(df["conditioning"].dropna().unique(), key=lambda x: cond_order.get(x, 99)):
        cond_df = df[df["conditioning"] == cond]
        if len(cond_df) > 0:
            summary_lines.append(f"\n{cond} - {len(cond_df)} samples:")
            for metric, metric_name in zip(metrics, metric_names):
                data = cond_df[metric].dropna()
                if len(data) > 0:
                    summary_lines.append(f"  {metric_name:20s}: {data.mean():.4f} ± {data.std():.4f}")
    summary_lines.append("")
    
    # Empty vs Furnished
    summary_lines.extend([
        "",
        "ROOM TYPE COMPARISON (Empty vs Furnished)",
        "-" * 100,
    ])
    df_copy = df.copy()
    if "is_empty" in df_copy.columns and df_copy["is_empty"].dtype == object:
        df_copy["is_empty"] = df_copy["is_empty"].astype(str).str.upper() == "TRUE"
    
    for room_type, room_label in [("empty", "Empty"), ("furnished", "Furnished")]:
        if room_type == "empty":
            room_df = df_copy[df_copy["is_empty"] == True]
        else:
            room_df = df_copy[df_copy["is_empty"] == False]
        
        if len(room_df) > 0:
            summary_lines.append(f"\n{room_label} - {len(room_df)} samples:")
            for metric, metric_name in zip(metrics, metric_names):
                data = room_df[metric].dropna()
                if len(data) > 0:
                    summary_lines.append(f"  {metric_name:20s}: {data.mean():.4f} ± {data.std():.4f}")
    summary_lines.append("")
    
    # Top performers
    summary_lines.extend([
        "",
        "=" * 100,
        "TOP 10 PERFORMING CONFIGURATIONS",
        "-" * 100,
    ])
    
    top_exps = df.groupby("experiment_label")[metrics].mean().reset_index()
    top_exps = top_exps.sort_values("unified_score", ascending=False)
    
    for i, (_, row) in enumerate(top_exps.head(10).iterrows(), 1):
        summary_lines.append(f"\n{i:2d}. {row['experiment_label']}")
        for metric, metric_name in zip(metrics, metric_names):
            summary_lines.append(f"    {metric_name:20s}: {row[metric]:.4f}")
    summary_lines.append("")
    
    # Figure descriptions
    summary_lines.extend([
        "",
        "=" * 100,
        "FIGURE DESCRIPTIONS",
        "=" * 100,
        "",
        "RESULTS CHAPTER FIGURES (01-08)",
        "-" * 100,
        "",
        "Figure 01: ARCHITECTURE COMPARISON (Deep-Narrow vs Wide-Shallow)",
        "  Type: 2×2 boxplot grid",
        "  Content:",
        "    - Top-left: Floor IoU (measures floor region detection accuracy)",
        "    - Top-right: Presence Accuracy (measures furniture presence prediction)",
        "    - Bottom-left: Detection F1 (measures furniture detection quality)",
        "    - Bottom-right: Unified Score (combined metric across all objectives)",
        "  What it shows:",
        "    - Compares architectural depth-vs-width tradeoff",
        "    - Shows which architecture family performs better overall",
        "    - Error bars indicate variance across all models of each architecture",
        "",
        "Figure 02: CAPACITY SCALING (Small → Medium → Large)",
        "  Type: 2×2 boxplot grid",
        "  Content: Same 4 metrics as Figure 01, but grouped by model capacity",
        "  What it shows:",
        "    - How performance improves with larger model capacity",
        "    - Whether scaling plateaus or continues improving",
        "    - Capacity-dependent performance trends per metric",
        "",
        "Figure 03: CONDITIONING MODALITY (POV vs Graph vs Both)",
        "  Type: 2×2 boxplot grid",
        "  Content: Same 4 metrics, grouped by conditioning type",
        "  What it shows:",
        "    - Relative contribution of spatial (POV) vs semantic (Graph) information",
        "    - Whether combining both modalities (Both) provides synergistic benefit",
        "    - Which input modality is more critical for generation quality",
        "",
        "Figure 04: EMPTY vs FURNISHED ROOMS",
        "  Type: 1×3 boxplot grid (3 key metrics)",
        "  Content:",
        "    - Left: Floor IoU (critical for navigation)",
        "    - Middle: Detection F1 (furniture localization)",
        "    - Right: Unified Score (overall quality)",
        "  What it shows:",
        "    - Whether models perform differently on simple (empty) vs complex (furnished) layouts",
        "    - P-values indicate statistical significance of differences",
        "    - Identifies which room type is more challenging for the models",
        "",
        "Figure 05: METRIC CORRELATIONS",
        "  Type: Heatmap (correlation matrix)",
        "  Content: Pearson correlations between all evaluation metrics",
        "  What it shows:",
        "    - Which metrics move together (strong positive correlation)",
        "    - Which metrics are independent (weak correlation)",
        "    - Interdependencies in model performance across objectives",
        "",
        "Figure 06: PER-EXPERIMENT PERFORMANCE RANKING",
        "  Type: 2×2 horizontal barplot grid",
        "  Content: All 15 model configurations ranked by each metric",
        "  What it shows:",
        "    - Overall performance ranking of all baseline configurations",
        "    - Whether best architecture changes depending on metric",
        "    - Color coding by conditioning type (POV/Graph/Both)",
        "",
        "Figure 07: METRIC DISTRIBUTIONS (Empty vs Furnished)",
        "  Type: 2×2 overlapping histogram grid",
        "  Content: Distribution of 4 metrics, with empty and furnished samples separated",
        "  What it shows:",
        "    - Whether empty rooms cluster at higher scores (easier task)",
        "    - Whether furnished rooms have wider spread (more variation)",
        "    - Actual distribution shapes and sample sizes per room type",
        "",
        "Figure 08: QUALITATIVE RESULTS",
        "  Type: 1×N image grid (target + POV + 6 models)",
        "  Content:",
        "    - Left 2 columns: Ground truth floorplan and POV conditioning input",
        "    - Remaining columns: Generated floorplans from each model",
        "    - One figure per conditioning type (POV, Graph, Both)",
        "  What it shows:",
        "    - Visual quality of generated layouts vs ground truth",
        "    - How different models interpret same conditioning input",
        "    - Qualitative differences in architecture/capacity choices",
        "    - Same sample used across all models for fair comparison",
        "",
        "APPENDIX FIGURES (A1-A18)",
        "-" * 100,
        "",
        "18 figures total (3 conditioning types × 2 room types × 3 quality levels)",
        "",
        "Each figure shows the same layout structure:",
        "  - Column 1: Ground truth (target floorplan to match)",
        "  - Column 2: Conditioning input (POV camera view for POV/Both, N/A for Graph)",
        "  - Columns 3-N: Model outputs (all models in conditioning group)",
        "",
        "Organization:",
        "  A1-A6:   POV conditioning (best/median/worst for empty & furnished)",
        "  A7-A12:  Graph conditioning (best/median/worst for empty & furnished)",
        "  A13-A18: Both conditioning (best/median/worst for empty & furnished)",
        "",
        "Sample selection:",
        "  - BEST: Sample with highest unified_score in group",
        "  - MEDIAN: Sample at 50th percentile of unified_score",
        "  - WORST: Sample with lowest unified_score in group",
        "",
        "What these show:",
        "  - Best: Models performing well on a favorable sample",
        "  - Median: Typical model performance on average-difficulty sample",
        "  - Worst: Challenging cases where models struggle",
        "  - Cross-model comparison: How architecture/capacity affects same input",
        "",
    ])
    
    summary_lines.extend([
        "=" * 100,
        "KEY FINDINGS & INTERPRETATION",
        "=" * 100,
        "",
        "1. ARCHITECTURE EFFECT",
        "   - Compare Figure 01 and Figure 06 to assess DN vs WS trade-offs",
        "   - Check which architecture is more consistent across metrics",
        "",
        "2. CAPACITY EFFECT",
        "   - Figure 02 shows scaling trajectory",
        "   - Diminishing returns indicate saturation point",
        "",
        "3. CONDITIONING COMPLEMENTARITY",
        "   - Figure 03 shows if POV + Graph > POV alone or Graph alone",
        "   - Large difference indicates complementary information",
        "",
        "4. ROOM COMPLEXITY",
        "   - Figure 04 reveals whether models handle complexity uniformly",
        "   - Larger gap favors architectural decisions targeting that room type",
        "",
        "5. METRIC RELATIONSHIPS",
        "   - Figure 05 shows if optimizing one metric optimizes others",
        "   - Strong correlations simplify multi-objective optimization",
        "",
        "6. DISTRIBUTION CHARACTERISTICS",
        "   - Figure 07 shows whether task difficulty is consistent",
        "   - Narrow distributions indicate predictable model behavior",
        "",
        "7. QUALITATIVE ASSESSMENT",
        "   - Figures 08 and A1-A18 provide visual validation",
        "   - Compare visual quality to metric scores",
        "   - Identify failure modes in worst-case examples",
        "",
        "=" * 100,
    ])
    
    summary_path = output_dir / "RESULTS_SUMMARY.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"✓ Comprehensive summary saved to: {summary_path}")
    return summary_path
    """Generate statistical summary."""
    report_lines = [
        "=" * 100,
        "BASELINE EVALUATION STATISTICAL SUMMARY",
        "=" * 100,
        "",
        f"Total samples analyzed: {len(df)}",
        f"Unique experiments: {df['experiment'].nunique()}",
        "",
    ]
    
    if "is_empty" in df.columns:
        is_empty = (df["is_empty"].astype(str).str.upper() == "TRUE").sum()
        report_lines.extend([
            f"Empty rooms: {is_empty}",
            f"Furnished rooms: {len(df) - is_empty}",
            "",
        ])
    
    report_lines.extend([
        "ARCHITECTURE COMPARISON",
        "-" * 100,
    ])
    for arch in ["DN", "WS"]:
        arch_df = df[df["architecture"] == arch]
        if len(arch_df) > 0:
            report_lines.append(f"\n{arch} ({len(arch_df)} samples):")
            for metric in ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]:
                data = arch_df[metric].dropna()
                if len(data) > 0:
                    report_lines.append(f"  {metric:25s}: {data.mean():.4f} ± {data.std():.4f}")
    
    report_lines.extend([
        "",
        "CAPACITY SCALING",
        "-" * 100,
    ])
    capacity_order = {"Small": 0, "Medium": 1, "Large": 2}
    for cap in sorted(df["capacity"].dropna().unique(), key=lambda x: capacity_order.get(x, 99)):
        cap_df = df[df["capacity"] == cap]
        if len(cap_df) > 0:
            report_lines.append(f"\n{cap} ({len(cap_df)} samples):")
            for metric in ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]:
                data = cap_df[metric].dropna()
                if len(data) > 0:
                    report_lines.append(f"  {metric:25s}: {data.mean():.4f} ± {data.std():.4f}")
    
    report_lines.extend([
        "",
        "CONDITIONING MODALITY",
        "-" * 100,
    ])
    cond_order = {"POV": 0, "Graph": 1, "Both": 2}
    for cond in sorted(df["conditioning"].dropna().unique(), key=lambda x: cond_order.get(x, 99)):
        cond_df = df[df["conditioning"] == cond]
        if len(cond_df) > 0:
            report_lines.append(f"\n{cond} ({len(cond_df)} samples):")
            for metric in ["floor_iou", "presence_accuracy", "detection_f1", "unified_score"]:
                data = cond_df[metric].dropna()
                if len(data) > 0:
                    report_lines.append(f"  {metric:25s}: {data.mean():.4f} ± {data.std():.4f}")
    
    report_lines.extend([
        "",
        "=" * 100,
        "TOP PERFORMING CONFIGURATIONS",
        "-" * 100,
    ])
    
    top_exps = df.groupby("experiment_label")[["floor_iou", "presence_accuracy", 
                                               "detection_f1", "unified_score"]].mean()
    top_exps = top_exps.sort_values("unified_score", ascending=False)
    
    for i, (exp_label, row) in enumerate(top_exps.head(10).iterrows(), 1):
        report_lines.append(f"{i:2d}. {exp_label:40s} | Unified: {row['unified_score']:.4f}")
    
    report_lines.append("=" * 100)
    
    report_path = output_dir / "STATISTICAL_SUMMARY.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print("\n" + "\n".join(report_lines))
    print(f"\n✓ Statistical summary saved to: {report_path}")


def generate_latex_snippets(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive LaTeX figure inclusion snippets."""
    
    latex_lines = [
        "% =============================================================================",
        "% RESULTS CHAPTER: Figures 1-9 + APPENDIX: Figures A1-A18",
        "% Paste these into your thesis .tex file",
        "% =============================================================================",
        "",
        "%% FIGURE 1: Architecture Comparison",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/results_chapter/01_architecture_comparison}",
        "  \\caption{Architecture comparison: Deep-Narrow vs Wide-Shallow.}",
        "  \\label{fig:arch_comparison}",
        "\\end{figure}",
        "",
        "%% FIGURE 2: Capacity Scaling",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/results_chapter/02_capacity_scaling}",
        "  \\caption{Capacity scaling effects: Small, Medium, Large models.}",
        "  \\label{fig:capacity_scaling}",
        "\\end{figure}",
        "",
        "%% FIGURE 3: Conditioning Modality",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/results_chapter/03_conditioning_comparison}",
        "  \\caption{Conditioning modality effects: POV-only, Graph-only, Both.}",
        "  \\label{fig:conditioning_comparison}",
        "\\end{figure}",
        "",
        "%% FIGURE 4: Empty vs Furnished Rooms",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/results_chapter/04_empty_vs_furnished}",
        "  \\caption{Empty vs Furnished room stratification with p-values.}",
        "  \\label{fig:empty_vs_furnished}",
        "\\end{figure}",
        "",
        "%% FIGURE 5: Metric Correlations",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.8\\textwidth]{thesis_figures/results_chapter/05_metric_correlations}",
        "  \\caption{Pearson correlation matrix between all evaluation metrics.}",
        "  \\label{fig:metric_correlations}",
        "\\end{figure}",
        "",
        "%% FIGURE 6: Per-Experiment Rankings",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/results_chapter/06_per_experiment_comparison}",
        "  \\caption{Ranking of all 15 baseline configurations by performance metric.}",
        "  \\label{fig:experiment_rankings}",
        "\\end{figure}",
        "",
        "%% FIGURE 7: Metric Distributions",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/results_chapter/07_metric_distributions}",
        "  \\caption{Metric distributions across all 7,500 evaluation samples.}",
        "  \\label{fig:metric_distributions}",
        "\\end{figure}",
        "",
        "%% FIGURE 8: Architecture × Capacity Matrix",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.75\\textwidth]{thesis_figures/results_chapter/08_heatmap_unified_score}",
        "  \\caption{Architecture and capacity interaction effects via heatmap.}",
        "  \\label{fig:arch_capacity_heatmap}",
        "\\end{figure}",
        "",
        "%% FIGURE 9: Qualitative Results (one per conditioning type)",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/results_chapter/09_qualitative_POV_*}",
        "  \\caption{Qualitative results: Target/POV inputs (top row) and model outputs in 2×3 grid (DN and WS with Small/Medium/Large).}",
        "  \\label{fig:qualitative_results}",
        "\\end{figure}",
        "",
        "\\appendix",
        "\\section{Appendix: Visual Comparisons Across All Models}",
        "",
        "Each figure (A1-A18) shows the same sample generated by all models within a conditioning type.",
        "Layout: [Ground Truth Target] [POV/Input Image] [DN-Small] [DN-Medium] [DN-Large] [WS-Small] [WS-Medium] [WS-Large]",
        "",
        "\\subsection{POV-only Conditioning (A1-A6)}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A1_POV_best_empty}",
        "  \\caption{A1: POV best empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A2_POV_best_furnished}",
        "  \\caption{A2: POV best furnished room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A3_POV_median_empty}",
        "  \\caption{A3: POV median empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A4_POV_median_furnished}",
        "  \\caption{A4: POV median furnished room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A5_POV_worst_empty}",
        "  \\caption{A5: POV worst empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A6_POV_worst_furnished}",
        "  \\caption{A6: POV worst furnished room.}",
        "\\end{figure}",
        "",
        "\\subsection{Graph-only Conditioning (A7-A12)}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A7_Graph_best_empty}",
        "  \\caption{A7: Graph best empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A8_Graph_best_furnished}",
        "  \\caption{A8: Graph best furnished room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A9_Graph_median_empty}",
        "  \\caption{A9: Graph median empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A10_Graph_median_furnished}",
        "  \\caption{A10: Graph median furnished room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A11_Graph_worst_empty}",
        "  \\caption{A11: Graph worst empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A12_Graph_worst_furnished}",
        "  \\caption{A12: Graph worst furnished room.}",
        "\\end{figure}",
        "",
        "\\subsection{Both (POV + Graph) Conditioning (A13-A18)}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A13_Both_best_empty}",
        "  \\caption{A13: Both best empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A14_Both_best_furnished}",
        "  \\caption{A14: Both best furnished room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A15_Both_median_empty}",
        "  \\caption{A15: Both median empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A16_Both_median_furnished}",
        "  \\caption{A16: Both median furnished room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A17_Both_worst_empty}",
        "  \\caption{A17: Both worst empty room.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{thesis_figures/appendix/A18_Both_worst_furnished}",
        "  \\caption{A18: Both worst furnished room.}",
        "\\end{figure}",
    ]

    
    latex_path = output_dir / "figures.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    print(f"✓ LaTeX snippets saved to: {latex_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete baseline evaluation analysis for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to evaluation_results directory"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Create output structure
    output_root = results_dir / "results_figures"
    results_chapter_dir = output_root / "results_chapter"
    appendix_dir = output_root / "appendix"
    latex_dir = output_root / "latex_snippets"
    
    results_chapter_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("BASELINE EVALUATION - THESIS RESULTS GENERATION")
    print("=" * 100)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_root}")
    print("")
    
    try:
        print("STEP 1: Loading metrics data...\n")
        df = load_all_metrics_from_csv(results_dir)
        print(f"✓ Loaded {len(df)} total samples\n")
        
        print("STEP 2: Generating results chapter figures 1-7...\n")
        plot_architecture_comparison(df, results_chapter_dir)
        plot_capacity_scaling(df, results_chapter_dir)
        plot_conditioning_comparison(df, results_chapter_dir)
        plot_empty_vs_furnished(df, results_chapter_dir)
        plot_metric_correlations(df, results_chapter_dir)
        plot_per_experiment_comparison(df, results_chapter_dir)
        plot_metric_distributions(df, results_chapter_dir)
        
        print("\nSTEP 3: Generating Figure 8 - Qualitative results...\n")
        create_qualitative_results(results_dir, results_chapter_dir, df)
        
        print("\nSTEP 4: Generating appendix figures A1-A18...\n")
        create_appendix_same_samples_per_conditioning(results_dir, appendix_dir, df)
        
        print("\nSTEP 5: Generating comprehensive summary...\n")
        generate_comprehensive_summary(df, output_root, results_dir)
        
        print("STEP 6: Generating statistical summary...\n")
        generate_statistical_summary(df, output_root)
        
        print("STEP 7: Generating LaTeX snippets...\n")
        generate_latex_snippets(df, latex_dir)
        
        csv_path = output_root / "all_samples.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ All samples CSV saved to: {csv_path}")
        
        print("\n" + "=" * 100)
        print("✓ COMPLETE! All figures generated successfully")
        print("=" * 100)
        print(f"\nFolder structure ready for Overleaf:")
        print(f"  {output_root}/")
        print(f"  ├── results_chapter/          (7 figures: statistical analysis + qualitative)")
        print(f"  ├── appendix/                 (18 figures: A1-A18 best/median/worst samples)")
        print(f"  ├── latex_snippets/           (LaTeX code for inclusion)")
        print(f"  ├── RESULTS_SUMMARY.txt       (Comprehensive summary with figure descriptions)")
        print(f"  ├── STATISTICAL_SUMMARY.txt   (Raw statistical breakdown)")
        print(f"  └── all_samples.csv           (Full dataset with all metrics)")
        print(f"\nYou can now:")
        print(f"  1. Copy entire '{output_root.name}' folder")
        print(f"  2. Upload to Overleaf in your project")
        print(f"  3. Include figures using LaTeX snippets in latex_snippets/figures.tex")
        print("=" * 100)
    
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()