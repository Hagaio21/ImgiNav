#!/usr/bin/env python3
"""
Baseline Evaluation Script for Floorplan Generation.

Evaluates a trained diffusion model with supercategory-level metrics:
1. Loads checkpoint and validation data
2. Evaluates BOTH empty and furnished rooms in one run
3. Computes supercategory-level metrics: Floor/Wall/Openings IoU, furniture presence/count/L1
4. Generates visualizations for best/worst/median samples (with 80° FOV beam)
5. Outputs: 1 CSV with all samples, 1 JSON summary with sorted per-supercategory averages

Usage:
    python evaluate_baseline.py --checkpoint path/to/checkpoint.pt --manifest path/to/manifest.csv
    
    # Evaluate only furnished rooms:
    python evaluate_baseline.py --checkpoint ckpt.pt --manifest val.csv --furnished-only
    
    # Evaluate only empty rooms:
    python evaluate_baseline.py --checkpoint ckpt.pt --manifest val.csv --empty-only
"""

import argparse
import json
import csv
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.datasets.datasets import ManifestDataset
from training.evaluation_metrics import (
    FloorplanEvaluator,
    tensor_to_numpy_rgb,
    load_taxonomy
)


def numpy_safe_json_default(obj):
    """JSON encoder default function that handles numpy types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def load_model(checkpoint_path: Path, device: str = "cuda") -> DiffusionModel:
    """Load diffusion model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = DiffusionModel.load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully")
    return model


def load_validation_data(
    manifest_path: Path,
    outputs: dict,
    include_empty: bool = True,
    include_furnished: bool = True,
    empty_threshold: int = 3,
):
    """Load evaluation dataset with both empty and furnished rooms.
    
    Args:
        manifest_path: Path to manifest CSV
        outputs: Column mapping for dataset
        include_empty: Include empty rooms (furniture_count < empty_threshold)
        include_furnished: Include furnished rooms (furniture_count >= empty_threshold)
        empty_threshold: Furniture count below which a room is considered empty (default: 3)
    
    Returns:
        Tuple of (dataset, valid_indices, manifest_df, manifest_dir)
    
    Note:
        Empty/furnished is determined by furniture_count, not is_empty column.
        Rooms with only doors/windows (furniture_count < 3) are considered empty.
    """
    print(f"Loading evaluation dataset from {manifest_path}...")
    
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    
    df = pd.read_csv(manifest_path)
    print(f"  Total rows in manifest: {len(df)}")
    
    # Always exclude rejected samples
    mask = pd.Series([True] * len(df))
    
    if "rejected" in df.columns:
        rejected = df["rejected"].fillna(False).astype(bool)
        rejected_count = rejected.sum()
        mask &= ~rejected
        print(f"  Excluding {rejected_count} rejected samples")
    
    # Determine empty/furnished based on furniture_count
    # Rooms with furniture_count < threshold are considered "empty" (only structure + openings)
    if "furniture_count" in df.columns:
        furniture_count = df["furniture_count"].fillna(0).astype(int)
        is_empty_by_furniture = furniture_count < empty_threshold
        
        print(f"  Empty threshold: furniture_count < {empty_threshold}")
        
        if include_empty and include_furnished:
            # Include both
            print(f"  Including both empty and furnished rooms")
        elif include_empty and not include_furnished:
            # Empty only (furniture_count < threshold)
            mask &= is_empty_by_furniture
            print(f"  Including only empty rooms (furniture_count < {empty_threshold})")
        elif include_furnished and not include_empty:
            # Furnished only (furniture_count >= threshold)
            mask &= ~is_empty_by_furniture
            print(f"  Including only furnished rooms (furniture_count >= {empty_threshold})")
        else:
            raise ValueError("Must include at least one of empty or furnished rooms")
        
        empty_count = (mask & is_empty_by_furniture).sum()
        furnished_count = (mask & ~is_empty_by_furniture).sum()
        print(f"  Empty rooms: {empty_count}, Furnished rooms: {furnished_count}")
    else:
        print(f"  WARNING: 'furniture_count' column not found, using is_empty column as fallback")
        if "is_empty" in df.columns:
            is_empty_by_furniture = df["is_empty"].fillna(False).astype(bool)
            if include_empty and not include_furnished:
                mask &= is_empty_by_furniture
            elif include_furnished and not include_empty:
                mask &= ~is_empty_by_furniture
    
    valid_indices = df[mask].index.tolist()
    print(f"  Valid samples after filtering: {len(valid_indices)}")
    
    dataset = ManifestDataset(
        manifest=str(manifest_path),
        outputs=outputs,
        filters=None,
        return_path=True
    )
    
    return dataset, valid_indices, df, manifest_dir


def infer_conditioning_type(checkpoint_path: Path, config_path: Path = None) -> str:
    """Infer conditioning type from checkpoint path or config."""
    
    if config_path and config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            exp_name = config.get("experiment", {}).get("name", "").lower()
            
            if "_both_" in exp_name or "_both" in exp_name:
                return "both"
            elif "_graph" in exp_name and "_pov" not in exp_name:
                return "graph"
            elif "_pov" in exp_name and "_graph" not in exp_name:
                return "pov"
        except Exception as e:
            print(f"  Warning: Could not parse config: {e}")
    
    ckpt_str = str(checkpoint_path).lower()
    
    if "_both_" in ckpt_str or "_both/" in ckpt_str or "/both_" in ckpt_str:
        return "both"
    elif "_graphs_" in ckpt_str or "_graph_" in ckpt_str or "/graph" in ckpt_str:
        return "graph"
    elif "_povs_" in ckpt_str or "_pov_" in ckpt_str or "/pov" in ckpt_str:
        return "pov"
    
    print("  Warning: Could not infer conditioning type, defaulting to 'both'")
    return "both"


def generate_single(
    model: DiffusionModel,
    sample: dict,
    device: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    conditioning: str = "both"
) -> dict:
    """Generate a single floorplan from conditioning."""
    
    text_emb = None
    pov_emb = None
    
    if conditioning in ["graph", "both"]:
        text_emb = sample.get("text_emb")
        if isinstance(text_emb, str):
            try:
                text_emb = torch.load(text_emb, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"Warning: Could not load text embedding from {text_emb}: {e}")
                text_emb = None
        if text_emb is not None:
            text_emb = text_emb.unsqueeze(0).to(device) if text_emb.dim() == 1 else text_emb.to(device)
    
    if conditioning in ["pov", "both"]:
        pov_emb = sample.get("pov_emb")
        if isinstance(pov_emb, str):
            try:
                pov_emb = torch.load(pov_emb, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"Warning: Could not load pov embedding from {pov_emb}: {e}")
                pov_emb = None
        if pov_emb is not None:
            pov_emb = pov_emb.unsqueeze(0).to(device) if pov_emb.dim() == 1 else pov_emb.to(device)
    
    with torch.no_grad():
        output = model.sample(
            batch_size=1,
            num_steps=num_steps,
            method="ddim",
            guidance_scale=guidance_scale,
            text_emb=text_emb,
            pov_emb=pov_emb,
            verbose=False
        )
    
    return output


def decode_target(model: DiffusionModel, latent: torch.Tensor, device: str) -> np.ndarray:
    """Decode target latent to RGB image."""
    latent = latent.unsqueeze(0).to(device) if latent.dim() == 3 else latent.to(device)
    
    with torch.no_grad():
        decoded = model.decoder({"latent": latent})
        rgb = decoded["rgb"]
        rgb = (rgb + 1.0) / 2.0
    
    return tensor_to_numpy_rgb(rgb[0])


def run_evaluation(
    model: DiffusionModel,
    dataset,
    valid_indices: list,
    manifest_df,
    manifest_dir: Path,
    evaluator: FloorplanEvaluator,
    device: str,
    experiment_name: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    max_samples: int = None,
    save_images: bool = False,
    output_dir: Path = None,
    conditioning: str = "both",
    seed: int = None,
    empty_threshold: int = 3
) -> dict:
    """Run evaluation on both empty and furnished rooms, output single CSV.
    
    Args:
        empty_threshold: Furniture count below which room is considered empty (default: 3)
    """
    
    n_valid = len(valid_indices)
    n_samples = n_valid if max_samples is None else min(max_samples, n_valid)
    
    # Random sampling with seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if max_samples is not None and max_samples < n_valid:
        sampled_valid_indices = random.sample(valid_indices, n_samples)
        sampled_valid_indices.sort()
    else:
        sampled_valid_indices = valid_indices[:n_samples]
    
    # Setup directories - per-experiment folder structure
    exp_dir = output_dir / experiment_name if output_dir else None
    images_dir = None
    conditions_dir = None
    if save_images and exp_dir:
        exp_dir.mkdir(parents=True, exist_ok=True)
        images_dir = exp_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        conditions_dir = images_dir / "conditions"
        conditions_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if POV conditioning (for FOV visualization)
    show_fov = conditioning in ["pov", "both"]
    
    print(f"\nEvaluating {n_samples} samples with conditioning: {conditioning}")
    print(f"  Empty threshold: furniture_count < {empty_threshold}")
    
    # CSV file for per-sample metrics (in experiment folder)
    # We stream results to CSV as we go - no accumulation in memory
    csv_path = exp_dir / "metrics.csv" if exp_dir else None
    csv_file = None
    csv_writer = None
    
    conditions_saved = {"pov": 0, "text": 0}
    
    # Get empty threshold from function parameter
    for eval_idx, dataset_idx in enumerate(tqdm(sampled_valid_indices, desc="Evaluating")):
        sample = dataset[dataset_idx]
        manifest_row = manifest_df.iloc[dataset_idx]
        
        # Determine if empty room based on furniture_count
        if "furniture_count" in manifest_df.columns:
            furniture_count = int(manifest_row.get("furniture_count", 0))
            is_empty = furniture_count < empty_threshold
        else:
            # Fallback to is_empty column
            is_empty = bool(manifest_row.get("is_empty", False)) if "is_empty" in manifest_df.columns else False
        
        # Generate
        output = generate_single(model, sample, device, guidance_scale, num_steps, conditioning=conditioning)
        pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
        
        # Decode target
        target_rgb = decode_target(model, sample["latent"], device)
        
        # Evaluate with is_empty flag
        metrics = evaluator.evaluate(pred_rgb, target_rgb, is_empty=is_empty)
        summary = metrics["summary"]
        
        # Write CSV row (streaming - no memory accumulation)
        if csv_path:
            if csv_file is None:
                csv_file = open(csv_path, "w", newline="")
                fieldnames = ["eval_idx", "dataset_idx", "is_empty"] + list(summary.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
            
            row = {"eval_idx": eval_idx, "dataset_idx": dataset_idx, "is_empty": is_empty}
            row.update(summary)
            csv_writer.writerow(row)
            csv_file.flush()
        
        # Save images
        if images_dir:
            prefix = "empty" if is_empty else "furnished"
            Image.fromarray(pred_rgb).save(images_dir / f"{eval_idx:04d}_{prefix}_pred.png")
            Image.fromarray(target_rgb).save(images_dir / f"{eval_idx:04d}_{prefix}_target.png")
            Image.fromarray(metrics["_cleaned_pred"]).save(images_dir / f"{eval_idx:04d}_{prefix}_pred_cleaned.png")
            Image.fromarray(metrics["_cleaned_target"]).save(images_dir / f"{eval_idx:04d}_{prefix}_target_cleaned.png")
            
            # Save conditions
            if conditions_dir:
                if "pov_path" in manifest_df.columns and pd.notna(manifest_row["pov_path"]):
                    pov_path = str(manifest_row["pov_path"])
                    # Try multiple base directories
                    possible_bases = [
                        Path(pov_path),  # Absolute path
                        manifest_dir / pov_path,  # Relative to manifest dir
                        manifest_dir.parent / pov_path,  # Relative to parent (dataset root)
                    ]
                    for p in possible_bases:
                        if p.exists():
                            shutil.copy(p, conditions_dir / f"{eval_idx:04d}_pov.png")
                            conditions_saved["pov"] += 1
                            break
                
                if "graph_text_path" in manifest_df.columns and pd.notna(manifest_row["graph_text_path"]):
                    text_path = str(manifest_row["graph_text_path"])
                    # Try multiple base directories
                    possible_bases = [
                        Path(text_path),  # Absolute path
                        manifest_dir / text_path,  # Relative to manifest dir
                        manifest_dir.parent / text_path,  # Relative to parent (dataset root)
                    ]
                    for p in possible_bases:
                        if p.exists():
                            shutil.copy(p, conditions_dir / f"{eval_idx:04d}_text.txt")
                            conditions_saved["text"] += 1
                            break
        
        del pred_rgb, target_rgb, metrics, output, sample
    
    if csv_file:
        csv_file.close()
    
    print(f"  Conditions saved: {conditions_saved['pov']} POV, {conditions_saved['text']} text")
    
    # Read CSV and compute stats
    df = pd.read_csv(csv_path)
    
    # Separate empty and furnished
    df_empty = df[df["is_empty"] == True]
    df_furnished = df[df["is_empty"] == False]
    
    print(f"  Empty samples: {len(df_empty)}, Furnished samples: {len(df_furnished)}")
    
    # Generate visualizations for best/median/worst - separately for empty and furnished (6 total)
    if exp_dir and images_dir:
        viz_dir = exp_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Helper to create visualizations for a subset
        def create_viz_for_subset(subset_df: pd.DataFrame, subset_name: str):
            if len(subset_df) == 0:
                print(f"  No {subset_name} samples for visualization")
                return
            
            sorted_df = subset_df.sort_values("unified_score")
            
            samples_to_viz = [
                ("best", sorted_df.iloc[-1]),
                ("median", sorted_df.iloc[len(sorted_df) // 2]),
                ("worst", sorted_df.iloc[0]),
            ]
            
            for label, row in samples_to_viz:
                viz_label = f"{label}_{subset_name}"  # e.g., "best_furnished"
                
                # Create subfolder for this visualization
                sample_viz_dir = viz_dir / viz_label
                sample_viz_dir.mkdir(parents=True, exist_ok=True)
                
                idx = int(row["eval_idx"])
                dataset_idx = int(row["dataset_idx"])
                score = float(row["unified_score"])
                is_empty = subset_name == "empty"
                prefix = "empty" if is_empty else "furnished"
                
                # Load images
                pred_path = images_dir / f"{idx:04d}_{prefix}_pred.png"
                target_path = images_dir / f"{idx:04d}_{prefix}_target.png"
                
                if not pred_path.exists():
                    print(f"  Warning: images not found for {viz_label}")
                    continue
                
                pred_img = np.array(Image.open(pred_path))
                target_img = np.array(Image.open(target_path))
                
                # Re-evaluate for visualization
                metrics = evaluator.evaluate(pred_img, target_img, is_empty=is_empty)
                
                # Create visualization with FOV beam
                sample_label = f"{label.capitalize()} {subset_name} (idx={idx}, score={score:.3f})"
                detailed, summary_viz = evaluator.create_visualization(
                    pred_img, target_img, metrics, sample_label, show_fov=show_fov
                )
                
                # Save all files in the subfolder
                detailed.save(sample_viz_dir / "detailed.png")
                summary_viz.save(sample_viz_dir / "summary.png")
                
                # Copy simple images
                shutil.copy(pred_path, sample_viz_dir / "pred.png")
                shutil.copy(target_path, sample_viz_dir / "target.png")
                
                # Also copy cleaned versions if they exist
                cleaned_pred = images_dir / f"{idx:04d}_{prefix}_pred_cleaned.png"
                cleaned_target = images_dir / f"{idx:04d}_{prefix}_target_cleaned.png"
                if cleaned_pred.exists():
                    shutil.copy(cleaned_pred, sample_viz_dir / "pred_cleaned.png")
                if cleaned_target.exists():
                    shutil.copy(cleaned_target, sample_viz_dir / "target_cleaned.png")
                
                # Copy conditions
                if conditions_dir:
                    pov = conditions_dir / f"{idx:04d}_pov.png"
                    txt = conditions_dir / f"{idx:04d}_text.txt"
                    if pov.exists():
                        shutil.copy(pov, sample_viz_dir / "pov.png")
                    if txt.exists():
                        shutil.copy(txt, sample_viz_dir / "text.txt")
                
                # Save info JSON
                info = {
                    "eval_idx": int(idx),
                    "dataset_idx": int(dataset_idx),
                    "is_empty": bool(is_empty),
                    "subset": subset_name,
                    "rank": label,
                    "unified_score": float(score),
                    "metrics": {k: numpy_safe_json_default(v) if not isinstance(v, (int, float, str, bool)) else v 
                               for k, v in row.items() if k not in ["eval_idx", "dataset_idx"]}
                }
                with open(sample_viz_dir / "info.json", "w") as f:
                    json.dump(info, f, indent=2, default=numpy_safe_json_default)
                
                del pred_img, target_img, metrics
            
            # Print summary for this subset
            best_row = sorted_df.iloc[-1]
            median_row = sorted_df.iloc[len(sorted_df) // 2]
            worst_row = sorted_df.iloc[0]
            print(f"  {subset_name.capitalize()}: best={best_row['unified_score']:.3f}, "
                  f"median={median_row['unified_score']:.3f}, worst={worst_row['unified_score']:.3f}")
        
        # Create visualizations for both subsets
        print("\nGenerating visualizations...")
        create_viz_for_subset(df_furnished, "furnished")
        create_viz_for_subset(df_empty, "empty")
    
    # Compute aggregated statistics from CSV (not from memory)
    aggregated = compute_aggregated_stats(df, df_empty, df_furnished)
    aggregated["conditioning"] = conditioning
    aggregated["seed"] = seed
    
    return {"aggregated": aggregated, "csv_path": str(csv_path)}


def compute_aggregated_stats(df: pd.DataFrame, df_empty: pd.DataFrame, df_furnished: pd.DataFrame) -> dict:
    """Compute aggregated statistics with sorting."""
    
    metric_cols = [c for c in df.columns if c not in ["eval_idx", "dataset_idx", "is_empty"]]
    
    aggregated = {
        "num_samples": int(len(df)),
        "num_empty": int(len(df_empty)),
        "num_furnished": int(len(df_furnished)),
    }
    
    # Overall stats
    for col in metric_cols:
        aggregated[f"{col}_mean"] = float(df[col].mean())
        aggregated[f"{col}_std"] = float(df[col].std())
    
    # Structure metrics (all samples)
    aggregated["structure"] = {
        "floor_iou": {"mean": float(df["floor_iou"].mean()), "std": float(df["floor_iou"].std())},
        "wall_iou": {"mean": float(df["wall_iou"].mean()), "std": float(df["wall_iou"].std())},
        "openings_iou": {"mean": float(df["openings_iou"].mean()), "std": float(df["openings_iou"].std())},
    }
    
    # Empty room stats
    if len(df_empty) > 0:
        aggregated["empty"] = {
            "num_samples": int(len(df_empty)),
            "floor_iou": {"mean": float(df_empty["floor_iou"].mean()), "std": float(df_empty["floor_iou"].std())},
            "wall_iou": {"mean": float(df_empty["wall_iou"].mean()), "std": float(df_empty["wall_iou"].std())},
            "openings_iou": {"mean": float(df_empty["openings_iou"].mean()), "std": float(df_empty["openings_iou"].std())},
            "unified_score": {"mean": float(df_empty["unified_score"].mean()), "std": float(df_empty["unified_score"].std())},
        }
    
    # Furnished room stats
    if len(df_furnished) > 0:
        aggregated["furnished"] = {
            "num_samples": int(len(df_furnished)),
            "floor_iou": {"mean": float(df_furnished["floor_iou"].mean()), "std": float(df_furnished["floor_iou"].std())},
            "wall_iou": {"mean": float(df_furnished["wall_iou"].mean()), "std": float(df_furnished["wall_iou"].std())},
            "openings_iou": {"mean": float(df_furnished["openings_iou"].mean()), "std": float(df_furnished["openings_iou"].std())},
            "presence_accuracy": {"mean": float(df_furnished["presence_accuracy"].mean()), "std": float(df_furnished["presence_accuracy"].std())},
            "count_accuracy": {"mean": float(df_furnished["count_accuracy"].mean()), "std": float(df_furnished["count_accuracy"].std())},
            "detection_f1": {"mean": float(df_furnished["detection_f1"].mean()), "std": float(df_furnished["detection_f1"].std())},
            "mean_l1_distance": {"mean": float(df_furnished["mean_l1_distance"].mean()), "std": float(df_furnished["mean_l1_distance"].std())},
            "mean_bbox_iou": {"mean": float(df_furnished["mean_bbox_iou"].mean()), "std": float(df_furnished["mean_bbox_iou"].std())},
            "unified_score": {"mean": float(df_furnished["unified_score"].mean()), "std": float(df_furnished["unified_score"].std())},
        }
    
    # Sorted metrics (by mean, descending)
    sorted_metrics = []
    for col in metric_cols:
        sorted_metrics.append({
            "metric": col,
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        })
    sorted_metrics.sort(key=lambda x: x["mean"], reverse=True)
    aggregated["sorted_metrics"] = sorted_metrics
    
    return aggregated


def save_results(results: dict, output_dir: Path, experiment_name: str):
    """Save evaluation results to per-experiment folder."""
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    full_path = exp_dir / f"results_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, default=numpy_safe_json_default)
    print(f"  Full results saved to: {full_path}")
    
    # Save summary
    summary_path = exp_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(results["aggregated"], f, indent=2, default=numpy_safe_json_default)
    print(f"  Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    agg = results["aggregated"]
    
    print(f"\n[Samples]")
    print(f"  Total: {agg['num_samples']} (Empty: {agg['num_empty']}, Furnished: {agg['num_furnished']})")
    
    print("\n[Structure Metrics (All Samples)]")
    struct = agg.get("structure", {})
    print(f"  Floor IoU:    {struct.get('floor_iou', {}).get('mean', 0):.4f} ± {struct.get('floor_iou', {}).get('std', 0):.4f}")
    print(f"  Wall IoU:     {struct.get('wall_iou', {}).get('mean', 0):.4f} ± {struct.get('wall_iou', {}).get('std', 0):.4f}")
    print(f"  Openings IoU: {struct.get('openings_iou', {}).get('mean', 0):.4f} ± {struct.get('openings_iou', {}).get('std', 0):.4f}")
    
    if "empty" in agg:
        print(f"\n[Empty Rooms ({agg['empty']['num_samples']} samples)]")
        print(f"  Unified Score: {agg['empty']['unified_score']['mean']:.4f} ± {agg['empty']['unified_score']['std']:.4f}")
    
    if "furnished" in agg:
        print(f"\n[Furnished Rooms ({agg['furnished']['num_samples']} samples)]")
        furn = agg["furnished"]
        print(f"  Presence Acc:  {furn['presence_accuracy']['mean']:.4f} ± {furn['presence_accuracy']['std']:.4f}")
        print(f"  Count Acc:     {furn['count_accuracy']['mean']:.4f} ± {furn['count_accuracy']['std']:.4f}")
        print(f"  Detection F1:  {furn['detection_f1']['mean']:.4f} ± {furn['detection_f1']['std']:.4f}")
        print(f"  Mean L1 Dist:  {furn['mean_l1_distance']['mean']:.4f} ± {furn['mean_l1_distance']['std']:.4f}")
        print(f"  BBox IoU:      {furn['mean_bbox_iou']['mean']:.4f} ± {furn['mean_bbox_iou']['std']:.4f}")
        print(f"  Unified Score: {furn['unified_score']['mean']:.4f} ± {furn['unified_score']['std']:.4f}")
    
    print("\n[Sorted Metrics (by mean, descending)]")
    for item in agg.get("sorted_metrics", [])[:5]:
        print(f"  {item['metric']:20s}: {item['mean']:.4f} ± {item['std']:.4f}")
    
    print("\n" + "=" * 70)
    print(f"OVERALL UNIFIED SCORE: {agg.get('unified_score_mean', 0):.4f} ± {agg.get('unified_score_std', 0):.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for floorplan generation")
    
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to training config")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to manifest CSV")
    parser.add_argument("--taxonomy", type=Path, default=None,
                        help="Path to taxonomy.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results")
    parser.add_argument("--conditioning", type=str, choices=["pov", "graph", "both"],
                        default=None, help="Conditioning type")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="CFG guidance scale")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of DDIM steps")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (randomly sampled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--fov", type=float, default=80.0,
                        help="Camera field of view in degrees (default: 80)")
    parser.add_argument("--empty-threshold", type=int, default=3,
                        help="Furniture count below which room is considered empty (default: 3)")
    
    # Room type selection
    parser.add_argument("--empty-only", action="store_true",
                        help="Evaluate only empty rooms")
    parser.add_argument("--furnished-only", action="store_true",
                        help="Evaluate only furnished rooms")
    
    parser.add_argument("--save-images", action="store_true",
                        help="Save generated and target images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Validate room type selection
    if args.empty_only and args.furnished_only:
        raise ValueError("Cannot specify both --empty-only and --furnished-only")
    
    include_empty = not args.furnished_only
    include_furnished = not args.empty_only
    
    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        if args.manifest is None:
            args.manifest = Path(config["dataset"]["manifest"])
        if args.output_dir is None:
            args.output_dir = Path(config["experiment"]["save_path"]) / "evaluation"
    
    if args.manifest is None:
        raise ValueError("Must provide --manifest or --config")
    
    if args.output_dir is None:
        args.output_dir = Path("evaluation_results")
    
    # Find taxonomy
    if args.taxonomy is None:
        possible_paths = [
            args.manifest.parent / "taxonomy.json",
            args.manifest.parent.parent / "taxonomy.json",
            Path("data_preparation_v2/taxonomy.json"),
        ]
        for p in possible_paths:
            if p.exists():
                args.taxonomy = p
                break
    
    if args.taxonomy is None or not args.taxonomy.exists():
        raise ValueError("Could not find taxonomy.json. Specify with --taxonomy")
    
    if args.conditioning is None:
        args.conditioning = infer_conditioning_type(args.checkpoint, args.config)
    
    # Determine evaluation mode string
    if args.empty_only:
        eval_mode = "empty-only"
    elif args.furnished_only:
        eval_mode = "furnished-only"
    else:
        eval_mode = "all (empty + furnished)"
    
    print("=" * 70)
    print("EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Manifest: {args.manifest}")
    print(f"  Taxonomy: {args.taxonomy}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Conditioning: {args.conditioning}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Seed: {args.seed}")
    print(f"  FOV: {args.fov}°")
    print(f"  Empty threshold: furniture_count < {args.empty_threshold}")
    print(f"  Evaluation mode: {eval_mode}")
    print("=" * 70)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Outputs
    outputs = {
        "latent": "latent_embedding_path",
        "text_emb": "graph_embedding_path",
        "pov_emb": "pov_embedding_path",
        "pov_path": "pov_path",
        "graph_text_path": "graph_text_path",
    }
    
    if args.config:
        config_outputs = config.get("dataset", {}).get("outputs", {})
        if config_outputs:
            outputs.update(config_outputs)
            outputs["pov_path"] = "pov_path"
            outputs["graph_text_path"] = "graph_text_path"
    
    # Load validation data
    dataset, valid_indices, manifest_df, manifest_dir = load_validation_data(
        args.manifest,
        outputs=outputs,
        include_empty=include_empty,
        include_furnished=include_furnished,
        empty_threshold=args.empty_threshold,
    )
    
    # Create evaluator
    evaluator = FloorplanEvaluator(args.taxonomy, fov_degrees=args.fov)
    
    # Get experiment name
    experiment_name = args.checkpoint.stem
    if experiment_name in ["best_checkpoint", "checkpoint"]:
        experiment_name = args.checkpoint.parent.parent.name
    
    # Add suffix for evaluation mode
    if args.empty_only:
        experiment_name += "_empty"
    elif args.furnished_only:
        experiment_name += "_furnished"
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        dataset=dataset,
        valid_indices=valid_indices,
        manifest_df=manifest_df,
        manifest_dir=manifest_dir,
        evaluator=evaluator,
        device=args.device,
        experiment_name=experiment_name,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        max_samples=args.max_samples,
        save_images=args.save_images,
        output_dir=args.output_dir,
        conditioning=args.conditioning,
        seed=args.seed,
        empty_threshold=args.empty_threshold
    )
    
    # Add metadata
    results["metadata"] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "conditioning": args.conditioning,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "fov_degrees": args.fov,
        "empty_threshold": args.empty_threshold,
        "include_empty": include_empty,
        "include_furnished": include_furnished,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    save_results(results, args.output_dir, experiment_name)


if __name__ == "__main__":
    main()