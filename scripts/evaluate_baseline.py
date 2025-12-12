#!/usr/bin/env python3
"""
Baseline Evaluation Script for Floorplan Generation.

Updated with navigation-focused metrics and visualization.

Evaluates a trained diffusion model on the validation set:
1. Loads checkpoint and validation data
2. Randomly samples from validation set (with seed for reproducibility)
3. Generates floorplans for each sample (single-shot)
4. Saves conditions (POV image, text description) alongside predictions
5. Computes metrics: class presence, counts, spatial (scale-invariant), camera-centric
6. Generates visualizations for best and median samples
7. Saves detailed results and summary statistics

Usage:
    python evaluate_baseline.py --checkpoint path/to/checkpoint.pt --manifest path/to/manifest.csv
    
    # With random sampling:
    python evaluate_baseline.py --checkpoint ckpt.pt --manifest val.csv --max-samples 100 --seed 42
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
    exclude_empty: bool = True,
    min_furniture: int = 0,
):
    """Load evaluation dataset directly from eval_manifest.csv.
    
    Args:
        manifest_path: Path to manifest CSV
        outputs: Column mapping for dataset
        exclude_empty: Whether to exclude empty rooms (is_empty=False)
        min_furniture: Minimum furniture count required
    
    Returns:
        Tuple of (dataset, valid_indices, manifest_df, manifest_dir) where:
        - valid_indices maps eval_idx to original row
        - manifest_df is the filtered dataframe for condition lookup
        - manifest_dir is the directory containing the manifest (for path resolution)
    """
    print(f"Loading evaluation dataset from {manifest_path}...")
    
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    
    # Load manifest with pandas for proper filtering
    df = pd.read_csv(manifest_path)
    print(f"  Total rows in manifest: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Debug: show is_empty column info
    if "is_empty" in df.columns:
        print(f"  is_empty dtype: {df['is_empty'].dtype}")
        print(f"  is_empty value_counts: {df['is_empty'].value_counts().to_dict()}")
    else:
        print(f"  WARNING: 'is_empty' column not found!")
    
    if "furniture_count" in df.columns:
        print(f"  furniture_count range: {df['furniture_count'].min()} - {df['furniture_count'].max()}")
        print(f"  furniture_count=0: {(df['furniture_count'] == 0).sum()}")
    
    # Apply filters
    mask = pd.Series([True] * len(df))
    
    # Always exclude rejected samples
    if "rejected" in df.columns:
        # Convert to bool and exclude where True
        rejected = df["rejected"].fillna(False).astype(bool)
        rejected_count = rejected.sum()
        mask &= ~rejected  # Keep where rejected is False
        print(f"  Excluding {rejected_count} rejected samples")
    
    # Optionally exclude empty rooms
    if exclude_empty and "is_empty" in df.columns:
        # Convert to bool and exclude where True
        is_empty = df["is_empty"].fillna(False).astype(bool)
        empty_count = is_empty.sum()
        mask &= ~is_empty  # Keep where is_empty is False
        print(f"  Excluding {empty_count} empty rooms")
    
    # Filter by minimum furniture count
    if min_furniture > 0 and "furniture_count" in df.columns:
        low_furniture = (df["furniture_count"] < min_furniture).sum()
        mask &= (df["furniture_count"] >= min_furniture)
        print(f"  Excluding {low_furniture} samples with <{min_furniture} furniture")
    
    # Get valid indices
    valid_indices = df[mask].index.tolist()
    print(f"  Valid samples after filtering: {len(valid_indices)}")
    
    # Create dataset with no filters (we'll index directly)
    dataset = ManifestDataset(
        manifest=str(manifest_path),
        outputs=outputs,
        filters=None,  # No filters, we handle it ourselves
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
    seed: int = None
) -> dict:
    """Run evaluation, stream results to CSV, compute stats at end."""
    
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
    
    # Setup directories
    images_dir = None
    conditions_dir = None
    if save_images and output_dir:
        images_dir = output_dir / "images" / experiment_name
        images_dir.mkdir(parents=True, exist_ok=True)
        conditions_dir = images_dir / "conditions"
        conditions_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating {n_samples} samples with conditioning: {conditioning}")
    
    # CSV file for metrics
    csv_path = output_dir / f"{experiment_name}_metrics.csv" if output_dir else None
    csv_file = None
    csv_writer = None
    conditions_saved = {"pov": 0, "text": 0}
    
    for eval_idx, dataset_idx in enumerate(tqdm(sampled_valid_indices, desc="Evaluating")):
        sample = dataset[dataset_idx]
        
        # Generate
        output = generate_single(model, sample, device, guidance_scale, num_steps, conditioning=conditioning)
        pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
        
        # Decode target
        target_rgb = decode_target(model, sample["latent"], device)
        
        # Evaluate
        metrics = evaluator.evaluate(pred_rgb, target_rgb)
        summary = metrics["summary"]
        
        # Write CSV row
        if csv_path:
            if csv_file is None:
                csv_file = open(csv_path, "w", newline="")
                fieldnames = ["eval_idx", "dataset_idx", "is_empty"] + list(summary.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
            
            manifest_row = manifest_df.iloc[dataset_idx]
            is_empty = bool(manifest_row["is_empty"]) if "is_empty" in manifest_df.columns else False
            
            row = {"eval_idx": eval_idx, "dataset_idx": dataset_idx, "is_empty": is_empty}
            row.update(summary)
            csv_writer.writerow(row)
            csv_file.flush()
        
        # Save images
        if images_dir:
            Image.fromarray(pred_rgb).save(images_dir / f"{eval_idx:04d}_pred.png")
            Image.fromarray(target_rgb).save(images_dir / f"{eval_idx:04d}_target.png")
            Image.fromarray(metrics["_cleaned_pred"]).save(images_dir / f"{eval_idx:04d}_pred_cleaned.png")
            Image.fromarray(metrics["_cleaned_target"]).save(images_dir / f"{eval_idx:04d}_target_cleaned.png")
            
            # Save conditions
            if conditions_dir:
                manifest_row = manifest_df.iloc[dataset_idx]
                if "pov_path" in manifest_df.columns and pd.notna(manifest_row["pov_path"]):
                    pov_path = str(manifest_row["pov_path"])
                    if not Path(pov_path).exists():
                        pov_path = str(manifest_dir / pov_path)
                    if Path(pov_path).exists():
                        shutil.copy(pov_path, conditions_dir / f"{eval_idx:04d}_pov.png")
                        conditions_saved["pov"] += 1
                
                if "graph_text_path" in manifest_df.columns and pd.notna(manifest_row["graph_text_path"]):
                    text_path = str(manifest_row["graph_text_path"])
                    if not Path(text_path).exists():
                        text_path = str(manifest_dir / text_path)
                    if Path(text_path).exists():
                        shutil.copy(text_path, conditions_dir / f"{eval_idx:04d}_text.txt")
                        conditions_saved["text"] += 1
        
        del pred_rgb, target_rgb, metrics, output, sample
    
    if csv_file:
        csv_file.close()
    
    print(f"  Conditions saved: {conditions_saved['pov']} POV, {conditions_saved['text']} text")
    
    # Read CSV and compute stats
    df = pd.read_csv(csv_path)
    
    # Best/median/worst - only non-empty rooms
    df_nonempty = df[df["is_empty"] == False]
    if len(df_nonempty) == 0:
        df_nonempty = df
    
    sorted_df = df_nonempty.sort_values("unified_score")
    worst_row = sorted_df.iloc[0]
    median_row = sorted_df.iloc[len(sorted_df) // 2]
    best_row = sorted_df.iloc[-1]
    
    if output_dir and images_dir:
        viz_dir = output_dir / "visualizations" / experiment_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for label, row in [("best", best_row), ("median", median_row), ("worst", worst_row)]:
            idx = int(row["eval_idx"])
            dataset_idx = int(row["dataset_idx"])
            score = row["unified_score"]
            
            # Load images
            pred_path = images_dir / f"{idx:04d}_pred.png"
            target_path = images_dir / f"{idx:04d}_target.png"
            cleaned_pred_path = images_dir / f"{idx:04d}_pred_cleaned.png"
            cleaned_target_path = images_dir / f"{idx:04d}_target_cleaned.png"
            
            if not pred_path.exists():
                print(f"  Warning: images not found for {label}")
                continue
            
            pred_img = np.array(Image.open(pred_path))
            target_img = np.array(Image.open(target_path))
            cleaned_pred = np.array(Image.open(cleaned_pred_path))
            cleaned_target = np.array(Image.open(cleaned_target_path))
            
            # Re-evaluate to get full metrics for visualization
            metrics = evaluator.evaluate(pred_img, target_img)
            
            # Create detailed visualization
            sample_label = f"{label.capitalize()} (eval_idx={idx}, score={score:.3f})"
            detailed, summary_viz = evaluator.create_visualization(
                pred_img, target_img, metrics, sample_label
            )
            
            detailed.save(viz_dir / f"{label}_detailed.png")
            summary_viz.save(viz_dir / f"{label}_summary.png")
            
            # Copy simple images
            shutil.copy(pred_path, viz_dir / f"{label}_pred.png")
            shutil.copy(target_path, viz_dir / f"{label}_target.png")
            
            # Copy conditions
            if conditions_dir:
                pov = conditions_dir / f"{idx:04d}_pov.png"
                txt = conditions_dir / f"{idx:04d}_text.txt"
                if pov.exists():
                    shutil.copy(pov, viz_dir / f"{label}_pov.png")
                if txt.exists():
                    shutil.copy(txt, viz_dir / f"{label}_text.txt")
            
            # Save info JSON
            info = {
                "eval_idx": idx,
                "dataset_idx": dataset_idx,
                "unified_score": float(score),
                "metrics": {k: float(v) for k, v in row.items() if k not in ["eval_idx", "dataset_idx"]}
            }
            with open(viz_dir / f"{label}_info.json", "w") as f:
                json.dump(info, f, indent=2)
            
            del pred_img, target_img, cleaned_pred, cleaned_target, metrics
        
        print(f"  Best: idx={int(best_row['eval_idx'])}, score={best_row['unified_score']:.3f}")
        print(f"  Median: idx={int(median_row['eval_idx'])}, score={median_row['unified_score']:.3f}")
        print(f"  Worst: idx={int(worst_row['eval_idx'])}, score={worst_row['unified_score']:.3f}")
    
    # Aggregated stats
    metric_cols = [c for c in df.columns if c not in ["eval_idx", "dataset_idx", "is_empty"]]
    aggregated = {}
    for col in metric_cols:
        aggregated[f"{col}_mean"] = float(df[col].mean())
        aggregated[f"{col}_std"] = float(df[col].std())
    
    aggregated["num_samples"] = len(df)
    aggregated["conditioning"] = conditioning
    aggregated["seed"] = seed
    
    return {"aggregated": aggregated, "csv_path": str(csv_path)}


def save_results(results: dict, output_dir: Path, experiment_name: str):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    full_path = output_dir / f"{experiment_name}_results_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Full results saved to: {full_path}")
    
    # Save summary
    summary_path = output_dir / f"{experiment_name}_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(results["aggregated"], f, indent=2)
    print(f"  Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    agg = results["aggregated"]
    
    print("\n[Class & Count Metrics]")
    print(f"  Class F1:          {agg.get('class_f1_mean', 0):.4f} ± {agg.get('class_f1_std', 0):.4f}")
    print(f"  Count Accuracy:    {agg.get('count_accuracy_mean', 0):.4f} ± {agg.get('count_accuracy_std', 0):.4f}")
    
    print("\n[Detection Metrics]")
    print(f"  Detection F1:      {agg.get('detection_f1_mean', 0):.4f} ± {agg.get('detection_f1_std', 0):.4f}")
    print(f"  Detection Recall:  {agg.get('detection_recall_mean', 0):.4f} ± {agg.get('detection_recall_std', 0):.4f}")
    
    print("\n[Spatial Metrics (Scale-Invariant)]")
    print(f"  BBox IoU:          {agg.get('mean_bbox_iou_mean', 0):.4f} ± {agg.get('mean_bbox_iou_std', 0):.4f}")
    print(f"  Centroid Accuracy: {agg.get('mean_centroid_accuracy_mean', 0):.4f} ± {agg.get('mean_centroid_accuracy_std', 0):.4f}")
    print(f"  Density Sim:       {agg.get('mean_density_sim_mean', 0):.4f} ± {agg.get('mean_density_sim_std', 0):.4f}")
    
    print("\n[Camera-Centric Metrics]")
    print(f"  Camera Similarity: {agg.get('mean_camera_similarity_mean', 0):.4f} ± {agg.get('mean_camera_similarity_std', 0):.4f}")
    
    print("\n[Pixel-Level Metrics]")
    print(f"  Object Accuracy:   {agg.get('object_pixel_accuracy_mean', 0):.4f} ± {agg.get('object_pixel_accuracy_std', 0):.4f}")
    print(f"  Mean IoU:          {agg.get('mean_iou_mean', 0):.4f} ± {agg.get('mean_iou_std', 0):.4f}")
    print(f"  Wall IoU:          {agg.get('wall_iou_mean', 0):.4f} ± {agg.get('wall_iou_std', 0):.4f}")
    
    print("\n" + "=" * 70)
    print(f"UNIFIED SCORE:       {agg.get('unified_score_mean', 0):.4f} ± {agg.get('unified_score_std', 0):.4f}")
    print("=" * 70)
    
    print(f"\n  Num samples: {agg['num_samples']}")
    print(f"  Conditioning: {agg['conditioning']}")


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
                        help="Maximum samples to evaluate (randomly sampled from dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--include-empty", action="store_true",
                        help="Include empty rooms (excluded by default)")
    parser.add_argument("--min-furniture", type=int, default=0,
                        help="Minimum furniture count required (default: 0)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save generated and target images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
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
    print(f"  Exclude empty rooms: {not args.include_empty}")
    print(f"  Min furniture: {args.min_furniture}")
    print("=" * 70)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Outputs - include paths for conditions
    outputs = {
        "latent": "latent_embedding_path",
        "text_emb": "graph_embedding_path",
        "pov_emb": "pov_embedding_path",
        "pov_path": "pov_path",              # Actual POV image
        "graph_text_path": "graph_text_path", # Text description
    }
    
    if args.config:
        config_outputs = config.get("dataset", {}).get("outputs", {})
        if config_outputs:
            # Merge, keeping our condition paths
            outputs.update(config_outputs)
            outputs["pov_path"] = "pov_path"
            outputs["graph_text_path"] = "graph_text_path"
    
    # Load validation data
    dataset, valid_indices, manifest_df, manifest_dir = load_validation_data(
        args.manifest,
        outputs=outputs,
        exclude_empty=not args.include_empty,
        min_furniture=args.min_furniture,
    )
    
    # Create evaluator
    evaluator = FloorplanEvaluator(args.taxonomy)
    
    # Get experiment name
    experiment_name = args.checkpoint.stem
    if experiment_name in ["best_checkpoint", "checkpoint"]:
        experiment_name = args.checkpoint.parent.parent.name
    
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
        seed=args.seed
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
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    save_results(results, args.output_dir, experiment_name)


if __name__ == "__main__":
    main()