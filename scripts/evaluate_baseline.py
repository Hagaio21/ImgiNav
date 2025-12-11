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
import sys
import yaml
import torch
import numpy as np
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
    filters: dict = None,
):
    """Load evaluation dataset directly from eval_manifest.csv."""
    print(f"Loading evaluation dataset from {manifest_path}...")
    
    if filters is None:
        filters = {"rejected": False}
    
    dataset = ManifestDataset(
        manifest=str(manifest_path),
        outputs=outputs,
        filters=filters,
        return_path=True
    )
    
    print(f"  Evaluation samples: {len(dataset)}")
    
    return dataset


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
    """
    Run evaluation on dataset.
    
    Args:
        model: Diffusion model
        dataset: Validation dataset
        evaluator: Floorplan evaluator
        device: Device to use
        experiment_name: Name for output folders
        guidance_scale: CFG scale
        num_steps: DDIM steps
        max_samples: Max samples to evaluate (randomly sampled if < len(dataset))
        save_images: Whether to save all images
        output_dir: Output directory
        conditioning: Conditioning type (pov/graph/both)
        seed: Random seed for reproducible sampling
    
    Returns:
        Dictionary with per-sample and aggregated metrics
    """
    all_results = []
    all_pred_images = []
    all_target_images = []
    all_metrics = []
    
    n_total = len(dataset)
    n_samples = n_total if max_samples is None else min(max_samples, n_total)
    
    # Random sampling with seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if max_samples is not None and max_samples < n_total:
        sample_indices = random.sample(range(n_total), n_samples)
        sample_indices.sort()  # Sort for reproducibility in logs
    else:
        sample_indices = list(range(n_samples))
    
    if save_images and output_dir:
        images_dir = output_dir / "images" / experiment_name
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create conditions subdirectory
        conditions_dir = images_dir / "conditions"
        conditions_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_dir = None
        conditions_dir = None
    
    print(f"\nEvaluating {n_samples} samples (from {n_total} total) with conditioning: {conditioning}")
    if seed is not None:
        print(f"  Random seed: {seed}")
    
    for eval_idx, dataset_idx in enumerate(tqdm(sample_indices, desc="Evaluating")):
        sample = dataset[dataset_idx]
        
        # Generate
        output = generate_single(
            model, sample, device, guidance_scale, num_steps, conditioning=conditioning
        )
        pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
        
        # Decode target
        target_latent = sample["latent"]
        target_rgb = decode_target(model, target_latent, device)
        
        # Evaluate
        metrics = evaluator.evaluate(pred_rgb, target_rgb)
        
        # Build result with condition info
        result = {
            "eval_idx": eval_idx,
            "dataset_idx": dataset_idx,
            "metrics": metrics["summary"],
            "class_presence": metrics["class_presence"],
            "counts": metrics["counts"],
            "spatial": metrics["spatial"],
            "pixels": metrics["pixels"],
            "conditions": {}
        }
        
        # Extract condition paths from sample
        if "paths" in sample:
            result["sample_paths"] = sample["paths"]
        
        # Store condition information
        if "pov_emb" in sample:
            pov_path = sample.get("paths", {}).get("pov_emb", None)
            if pov_path:
                result["conditions"]["pov_embedding_path"] = str(pov_path)
                
                # Try to find corresponding POV image
                pov_emb_path = Path(pov_path)
                # Common pattern: embedding is .pt, image is .png/.jpg in similar location
                possible_pov_images = [
                    pov_emb_path.with_suffix(".png"),
                    pov_emb_path.with_suffix(".jpg"),
                    pov_emb_path.parent.parent / "povs" / (pov_emb_path.stem + ".png"),
                    pov_emb_path.parent.parent / "povs" / (pov_emb_path.stem + ".jpg"),
                ]
                for pov_img_path in possible_pov_images:
                    if pov_img_path.exists():
                        result["conditions"]["pov_image_path"] = str(pov_img_path)
                        break
        
        if "text_emb" in sample:
            text_path = sample.get("paths", {}).get("text_emb", None)
            if text_path:
                result["conditions"]["text_embedding_path"] = str(text_path)
                
                # Try to find corresponding text/graph description
                text_emb_path = Path(text_path)
                possible_text_files = [
                    text_emb_path.with_suffix(".txt"),
                    text_emb_path.with_suffix(".json"),
                    text_emb_path.parent.parent / "graphs" / (text_emb_path.stem + ".txt"),
                    text_emb_path.parent.parent / "graphs" / (text_emb_path.stem + ".json"),
                ]
                for text_file_path in possible_text_files:
                    if text_file_path.exists():
                        result["conditions"]["text_description_path"] = str(text_file_path)
                        # Try to read the text content
                        try:
                            if text_file_path.suffix == ".json":
                                with open(text_file_path, "r") as f:
                                    text_data = json.load(f)
                                    result["conditions"]["text_description"] = text_data
                            else:
                                with open(text_file_path, "r") as f:
                                    result["conditions"]["text_description"] = f.read()
                        except Exception:
                            pass
                        break
        
        all_results.append(result)
        all_pred_images.append(pred_rgb)
        all_target_images.append(target_rgb)
        all_metrics.append(metrics)
        
        # Save individual images
        if images_dir is not None:
            pred_img = Image.fromarray(pred_rgb)
            target_img = Image.fromarray(target_rgb)
            cleaned_pred = Image.fromarray(metrics["_cleaned_pred"])
            cleaned_target = Image.fromarray(metrics["_cleaned_target"])
            
            pred_img.save(images_dir / f"{eval_idx:04d}_pred.png")
            target_img.save(images_dir / f"{eval_idx:04d}_target.png")
            cleaned_pred.save(images_dir / f"{eval_idx:04d}_pred_cleaned.png")
            cleaned_target.save(images_dir / f"{eval_idx:04d}_target_cleaned.png")
            
            # Save conditions
            if conditions_dir:
                # Save POV image if available
                pov_img_path = result["conditions"].get("pov_image_path")
                if pov_img_path and Path(pov_img_path).exists():
                    shutil.copy(pov_img_path, conditions_dir / f"{eval_idx:04d}_pov.png")
                
                # Save text description if available
                text_desc = result["conditions"].get("text_description")
                if text_desc:
                    with open(conditions_dir / f"{eval_idx:04d}_text.txt", "w") as f:
                        if isinstance(text_desc, dict):
                            f.write(json.dumps(text_desc, indent=2))
                        else:
                            f.write(str(text_desc))
    
    # Find best and median samples
    unified_scores = [r["metrics"]["unified_score"] for r in all_results]
    sorted_indices = np.argsort(unified_scores)
    
    best_idx = sorted_indices[-1]
    worst_idx = sorted_indices[0]
    median_idx = sorted_indices[len(sorted_indices) // 2]
    
    # Create visualizations for best, median, worst
    if output_dir:
        viz_dir = output_dir / "visualizations" / experiment_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for label, sample_idx in [("best", best_idx), ("median", median_idx), ("worst", worst_idx)]:
            score = all_results[sample_idx]["metrics"]["unified_score"]
            dataset_idx = all_results[sample_idx]["dataset_idx"]
            sample_label = f"{label.capitalize()} Sample (eval_idx={sample_idx}, dataset_idx={dataset_idx}, score={score:.3f})"
            
            detailed, summary = evaluator.create_visualization(
                all_pred_images[sample_idx],
                all_target_images[sample_idx],
                all_metrics[sample_idx],
                sample_label
            )
            
            detailed.save(viz_dir / f"{label}_detailed.png")
            summary.save(viz_dir / f"{label}_summary.png")
            
            # Save individual images for these samples
            Image.fromarray(all_pred_images[sample_idx]).save(viz_dir / f"{label}_pred.png")
            Image.fromarray(all_target_images[sample_idx]).save(viz_dir / f"{label}_target.png")
            
            # Save conditions for visualization samples
            conditions = all_results[sample_idx].get("conditions", {})
            pov_img_path = conditions.get("pov_image_path")
            if pov_img_path and Path(pov_img_path).exists():
                shutil.copy(pov_img_path, viz_dir / f"{label}_condition_pov.png")
            
            text_desc = conditions.get("text_description")
            if text_desc:
                with open(viz_dir / f"{label}_condition_text.txt", "w") as f:
                    if isinstance(text_desc, dict):
                        f.write(json.dumps(text_desc, indent=2))
                    else:
                        f.write(str(text_desc))
        
        print(f"\n  Visualizations saved to: {viz_dir}")
        print(f"    Best sample: eval_idx={best_idx}, dataset_idx={all_results[best_idx]['dataset_idx']}, score={unified_scores[best_idx]:.3f}")
        print(f"    Median sample: eval_idx={median_idx}, dataset_idx={all_results[median_idx]['dataset_idx']}, score={unified_scores[median_idx]:.3f}")
        print(f"    Worst sample: eval_idx={worst_idx}, dataset_idx={all_results[worst_idx]['dataset_idx']}, score={unified_scores[worst_idx]:.3f}")
    
    # Aggregate metrics
    summary_keys = all_results[0]["metrics"].keys()
    aggregated = {}
    
    for key in summary_keys:
        values = [r["metrics"][key] for r in all_results]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
        aggregated[f"{key}_min"] = float(np.min(values))
        aggregated[f"{key}_max"] = float(np.max(values))
    
    aggregated["num_samples"] = n_samples
    aggregated["total_dataset_size"] = n_total
    aggregated["conditioning"] = conditioning
    aggregated["seed"] = seed
    aggregated["sample_indices"] = sample_indices
    aggregated["best_sample_idx"] = int(best_idx)
    aggregated["median_sample_idx"] = int(median_idx)
    aggregated["worst_sample_idx"] = int(worst_idx)
    
    return {
        "per_sample": all_results,
        "aggregated": aggregated
    }


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
    print("=" * 70)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Outputs
    outputs = {
        "latent": "latent_embedding_path",
        "text_emb": "graph_embedding_path",
        "pov_emb": "pov_embedding_path"
    }
    
    if args.config:
        config_outputs = config.get("dataset", {}).get("outputs", {})
        if config_outputs:
            outputs = config_outputs
    
    # Load validation data
    dataset = load_validation_data(
        args.manifest,
        outputs=outputs,
        filters={"rejected": False}
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