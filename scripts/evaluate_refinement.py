#!/usr/bin/env python3
"""
Iterative Refinement Experiment Script.

Tests two approaches to incorporating new observations:
1. ACCUMULATION: Average more POV embeddings, generate fresh each time
2. REFINEMENT: Use previous generation as prior, noise and denoise with new POV

This script directly addresses RQ2:
"Does iterative evidence assimilation improve plan quality?"

Usage:
    python evaluate_refinement.py \
        --checkpoint /path/to/checkpoint.pt \
        --manifest /path/to/manifest.csv \
        --taxonomy /path/to/taxonomy.json \
        --output-dir /path/to/results
"""

import argparse
import json
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from collections import defaultdict
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.datasets.multi_pov_dataset import MultiPOVDataset
from training.evaluation_metrics import (
    FloorplanEvaluator,
    tensor_to_numpy_rgb
)


def load_model(checkpoint_path: Path, device: str = "cuda") -> DiffusionModel:
    """Load diffusion model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = DiffusionModel.load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def decode_latent(model: DiffusionModel, latent: torch.Tensor, device: str) -> np.ndarray:
    """Decode latent to RGB image."""
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    latent = latent.to(device)
    
    with torch.no_grad():
        decoded = model.decoder({"latent": latent})
        rgb = decoded["rgb"]
        rgb = (rgb + 1.0) / 2.0
    
    return tensor_to_numpy_rgb(rgb[0])


def run_accumulation_experiment(
    model: DiffusionModel,
    sample: dict,
    evaluator: FloorplanEvaluator,
    device: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    max_povs: int = 5
) -> Tuple[list, np.ndarray, List[np.ndarray]]:
    """
    Run accumulation experiment: progressively add more POVs.
    
    At each step, average the first N POV embeddings and generate fresh.
    
    Returns:
        results: List of metrics per n_povs
        target_rgb: Target image
        pred_images: List of predicted images per n_povs
    """
    results = []
    pred_images = []
    all_povs = sample.get("all_pov_embs", [])
    text_emb = sample.get("text_emb")
    target_latent = sample.get("latent")
    
    # Decode target once
    target_rgb = decode_latent(model, target_latent, device)
    
    # Prepare text embedding
    if text_emb is not None:
        text_emb_batch = text_emb.unsqueeze(0).to(device) if text_emb.dim() == 1 else text_emb.to(device)
    else:
        text_emb_batch = None
    
    max_povs = min(max_povs, len(all_povs))
    
    for n_povs in range(0, max_povs + 1):
        # Combine POV embeddings
        if n_povs == 0 or len(all_povs) == 0:
            pov_emb_batch = None
        else:
            povs_to_use = all_povs[:n_povs]
            combined_pov = torch.stack(povs_to_use).mean(dim=0)
            pov_emb_batch = combined_pov.unsqueeze(0).to(device)
        
        # Generate fresh
        with torch.no_grad():
            output = model.sample(
                batch_size=1,
                num_steps=num_steps,
                method="ddim",
                guidance_scale=guidance_scale,
                text_emb=text_emb_batch,
                pov_emb=pov_emb_batch,
                verbose=False
            )
        
        pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
        pred_images.append(pred_rgb)
        
        # Evaluate
        metrics = evaluator.evaluate(pred_rgb, target_rgb, compute_paths=True)
        
        results.append({
            "n_povs": n_povs,
            "method": "accumulation",
            "metrics": metrics["summary"]
        })
    
    return results, target_rgb, pred_images


def run_refinement_experiment(
    model: DiffusionModel,
    sample: dict,
    evaluator: FloorplanEvaluator,
    device: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    max_povs: int = 5,
    noise_strength: float = 0.3
) -> Tuple[list, np.ndarray, List[np.ndarray]]:
    """
    Run refinement experiment: each generation builds on the previous.
    
    At each step after the first:
    1. Take previous generated latent
    2. Add noise (controlled by noise_strength)
    3. Denoise with new POV conditioning
    
    Returns:
        results: List of metrics per n_povs
        target_rgb: Target image
        pred_images: List of predicted images per n_povs
    """
    results = []
    pred_images = []
    all_povs = sample.get("all_pov_embs", [])
    text_emb = sample.get("text_emb")
    target_latent = sample.get("latent")
    
    # Decode target once
    target_rgb = decode_latent(model, target_latent, device)
    
    # Prepare text embedding
    if text_emb is not None:
        text_emb_batch = text_emb.unsqueeze(0).to(device) if text_emb.dim() == 1 else text_emb.to(device)
    else:
        text_emb_batch = None
    
    max_povs = min(max_povs, len(all_povs))
    prior_latent = None
    
    for n_povs in range(0, max_povs + 1):
        # Get current POV embedding (single, not accumulated)
        if n_povs == 0 or len(all_povs) == 0:
            pov_emb_batch = None
        else:
            # Use the N-th POV (0-indexed, so n_povs-1)
            current_pov = all_povs[n_povs - 1]
            pov_emb_batch = current_pov.unsqueeze(0).to(device)
        
        with torch.no_grad():
            if prior_latent is None:
                # First generation: start from noise
                output = model.sample(
                    batch_size=1,
                    num_steps=num_steps,
                    method="ddim",
                    guidance_scale=guidance_scale,
                    text_emb=text_emb_batch,
                    pov_emb=pov_emb_batch,
                    verbose=False
                )
            else:
                # Refinement: start from noised prior
                start_step = int(model.scheduler.num_steps * noise_strength)
                noised_prior = model.add_noise_to_latent(prior_latent, start_step)
                
                output = model.sample_from(
                    x_t=noised_prior,
                    start_step=start_step,
                    num_steps=num_steps,
                    method="ddim",
                    guidance_scale=guidance_scale,
                    text_emb=text_emb_batch,
                    pov_emb=pov_emb_batch,
                    verbose=False
                )
        
        # Store latent for next iteration
        prior_latent = output["latent"].clone()
        
        pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
        pred_images.append(pred_rgb)
        
        # Evaluate
        metrics = evaluator.evaluate(pred_rgb, target_rgb, compute_paths=True)
        
        results.append({
            "n_povs": n_povs,
            "method": "refinement",
            "noise_strength": noise_strength,
            "metrics": metrics["summary"]
        })
    
    return results, target_rgb, pred_images


def run_full_experiment(
    model: DiffusionModel,
    dataset: MultiPOVDataset,
    evaluator: FloorplanEvaluator,
    device: str,
    experiment_name: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    max_povs: int = 5,
    noise_strengths: list = [0.3],
    max_samples: int = None,
    save_images: bool = False,
    output_dir: Path = None
) -> dict:
    """
    Run full refinement experiment on dataset.
    """
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    all_accumulation_results = []
    all_refinement_results = defaultdict(list)
    
    # Setup image saving directory
    images_dir = None
    if save_images and output_dir:
        images_dir = output_dir / "images" / experiment_name
        images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning refinement experiment on {n_samples} samples...")
    print(f"  Max POVs: {max_povs}")
    print(f"  Noise strengths: {noise_strengths}")
    if images_dir:
        print(f"  Saving images to: {images_dir}")
    
    for idx in tqdm(range(n_samples), desc="Samples"):
        sample = dataset[idx]
        
        # Skip samples with too few POVs
        if sample["num_povs"] < 2:
            continue
        
        # Run accumulation
        acc_results, target_rgb, acc_images = run_accumulation_experiment(
            model, sample, evaluator, device,
            guidance_scale, num_steps, max_povs
        )
        for r in acc_results:
            r["sample_idx"] = idx
        all_accumulation_results.extend(acc_results)
        
        # Run refinement with different noise strengths
        ref_images_by_ns = {}
        for ns in noise_strengths:
            ref_results, _, ref_images = run_refinement_experiment(
                model, sample, evaluator, device,
                guidance_scale, num_steps, max_povs, ns
            )
            for r in ref_results:
                r["sample_idx"] = idx
            all_refinement_results[ns].extend(ref_results)
            ref_images_by_ns[ns] = ref_images
        
        # Save images for this sample
        if images_dir:
            sample_dir = images_dir / f"sample_{idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Save target
            Image.fromarray(target_rgb).save(sample_dir / "target.png")
            
            # Save accumulation progression
            for pov_idx, img in enumerate(acc_images):
                Image.fromarray(img).save(sample_dir / f"accum_pov{pov_idx}.png")
            
            # Save refinement progression (for each noise strength)
            for ns, ref_imgs in ref_images_by_ns.items():
                ns_str = str(ns).replace(".", "p")
                for pov_idx, img in enumerate(ref_imgs):
                    Image.fromarray(img).save(sample_dir / f"refine_ns{ns_str}_pov{pov_idx}.png")
    
    # Aggregate results by n_povs
    def aggregate_by_povs(results_list):
        by_povs = defaultdict(list)
        for r in results_list:
            by_povs[r["n_povs"]].append(r["metrics"])
        
        aggregated = {}
        for n_povs, metrics_list in sorted(by_povs.items()):
            agg = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))
            agg["n_samples"] = len(metrics_list)
            aggregated[n_povs] = agg
        
        return aggregated
    
    results = {
        "accumulation": {
            "per_sample": all_accumulation_results,
            "by_n_povs": aggregate_by_povs(all_accumulation_results)
        },
        "refinement": {}
    }
    
    for ns, ref_results in all_refinement_results.items():
        results["refinement"][f"ns_{ns}"] = {
            "per_sample": ref_results,
            "by_n_povs": aggregate_by_povs(ref_results)
        }
    
    return results


def print_comparison_table(results: dict):
    """Print comparison table of accumulation vs refinement."""
    print("\n" + "=" * 80)
    print("COMPARISON: ACCUMULATION vs REFINEMENT")
    print("=" * 80)
    
    acc_by_povs = results["accumulation"]["by_n_povs"]
    
    # Get refinement results (first noise strength)
    ref_key = list(results["refinement"].keys())[0]
    ref_by_povs = results["refinement"][ref_key]["by_n_povs"]
    
    # Header
    print(f"{'N_POVs':<8} {'Metric':<25} {'Accumulation':<15} {'Refinement':<15} {'Δ':<10}")
    print("-" * 80)
    
    metrics_to_show = ["palette_jaccard", "mean_bbox_iou", "path_overlap"]
    
    for n_povs in sorted(acc_by_povs.keys()):
        if n_povs not in ref_by_povs:
            continue
        
        for metric in metrics_to_show:
            acc_val = acc_by_povs[n_povs].get(f"{metric}_mean", 0)
            ref_val = ref_by_povs[n_povs].get(f"{metric}_mean", 0)
            delta = ref_val - acc_val
            
            print(f"{n_povs:<8} {metric:<25} {acc_val:<15.4f} {ref_val:<15.4f} {delta:+.4f}")
        print()


def save_results(results: dict, output_dir: Path, experiment_name: str):
    """Save experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    full_path = output_dir / f"{experiment_name}_refinement_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {full_path}")
    
    # Save summary CSV for easy plotting
    import csv
    csv_path = output_dir / f"{experiment_name}_refinement_summary_{timestamp}.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        header = ["method", "noise_strength", "n_povs", 
                  "palette_jaccard_mean", "palette_jaccard_std",
                  "mean_bbox_iou_mean", "mean_bbox_iou_std",
                  "path_overlap_mean", "path_overlap_std",
                  "n_samples"]
        writer.writerow(header)
        
        # Accumulation rows
        for n_povs, agg in results["accumulation"]["by_n_povs"].items():
            row = ["accumulation", "N/A", n_povs,
                   agg.get("palette_jaccard_mean", ""),
                   agg.get("palette_jaccard_std", ""),
                   agg.get("mean_bbox_iou_mean", ""),
                   agg.get("mean_bbox_iou_std", ""),
                   agg.get("path_overlap_mean", ""),
                   agg.get("path_overlap_std", ""),
                   agg.get("n_samples", "")]
            writer.writerow(row)
        
        # Refinement rows
        for ns_key, ref_data in results["refinement"].items():
            ns = ns_key.replace("ns_", "")
            for n_povs, agg in ref_data["by_n_povs"].items():
                row = ["refinement", ns, n_povs,
                       agg.get("palette_jaccard_mean", ""),
                       agg.get("palette_jaccard_std", ""),
                       agg.get("mean_bbox_iou_mean", ""),
                       agg.get("mean_bbox_iou_std", ""),
                       agg.get("path_overlap_mean", ""),
                       agg.get("path_overlap_std", ""),
                       agg.get("n_samples", "")]
                writer.writerow(row)
    
    print(f"Summary CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Iterative refinement experiment")
    
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("refinement_results"))
    
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--max-povs", type=int, default=5)
    parser.add_argument("--noise-strengths", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--group-by", type=str, default="room_id",
                        help="Column to group POVs by")
    parser.add_argument("--save-images", action="store_true",
                        help="Save progression images for each sample")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load multi-POV dataset
    print(f"\nLoading multi-POV dataset from {args.manifest}...")
    dataset = MultiPOVDataset(
        manifest_path=args.manifest,
        outputs={
            "latent": "latent_embedding_path",
            "text_emb": "graph_embedding_path",
            "pov_emb": "pov_embedding_path"
        },
        group_by=args.group_by,
        filters={"rejected": False}
    )
    print(f"  Dataset size: {len(dataset)} rooms/scenes")
    
    # Create evaluator
    evaluator = FloorplanEvaluator(args.taxonomy)
    
    # Get experiment name from checkpoint path
    experiment_name = args.checkpoint.stem
    if experiment_name in ["best_checkpoint", "checkpoint"]:
        experiment_name = args.checkpoint.parent.parent.name
    
    # Run experiment
    results = run_full_experiment(
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        device=args.device,
        experiment_name=experiment_name,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        max_povs=args.max_povs,
        noise_strengths=args.noise_strengths,
        max_samples=args.max_samples,
        save_images=args.save_images,
        output_dir=args.output_dir
    )
    
    # Add metadata
    results["metadata"] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "max_povs": args.max_povs,
        "noise_strengths": args.noise_strengths,
        "timestamp": datetime.now().isoformat()
    }
    
    # Print comparison
    print_comparison_table(results)
    
    # Save
    save_results(results, args.output_dir, experiment_name)


if __name__ == "__main__":
    main()
