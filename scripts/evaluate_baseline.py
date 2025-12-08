#!/usr/bin/env python3
"""
Baseline Evaluation Script for Floorplan Generation.

Evaluates a trained diffusion model on the validation set:
1. Loads checkpoint and validation data
2. Generates floorplans for each sample (single-shot)
3. Computes metrics: palette, object count, bbox IoU, centroid L1, path similarity
4. Saves detailed results and summary statistics

Usage:
    python evaluate_baseline.py --config path/to/config.yaml --checkpoint path/to/checkpoint.pt
    
    # Or specify paths directly:
    python evaluate_baseline.py \
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


def generate_single(
    model: DiffusionModel,
    sample: dict,
    device: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50
) -> dict:
    """Generate a single floorplan from conditioning."""
    
    # Get conditioning
    text_emb = sample.get("text_emb")
    pov_emb = sample.get("pov_emb")
    
    if text_emb is not None:
        text_emb = text_emb.unsqueeze(0).to(device) if text_emb.dim() == 1 else text_emb.to(device)
    if pov_emb is not None:
        pov_emb = pov_emb.unsqueeze(0).to(device) if pov_emb.dim() == 1 else pov_emb.to(device)
    
    # Generate
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
        rgb = (rgb + 1.0) / 2.0  # [-1,1] -> [0,1]
    
    return tensor_to_numpy_rgb(rgb[0])


def run_evaluation(
    model: DiffusionModel,
    dataset,
    evaluator: FloorplanEvaluator,
    device: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    max_samples: int = None,
    save_images: bool = False,
    output_dir: Path = None
) -> dict:
    """
    Run evaluation on dataset.
    
    Returns:
        Dictionary with per-sample and aggregated metrics
    """
    all_results = []
    
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    if save_images and output_dir:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating {n_samples} samples...")
    
    for idx in tqdm(range(n_samples), desc="Evaluating"):
        sample = dataset[idx]
        
        # Generate
        output = generate_single(model, sample, device, guidance_scale, num_steps)
        pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
        
        # Decode target
        target_latent = sample["latent"]
        target_rgb = decode_target(model, target_latent, device)
        
        # Evaluate
        metrics = evaluator.evaluate(pred_rgb, target_rgb, compute_paths=True)
        
        result = {
            "idx": idx,
            "metrics": metrics["summary"],
            "palette": metrics["palette"],
            "object_counts": metrics["object_counts"],
            "blob_matching": metrics["blob_matching"],
            "paths": metrics.get("paths", {})
        }
        
        # Add path info if available
        if "paths" in sample:
            result["sample_paths"] = sample["paths"]
        
        all_results.append(result)
        
        # Save images if requested
        if save_images and output_dir:
            pred_img = Image.fromarray(pred_rgb)
            target_img = Image.fromarray(target_rgb)
            cleaned_pred = Image.fromarray(metrics["_cleaned_pred"])
            cleaned_target = Image.fromarray(metrics["_cleaned_target"])
            
            pred_img.save(images_dir / f"{idx:04d}_pred.png")
            target_img.save(images_dir / f"{idx:04d}_target.png")
            cleaned_pred.save(images_dir / f"{idx:04d}_pred_cleaned.png")
            cleaned_target.save(images_dir / f"{idx:04d}_target_cleaned.png")
    
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
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in results["aggregated"].items():
        if key != "num_samples":
            print(f"  {key}: {value:.4f}")
    print(f"  num_samples: {results['aggregated']['num_samples']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for floorplan generation")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint")
    
    # Optional: load from config
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to training config (to get manifest, etc.)")
    
    # Or specify directly
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to manifest CSV")
    parser.add_argument("--taxonomy", type=Path, default=None,
                        help="Path to taxonomy.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results")
    
    # Evaluation settings
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="CFG guidance scale")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of DDIM steps")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (None = all)")
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
    
    # Validate required paths
    if args.manifest is None:
        raise ValueError("Must provide --manifest or --config")
    
    if args.output_dir is None:
        args.output_dir = Path("evaluation_results")
    
    # Find taxonomy
    if args.taxonomy is None:
        # Try common locations
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
    
    print(f"Taxonomy: {args.taxonomy}")
    print(f"Manifest: {args.manifest}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Determine outputs from config or defaults
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
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
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
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    experiment_name = args.checkpoint.stem
    save_results(results, args.output_dir, experiment_name)


if __name__ == "__main__":
    main()
