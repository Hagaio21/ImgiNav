#!/usr/bin/env python3
"""
Iterative Refinement Experiment Script.

Updated with navigation-focused metrics and visualization.

Tests two approaches to incorporating new observations:
1. ACCUMULATION: Average more POV embeddings, generate fresh each time
2. REFINEMENT: Use previous generation as prior, noise and denoise with new POV

Features:
- Random sampling from validation set with seed for reproducibility
- Saves conditions (POV images, text descriptions) for each sample
- Generates progression visualizations
- Saves per-sample metrics in CSV format

Usage:
    python evaluate_refinement.py \
        --checkpoint /path/to/checkpoint.pt \
        --manifest /path/to/manifest.csv \
        --taxonomy /path/to/taxonomy.json \
        --max-samples 50 --seed 42
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
from collections import defaultdict
from typing import List, Tuple, Dict
import shutil

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
    max_povs: int = 5,
    baseline: dict = None
) -> Tuple[list, np.ndarray, List[np.ndarray], List[Dict]]:
    """
    Run accumulation experiment: progressively add more POVs.
    
    Args:
        baseline: Optional dict with 'latent', 'rgb', 'metrics' for shared baseline (0 POV)
    
    Returns:
        results: List of metrics per n_povs
        target_rgb: Target image
        pred_images: List of predicted images per n_povs
        all_metrics: List of full metrics dicts
    """
    results = []
    pred_images = []
    all_metrics = []
    all_povs = sample.get("all_pov_embs", [])
    text_emb = sample.get("text_emb")
    target_latent = sample.get("latent")
    
    target_rgb = decode_latent(model, target_latent, device)
    
    if text_emb is not None:
        text_emb_batch = text_emb.unsqueeze(0).to(device) if text_emb.dim() == 1 else text_emb.to(device)
    else:
        text_emb_batch = None
    
    max_povs = min(max_povs, len(all_povs))
    
    for n_povs in range(0, max_povs + 1):
        if n_povs == 0:
            # Use shared baseline
            if baseline is not None:
                pred_rgb = baseline["rgb"]
                metrics = baseline["metrics"]
            else:
                # Generate baseline (should not happen if called correctly)
                with torch.no_grad():
                    output = model.sample(
                        batch_size=1,
                        num_steps=num_steps,
                        method="ddim",
                        guidance_scale=guidance_scale,
                        text_emb=text_emb_batch,
                        pov_emb=None,
                        verbose=False
                    )
                pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
                metrics = evaluator.evaluate(pred_rgb, target_rgb)
        else:
            povs_to_use = all_povs[:n_povs]
            combined_pov = torch.stack(povs_to_use).mean(dim=0)
            pov_emb_batch = combined_pov.unsqueeze(0).to(device)
            
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
            metrics = evaluator.evaluate(pred_rgb, target_rgb)
        
        pred_images.append(pred_rgb)
        all_metrics.append(metrics)
        
        results.append({
            "n_povs": n_povs,
            "method": "accumulation",
            "metrics": metrics["summary"]
        })
    
    return results, target_rgb, pred_images, all_metrics


def run_refinement_experiment(
    model: DiffusionModel,
    sample: dict,
    evaluator: FloorplanEvaluator,
    device: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    max_povs: int = 5,
    noise_strength: float = 0.3,
    baseline: dict = None
) -> Tuple[list, np.ndarray, List[np.ndarray], List[Dict]]:
    """
    Run refinement experiment: each generation builds on the previous.
    
    Args:
        baseline: Optional dict with 'latent', 'rgb', 'metrics' for shared baseline (0 POV)
    
    Returns:
        results: List of metrics per n_povs
        target_rgb: Target image
        pred_images: List of predicted images per n_povs
        all_metrics: List of full metrics dicts
    """
    results = []
    pred_images = []
    all_metrics = []
    all_povs = sample.get("all_pov_embs", [])
    text_emb = sample.get("text_emb")
    target_latent = sample.get("latent")
    
    target_rgb = decode_latent(model, target_latent, device)
    
    if text_emb is not None:
        text_emb_batch = text_emb.unsqueeze(0).to(device) if text_emb.dim() == 1 else text_emb.to(device)
    else:
        text_emb_batch = None
    
    max_povs = min(max_povs, len(all_povs))
    prior_latent = None
    
    for n_povs in range(0, max_povs + 1):
        if n_povs == 0:
            # Use shared baseline
            if baseline is not None:
                pred_rgb = baseline["rgb"]
                metrics = baseline["metrics"]
                prior_latent = baseline["latent"].clone()
            else:
                # Generate baseline (should not happen if called correctly)
                with torch.no_grad():
                    output = model.sample(
                        batch_size=1,
                        num_steps=num_steps,
                        method="ddim",
                        guidance_scale=guidance_scale,
                        text_emb=text_emb_batch,
                        pov_emb=None,
                        verbose=False
                    )
                pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
                metrics = evaluator.evaluate(pred_rgb, target_rgb)
                prior_latent = output["latent"].clone()
        else:
            current_pov = all_povs[n_povs - 1]
            pov_emb_batch = current_pov.unsqueeze(0).to(device)
            
            with torch.no_grad():
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
            
            prior_latent = output["latent"].clone()
            pred_rgb = tensor_to_numpy_rgb(output["rgb"][0])
            metrics = evaluator.evaluate(pred_rgb, target_rgb)
        
        pred_images.append(pred_rgb)
        all_metrics.append(metrics)
        
        results.append({
            "n_povs": n_povs,
            "method": "refinement",
            "noise_strength": noise_strength,
            "metrics": metrics["summary"]
        })
    
    return results, target_rgb, pred_images, all_metrics


def create_progression_visualization(
    pred_images: List[np.ndarray],
    target_rgb: np.ndarray,
    method: str,
    scores: List[float],
    output_path: Path
):
    """Create a progression visualization showing improvement over POVs."""
    from PIL import Image, ImageDraw, ImageFont
    
    n_images = len(pred_images)
    h, w = pred_images[0].shape[:2]
    
    # Layout: target + all preds in a row
    canvas_width = (n_images + 1) * (w + 10) + 10
    canvas_height = h + 80
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Paste target
    canvas.paste(Image.fromarray(target_rgb), (10, 40))
    draw.text((10, 10), "Target", fill=(0, 0, 0), font=font)
    
    # Paste predictions
    for i, (pred, score) in enumerate(zip(pred_images, scores)):
        x_offset = (i + 1) * (w + 10) + 10
        canvas.paste(Image.fromarray(pred), (x_offset, 40))
        draw.text((x_offset, 10), f"{method} POV={i}", fill=(0, 0, 0), font=font)
        draw.text((x_offset, h + 45), f"Score: {score:.3f}", fill=(50, 50, 50), font=small_font)
    
    # Draw improvement arrow if scores improved
    if len(scores) > 1 and scores[-1] > scores[0]:
        improvement = scores[-1] - scores[0]
        arrow_y = h + 65
        draw.line([(w + 20, arrow_y), (canvas_width - 20, arrow_y)], fill=(0, 128, 0), width=2)
        draw.text((canvas_width // 2, arrow_y + 5), f"+{improvement:.3f}", fill=(0, 128, 0), font=small_font)
    
    canvas.save(output_path)


def create_comparison_visualization(
    target_rgb: np.ndarray,
    baseline_image: np.ndarray,
    acc_image: np.ndarray,
    ref_image: np.ndarray,
    baseline_score: float,
    acc_score: float,
    ref_score: float,
    noise_strength: float,
    output_path: Path
):
    """Create a comparison visualization showing [Target, Baseline, Accumulation, Refinement] side by side."""
    from PIL import Image, ImageDraw, ImageFont
    
    h, w = target_rgb.shape[:2]
    padding = 10
    
    # Layout: 4 images in a row
    canvas_width = 4 * w + 5 * padding
    canvas_height = h + 80
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    images = [target_rgb, baseline_image, acc_image, ref_image]
    titles = ["Target", "Baseline (0 POV)", "Accumulation", f"Refinement (ns={noise_strength})"]
    scores = [None, baseline_score, acc_score, ref_score]
    
    for i, (img, title, score) in enumerate(zip(images, titles, scores)):
        x_offset = padding + i * (w + padding)
        canvas.paste(Image.fromarray(img), (x_offset, 40))
        draw.text((x_offset, 10), title, fill=(0, 0, 0), font=font)
        if score is not None:
            draw.text((x_offset, h + 45), f"Score: {score:.3f}", fill=(50, 50, 50), font=small_font)
    
    # Show improvements over baseline
    acc_delta = acc_score - baseline_score
    ref_delta = ref_score - baseline_score
    
    acc_color = (0, 128, 0) if acc_delta > 0 else (200, 0, 0)
    ref_color = (0, 128, 0) if ref_delta > 0 else (200, 0, 0)
    
    acc_x = padding + 2 * (w + padding)
    ref_x = padding + 3 * (w + padding)
    
    acc_sign = "+" if acc_delta > 0 else ""
    ref_sign = "+" if ref_delta > 0 else ""
    
    draw.text((acc_x, h + 62), f"vs baseline: {acc_sign}{acc_delta:.3f}", fill=acc_color, font=small_font)
    draw.text((ref_x, h + 62), f"vs baseline: {ref_sign}{ref_delta:.3f}", fill=ref_color, font=small_font)
    
    canvas.save(output_path)


def create_metrics_progression_plot(
    acc_metrics: List[Dict],
    ref_metrics_by_ns: Dict[float, List[Dict]],
    output_path: Path,
    metric_key: str = "unified_score"
):
    """Create a line plot showing metrics at each POV step for accumulation and refinement."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_povs = len(acc_metrics)
    x = list(range(n_povs))
    
    # Extract metrics
    acc_scores = [m["summary"][metric_key] for m in acc_metrics]
    
    plt.figure(figsize=(10, 6))
    
    # Plot accumulation
    plt.plot(x, acc_scores, 'b-o', label='Accumulation', linewidth=2, markersize=8)
    
    # Plot refinement for each noise strength
    colors = ['r', 'g', 'm', 'c']
    for i, (ns, ref_metrics) in enumerate(ref_metrics_by_ns.items()):
        ref_scores = [m["summary"][metric_key] for m in ref_metrics]
        color = colors[i % len(colors)]
        plt.plot(x, ref_scores, f'{color}-s', label=f'Refinement (ns={ns})', linewidth=2, markersize=8)
    
    plt.xlabel('Number of POVs', fontsize=12)
    plt.ylabel(metric_key.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric_key.replace("_", " ").title()} vs Number of POVs', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    
    # Add value annotations
    for i, score in enumerate(acc_scores):
        plt.annotate(f'{score:.3f}', (i, score), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_metrics_table_image(
    acc_metrics: List[Dict],
    ref_metrics_by_ns: Dict[float, List[Dict]],
    output_path: Path,
    metrics_to_show: List[str] = None
):
    """Create a table image showing metrics at each POV step."""
    from PIL import Image, ImageDraw, ImageFont
    
    if metrics_to_show is None:
        metrics_to_show = ["unified_score", "mean_bbox_iou", "detection_f1", "count_accuracy"]
    
    n_povs = len(acc_metrics)
    noise_strengths = list(ref_metrics_by_ns.keys())
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Table dimensions
    col_width = 90
    row_height = 22
    header_height = 25
    metric_col_width = 120
    
    n_methods = 1 + len(noise_strengths)  # accumulation + refinements
    n_cols = 1 + n_povs  # metric name + pov columns
    n_rows = len(metrics_to_show) * n_methods + n_methods  # metrics per method + method headers
    
    canvas_width = metric_col_width + n_povs * col_width + 20
    canvas_height = header_height + n_rows * row_height + 40
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Title
    draw.text((10, 5), "Metrics Progression by POV Step", fill=(0, 0, 0), font=font)
    
    # Column headers (POV numbers)
    y = header_height + 10
    draw.text((10, y), "Metric", fill=(0, 0, 0), font=font)
    for pov_idx in range(n_povs):
        x = metric_col_width + pov_idx * col_width
        draw.text((x, y), f"POV {pov_idx}", fill=(0, 0, 0), font=font)
    
    y += row_height
    
    # Accumulation section
    draw.rectangle([(5, y), (canvas_width - 5, y + row_height)], fill=(200, 200, 255))
    draw.text((10, y + 3), "ACCUMULATION", fill=(0, 0, 128), font=font)
    y += row_height
    
    for metric in metrics_to_show:
        draw.text((10, y + 3), metric.replace('_', ' ')[:15], fill=(50, 50, 50), font=small_font)
        for pov_idx, m in enumerate(acc_metrics):
            x = metric_col_width + pov_idx * col_width
            val = m["summary"].get(metric, 0)
            draw.text((x, y + 3), f"{val:.3f}", fill=(0, 0, 0), font=small_font)
        y += row_height
    
    # Refinement sections
    for ns in noise_strengths:
        draw.rectangle([(5, y), (canvas_width - 5, y + row_height)], fill=(255, 200, 200))
        draw.text((10, y + 3), f"REFINEMENT (ns={ns})", fill=(128, 0, 0), font=font)
        y += row_height
        
        ref_metrics = ref_metrics_by_ns[ns]
        for metric in metrics_to_show:
            draw.text((10, y + 3), metric.replace('_', ' ')[:15], fill=(50, 50, 50), font=small_font)
            for pov_idx, m in enumerate(ref_metrics):
                x = metric_col_width + pov_idx * col_width
                val = m["summary"].get(metric, 0)
                
                # Color code: green if better than accumulation, red if worse
                acc_val = acc_metrics[pov_idx]["summary"].get(metric, 0)
                if val > acc_val + 0.01:
                    color = (0, 128, 0)
                elif val < acc_val - 0.01:
                    color = (200, 0, 0)
                else:
                    color = (0, 0, 0)
                
                draw.text((x, y + 3), f"{val:.3f}", fill=color, font=small_font)
            y += row_height
    
    canvas.save(output_path)


def run_full_experiment(
    model: DiffusionModel,
    dataset,
    evaluator: FloorplanEvaluator,
    device: str,
    experiment_name: str,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    max_povs: int = 5,
    noise_strengths: List[float] = [0.3, 0.5],
    max_samples: int = None,
    save_images: bool = False,
    output_dir: Path = None,
    seed: int = None
) -> dict:
    """Run complete accumulation vs refinement experiment."""
    
    all_accumulation_results = []
    all_refinement_results = {ns: [] for ns in noise_strengths}
    
    # Store for finding best/median
    all_final_acc_scores = []
    all_final_ref_scores = {ns: [] for ns in noise_strengths}
    sample_data = []  # Store for visualization
    
    n_total = len(dataset)
    n_samples = n_total if max_samples is None else min(max_samples, n_total)
    
    # Random sampling with seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if max_samples is not None and max_samples < n_total:
        sample_indices = random.sample(range(n_total), n_samples)
        sample_indices.sort()
    else:
        sample_indices = list(range(n_samples))
    
    if save_images and output_dir:
        images_dir = output_dir / "images" / experiment_name
        images_dir.mkdir(parents=True, exist_ok=True)
        
        conditions_dir = images_dir / "conditions"
        conditions_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_dir = None
        conditions_dir = None
    
    print(f"\nRunning refinement experiment on {n_samples} samples (from {n_total} total)...")
    if seed is not None:
        print(f"  Random seed: {seed}")
    
    for eval_idx, dataset_idx in enumerate(tqdm(sample_indices, desc="Processing")):
        sample = dataset[dataset_idx]
        
        # Extract condition info
        condition_info = {
            "eval_idx": eval_idx,
            "dataset_idx": dataset_idx
        }
        
        # Get POV paths if available
        all_pov_paths = sample.get("all_pov_paths", [])
        if all_pov_paths:
            condition_info["pov_paths"] = [str(p) for p in all_pov_paths]
            
            # Try to find POV images
            pov_image_paths = []
            for pov_path in all_pov_paths:
                pov_path = Path(pov_path)
                possible_images = [
                    pov_path.with_suffix(".png"),
                    pov_path.with_suffix(".jpg"),
                    pov_path.parent.parent / "povs" / (pov_path.stem + ".png"),
                ]
                for img_path in possible_images:
                    if img_path.exists():
                        pov_image_paths.append(str(img_path))
                        break
            if pov_image_paths:
                condition_info["pov_image_paths"] = pov_image_paths
        
        # Get text/graph info if available
        text_path = sample.get("text_emb_path")
        if text_path:
            condition_info["text_embedding_path"] = str(text_path)
            text_path = Path(text_path)
            possible_text = [
                text_path.with_suffix(".txt"),
                text_path.with_suffix(".json"),
            ]
            for txt_path in possible_text:
                if txt_path.exists():
                    condition_info["text_description_path"] = str(txt_path)
                    try:
                        if txt_path.suffix == ".json":
                            with open(txt_path, "r") as f:
                                condition_info["text_description"] = json.load(f)
                        else:
                            with open(txt_path, "r") as f:
                                condition_info["text_description"] = f.read()
                    except:
                        pass
                    break
        
        # Generate baseline once (0 POV, text-only) - shared by both methods
        text_emb = sample.get("text_emb")
        target_latent = sample.get("latent")
        target_rgb = decode_latent(model, target_latent, device)
        
        if text_emb is not None:
            text_emb_batch = text_emb.unsqueeze(0).to(device) if text_emb.dim() == 1 else text_emb.to(device)
        else:
            text_emb_batch = None
        
        with torch.no_grad():
            baseline_output = model.sample(
                batch_size=1,
                num_steps=num_steps,
                method="ddim",
                guidance_scale=guidance_scale,
                text_emb=text_emb_batch,
                pov_emb=None,
                verbose=False
            )
        
        baseline_rgb = tensor_to_numpy_rgb(baseline_output["rgb"][0])
        baseline_metrics = evaluator.evaluate(baseline_rgb, target_rgb)
        baseline = {
            "latent": baseline_output["latent"].clone(),
            "rgb": baseline_rgb,
            "metrics": baseline_metrics
        }
        
        # Run accumulation with shared baseline
        acc_results, target_rgb, acc_images, acc_metrics = run_accumulation_experiment(
            model, sample, evaluator, device, guidance_scale, num_steps, max_povs, baseline=baseline
        )
        for r in acc_results:
            r["eval_idx"] = eval_idx
            r["dataset_idx"] = dataset_idx
            r["conditions"] = condition_info
        all_accumulation_results.extend(acc_results)
        
        # Store final accumulation score
        final_acc_score = acc_results[-1]["metrics"]["unified_score"] if acc_results else 0
        all_final_acc_scores.append(final_acc_score)
        
        # Run refinement for each noise strength with shared baseline
        ref_images_by_ns = {}
        ref_metrics_by_ns = {}
        for ns in noise_strengths:
            ref_results, _, ref_images, ref_metrics = run_refinement_experiment(
                model, sample, evaluator, device, guidance_scale, num_steps, max_povs, ns, baseline=baseline
            )
            for r in ref_results:
                r["eval_idx"] = eval_idx
                r["dataset_idx"] = dataset_idx
                r["conditions"] = condition_info
            all_refinement_results[ns].extend(ref_results)
            ref_images_by_ns[ns] = ref_images
            ref_metrics_by_ns[ns] = ref_metrics
            
            final_ref_score = ref_results[-1]["metrics"]["unified_score"] if ref_results else 0
            all_final_ref_scores[ns].append(final_ref_score)
        
        # Store for visualization
        sample_data.append({
            "eval_idx": eval_idx,
            "dataset_idx": dataset_idx,
            "target_rgb": target_rgb,
            "acc_images": acc_images,
            "acc_metrics": acc_metrics,
            "ref_images_by_ns": ref_images_by_ns,
            "ref_metrics_by_ns": ref_metrics_by_ns,
            "final_acc_score": final_acc_score,
            "conditions": condition_info
        })
        
        # Save progression images and conditions
        if images_dir:
            sample_dir = images_dir / f"sample_{eval_idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(target_rgb).save(sample_dir / "target.png")
            
            for pov_idx, img in enumerate(acc_images):
                Image.fromarray(img).save(sample_dir / f"accum_pov{pov_idx}.png")
            
            for ns, ref_imgs in ref_images_by_ns.items():
                ns_str = str(ns).replace(".", "p")
                for pov_idx, img in enumerate(ref_imgs):
                    Image.fromarray(img).save(sample_dir / f"refine_ns{ns_str}_pov{pov_idx}.png")
            
            # Save comparison image [Target, Baseline, Accumulation, Refinement]
            # Baseline is the 0-POV generation (same for both methods)
            baseline_image = acc_images[0]
            baseline_score = acc_metrics[0]["summary"]["unified_score"] if acc_metrics else 0
            
            for ns in noise_strengths:
                ns_str = str(ns).replace(".", "p")
                acc_final_score = acc_metrics[-1]["summary"]["unified_score"] if acc_metrics else 0
                ref_final_score = ref_metrics_by_ns[ns][-1]["summary"]["unified_score"] if ref_metrics_by_ns[ns] else 0
                create_comparison_visualization(
                    target_rgb,
                    baseline_image,
                    acc_images[-1],
                    ref_images_by_ns[ns][-1],
                    baseline_score,
                    acc_final_score,
                    ref_final_score,
                    ns,
                    sample_dir / f"comparison_ns{ns_str}.png"
                )
            
            # Save metrics progression plot and table
            create_metrics_progression_plot(
                acc_metrics,
                ref_metrics_by_ns,
                sample_dir / "metrics_progression.png"
            )
            create_metrics_table_image(
                acc_metrics,
                ref_metrics_by_ns,
                sample_dir / "metrics_table.png"
            )
            
            # Save conditions for this sample
            if conditions_dir:
                sample_cond_dir = conditions_dir / f"sample_{eval_idx:04d}"
                sample_cond_dir.mkdir(parents=True, exist_ok=True)
                
                # Save POV images
                pov_image_paths = condition_info.get("pov_image_paths", [])
                for pov_idx, pov_img_path in enumerate(pov_image_paths):
                    if Path(pov_img_path).exists():
                        shutil.copy(pov_img_path, sample_cond_dir / f"pov_{pov_idx}.png")
                
                # Save text description
                text_desc = condition_info.get("text_description")
                if text_desc:
                    with open(sample_cond_dir / "text_description.txt", "w") as f:
                        if isinstance(text_desc, dict):
                            f.write(json.dumps(text_desc, indent=2))
                        else:
                            f.write(str(text_desc))
                
                # Save condition metadata
                with open(sample_cond_dir / "condition_info.json", "w") as f:
                    json.dump(condition_info, f, indent=2, default=str)
        # ✅ MEMORY CLEANUP: Explicitly delete large objects after each sample
        del baseline, baseline_output, baseline_rgb, baseline_metrics
        del target_rgb, acc_results, acc_images, acc_metrics
        del ref_images_by_ns, ref_metrics_by_ns, condition_info, sample, text_emb_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create visualizations for best and median samples
    if output_dir:
        viz_dir = output_dir / "visualizations" / experiment_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        sorted_indices = np.argsort(all_final_acc_scores)
        best_idx = sorted_indices[-1]
        median_idx = sorted_indices[len(sorted_indices) // 2]
        
        for label, sample_idx in [("best", best_idx), ("median", median_idx)]:
            data = sample_data[sample_idx]
            
            # Accumulation progression
            acc_scores = [m["summary"]["unified_score"] for m in data["acc_metrics"]]
            create_progression_visualization(
                data["acc_images"],
                data["target_rgb"],
                "Accum",
                acc_scores,
                viz_dir / f"{label}_accumulation_progression.png"
            )
            
            # Refinement progression (first noise strength)
            ns = noise_strengths[0]
            ref_scores = [m["summary"]["unified_score"] for m in data["ref_metrics_by_ns"][ns]]
            create_progression_visualization(
                data["ref_images_by_ns"][ns],
                data["target_rgb"],
                f"Refine(ns={ns})",
                ref_scores,
                viz_dir / f"{label}_refinement_progression.png"
            )
            
            # Comparison visualization: [Target, Baseline, Accumulation, Refinement]
            baseline_image = data["acc_images"][0]  # 0-POV generation
            baseline_score = acc_scores[0]
            create_comparison_visualization(
                data["target_rgb"],
                baseline_image,
                data["acc_images"][-1],  # Final accumulation
                data["ref_images_by_ns"][ns][-1],  # Final refinement
                baseline_score,
                acc_scores[-1],
                ref_scores[-1],
                ns,
                viz_dir / f"{label}_comparison.png"
            )
            
            # Metrics progression plot and table
            create_metrics_progression_plot(
                data["acc_metrics"],
                data["ref_metrics_by_ns"],
                viz_dir / f"{label}_metrics_progression.png"
            )
            create_metrics_table_image(
                data["acc_metrics"],
                data["ref_metrics_by_ns"],
                viz_dir / f"{label}_metrics_table.png"
            )
            
            # Detailed visualization for final predictions
            final_acc_metrics = data["acc_metrics"][-1]
            detailed, summary = evaluator.create_visualization(
                data["acc_images"][-1],
                data["target_rgb"],
                final_acc_metrics,
                f"{label.capitalize()} Accumulation (score={acc_scores[-1]:.3f})"
            )
            summary.save(viz_dir / f"{label}_accumulation_summary.png")
            
            # Save conditions for viz samples
            conditions = data.get("conditions", {})
            pov_image_paths = conditions.get("pov_image_paths", [])
            for pov_idx, pov_img_path in enumerate(pov_image_paths[:3]):  # Save first 3 POVs
                if Path(pov_img_path).exists():
                    shutil.copy(pov_img_path, viz_dir / f"{label}_condition_pov{pov_idx}.png")
            
            text_desc = conditions.get("text_description")
            if text_desc:
                with open(viz_dir / f"{label}_condition_text.txt", "w") as f:
                    if isinstance(text_desc, dict):
                        f.write(json.dumps(text_desc, indent=2))
                    else:
                        f.write(str(text_desc))
        
        print(f"\n  Visualizations saved to: {viz_dir}")
    # ✅ MEMORY CLEANUP: Delete sample_data (no longer needed after visualization)
    del sample_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Aggregate results
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
        "refinement": {},
        "sample_indices": sample_indices,
        "seed": seed
    }
    
    for ns, ref_results in all_refinement_results.items():
        results["refinement"][f"ns_{ns}"] = {
            "per_sample": ref_results,
            "by_n_povs": aggregate_by_povs(ref_results)
        }
    
    # Create aggregated metrics plot (average across all samples)
    if output_dir:
        viz_dir = output_dir / "visualizations" / experiment_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        create_aggregated_metrics_plot(results, viz_dir / "aggregated_metrics.png")
    
    return results


def create_aggregated_metrics_plot(results: dict, output_path: Path):
    """Create a plot showing aggregated metrics (mean ± std) across all samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    acc_by_povs = results["accumulation"]["by_n_povs"]
    n_povs_list = sorted(acc_by_povs.keys())
    
    metrics_to_plot = ["unified_score", "mean_bbox_iou", "detection_f1", "count_accuracy"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric in zip(axes, metrics_to_plot):
        # Accumulation
        acc_means = [acc_by_povs[n][f"{metric}_mean"] for n in n_povs_list]
        acc_stds = [acc_by_povs[n][f"{metric}_std"] for n in n_povs_list]
        
        ax.errorbar(n_povs_list, acc_means, yerr=acc_stds, fmt='b-o', 
                    label='Accumulation', linewidth=2, markersize=8, capsize=4)
        
        # Refinement for each noise strength
        colors = ['r', 'g', 'm', 'c']
        markers = ['s', '^', 'D', 'v']
        for i, (ns_key, ref_data) in enumerate(results["refinement"].items()):
            ref_by_povs = ref_data["by_n_povs"]
            ref_means = [ref_by_povs[n][f"{metric}_mean"] for n in n_povs_list]
            ref_stds = [ref_by_povs[n][f"{metric}_std"] for n in n_povs_list]
            
            ns = ns_key.replace("ns_", "")
            ax.errorbar(n_povs_list, ref_means, yerr=ref_stds, 
                        fmt=f'{colors[i % len(colors)]}-{markers[i % len(markers)]}',
                        label=f'Refinement (ns={ns})', linewidth=2, markersize=8, capsize=4)
        
        ax.set_xlabel('Number of POVs', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_povs_list)
    
    plt.suptitle('Aggregated Metrics: Accumulation vs Refinement\n(mean ± std across all samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Aggregated metrics plot saved to: {output_path}")


def print_comparison_table(results: dict):
    """Print comparison table of accumulation vs refinement."""
    print("\n" + "=" * 90)
    print("COMPARISON: ACCUMULATION vs REFINEMENT")
    print("=" * 90)
    
    acc_by_povs = results["accumulation"]["by_n_povs"]
    
    ref_key = list(results["refinement"].keys())[0]
    ref_by_povs = results["refinement"][ref_key]["by_n_povs"]
    
    print(f"{'N_POVs':<8} {'Metric':<25} {'Accumulation':<15} {'Refinement':<15} {'Δ':<10}")
    print("-" * 90)
    
    metrics_to_show = [
        "unified_score", "mean_bbox_iou", "mean_camera_similarity",
        "detection_f1", "count_accuracy"
    ]
    
    for n_povs in sorted(acc_by_povs.keys()):
        if n_povs not in ref_by_povs:
            continue
        
        for metric in metrics_to_show:
            acc_val = acc_by_povs[n_povs].get(f"{metric}_mean", 0)
            ref_val = ref_by_povs[n_povs].get(f"{metric}_mean", 0)
            delta = ref_val - acc_val
            
            delta_color = "+" if delta > 0 else ""
            print(f"{n_povs:<8} {metric:<25} {acc_val:<15.4f} {ref_val:<15.4f} {delta_color}{delta:.4f}")
        print()


def save_per_sample_final_metrics_csv(results: dict, output_dir: Path, experiment_name: str, timestamp: str):
    """
    Save per-sample final metrics in CSV format (streaming to avoid OOM).
    Each row represents a sample with its final metrics for each method and noise strength.
    
    Args:
        results: Complete results dictionary from run_full_experiment
        output_dir: Output directory path
        experiment_name: Name of the experiment
        timestamp: Timestamp string for file naming
    """
    import csv
    
    csv_path = output_dir / f"{experiment_name}_per_sample_metrics_{timestamp}.csv"
    
    # First pass: collect all metric column names to write header
    all_metric_keys = set()
    
    for r in results["accumulation"]["per_sample"]:
        for metric_name in r["metrics"].keys():
            all_metric_keys.add(f"accumulation_{metric_name}")
    
    for ns_key, ref_data in results["refinement"].items():
        ns = ns_key.replace("ns_", "")
        for r in ref_data["per_sample"]:
            for metric_name in r["metrics"].keys():
                all_metric_keys.add(f"refinement_ns{ns}_{metric_name}")
    
    all_metric_keys = sorted(list(all_metric_keys))
    
    # Collect all unique eval_idx values
    all_eval_indices = set()
    for r in results["accumulation"]["per_sample"]:
        all_eval_indices.add(r.get("eval_idx"))
    for ns_key, ref_data in results["refinement"].items():
        for r in ref_data["per_sample"]:
            all_eval_indices.add(r.get("eval_idx"))
    all_eval_indices = sorted(list(all_eval_indices))
    
    # Stream write CSV - one row per sample
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["eval_idx", "dataset_idx"] + all_metric_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each sample and write immediately
        for eval_idx in all_eval_indices:
            row = {"eval_idx": eval_idx, "dataset_idx": ""}
            
            # Initialize all metric columns with empty strings
            for key in all_metric_keys:
                row[key] = ""
            
            # Get dataset_idx and final accumulation metrics
            acc_results_for_sample = [r for r in results["accumulation"]["per_sample"] if r.get("eval_idx") == eval_idx]
            if acc_results_for_sample:
                row["dataset_idx"] = acc_results_for_sample[0].get("dataset_idx", "")
                
                # Find the final (max POVs) result for accumulation
                max_povs = max([r.get("n_povs", 0) for r in acc_results_for_sample])
                final_acc = next((r for r in acc_results_for_sample if r.get("n_povs") == max_povs), None)
                
                if final_acc:
                    for metric_name, metric_val in final_acc["metrics"].items():
                        row[f"accumulation_{metric_name}"] = metric_val
            
            # Get final refinement metrics for each noise strength
            for ns_key, ref_data in results["refinement"].items():
                ns = ns_key.replace("ns_", "")
                ref_results_for_sample = [r for r in ref_data["per_sample"] if r.get("eval_idx") == eval_idx]
                
                if ref_results_for_sample:
                    # Set dataset_idx if not already set
                    if not row["dataset_idx"]:
                        row["dataset_idx"] = ref_results_for_sample[0].get("dataset_idx", "")
                    
                    # Find the final (max POVs) result for this refinement
                    max_povs = max([r.get("n_povs", 0) for r in ref_results_for_sample])
                    final_ref = next((r for r in ref_results_for_sample if r.get("n_povs") == max_povs), None)
                    
                    if final_ref:
                        for metric_name, metric_val in final_ref["metrics"].items():
                            row[f"refinement_ns{ns}_{metric_name}"] = metric_val
            
            # Write row immediately
            writer.writerow(row)
    
    print(f"Per-sample metrics CSV saved to: {csv_path}")


def save_results(results: dict, output_dir: Path, experiment_name: str):
    """Save experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    full_path = output_dir / f"{experiment_name}_refinement_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {full_path}")
    
    # Save CSV summary
    import csv
    csv_path = output_dir / f"{experiment_name}_refinement_summary_{timestamp}.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        header = [
            "method", "noise_strength", "n_povs",
            "unified_score_mean", "unified_score_std",
            "mean_bbox_iou_mean", "mean_bbox_iou_std",
            "mean_camera_similarity_mean", "mean_camera_similarity_std",
            "detection_f1_mean", "detection_f1_std",
            "count_accuracy_mean", "count_accuracy_std",
            "n_samples"
        ]
        writer.writerow(header)
        
        # Accumulation rows
        for n_povs, agg in results["accumulation"]["by_n_povs"].items():
            row = [
                "accumulation", "N/A", n_povs,
                agg.get("unified_score_mean", ""),
                agg.get("unified_score_std", ""),
                agg.get("mean_bbox_iou_mean", ""),
                agg.get("mean_bbox_iou_std", ""),
                agg.get("mean_camera_similarity_mean", ""),
                agg.get("mean_camera_similarity_std", ""),
                agg.get("detection_f1_mean", ""),
                agg.get("detection_f1_std", ""),
                agg.get("count_accuracy_mean", ""),
                agg.get("count_accuracy_std", ""),
                agg.get("n_samples", "")
            ]
            writer.writerow(row)
        
        # Refinement rows
        for ns_key, ref_data in results["refinement"].items():
            ns = ns_key.replace("ns_", "")
            for n_povs, agg in ref_data["by_n_povs"].items():
                row = [
                    "refinement", ns, n_povs,
                    agg.get("unified_score_mean", ""),
                    agg.get("unified_score_std", ""),
                    agg.get("mean_bbox_iou_mean", ""),
                    agg.get("mean_bbox_iou_std", ""),
                    agg.get("mean_camera_similarity_mean", ""),
                    agg.get("mean_camera_similarity_std", ""),
                    agg.get("detection_f1_mean", ""),
                    agg.get("detection_f1_std", ""),
                    agg.get("count_accuracy_mean", ""),
                    agg.get("count_accuracy_std", ""),
                    agg.get("n_samples", "")
                ]
                writer.writerow(row)
    
    print(f"Summary CSV saved to: {csv_path}")
    
    # Save per-sample per-step CSV (detailed metrics)
    detailed_csv_path = output_dir / f"{experiment_name}_refinement_detailed_{timestamp}.csv"
    
    with open(detailed_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        header = [
            "sample_idx", "dataset_idx", "method", "noise_strength", "n_povs",
            "unified_score", "mean_bbox_iou", "mean_camera_similarity",
            "detection_f1", "count_accuracy"
        ]
        writer.writerow(header)
        
        # Accumulation per-sample results
        for r in results["accumulation"]["per_sample"]:
            row = [
                r.get("eval_idx", ""),
                r.get("dataset_idx", ""),
                "accumulation",
                "N/A",
                r.get("n_povs", ""),
                r["metrics"].get("unified_score", ""),
                r["metrics"].get("mean_bbox_iou", ""),
                r["metrics"].get("mean_camera_similarity", ""),
                r["metrics"].get("detection_f1", ""),
                r["metrics"].get("count_accuracy", "")
            ]
            writer.writerow(row)
        
        # Refinement per-sample results
        for ns_key, ref_data in results["refinement"].items():
            ns = ns_key.replace("ns_", "")
            for r in ref_data["per_sample"]:
                row = [
                    r.get("eval_idx", ""),
                    r.get("dataset_idx", ""),
                    "refinement",
                    ns,
                    r.get("n_povs", ""),
                    r["metrics"].get("unified_score", ""),
                    r["metrics"].get("mean_bbox_iou", ""),
                    r["metrics"].get("mean_camera_similarity", ""),
                    r["metrics"].get("detection_f1", ""),
                    r["metrics"].get("count_accuracy", "")
                ]
                writer.writerow(row)
    
    print(f"Detailed CSV saved to: {detailed_csv_path}")
    
    # Save per-sample final metrics CSV (one row per sample)
    save_per_sample_final_metrics_csv(results, output_dir, experiment_name, timestamp)


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
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (randomly sampled from dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--group-by", type=str, default="room_id")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--pov-columns", type=str, nargs="+", default=None,
                        help="List of column names for POV embeddings (for column-based manifest format)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("REFINEMENT EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Manifest: {args.manifest}")
    print(f"  Taxonomy: {args.taxonomy}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Seed: {args.seed}")
    print(f"  Max POVs: {args.max_povs}")
    print(f"  Noise strengths: {args.noise_strengths}")
    print("=" * 70)
    
    model = load_model(args.checkpoint, args.device)
    
    print(f"\nLoading multi-POV dataset from {args.manifest}...")
    
    # Define POV columns for column-based manifest format
    pov_columns = None
    if args.pov_columns:
        pov_columns = args.pov_columns
    
    dataset = MultiPOVDataset(
        manifest_path=args.manifest,
        outputs={
            "latent": "latent_embedding_path",
            "text_emb": "graph_embedding_path",
            "pov_emb": "pov_embedding_path"
        },
        group_by=args.group_by,
        filters={"rejected": False} if not args.pov_columns else None,
        pov_columns=pov_columns
    )
    print(f"  Dataset size: {len(dataset)} rooms/scenes")
    
    evaluator = FloorplanEvaluator(args.taxonomy)
    
    experiment_name = args.checkpoint.stem
    if experiment_name in ["best_checkpoint", "checkpoint"]:
        experiment_name = args.checkpoint.parent.parent.name
    
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
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    results["metadata"] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "max_povs": args.max_povs,
        "noise_strengths": args.noise_strengths,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat()
    }
    
    print_comparison_table(results)
    save_results(results, args.output_dir, experiment_name)


if __name__ == "__main__":
    main()