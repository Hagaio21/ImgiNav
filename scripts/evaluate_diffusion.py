#!/usr/bin/env python3
"""
Evaluation script for diffusion models.
Generates a population of images (unconditional, conditioned targets vs generated)
using DDPM sampling from the best checkpoint.
"""

import argparse
import sys
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import (
    load_config,
    get_device,
    build_dataset,
    split_dataset,
    to_device,
    move_batch_to_device,
)
from models.diffusion import DiffusionModel


def create_image_grid(images, nrow=4, padding=2):
    """Create a grid of images."""
    if len(images) == 0:
        return None
    
    img_size = images[0].size[0]
    num_images = len(images)
    num_rows = (num_images + nrow - 1) // nrow
    
    grid_width = nrow * img_size + (nrow + 1) * padding
    grid_height = num_rows * img_size + (num_rows + 1) * padding
    
    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        x = col * img_size + (col + 1) * padding
        y = row * img_size + (row + 1) * padding
        grid.paste(img, (x, y))
    
    return grid


def save_comparison_grid(target_images, generated_images, output_path, nrow=4):
    """Create side-by-side comparison grid: target (left) | generated (right)."""
    if len(target_images) != len(generated_images):
        raise ValueError(f"Mismatch: {len(target_images)} targets vs {len(generated_images)} generated")
    
    img_size = target_images[0].size[0]
    num_images = len(target_images)
    num_rows = (num_images + nrow - 1) // nrow
    padding = 2
    
    # Create target grid
    target_grid = create_image_grid(target_images, nrow=nrow, padding=padding)
    
    # Create generated grid
    generated_grid = create_image_grid(generated_images, nrow=nrow, padding=padding)
    
    # Concatenate horizontally
    comparison_width = target_grid.width * 2
    comparison_height = target_grid.height
    comparison = Image.new('RGB', (comparison_width, comparison_height), color=(255, 255, 255))
    comparison.paste(target_grid, (0, 0))
    comparison.paste(generated_grid, (target_grid.width, 0))
    
    comparison.save(output_path)
    print(f"[SAVED] Comparison grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion model and generate population of images")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate (default: 64)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation (default: 16)")
    parser.add_argument("--unconditional_samples", type=int, default=32, help="Number of unconditional samples (default: 32)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: experiment_dir/outputs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Load config
    print(f"[CONFIG] Loading config from {args.config}")
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    print(f"[CONFIG] Experiment: {exp_name}")
    
    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_output_dir = config.get("experiment", {}).get("save_path")
        if exp_output_dir:
            output_dir = Path(exp_output_dir) / "outputs"
        else:
            output_dir = Path("outputs") / exp_name / "outputs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Output directory: {output_dir}")
    
    # Find best checkpoint
    exp_output_dir = config.get("experiment", {}).get("save_path")
    if exp_output_dir:
        checkpoint_dir = Path(exp_output_dir) / "checkpoints"
    else:
        checkpoint_dir = Path("outputs") / exp_name / "checkpoints"
    
    best_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_best.pt"
    if not best_checkpoint.exists():
        print(f"[ERROR] Best checkpoint not found: {best_checkpoint}")
        print(f"[INFO] Looking for latest checkpoint instead...")
        latest_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
        if latest_checkpoint.exists():
            best_checkpoint = latest_checkpoint
            print(f"[INFO] Using latest checkpoint: {best_checkpoint}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    print(f"[CHECKPOINT] Loading best checkpoint: {best_checkpoint}")
    
    # Get device
    device = get_device(config)
    device_obj = to_device(device)
    print(f"[DEVICE] Using device: {device}")
    
    # Load model from checkpoint
    print("[MODEL] Loading model from checkpoint...")
    model, extra_state = DiffusionModel.load_checkpoint(
        best_checkpoint,
        map_location=device,
        return_extra=True,
        config=config
    )
    model = model.to(device_obj)
    model.eval()
    
    val_loss = extra_state.get("best_val_loss", float("inf"))
    epoch = extra_state.get("epoch", "unknown")
    print(f"[MODEL] Model loaded - Epoch: {epoch}, Best Val Loss: {val_loss:.6f}")
    
    # Build dataset
    print("[DATASET] Building dataset...")
    dataset = build_dataset(config)
    train_dataset, val_dataset = split_dataset(dataset, config["training"])
    print(f"[DATASET] Validation dataset size: {len(val_dataset)}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Get guidance scale from config
    guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
    print(f"[SAMPLING] Using guidance scale: {guidance_scale}")
    
    # ============================================================================
    # Part 1: Generate unconditional samples
    # ============================================================================
    num_steps = model.scheduler.num_steps
    print(f"\n[UNCONDITIONAL] Generating {args.unconditional_samples} unconditional samples using DDPM ({num_steps} steps)...")
    unconditional_dir = output_dir / "unconditional"
    unconditional_dir.mkdir(parents=True, exist_ok=True)
    
    num_batches = (args.unconditional_samples + args.batch_size - 1) // args.batch_size
    all_unconditional_images = []
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Unconditional sampling"):
            batch_size = min(args.batch_size, args.unconditional_samples - batch_idx * args.batch_size)
            
            unconditional_output = model.sample(
                batch_size=batch_size,
                num_steps=num_steps,
                method="ddpm",
                eta=1.0,
                cond=None,
                guidance_scale=1.0,
                text_emb=None,
                pov_emb=None,
                device=device_obj,
                verbose=False
            )
            
            # Decode to RGB
            if "rgb" in unconditional_output:
                unconditional_rgb = unconditional_output["rgb"]
                if unconditional_rgb.min() < 0:  # [-1, 1] range
                    unconditional_rgb = (unconditional_rgb + 1.0) / 2.0
                unconditional_rgb = torch.clamp(unconditional_rgb, 0.0, 1.0)
            else:
                decoded = model.decoder({"latent": unconditional_output["latent"]})
                if "rgb" in decoded:
                    unconditional_rgb = (decoded["rgb"] + 1.0) / 2.0
                    unconditional_rgb = torch.clamp(unconditional_rgb, 0.0, 1.0)
                else:
                    print(f"[WARNING] Batch {batch_idx}: Decoder did not produce RGB output")
                    continue
            
            # Convert to PIL images
            unconditional_np = (unconditional_rgb.cpu().numpy() * 255.0).astype(np.uint8)
            for i in range(batch_size):
                img = Image.fromarray(unconditional_np[i].transpose(1, 2, 0))
                all_unconditional_images.append(img)
                
                # Save individual image
                img.save(unconditional_dir / f"unconditional_{batch_idx * args.batch_size + i:04d}.png")
    
    # Save grid
    if all_unconditional_images:
        grid = create_image_grid(all_unconditional_images, nrow=8)
        if grid:
            grid_path = output_dir / "unconditional_grid.png"
            grid.save(grid_path)
            print(f"[SAVED] Unconditional grid: {grid_path} ({len(all_unconditional_images)} images)")
    
    # ============================================================================
    # Part 2: Generate conditioned samples (targets vs generated)
    # ============================================================================
    print(f"\n[CONDITIONED] Generating {args.num_samples} conditioned samples using DDPM ({num_steps} steps)...")
    conditioned_dir = output_dir / "conditioned"
    conditioned_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample from validation dataset
    num_samples = min(args.num_samples, len(val_dataset))
    sampled_indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    all_target_images = []
    all_generated_images = []
    all_conditions = []
    
    # Process in batches
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Conditioned sampling"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, num_samples)
            batch_indices = sampled_indices[start_idx:end_idx]
            current_batch_size = len(batch_indices)
            
            # Load batch from dataset
            batch_data = {}
            for idx in batch_indices:
                sample = val_dataset[idx]
                for key, value in sample.items():
                    if key not in batch_data:
                        batch_data[key] = []
                    batch_data[key].append(value)
            
            # Convert to tensors
            batch = {}
            for key, values in batch_data.items():
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values
            
            batch = move_batch_to_device(batch, device_obj, non_blocking=False)
            
            # Get latents and conditioning
            target_latents = batch.get("latent", None)
            text_emb = batch.get("text_emb", None)
            pov_emb = batch.get("pov_emb", None)
            
            if target_latents is None:
                print(f"[WARNING] Batch {batch_idx}: Missing latents, skipping")
                continue
            
            # Decode target latents
            target_decoded = model.decoder({"latent": target_latents})
            if "rgb" in target_decoded:
                target_rgb = (target_decoded["rgb"] + 1.0) / 2.0
                target_rgb = torch.clamp(target_rgb, 0.0, 1.0)
            else:
                print(f"[WARNING] Batch {batch_idx}: Target decoder did not produce RGB")
                continue
            
            # Prepare conditioning
            cond = None
            if text_emb is not None:
                if text_emb.dim() > 1:
                    text_emb = text_emb.flatten(start_dim=1)
            if pov_emb is not None:
                if pov_emb.dim() > 1:
                    pov_emb = pov_emb.flatten(start_dim=1)
            
            # Generate conditioned samples using DDPM
            num_steps = model.scheduler.num_steps
            conditioned_output = model.sample(
                batch_size=current_batch_size,
                num_steps=num_steps,
                method="ddpm",
                eta=1.0,
                cond=cond,
                guidance_scale=guidance_scale,
                text_emb=text_emb,
                pov_emb=pov_emb,
                device=device_obj,
                verbose=False
            )
            
            # Decode generated latents
            if "rgb" in conditioned_output:
                generated_rgb = conditioned_output["rgb"]
                if generated_rgb.min() < 0:
                    generated_rgb = (generated_rgb + 1.0) / 2.0
                generated_rgb = torch.clamp(generated_rgb, 0.0, 1.0)
            else:
                decoded = model.decoder({"latent": conditioned_output["latent"]})
                if "rgb" in decoded:
                    generated_rgb = (decoded["rgb"] + 1.0) / 2.0
                    generated_rgb = torch.clamp(generated_rgb, 0.0, 1.0)
                else:
                    print(f"[WARNING] Batch {batch_idx}: Generated decoder did not produce RGB")
                    continue
            
            # Convert to PIL images and save
            target_np = (target_rgb.cpu().numpy() * 255.0).astype(np.uint8)
            generated_np = (generated_rgb.cpu().numpy() * 255.0).astype(np.uint8)
            
            for i in range(current_batch_size):
                target_img = Image.fromarray(target_np[i].transpose(1, 2, 0))
                generated_img = Image.fromarray(generated_np[i].transpose(1, 2, 0))
                
                all_target_images.append(target_img)
                all_generated_images.append(generated_img)
                
                # Save individual images
                sample_idx = start_idx + i
                target_img.save(conditioned_dir / f"target_{sample_idx:04d}.png")
                generated_img.save(conditioned_dir / f"generated_{sample_idx:04d}.png")
                
                # Save conditioning info
                condition_info = {
                    "sample_idx": int(sample_idx),
                    "dataset_idx": int(batch_indices[i]),
                    "has_text_emb": text_emb is not None,
                    "has_pov_emb": pov_emb is not None,
                    "text_emb_shape": list(text_emb[i].shape) if text_emb is not None else None,
                    "pov_emb_shape": list(pov_emb[i].shape) if pov_emb is not None else None,
                }
                all_conditions.append(condition_info)
                
                # Save condition info as JSON
                with open(conditioned_dir / f"condition_{sample_idx:04d}.json", 'w') as f:
                    json.dump(condition_info, f, indent=2)
    
    # Save comparison grid
    if all_target_images and all_generated_images:
        comparison_path = output_dir / "conditioned_comparison.png"
        save_comparison_grid(all_target_images, all_generated_images, comparison_path, nrow=8)
        
        # Save separate grids
        target_grid = create_image_grid(all_target_images, nrow=8)
        if target_grid:
            target_grid.save(output_dir / "conditioned_targets.png")
        
        generated_grid = create_image_grid(all_generated_images, nrow=8)
        if generated_grid:
            generated_grid.save(output_dir / "conditioned_generated.png")
        
        # Save conditions summary
        conditions_summary = {
            "total_samples": len(all_conditions),
            "guidance_scale": guidance_scale,
            "sampling_method": "ddpm",
            "num_steps": model.scheduler.num_steps,
            "conditions": all_conditions
        }
        with open(output_dir / "conditions_summary.json", 'w') as f:
            json.dump(conditions_summary, f, indent=2)
        
        print(f"[SAVED] Conditioned comparison: {comparison_path} ({len(all_target_images)} samples)")
    
    print(f"\n[COMPLETE] Evaluation complete!")
    print(f"[OUTPUT] All outputs saved to: {output_dir}")
    print(f"  - Unconditional samples: {unconditional_dir}")
    print(f"  - Conditioned samples: {conditioned_dir}")
    print(f"  - Grids and summaries: {output_dir}")


if __name__ == "__main__":
    main()

