#!/usr/bin/env python3
"""
Unified training script for diffusion models (Stage 1, Stage 2, Stage 3).
Uses CompositeLoss from config to combine noise and semantic losses.
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import math
import numpy as np
from PIL import Image

from training.utils import (
    load_config,
    set_deterministic,
    get_device,
    build_optimizer,
    build_loss,
    build_dataset,
    build_scheduler,
    to_device,
    move_batch_to_device,
    split_dataset,
    create_grad_scaler,
    save_metrics_csv,
)
from training.plotting_utils import plot_diffusion_metrics_epochs
from models.diffusion import DiffusionModel
from models.losses.base_loss import LOSS_REGISTRY


def compute_loss(
    model, batch, latents, t, noise, cond, loss_fn, 
    use_amp=False, device_obj=None, needs_decoding=False
):
    """
    Compute loss using CompositeLoss from config.
    
    This function prepares preds and targets dictionaries that are passed to
    CompositeLoss, which then distributes them to each sub-loss component.
    
    Each loss component expects specific keys:
    - SNRWeightedNoiseLoss: 
        preds["pred_noise"], preds["scheduler"], preds["timesteps"]
        targets["noise"]
    - LatentStructuralLoss:
        preds["pred_noise"], preds["scheduler"], preds["timesteps"], preds["noisy_latent"]
        targets["latent"]
    - SemanticLoss:
        preds["decoded_rgb"]
        targets["rgb"]
    
    Args:
        model: Diffusion model
        batch: Batch dictionary
        latents: Latent tensors [B, C, H, W]
        t: Timesteps [B]
        noise: Noise tensor [B, C, H, W]
        cond: Conditioning (optional)
        loss_fn: CompositeLoss built from config
        use_amp: Whether to use mixed precision
        device_obj: Device object
    
    Returns:
        (total_loss, logs_dict)
    """
    # Forward pass through model
    outputs = model(latents, t, cond=cond, noise=noise)
    
    # Prepare preds dict for loss computation
    # All loss components will receive this dict, but only use the keys they need
    preds = {
        "pred_noise": outputs["pred_noise"],      # For MSELoss
        "scheduler": model.scheduler,            # For LatentStructuralLoss (if used)
        "timesteps": t,                          # For LatentStructuralLoss (if used)
        "noisy_latent": outputs.get("noisy_latent"),  # For LatentStructuralLoss (if used)
    }
    
    # Decode latents if semantic losses are needed (needs_decoding is cached from train_epoch)
    if needs_decoding and "rgb" in batch:
        decoded = model.decoder({"latent": latents})
        preds["decoded_rgb"] = decoded.get("rgb")              # For SemanticLoss (perceptual)
    
    # Prepare targets dict
    # All loss components will receive this dict, but only use the keys they need
    targets = {
        "noise": noise,      # For MSELoss
        "latent": latents,   # For LatentStructuralLoss (ground-truth clean latents, if used)
    }
    
    # Add RGB if available (for SemanticLoss)
    if "rgb" in batch:
        targets["rgb"] = batch["rgb"]  # For SemanticLoss (perceptual)
    
    # Compute loss using CompositeLoss
    if use_amp and device_obj.type == "cuda":
        with torch.amp.autocast('cuda'):
            total_loss, logs = loss_fn(preds, targets)
    else:
        total_loss, logs = loss_fn(preds, targets)
    
    return total_loss, logs


def train_epoch(
    model, dataloader, scheduler, loss_fn, 
    optimizer, device, epoch, use_amp=False, max_grad_norm=None, use_non_uniform_sampling=False, cfg_dropout_rate=0.0
):
    """Train for one epoch using CompositeLoss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    device_obj = to_device(device)
    
    # Cache loss class lookups and check if decoding is needed (once per epoch, not per batch)
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    SemanticLossClass = LOSS_REGISTRY.get("SemanticLoss")
    needs_decoding = False
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        for sub_loss in loss_fn.losses:
            if SemanticLossClass and isinstance(sub_loss, SemanticLossClass):
                needs_decoding = True
                break
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device_obj)
        
        # Get latents
        latents = batch.get("latent")
        if latents is None:
            if "rgb" in batch and model._has_encoder:
                with torch.no_grad():
                    encoder_out = model.encoder(batch["rgb"])
                    if "latent" in encoder_out:
                        latents = encoder_out["latent"]
                    elif "mu" in encoder_out:
                        latents = encoder_out["mu"]
                    else:
                        raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
            else:
                raise ValueError("Dataset must provide 'latent' key (for pre-embedded) or 'rgb' key (for on-the-fly encoding)")
        
        # Sample random timesteps
        num_steps = model.scheduler.num_steps
        # Support non-uniform timestep sampling (favors high-noise timesteps for better generalization)
        # This helps with low-diversity datasets by focusing on harder denoising tasks
        if use_non_uniform_sampling:
            # Higher probability for early timesteps (high noise)
            # Exponential decay: early timesteps have higher probability
            probs = torch.exp(-torch.linspace(0, 2, num_steps, device=device_obj))
            probs = probs / probs.sum()
            t = torch.multinomial(probs, latents.shape[0], replacement=True)
        else:
            # Uniform sampling (default)
            t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
        noise = model.scheduler.randn_like(latents)
        
        # Extract and convert type labels (room/scene) to condition indices
        # type column contains "room" or "scene" strings, convert to 0=ROOM, 1=SCENE
        cond = None
        type_labels = batch.get("type", None)
        if type_labels is not None:
            # Handle both string and tensor types
            if isinstance(type_labels, (list, tuple)) and len(type_labels) > 0:
                if isinstance(type_labels[0], str):
                    # Convert string labels to indices: "room" -> 0, "scene" -> 1
                    cond = torch.tensor(
                        [0 if t.lower().strip() == "room" else 1 for t in type_labels],
                        device=device_obj, dtype=torch.long
                    )
                else:
                    # Already tensor indices
                    cond = type_labels.to(device_obj) if isinstance(type_labels, torch.Tensor) else torch.tensor(type_labels, device=device_obj, dtype=torch.long)
            elif isinstance(type_labels, torch.Tensor):
                # Tensor might contain strings or indices
                if type_labels.dtype == torch.long or type_labels.dtype == torch.int:
                    cond = type_labels.to(device_obj)
                else:
                    # Convert string tensor to indices
                    cond = torch.tensor(
                        [0 if str(t).lower().strip() == "room" else 1 for t in type_labels.cpu().tolist()],
                        device=device_obj, dtype=torch.long
                    )
        
        # Apply CFG condition dropout (randomly drop condition with cfg_dropout_rate probability)
        # This teaches the model to work both conditionally and unconditionally
        # Use per-batch dropout: randomly set entire batch to None with probability cfg_dropout_rate
        if cfg_dropout_rate > 0.0 and cond is not None:
            if torch.rand(1, device=device_obj).item() < cfg_dropout_rate:
                cond = None  # Drop entire batch condition for CFG training
        
        # Compute loss
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    use_amp, device_obj, needs_decoding
                )
            
            optimizer.zero_grad()
            scaler = getattr(train_epoch, '_scaler', None)
            if scaler is None:
                scaler = create_grad_scaler(use_amp, device_obj)
                train_epoch._scaler = scaler
            
            if scaler:
                scaler.scale(total_loss_val).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss_val.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            
            # Update EMA after optimizer step
            if hasattr(model, 'update_ema'):
                model.update_ema()
        else:
            total_loss_val, logs = compute_loss(
                model, batch, latents, t, noise, cond, loss_fn,
                use_amp, device_obj, needs_decoding
            )
            
            optimizer.zero_grad()
            total_loss_val.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            # Update EMA after optimizer step
            if hasattr(model, 'update_ema'):
                model.update_ema()
        
        batch_size = latents.shape[0]
        loss_val = total_loss_val.detach().item()
        total_loss += loss_val * batch_size
        total_samples += batch_size
        
        for k, v in logs.items():
            if k not in log_dict:
                log_dict[k] = 0.0
            if isinstance(v, torch.Tensor):
                log_dict[k] += v.item() * batch_size
            else:
                log_dict[k] += v * batch_size
        
        pbar.set_postfix({"loss": loss_val, **{k: v/total_samples for k, v in log_dict.items()}})
    
    if total_samples == 0:
        raise RuntimeError("No samples processed in training epoch! Check dataloader.")
    
    avg_loss = total_loss / total_samples
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    # Step scheduler once per epoch (not per batch)
    # Most schedulers (CosineAnnealingLR, LinearLR, StepLR) are epoch-based
    if scheduler:
        scheduler.step()
    
    return avg_loss, avg_logs


def eval_epoch(
    model, dataloader, scheduler, loss_fn, 
    device, use_amp=False
):
    """Evaluate for one epoch using CompositeLoss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    device_obj = to_device(device)
    
    # Cache loss class lookups and check if decoding is needed (once per epoch)
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    SemanticLossClass = LOSS_REGISTRY.get("SemanticLoss")
    needs_decoding = False
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        for sub_loss in loss_fn.losses:
            if SemanticLossClass and isinstance(sub_loss, SemanticLossClass):
                needs_decoding = True
                break
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = move_batch_to_device(batch, device_obj)
            
            latents = batch.get("latent")
            if latents is None:
                if "rgb" in batch and model._has_encoder:
                    encoder_out = model.encoder(batch["rgb"])
                    if "latent" in encoder_out:
                        latents = encoder_out["latent"]
                    elif "mu" in encoder_out:
                        latents = encoder_out["mu"]
                    else:
                        raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
                else:
                    raise ValueError("Dataset must provide 'latent' key (for pre-embedded) or 'rgb' key (for on-the-fly encoding)")
            
            num_steps = model.scheduler.num_steps
            t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
            noise = model.scheduler.randn_like(latents)
            
            # Extract and convert type labels (room/scene) to condition indices
            # type column contains "room" or "scene" strings, convert to 0=ROOM, 1=SCENE
            cond = None
            type_labels = batch.get("type", None)
            if type_labels is not None:
                # Handle both string and tensor types
                if isinstance(type_labels, (list, tuple)) and len(type_labels) > 0:
                    if isinstance(type_labels[0], str):
                        # Convert string labels to indices: "room" -> 0, "scene" -> 1
                        cond = torch.tensor(
                            [0 if t.lower().strip() == "room" else 1 for t in type_labels],
                            device=device_obj, dtype=torch.long
                        )
                    else:
                        # Already tensor indices
                        cond = type_labels.to(device_obj) if isinstance(type_labels, torch.Tensor) else torch.tensor(type_labels, device=device_obj, dtype=torch.long)
                elif isinstance(type_labels, torch.Tensor):
                    # Tensor might contain strings or indices
                    if type_labels.dtype == torch.long or type_labels.dtype == torch.int:
                        cond = type_labels.to(device_obj)
                    else:
                        # Convert string tensor to indices
                        cond = torch.tensor(
                            [0 if str(t).lower().strip() == "room" else 1 for t in type_labels.cpu().tolist()],
                            device=device_obj, dtype=torch.long
                        )
            
            if use_amp and device_obj.type == "cuda":
                with torch.amp.autocast('cuda'):
                    total_loss_val, logs = compute_loss(
                        model, batch, latents, t, noise, cond, loss_fn,
                        use_amp, device_obj, needs_decoding
                    )
            else:
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    use_amp, device_obj, needs_decoding
                )
            
            batch_size = latents.shape[0]
            loss_val = total_loss_val.item()
            total_loss += loss_val * batch_size
            total_samples += batch_size
            
            for k, v in logs.items():
                if k not in log_dict:
                    log_dict[k] = 0.0
                if isinstance(v, torch.Tensor):
                    log_dict[k] += v.item() * batch_size
                else:
                    log_dict[k] += v * batch_size
    
    if total_samples == 0:
        raise RuntimeError("No samples processed in evaluation epoch! Check dataloader.")
    
    avg_loss = total_loss / total_samples
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    return avg_loss, avg_logs


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=64, exp_name=None, guidance_scale=1.0):
    """Generate and save sample images.
    
    Generates 3 types of samples, each in a 4x4 grid:
    - Unconditioned: 16 samples (4x4 grid, cond=None)
    - Rooms: 16 samples (4x4 grid, cond=0)
    - Scenes: 16 samples (4x4 grid, cond=1)
    Total: 48 samples arranged in a 12x4 grid (3 sections of 4x4 stacked vertically)
    """
    model.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = to_device(device)
    
    try:
        batch_iter = iter(val_loader)
        batch = next(batch_iter)
    except StopIteration:
        return
    
    num_steps = model.scheduler.num_steps
    samples_per_type = 16  # 4x4 = 16 samples for each type
    grid_size = 4  # 4x4 grid for each type
    
    # Check if model supports conditioning (room/scene)
    supports_conditioning = hasattr(model.unet, 'cond_embedding') and model.unet.cond_embedding is not None
    
    all_samples = []
    cfg_info = f" with CFG scale={guidance_scale}" if guidance_scale > 1.0 else ""
    
    with torch.no_grad():
        # Generate unconditioned samples (if CFG is enabled, otherwise skip)
        if supports_conditioning and guidance_scale > 1.0:
            print(f"  Generating {samples_per_type} unconditioned samples (4x4) using DDPM ({num_steps} steps){cfg_info}...")
            sample_output = model.sample(
                batch_size=samples_per_type,
                num_steps=num_steps,
                method="ddpm",
                eta=1.0,
                cond=None,  # Unconditioned
                guidance_scale=guidance_scale,
                device=device_obj,
                verbose=False
            )
            all_samples.append(sample_output)
        elif not supports_conditioning:
            # If no conditioning support, generate unconditioned samples
            print(f"  Generating {samples_per_type} unconditioned samples (4x4) using DDPM ({num_steps} steps)...")
            sample_output = model.sample(
                batch_size=samples_per_type,
                num_steps=num_steps,
                method="ddpm",
                eta=1.0,
                cond=None,
                guidance_scale=1.0,
                device=device_obj,
                verbose=False
            )
            all_samples.append(sample_output)
        
        # Generate room samples (cond=0)
        if supports_conditioning:
            print(f"  Generating {samples_per_type} ROOM samples (4x4) using DDPM ({num_steps} steps){cfg_info}...")
            cond_room = torch.zeros(samples_per_type, dtype=torch.long, device=device_obj)
            sample_output = model.sample(
                batch_size=samples_per_type,
                num_steps=num_steps,
                method="ddpm",
                eta=1.0,
                cond=cond_room,
                guidance_scale=guidance_scale,
                device=device_obj,
                verbose=False
            )
            all_samples.append(sample_output)
        
        # Generate scene samples (cond=1)
        if supports_conditioning:
            print(f"  Generating {samples_per_type} SCENE samples (4x4) using DDPM ({num_steps} steps){cfg_info}...")
            cond_scene = torch.ones(samples_per_type, dtype=torch.long, device=device_obj)
            sample_output = model.sample(
                batch_size=samples_per_type,
                num_steps=num_steps,
                method="ddpm",
                eta=1.0,
                cond=cond_scene,
                guidance_scale=guidance_scale,
                device=device_obj,
                verbose=False
            )
            all_samples.append(sample_output)
    
    # Process all samples: decode and convert to [0, 255]
    processed_samples = []
    for sample_output in all_samples:
        if "rgb" in sample_output:
            # Already decoded, in [0, 1] range
            samples = sample_output["rgb"] * 255.0
        else:
            # Decode from latents
            outputs = model.decoder({"latent": sample_output["latent"]})
            samples = outputs["rgb"]  # [-1, 1] range from tanh
            samples = (samples + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            samples = samples * 255.0  # [0, 1] -> [0, 255]
        processed_samples.append(samples)
    
    # Concatenate all samples: [unconditioned, rooms, scenes]
    all_samples_tensor = torch.cat(processed_samples, dim=0)  # [48, C, H, W] if all 3 types
    
    # Convert to numpy
    samples_np = all_samples_tensor.cpu().numpy()
    samples_np = np.clip(samples_np, 0, 255).astype(np.uint8)
    
    # Create grid: 3 sections stacked vertically, each 4x4
    # Total: 12 rows x 4 columns (if all 3 types) or 4 rows x 4 columns (if only 1 type)
    num_sections = len(processed_samples)  # Number of types (1, 2, or 3)
    num_rows_total = num_sections * grid_size  # Total rows: 12 if 3 types, 4 if 1 type
    num_cols = grid_size  # 4 columns
    
    images = []
    for i in range(all_samples_tensor.shape[0]):
        img = samples_np[i].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        images.append(Image.fromarray(img))
    
    # Create grid image: stacked 4x4 grids
    img_size = images[0].size[0]
    grid_width = img_size * num_cols
    grid_height = img_size * num_rows_total
    grid_img = Image.new('RGB', (grid_width, grid_height))
    
    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        grid_img.paste(img, (col * img_size, row * img_size))
    
    grid_path = samples_dir / (f"{exp_name}_epoch_{epoch:03d}_samples.png" if exp_name else f"epoch_{epoch:03d}_samples.png")
    grid_img.save(grid_path)
    print(f"  Saved {all_samples_tensor.shape[0]} samples ({num_sections} types x {grid_size}x{grid_size} grids = {num_rows_total}x{num_cols} total grid) to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model (unified for all stages)")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--no-resume", action="store_true", help="Force start from scratch")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    print(f"Experiment: {exp_name}")
    
    # Set deterministic behavior
    training_seed = config.get("training", {}).get("seed", None)
    if training_seed is not None:
        set_deterministic(training_seed)
        print(f"Set deterministic mode with seed: {training_seed}")
    
    # Get device
    device = get_device(config)
    print(f"Device: {device}")
    
    # Get output directory
    output_dir = config.get("experiment", {}).get("save_path")
    if output_dir is None:
        output_dir = Path("outputs") / exp_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check for latest checkpoint (in checkpoints subdirectory)
    checkpoint_dir = output_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    
    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(config)
    
    # Build validation dataset
    train_dataset, val_dataset = split_dataset(dataset, config["training"])
    
    device_obj = to_device(device)
    
    # Check if we should resume or start fresh
    should_resume = not args.no_resume and latest_checkpoint.exists()
    
    # Load checkpoint (Stage 1, Stage 2, or resume)
    stage1_checkpoint = config.get("diffusion", {}).get("stage1_checkpoint")
    stage2_checkpoint = config.get("diffusion", {}).get("stage2_checkpoint")
    
    if stage2_checkpoint and not should_resume:
        print(f"\nLoading Stage 2 checkpoint from: {stage2_checkpoint}")
        model, _ = DiffusionModel.load_checkpoint(
            stage2_checkpoint,
            map_location=device,
            return_extra=True,
            config=config
        )
        model = model.to(device_obj)
        print("Stage 2 checkpoint loaded successfully")
    elif stage1_checkpoint and not should_resume:
        print(f"\nLoading Stage 1 checkpoint from: {stage1_checkpoint}")
        model, _ = DiffusionModel.load_checkpoint(
            stage1_checkpoint,
            map_location=device,
            return_extra=True,
            config=config
        )
        model = model.to(device_obj)
        print("Stage 1 checkpoint loaded successfully")
    elif should_resume:
        print(f"\nFound latest checkpoint: {latest_checkpoint}")
        print("Resuming training...")
        
        model, extra_state = DiffusionModel.load_checkpoint(
            latest_checkpoint,
            map_location=device,
            return_extra=True,
            config=config
        )
        model = model.to(device_obj)
        
        start_epoch = extra_state.get("epoch", 1) - 1
        best_val_loss = extra_state.get("best_val_loss", float("inf"))
        training_history = extra_state.get("training_history", [])
        
        if not training_history and metrics_csv_path.exists():
            try:
                df = pd.read_csv(metrics_csv_path)
                df_filtered = df[df['epoch'] < (start_epoch + 1)]
                training_history = df_filtered.to_dict('records')
                print(f"  Loaded {len(training_history)} epochs from CSV file")
            except Exception as e:
                print(f"  Warning: Could not load metrics from CSV: {e}")
        
        print(f"  Resuming from epoch {start_epoch + 1}")
        print(f"  Best validation loss so far: {best_val_loss:.6f}")
    else:
        # Build model from config (fresh start)
        print("\nBuilding model from config...")
        diffusion_cfg = config.get("diffusion", {})
        
        # If no diffusion section, extract from top-level config (for ablation configs)
        if not diffusion_cfg:
            diffusion_cfg = {
                "autoencoder": config.get("autoencoder"),
                "unet": config.get("unet", {}),
                "scheduler": config.get("scheduler", {})
            }
        
        if "type" in diffusion_cfg:
            diffusion_cfg = {k: v for k, v in diffusion_cfg.items() if k != "type"}
        # Pass save_path from experiment config so model can write statistics
        exp_cfg = config.get("experiment", {})
        if exp_cfg.get("save_path"):
            diffusion_cfg["save_path"] = exp_cfg["save_path"]
        model = DiffusionModel(**diffusion_cfg)
        model = model.to(device_obj)
    
    # Keep decoder frozen - only UNet is trained
    if hasattr(model, 'decoder'):
        print("Keeping decoder frozen - only UNet will be trained...")
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    # CRITICAL: Ensure UNet is trainable (explicitly set requires_grad=True)
    # This overrides any frozen settings from config or checkpoint
    if hasattr(model, 'unet'):
        print("Ensuring UNet is trainable...")
        trainable_params = 0
        frozen_params = 0
        frozen_tensors = 0
        for param in model.unet.parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
                frozen_tensors += 1
                param.requires_grad = True  # Force trainable
        
        if frozen_tensors > 0:
            print(f"  WARNING: Found {frozen_tensors} frozen UNet parameter tensors ({frozen_params:,} params) - setting them to trainable!")
        total_params = trainable_params + frozen_params
        print(f"  UNet parameters: {total_params:,} total, {total_params:,} trainable")
    
    # Build data loaders
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 8)
    shuffle = config["training"].get("shuffle", True)
    use_weighted_sampling = config["training"].get("use_weighted_sampling", False)
    
    # Auto-generate weight stats if needed
    weights_stats_path = None
    if use_weighted_sampling:
        # Support both "weight_column" (new) and "column" (old) for backward compatibility
        weight_column = config["training"].get("weight_column", None) or config["training"].get("column", None)
        if weight_column:
            # Get manifest path from dataset config
            manifest_path = Path(config["dataset"]["manifest"])
            
            # Get filters from dataset config to apply before computing weights
            # This ensures weights are computed on the same filtered dataset used for training
            dataset_filters = config["dataset"].get("filters", None)
            
            # Ensure weight stats exist (will generate if needed)
            from training.utils import ensure_weight_stats_exist
            weights_stats_path = ensure_weight_stats_exist(
                manifest_path=manifest_path,
                column_name=weight_column,
                output_dir=output_dir,
                rare_threshold_percentile=config["training"].get("rare_threshold_percentile", 10.0),
                min_samples_threshold=config["training"].get("min_samples_threshold", 50),
                weighting_method=config["training"].get("weighting_method", "inverse_frequency"),
                max_weight=config["training"].get("max_weight", None),
                min_weight=config["training"].get("min_weight", 1.0),
                filters=dataset_filters  # Apply same filters as dataset
            )
    
    # Use dataset's make_dataloader to support weighted sampling
    train_loader = train_dataset.make_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
        persistent_workers=num_workers > 0,
        use_weighted_sampling=use_weighted_sampling,
        weight_column=config["training"].get("weight_column", None) or config["training"].get("column", None),
        weights_stats_path=weights_stats_path,
        use_grouped_weights=config["training"].get("use_grouped_weights", False),
        group_rare_classes=config["training"].get("group_rare_classes", False),
        class_grouping_path=config["training"].get("class_grouping_path", None),
        max_weight=config["training"].get("max_weight", None),
        exclude_extremely_rare=config["training"].get("exclude_extremely_rare", False),
        min_samples_threshold=config["training"].get("min_samples_threshold", 50)
    )
    print(f"Training dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    
    val_loader = None
    if val_dataset:
        val_loader = val_dataset.make_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device_obj.type == "cuda",
            persistent_workers=num_workers > 0,
            use_weighted_sampling=False  # No weighted sampling for validation
        )
        print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)}")
    
    # Build loss function from config (uses CompositeLoss)
    print("Building loss function from config...")
    loss_fn = build_loss(config)
    print(f"  Loss type: {type(loss_fn).__name__}")
    if hasattr(loss_fn, 'losses'):
        print(f"  Loss components: {[type(l).__name__ for l in loss_fn.losses]}")
    
    # Build optimizer
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    
    # Verify optimizer has trainable parameters
    total_optimizer_params = sum(len(group['params']) for group in optimizer.param_groups)
    total_trainable_model_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"  Optimizer parameter groups: {len(optimizer.param_groups)}")
    print(f"  Total trainable parameters in model: {total_trainable_model_params:,}")
    
    # Count UNet parameters specifically
    if hasattr(model, 'unet'):
        unet_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        unet_total = sum(p.numel() for p in model.unet.parameters())
        print(f"  UNet parameters: {unet_trainable:,} trainable / {unet_total:,} total")
        if unet_trainable == 0:
            print("  ERROR: UNet has no trainable parameters! Training will not work!")
    
    # Build learning rate scheduler (account for already-trained epochs when resuming)
    # Use last_epoch=-1 for fresh start, or start_epoch for resume
    last_epoch = start_epoch if start_epoch > 0 else -1
    scheduler = build_scheduler(optimizer, config, last_epoch=last_epoch)
    
    # Training settings
    epochs = config["training"].get("epochs", 100)
    use_amp = config["training"].get("use_amp", False)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    eval_interval = config["training"].get("eval_interval", 5)
    sample_interval = config["training"].get("sample_interval", 10)
    use_non_uniform_sampling = config["training"].get("use_non_uniform_sampling", False)  # Default False for uniform sampling
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    
    # Early stopping state
    epochs_without_improvement = 0
    
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Non-uniform timestep sampling: {use_non_uniform_sampling}")
    cfg_dropout_rate = config.get("training", {}).get("cfg_dropout_rate", 0.0)
    guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
    if cfg_dropout_rate > 0.0:
        print(f"  CFG dropout rate: {cfg_dropout_rate} (condition randomly dropped {cfg_dropout_rate*100:.1f}% of the time)")
    if guidance_scale > 1.0:
        print(f"  CFG guidance scale: {guidance_scale} (used during sampling)")
    if early_stopping_patience is not None:
        print(f"  Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        # Get CFG parameters from config
        cfg_dropout_rate = config.get("training", {}).get("cfg_dropout_rate", 0.0)
        guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
        
        train_loss, train_logs = train_epoch(
            model, train_loader, scheduler, loss_fn,
            optimizer, device_obj, epoch + 1, use_amp=use_amp, max_grad_norm=max_grad_norm,
            use_non_uniform_sampling=use_non_uniform_sampling, cfg_dropout_rate=cfg_dropout_rate
        )
        
        print(f"Train Loss: {train_loss:.6f}")
        for k, v in train_logs.items():
            print(f"  {k}: {v:.6f}")
        
        # Validate
        val_loss = float("inf")
        val_logs = {}
        if val_loader and (epoch + 1) % eval_interval == 0:
            val_loss, val_logs = eval_epoch(
                model, val_loader, scheduler, loss_fn,
                device_obj, use_amp=use_amp
            )
            print(f"Val Loss: {val_loss:.6f}")
            for k, v in val_logs.items():
                print(f"  {k}: {v:.6f}")
            
            # Check if this is the best validation loss BEFORE updating best_val_loss
            is_best = val_loss < best_val_loss
            
            # Early stopping logic
            if early_stopping_patience is not None:
                improvement = best_val_loss - val_loss
                if improvement > early_stopping_min_delta:
                    epochs_without_improvement = 0
                    if is_best:
                        best_val_loss = val_loss
                    print(f"  Improvement: {improvement:.6f} (new best: {best_val_loss:.6f})")
                else:
                    epochs_without_improvement += 1
                    print(f"  No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
            elif is_best:
                # Update best_val_loss if not using early stopping
                best_val_loss = val_loss
        
        # Save samples
        # Always save at epoch 1, then every sample_interval epochs
        if val_loader and ((epoch + 1 == 1) or ((epoch + 1) % sample_interval == 0)):
            # Get guidance_scale from config (default 1.0 = no CFG)
            guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
            save_samples(model, val_loader, device_obj, output_dir, epoch + 1, sample_batch_size=64, exp_name=exp_name, guidance_scale=guidance_scale)
        
        # Save checkpoint (is_best was already determined above if validation ran)
        if val_loader and (epoch + 1) % eval_interval == 0:
            # is_best already determined above
            pass
        else:
            # No validation this epoch, so not best
            is_best = False
        
        # Record history
        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_logs.items()},
            **{f"val_{k}": v for k, v in val_logs.items()}
        }
        training_history.append(history_entry)
        
        # Save metrics to CSV
        save_metrics_csv(training_history, metrics_csv_path)
        
        # Plot metrics with loss breakdown
        if len(training_history) > 0:
            try:
                df = pd.DataFrame(training_history)
                plot_diffusion_metrics_epochs(df, output_dir, exp_name=exp_name)
            except Exception as e:
                print(f"  Warning: Could not plot metrics: {e}")
        
        # Save checkpoint
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
        model.save_checkpoint(
            checkpoint_path,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            training_history=training_history
        )
        
        if is_best:
            best_checkpoint_path = checkpoint_dir / f"{exp_name}_checkpoint_best.pt"
            model.save_checkpoint(
                best_checkpoint_path,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                training_history=training_history
            )
            print(f"  Saved best checkpoint (val_loss={best_val_loss:.6f})")
        
        # Early stopping check
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"No improvement for {epochs_without_improvement} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"{'='*60}")
            break
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
