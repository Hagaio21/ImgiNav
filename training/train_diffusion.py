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
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def calculate_scale_factor_from_dataset(dataset, num_samples=100, seed=42):
    """
    Calculate scale_factor from dataset latents.
    
    Samples random latents from the dataset and calculates their global standard deviation.
    Returns scale_factor = 1.0 / std to normalize latents to unit variance.
    
    Args:
        dataset: Dataset with 'latent' key in samples
        num_samples: Number of random samples to use (default: 100)
        seed: Random seed for reproducibility
    
    Returns:
        float: scale_factor (1.0 / std)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Sample random indices
    dataset_size = len(dataset)
    num_samples = min(num_samples, dataset_size)
    sampled_indices = random.sample(range(dataset_size), num_samples)
    
    print(f"Calculating scale_factor from {num_samples} random latents...")
    
    all_latent_values = []
    loaded_count = 0
    
    for idx in tqdm(sampled_indices, desc="Loading latents"):
        try:
            sample = dataset[idx]
            latents = sample.get("latent")
            if latents is None:
                continue
            
            # Convert to numpy and flatten
            if isinstance(latents, torch.Tensor):
                latent_np = latents.cpu().numpy()
            else:
                latent_np = np.array(latents)
            
            # Flatten to 1D array
            latent_flat = latent_np.flatten()
            all_latent_values.append(latent_flat)
            loaded_count += 1
        except Exception as e:
            print(f"Warning: Failed to load sample {idx}: {e}")
            continue
    
    if loaded_count == 0:
        raise RuntimeError("Failed to load any latents from dataset")
    
    # Concatenate all latents and calculate global statistics
    all_latents = np.concatenate(all_latent_values)
    
    mean = np.mean(all_latents)
    std = np.std(all_latents)
    
    # Calculate scale factor (1.0 / std to normalize to unit variance)
    scale_factor = 1.0 / std if std > 0 else 1.0
    
    print(f"  Loaded {loaded_count} latents")
    print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
    print(f"  Calculated scale_factor: {scale_factor:.6f}")
    
    return scale_factor


def compute_loss(
    model, batch, latents, t, noise, cond, loss_fn, 
    use_amp=False, device_obj=None, needs_decoding=False, cfg_dropout_rate=0.0
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
    # Extract embeddings if available (for cross-attention conditioning)
    text_emb = batch.get("text_emb", None)
    pov_emb = batch.get("pov_emb", None)
    
    # Ensure embeddings are 1D (flatten if needed)
    if text_emb is not None:
        if text_emb.dim() > 1:
            text_emb = text_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
    if pov_emb is not None:
        if pov_emb.dim() > 1:
            pov_emb = pov_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
    
    # Apply CFG dropout for embeddings (randomly drop embeddings with cfg_dropout_rate probability)
    # This teaches the model to work both with and without cross-attention conditioning
    if cfg_dropout_rate > 0.0 and text_emb is not None and pov_emb is not None:
        if torch.rand(1, device=device_obj).item() < cfg_dropout_rate:
            text_emb = None  # Drop embeddings for CFG training
            pov_emb = None
    
    # Forward pass through model
    outputs = model(latents, t, cond=cond, noise=noise, text_emb=text_emb, pov_emb=pov_emb)
    
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
                    use_amp, device_obj, needs_decoding, cfg_dropout_rate
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
                use_amp, device_obj, needs_decoding, cfg_dropout_rate
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
                        use_amp, device_obj, needs_decoding, cfg_dropout_rate
                    )
            else:
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    use_amp, device_obj, needs_decoding, cfg_dropout_rate
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


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=64, exp_name=None, guidance_scale=1.0, cfg_dropout_rate=0.0):
    """Generate and save sample images.
    
    If cfg_dropout_rate == 1.0 (fully unconditional training), only generates unconditioned samples.
    Otherwise generates 3 types of samples, each in a 4x4 grid:
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
    
    # If cfg_dropout_rate == 1.0, model was trained fully unconditionally - only generate unconditioned samples
    is_fully_unconditional = (cfg_dropout_rate >= 1.0)
    
    all_samples = []
    cfg_info = f" with CFG scale={guidance_scale}" if guidance_scale > 1.0 else ""
    
    # ============================================================================
    # Part 1: Generate 8x8 grid of unconditioned samples (64 samples)
    # ============================================================================
    print(f"  Generating 64 unconditioned samples (8x8 grid) using DDPM ({num_steps} steps){cfg_info}...")
    
    # Use epoch-based seed for sampling
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = None
    if torch.cuda.is_available():
        cuda_rng_states = torch.cuda.get_rng_state_all()
    
    sampling_seed = 42 + epoch
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)
    
    with torch.no_grad():
        unconditioned_output = model.sample(
            batch_size=64,
            num_steps=num_steps,
            method="ddpm",
            eta=1.0,
            cond=None,
            guidance_scale=guidance_scale if guidance_scale > 1.0 else 1.0,
            text_emb=None,
            pov_emb=None,
            device=device_obj,
            verbose=False
        )
        
        # Decode unconditioned samples
        if "rgb" in unconditioned_output:
            unconditioned_rgb = unconditioned_output["rgb"]
            if unconditioned_rgb.min() < 0:  # [-1, 1] range
                unconditioned_rgb = (unconditioned_rgb + 1.0) / 2.0
            unconditioned_rgb = torch.clamp(unconditioned_rgb, 0.0, 1.0)
        else:
            decoded = model.decoder({"latent": unconditioned_output["latent"]})
            if "rgb" in decoded:
                unconditioned_rgb = (decoded["rgb"] + 1.0) / 2.0
                unconditioned_rgb = torch.clamp(unconditioned_rgb, 0.0, 1.0)
            else:
                print("  Warning: Decoder did not produce RGB output for unconditioned samples")
                unconditioned_rgb = None
    
    # Save 8x8 unconditioned grid
    if unconditioned_rgb is not None:
        unconditioned_np = (unconditioned_rgb.cpu().numpy() * 255.0).astype(np.uint8)
        unconditioned_images = []
        for i in range(64):
            img = Image.fromarray(unconditioned_np[i].transpose(1, 2, 0))
            unconditioned_images.append(img)
        
        img_size = unconditioned_images[0].size[0]
        grid_n = 8  # 8x8 grid
        unconditioned_grid = Image.new('RGB', (img_size * grid_n, img_size * grid_n))
        for idx, img in enumerate(unconditioned_images):
            row = idx // grid_n
            col = idx % grid_n
            unconditioned_grid.paste(img, (col * img_size, row * img_size))
        
        epoch_prefix = f"{exp_name}_epoch_{epoch:03d}" if exp_name else f"epoch_{epoch:03d}"
        unconditioned_path = samples_dir / f"{epoch_prefix}_unconditioned_8x8.png"
        unconditioned_grid.save(unconditioned_path)
        print(f"  Saved 64 unconditioned samples (8x8 grid) to {unconditioned_path}")
    
    # Remove old logic that processed all_samples - we now handle unconditioned separately
    all_samples = []
    

    # ============================================================================
    # Part 2: Targets vs Generated comparison (4 rooms + 4 scenes)
    # ============================================================================
    # Get dataset to find rooms and scenes
    dataset = val_loader.dataset
    
    # Find 4 rooms and 4 scenes from the validation dataset
    room_indices = []
    scene_indices = []
    
    # Check if dataset has 'type' column
    if hasattr(dataset, 'df') and 'type' in dataset.df.columns:
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]
            sample_type = str(row.get('type', '')).lower().strip()
            if sample_type == 'room' and len(room_indices) < 4:
                room_indices.append(idx)
            elif sample_type == 'scene' and len(scene_indices) < 4:
                scene_indices.append(idx)
            if len(room_indices) >= 4 and len(scene_indices) >= 4:
                break
    else:
        # Fallback: use first 8 samples if type column not available
        room_indices = list(range(min(4, len(dataset))))
        scene_indices = list(range(min(4, len(dataset))))
    
    # Combine indices: 4 rooms + 4 scenes = 8 total
    selected_indices = room_indices + scene_indices
    batch_size = len(selected_indices)
    
    if batch_size < 8:
        print(f"  Warning: Could only find {len(room_indices)} rooms and {len(scene_indices)} scenes. Using available samples.")
    
    if batch_size == 0:
        print("  Warning: No samples found in validation dataset for comparison")
        return
    
    # Load selected samples from dataset
    batch_data = {}
    batch_indices = []
    
    for idx in selected_indices:
        sample = dataset[idx]
        for key, value in sample.items():
            if key not in batch_data:
                batch_data[key] = []
            batch_data[key].append(value)
        batch_indices.append(idx)
    
    # Convert lists to tensors
    batch = {}
    for key, values in batch_data.items():
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    
    batch = move_batch_to_device(batch, device_obj, non_blocking=False)
    
    print(f"  Selected {len(room_indices)} rooms and {len(scene_indices)} scenes for comparison")
    
    # Extract embeddings and conditions
    text_emb = batch.get("text_emb", None)
    pov_emb = batch.get("pov_emb", None)
    target_latents = batch.get("latent", None)
    
    # Extract cond (room/scene type) if available
    cond = None
    type_labels = batch.get("type", None)
    if type_labels is not None:
        if isinstance(type_labels, (list, tuple)) and len(type_labels) > 0:
            if isinstance(type_labels[0], str):
                cond = torch.tensor(
                    [0 if t.lower().strip() == "room" else 1 for t in type_labels],
                    device=device_obj, dtype=torch.long
                )
            else:
                cond = torch.tensor(type_labels, device=device_obj, dtype=torch.long)
        elif isinstance(type_labels, torch.Tensor):
            cond = type_labels.to(device_obj)
    
    # Ensure embeddings are 1D (flatten if needed)
    if text_emb is not None:
        if text_emb.dim() > 1:
            text_emb = text_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
    if pov_emb is not None:
        if pov_emb.dim() > 1:
            pov_emb = pov_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
    
    # Check if we have embeddings for cross-attention
    has_embeddings = text_emb is not None and pov_emb is not None
    
    if not has_embeddings:
        print("  Warning: Cannot generate conditioned samples without text_emb and pov_emb")
        text_emb = None
        pov_emb = None
    
    if target_latents is None:
        print("  Warning: Cannot decode target latents - missing 'latent' in batch")
        return
    
    # Decode target latents to RGB (for comparison)
    print(f"  Decoding {batch_size} target layouts...")
    with torch.no_grad():
        target_decoded = model.decoder({"latent": target_latents})
        if "rgb" in target_decoded:
            target_rgb = (target_decoded["rgb"] + 1.0) / 2.0
            target_rgb = torch.clamp(target_rgb, 0.0, 1.0)
        else:
            print("  Warning: Decoder did not produce RGB output for targets")
            target_rgb = None
    
    # Generate conditioned samples using DDPM
    print(f"  Generating {batch_size} conditioned samples using cross-attention diffusion with DDPM ({num_steps} steps){cfg_info}...")
    
    with torch.no_grad():
        conditioned_output = model.sample(
            batch_size=batch_size,
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
        
        # Decode generated latents to RGB
        if "rgb" in conditioned_output:
            generated_rgb = conditioned_output["rgb"]
            if generated_rgb.min() < 0:  # [-1, 1] range
                generated_rgb = (generated_rgb + 1.0) / 2.0
            generated_rgb = torch.clamp(generated_rgb, 0.0, 1.0)
        else:
            decoded = model.decoder({"latent": conditioned_output["latent"]})
            if "rgb" in decoded:
                generated_rgb = (decoded["rgb"] + 1.0) / 2.0
                generated_rgb = torch.clamp(generated_rgb, 0.0, 1.0)
            else:
                print("  Warning: Decoder did not produce RGB output for generated samples")
                generated_rgb = None
    
    # Create side-by-side comparison: target (left) | generated (right)
    if target_rgb is not None and generated_rgb is not None:
        # Convert to [0, 255] for PIL
        target_np = (target_rgb.cpu().numpy() * 255.0).astype(np.uint8)
        generated_np = (generated_rgb.cpu().numpy() * 255.0).astype(np.uint8)
        
        # Create images
        target_images = []
        generated_images = []
        for i in range(batch_size):
            target_img = Image.fromarray(target_np[i].transpose(1, 2, 0))
            generated_img = Image.fromarray(generated_np[i].transpose(1, 2, 0))
            target_images.append(target_img)
            generated_images.append(generated_img)
        
        # Create side-by-side comparison
        img_size = target_images[0].size[0]
        grid_n = 4  # 4 columns (2 rows: 4 rooms on top, 4 scenes on bottom)
        num_rows = (batch_size + grid_n - 1) // grid_n
        
        # Create target grid
        target_grid = Image.new('RGB', (img_size * grid_n, img_size * num_rows))
        for idx, img in enumerate(target_images):
            row = idx // grid_n
            col = idx % grid_n
            target_grid.paste(img, (col * img_size, row * img_size))
        
        # Create generated grid
        generated_grid = Image.new('RGB', (img_size * grid_n, img_size * num_rows))
        for idx, img in enumerate(generated_images):
            row = idx // grid_n
            col = idx % grid_n
            generated_grid.paste(img, (col * img_size, row * img_size))
        
        # Concatenate horizontally (side by side)
        comparison_width = img_size * grid_n * 2
        comparison_height = img_size * num_rows
        comparison_img = Image.new('RGB', (comparison_width, comparison_height))
        comparison_img.paste(target_grid, (0, 0))
        comparison_img.paste(generated_grid, (img_size * grid_n, 0))
        
        # Save comparison
        epoch_prefix = f"{exp_name}_epoch_{epoch:03d}" if exp_name else f"epoch_{epoch:03d}"
        comparison_path = samples_dir / f"{epoch_prefix}_comparison.png"
        comparison_img.save(comparison_path)
        print(f"  Saved target vs generated comparison ({batch_size} samples: {len(room_indices)} rooms + {len(scene_indices)} scenes) to {comparison_path}")
        
        # Also save generated samples only
        samples_path = samples_dir / f"{epoch_prefix}_samples.png"
        generated_grid.save(samples_path)
        print(f"  Saved generated samples ({batch_size} samples) to {samples_path}")
        
        # ============================================================================
        # Part 3: Save individual images with conditions
        # ============================================================================
        # Create directory for individual samples with conditions
        data_dir = output_dir / "sample_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = data_dir / f"{epoch_prefix}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get paths from dataset
        graph_texts = []
        layout_paths = []
        pov_paths = []
        
        for i in range(batch_size):
            idx = batch_indices[i] if i < len(batch_indices) else i
            row = dataset.df.iloc[idx]
            
            # Get graph text path
            graph_text_path = row.get("graph_text_path", "")
            if graph_text_path and Path(graph_text_path).exists():
                try:
                    with open(graph_text_path, 'r') as f:
                        graph_text = f.read()
                    graph_texts.append(graph_text)
                    # Save graph text
                    text_file = images_dir / f"sample_{i:03d}_graph_text.txt"
                    with open(text_file, 'w') as f:
                        f.write(graph_text)
                except Exception as e:
                    print(f"  Warning: Could not read graph text from {graph_text_path}: {e}")
                    graph_texts.append("")
            else:
                graph_texts.append("")
            
            # Get layout path (for reference)
            layout_path = row.get("layout_path", "")
            layout_paths.append(layout_path)
            
            # Get POV path
            pov_path = row.get("pov_path", "")
            pov_paths.append(pov_path)
            
            # Save POV image if available
            if pov_path and Path(pov_path).exists():
                try:
                    pov_img = Image.open(pov_path)
                    pov_img.save(images_dir / f"sample_{i:03d}_pov.png")
                except Exception as e:
                    print(f"  Warning: Could not save POV image from {pov_path}: {e}")
        
        # Save target and generated images individually
        for i in range(batch_size):
            target_img = target_images[i]
            generated_img = generated_images[i]
            target_img.save(images_dir / f"sample_{i:03d}_target.png")
            generated_img.save(images_dir / f"sample_{i:03d}_generated.png")
        
        # Save conditioning embeddings
        if text_emb is not None:
            torch.save(text_emb.cpu(), images_dir / "text_embeddings.pt")
        if pov_emb is not None:
            torch.save(pov_emb.cpu(), images_dir / "pov_embeddings.pt")
        if cond is not None:
            torch.save(cond.cpu(), images_dir / "cond_types.pt")
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "batch_size": batch_size,
            "room_indices": room_indices,
            "scene_indices": scene_indices,
            "batch_indices": batch_indices,
            "layout_paths": layout_paths,
            "pov_paths": pov_paths,
            "graph_texts": graph_texts,
            "cond_types": cond.cpu().tolist() if cond is not None else None,
            "has_text_emb": text_emb is not None,
            "has_pov_emb": pov_emb is not None,
            "guidance_scale": guidance_scale,
            "cfg_dropout_rate": cfg_dropout_rate,
        }
        with open(images_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved individual samples with conditions to {images_dir}")


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
    
    # Calculate scale_factor if not provided in config
    # Check both diffusion section and top-level config
    diffusion_cfg = config.get("diffusion", {})
    if not diffusion_cfg:
        diffusion_cfg = {}
    
    scale_factor = diffusion_cfg.get("scale_factor") or config.get("scale_factor")
    
    if scale_factor is None:
        print("\n" + "="*60)
        print("scale_factor not found in config - calculating automatically from dataset")
        print("="*60)
        try:
            scale_factor = calculate_scale_factor_from_dataset(
                train_dataset,
                num_samples=config.get("training", {}).get("scale_factor_samples", 100),
                seed=config.get("training", {}).get("seed", 42)
            )
            # Add to config for model building
            if "diffusion" in config:
                config["diffusion"]["scale_factor"] = scale_factor
            else:
                config["scale_factor"] = scale_factor
            print(f"  Auto-calculated scale_factor: {scale_factor:.6f}")
            print("  (Add this to your config to avoid recalculating)")
            print("="*60 + "\n")
        except Exception as e:
            print(f"Warning: Failed to calculate scale_factor automatically: {e}")
            print("  Using default scale_factor=1.0 (no scaling)")
            scale_factor = 1.0
            if "diffusion" in config:
                config["diffusion"]["scale_factor"] = scale_factor
            else:
                config["scale_factor"] = scale_factor
    else:
        print(f"\nUsing scale_factor from config: {scale_factor}")
    
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
        
        # Add scale_factor if it was calculated or provided
        if scale_factor is not None:
            diffusion_cfg["scale_factor"] = scale_factor
        
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
    cfg_dropout_config = config.get("training", {}).get("cfg_dropout_rate", 0.0)
    guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
    if isinstance(cfg_dropout_config, dict):
        start_rate = cfg_dropout_config.get("start", 1.0)
        end_rate = cfg_dropout_config.get("end", 0.1)
        schedule_type = cfg_dropout_config.get("schedule", "linear")
        step_size = cfg_dropout_config.get("step_size", 1)
        plateau_epoch = cfg_dropout_config.get("plateau_epoch", None)
        schedule_info = f"({schedule_type} schedule"
        if step_size > 1:
            schedule_info += f", changes every {step_size} epochs"
        if plateau_epoch is not None:
            schedule_info += f", plateaus at {end_rate} after epoch {plateau_epoch}"
        schedule_info += ")"
        print(f"  CFG dropout rate: scheduled from {start_rate} to {end_rate} {schedule_info}")
    elif cfg_dropout_config > 0.0:
        print(f"  CFG dropout rate: {cfg_dropout_config} (condition randomly dropped {cfg_dropout_config*100:.1f}% of the time)")
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
        # Support scheduled CFG dropout (decreasing over epochs)
        cfg_dropout_config = config.get("training", {}).get("cfg_dropout_rate", 0.0)
        if isinstance(cfg_dropout_config, dict):
            # Schedule format: {start: 1.0, end: 0.1, schedule: "linear", step_size: 10, plateau_epoch: 200}
            start_rate = cfg_dropout_config.get("start", 1.0)
            end_rate = cfg_dropout_config.get("end", 0.1)
            schedule_type = cfg_dropout_config.get("schedule", "linear")
            step_size = cfg_dropout_config.get("step_size", 1)  # Change every N epochs (default: 1 for backward compatibility)
            plateau_epoch = cfg_dropout_config.get("plateau_epoch", None)  # After this epoch, stay at end_rate
            
            # If we've passed the plateau epoch, just use the end rate
            if plateau_epoch is not None and epoch >= plateau_epoch:
                cfg_dropout_rate = end_rate
            else:
                # Calculate current rate based on schedule with step-based updates
                # Use floor division to get the current step, so rate stays constant for step_size epochs
                # If plateau_epoch is set, calculate progress based on plateau_epoch instead of total epochs
                effective_max_epoch = plateau_epoch if plateau_epoch is not None else epochs
                current_step = epoch // step_size
                max_step = (effective_max_epoch - 1) // step_size  # Maximum step before plateau
                if max_step > 0:
                    progress = min(current_step / max_step, 1.0)  # Clamp to 1.0
                else:
                    progress = 0.0
                
                if schedule_type == "linear":
                    cfg_dropout_rate = start_rate + (end_rate - start_rate) * progress
                elif schedule_type == "cosine":
                    import math
                    cfg_dropout_rate = end_rate + (start_rate - end_rate) * (1 + math.cos(math.pi * progress)) / 2
                else:
                    # Default to linear
                    cfg_dropout_rate = start_rate + (end_rate - start_rate) * progress
        else:
            # Fixed rate (backward compatible)
            cfg_dropout_rate = cfg_dropout_config
        
        guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
        
        train_loss, train_logs = train_epoch(
            model, train_loader, scheduler, loss_fn,
            optimizer, device_obj, epoch + 1, use_amp=use_amp, max_grad_norm=max_grad_norm,
            use_non_uniform_sampling=use_non_uniform_sampling, cfg_dropout_rate=cfg_dropout_rate
        )
        
        # Print current CFG dropout rate if using schedule
        if isinstance(cfg_dropout_config, dict):
            print(f"  Current CFG dropout rate: {cfg_dropout_rate:.4f}")
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
            save_samples(model, val_loader, device_obj, output_dir, epoch + 1, sample_batch_size=64, exp_name=exp_name, guidance_scale=guidance_scale, cfg_dropout_rate=cfg_dropout_rate)
        
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
            "cfg_dropout_rate": cfg_dropout_rate,  # Track CFG dropout rate
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
