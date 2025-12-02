#!/usr/bin/env python3
"""
Unified training script for diffusion models (Stage 1, Stage 2, Stage 3).
Uses CompositeLoss from config to combine losses (MSE or SNR weighted).
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import sys
import json
import time

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
from training.engine import Trainer
from training.plotting_utils import plot_loss_curves
from models.diffusion import DiffusionModel
from models.autoencoder import Autoencoder
from models.losses.base_loss import LOSS_REGISTRY


def latents2rgb(model, output_dict, warning_prefix="Decoder"):
    """
    Convert latents to RGB images from model output.
    
    Handles two cases:
    1. Output already contains "rgb" key - use it directly
    2. Output contains "latent" key - decode using model.decoder
    
    Args:
        model: DiffusionModel with decoder
        output_dict: Dictionary containing either "rgb" or "latent" key
        warning_prefix: Prefix for warning message if decoding fails
    
    Returns:
        torch.Tensor: RGB images in [0, 1] range, or None if decoding fails
    """
    if "rgb" in output_dict:
        rgb = output_dict["rgb"]
        # Normalize from [-1, 1] to [0, 1] if needed
        if rgb.min() < 0:
            rgb = (rgb + 1.0) / 2.0
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb
    elif "latent" in output_dict:
        with torch.no_grad():
            decoded = model.decoder({"latent": output_dict["latent"]})
            if "rgb" in decoded:
                rgb = (decoded["rgb"] + 1.0) / 2.0
                rgb = torch.clamp(rgb, 0.0, 1.0)
                return rgb
            else:
                print(f"  Warning: {warning_prefix} did not produce RGB output")
                return None
    else:
        print(f"  Warning: {warning_prefix} output missing both 'rgb' and 'latent' keys")
        return None


# Step function for Trainer
def diffusion_step_fn(model, batch, batch_idx, loss_fn, trainer):
    """Step function for diffusion training - computes loss only (Trainer handles backward/step)."""
    device_obj = trainer.device
    
    # Get latents - must be provided in batch (pre-encoded)
    latents = batch.get("latent")
    if latents is None:
        raise ValueError("Dataset must provide 'latent' key. Encoding should be done before training.")
    
    # Get training config from trainer or use defaults
    use_non_uniform_sampling = getattr(trainer, 'use_non_uniform_sampling', False)
    cfg_dropout_rate = getattr(trainer, 'cfg_dropout_rate', 0.0)
    
    # Sample random timesteps
    num_steps = model.scheduler.num_steps
    if use_non_uniform_sampling:
        # Higher probability for early timesteps (high noise)
        probs = torch.exp(-torch.linspace(0, 2, num_steps, device=device_obj))
        probs = probs / probs.sum()
        t = torch.multinomial(probs, latents.shape[0], replacement=True)
    else:
        # Uniform sampling (default)
        t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
    noise = model.scheduler.randn_like(latents)

    # Extract embeddings if available (for cross-attention conditioning)
    text_emb = batch.get("text_emb", None)
    pov_emb = batch.get("pov_emb", None)
    
    # Get batch size from latents (ground truth for batch size)
    batch_size = latents.shape[0]
    
    # Ensure embeddings are 1D (flatten if needed) and match batch size
    if text_emb is not None:
        if text_emb.dim() > 2:
            text_emb = text_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
        # Ensure batch size matches latents
        if text_emb.shape[0] != batch_size:
            raise ValueError(f"text_emb batch size {text_emb.shape[0]} doesn't match latents batch size {batch_size}")
    if pov_emb is not None:
        if pov_emb.dim() > 2:
            pov_emb = pov_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
        # Ensure batch size matches latents
        if pov_emb.shape[0] != batch_size:
            raise ValueError(f"pov_emb batch size {pov_emb.shape[0]} doesn't match latents batch size {batch_size}")
    
    # Apply CFG dropout during training
    if cfg_dropout_rate > 0.0 and (text_emb is not None or pov_emb is not None):
        if torch.rand(1, device=device_obj).item() < cfg_dropout_rate:
            # When dropping condition, set both to zeros_like if they exist (do NOT set to None)
            if text_emb is not None:
                text_emb = torch.zeros_like(text_emb)
            if pov_emb is not None:
                pov_emb = torch.zeros_like(pov_emb)
            
    # Handle embedding projection requirements based on config
    # Only create zero tensors for embeddings that are configured but missing
    # Always use batch_size from latents to ensure consistency
    if hasattr(model, 'embedding_proj') and model.embedding_proj is not None:
        # Get dtype from model parameters for consistency (handles mixed precision)
        param_dtype = next(model.embedding_proj.parameters()).dtype
        
        # Get which embeddings are configured from trainer
        use_text_emb = getattr(trainer, 'use_text_emb', True)  # Default True for backward compatibility
        use_pov_emb = getattr(trainer, 'use_pov_emb', True)  # Default True for backward compatibility
        
        # Only create zero tensors if the embedding is configured but missing
        if use_text_emb and text_emb is None:
            # text_emb is configured but missing - create zero tensor
            text_emb = torch.zeros((batch_size, 384), device=device_obj, dtype=param_dtype)
        elif not use_text_emb:
            # text_emb is not configured - keep as None
            text_emb = None
        
        if use_pov_emb and pov_emb is None:
            # pov_emb is configured but missing - create zero tensor
            pov_emb = torch.zeros((batch_size, 512), device=device_obj, dtype=param_dtype)
        elif not use_pov_emb:
            # pov_emb is not configured - keep as None
            pov_emb = None
    
    # Forward pass and loss computation (Trainer handles scaling for gradient accumulation)
    from models.components.dataflow import DataFlow
    
    if trainer.use_amp:
        with torch.amp.autocast('cuda'):
            outputs = model(latents, t, noise=noise, text_emb=text_emb, pov_emb=pov_emb)
            # outputs is now a DataFlow, extract values
            if isinstance(outputs, DataFlow):
                preds = DataFlow({
                    "pred_noise": outputs["pred_noise"],
                    "scheduler": model.scheduler,
                    "timesteps": t,
                }, source_component="DiffusionModel")
            else:
                preds = DataFlow({
                    "pred_noise": outputs["pred_noise"],
                    "scheduler": model.scheduler,
                    "timesteps": t,
                }, source_component="DiffusionModel")
            targets = DataFlow({"noise": noise}, source_component="Dataset")
            # Loss functions work with DataFlow since it's dict-like
            loss, logs = loss_fn(preds, targets)
    else:
        outputs = model(latents, t, noise=noise, text_emb=text_emb, pov_emb=pov_emb)
        # outputs is now a DataFlow, extract values
        if isinstance(outputs, DataFlow):
            preds = DataFlow({
                "pred_noise": outputs["pred_noise"],
                "scheduler": model.scheduler,
                "timesteps": t,
            }, source_component="DiffusionModel")
        else:
            preds = DataFlow({
                "pred_noise": outputs["pred_noise"],
                "scheduler": model.scheduler,
                "timesteps": t,
            }, source_component="DiffusionModel")
        targets = DataFlow({"noise": noise}, source_component="Dataset")
        # Loss functions work with DataFlow since it's dict-like
        loss, logs = loss_fn(preds, targets)
    
    # Return outputs for potential collection (though diffusion doesn't collect latents)
    return loss, logs, {"latents": latents, "t": t, "noise": noise}


# Eval step function for Trainer
def diffusion_eval_step_fn(model, batch, batch_idx, loss_fn, trainer):
    """Step function for diffusion evaluation - computes loss only."""
    from models.components.dataflow import DataFlow
    device_obj = trainer.device
    
    # Get latents - must be provided in batch (pre-encoded)
    latents = batch.get("latent")
    if latents is None:
        raise ValueError("Dataset must provide 'latent' key. Encoding should be done before training.")
    
    # Sample random timesteps (uniform for evaluation)
    num_steps = model.scheduler.num_steps
    t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
    noise = model.scheduler.randn_like(latents)
    
    # Extract embeddings if available (for cross-attention conditioning)
    text_emb = batch.get("text_emb", None)
    pov_emb = batch.get("pov_emb", None)
    
    # Get batch size from latents (ground truth for batch size)
    batch_size = latents.shape[0]
    
    # Ensure embeddings are 1D (flatten if needed) and match batch size
    if text_emb is not None:
        if text_emb.dim() > 2:
            text_emb = text_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
        # Ensure batch size matches latents
        if text_emb.shape[0] != batch_size:
            raise ValueError(f"text_emb batch size {text_emb.shape[0]} doesn't match latents batch size {batch_size}")
    if pov_emb is not None:
        if pov_emb.dim() > 2:
            pov_emb = pov_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
        # Ensure batch size matches latents
        if pov_emb.shape[0] != batch_size:
            raise ValueError(f"pov_emb batch size {pov_emb.shape[0]} doesn't match latents batch size {batch_size}")
    
    # No CFG dropout during evaluation
    
    # Handle embedding projection requirements based on config
    # Only create zero tensors for embeddings that are configured but missing
    # Always use batch_size from latents to ensure consistency
    if hasattr(model, 'embedding_proj') and model.embedding_proj is not None:
        # Get dtype from model parameters for consistency (handles mixed precision)
        param_dtype = next(model.embedding_proj.parameters()).dtype
        
        # Get which embeddings are configured from trainer
        use_text_emb = getattr(trainer, 'use_text_emb', True)  # Default True for backward compatibility
        use_pov_emb = getattr(trainer, 'use_pov_emb', True)  # Default True for backward compatibility
        
        # Only create zero tensors if the embedding is configured but missing
        if use_text_emb and text_emb is None:
            # text_emb is configured but missing - create zero tensor
            text_emb = torch.zeros((batch_size, 384), device=device_obj, dtype=param_dtype)
        elif not use_text_emb:
            # text_emb is not configured - keep as None
            text_emb = None
        
        if use_pov_emb and pov_emb is None:
            # pov_emb is configured but missing - create zero tensor
            pov_emb = torch.zeros((batch_size, 512), device=device_obj, dtype=param_dtype)
        elif not use_pov_emb:
            # pov_emb is not configured - keep as None
            pov_emb = None
    
    # Forward pass and loss computation (no cfg_dropout during evaluation)
    if trainer.use_amp:
        with torch.amp.autocast('cuda'):
            outputs = model(latents, t, noise=noise, text_emb=text_emb, pov_emb=pov_emb)
            # outputs is now a DataFlow, extract values
            if isinstance(outputs, DataFlow):
                preds = DataFlow({
                    "pred_noise": outputs["pred_noise"],
                    "scheduler": model.scheduler,
                    "timesteps": t,
                }, source_component="DiffusionModel")
            else:
                preds = DataFlow({
                    "pred_noise": outputs["pred_noise"],
                    "scheduler": model.scheduler,
                    "timesteps": t,
                }, source_component="DiffusionModel")
            targets = DataFlow({"noise": noise}, source_component="Dataset")
            # Loss functions work with DataFlow since it's dict-like
            loss, logs = loss_fn(preds, targets)
    else:
        outputs = model(latents, t, noise=noise, text_emb=text_emb, pov_emb=pov_emb)
        # outputs is now a DataFlow, extract values
        if isinstance(outputs, DataFlow):
            preds = DataFlow({
                "pred_noise": outputs["pred_noise"],
                "scheduler": model.scheduler,
                "timesteps": t,
            }, source_component="DiffusionModel")
        else:
            preds = DataFlow({
                "pred_noise": outputs["pred_noise"],
                "scheduler": model.scheduler,
                "timesteps": t,
            }, source_component="DiffusionModel")
        targets = DataFlow({"noise": noise}, source_component="Dataset")
        # Loss functions work with DataFlow since it's dict-like
        loss, logs = loss_fn(preds, targets)
    
    # Return outputs for potential collection
    return loss, logs, {"latents": latents, "t": t, "noise": noise}


def save_targets_and_conditions(model, val_loader, device, output_dir, config, exp_name=None):
    """Save target images and their conditioning information once.
    
    Creates per-sample structure:
    samples/conditioned/sample_X/
        ├── conditions/  (text_emb, pov_emb, graph_text [if text_emb used], pov image [if pov_emb used])
        └── target/      (target image)
    
    Note: 
    - Original conditions (graph_text, pov image) are saved only for conditions actually used in the experiment.
    - Embeddings (text_emb, pov_emb) are only saved if they exist in the batch.
    
    Args:
        model: DiffusionModel
        val_loader: Validation dataloader
        device: Device string
        output_dir: Output directory
        config: Experiment configuration dict (must contain dataset.outputs)
        exp_name: Experiment name prefix
    """
    model.eval()
    samples_dir = output_dir / "samples"
    conditioned_dir = samples_dir / "conditioned"
    conditioned_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = to_device(device)
    
    # Check if targets already saved (check first sample)
    sample_0_dir = conditioned_dir / "sample_0"
    if sample_0_dir.exists() and (sample_0_dir / "target").exists():
        print("  Targets and conditions already saved, skipping...")
        return
    
    try:
        batch_iter = iter(val_loader)
        batch = next(batch_iter)
    except StopIteration:
        return
    
    # Get dataset to find rooms and scenes
    dataset = val_loader.dataset
    
    # Get number of samples per type from config (default: 4)
    num_samples_per_type = config.get("training", {}).get("num_conditioned_samples_per_type", 4)
    
    # Select samples from the filtered dataset
    # The dataset is already filtered by config (type, rejected, etc.)
    # For "both" experiments (type filter is empty), take num_samples_per_type of each type
    # For single-type experiments, just take num_samples_per_type
    selected_indices = []
    
    if hasattr(dataset, 'df') and 'type' in dataset.df.columns:
        room_indices = []
        scene_indices = []
        
        # Collect indices by type from the filtered dataset
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]
            sample_type = str(row.get('type', '')).lower().strip()
            if sample_type == 'room':
                room_indices.append(idx)
            elif sample_type == 'scene':
                scene_indices.append(idx)
        
        # If both types exist in filtered dataset, take num_samples_per_type of each
        if len(room_indices) > 0 and len(scene_indices) > 0:
            selected_indices = room_indices[:num_samples_per_type] + scene_indices[:num_samples_per_type]
        elif len(room_indices) > 0:
            selected_indices = room_indices[:num_samples_per_type]
        elif len(scene_indices) > 0:
            selected_indices = scene_indices[:num_samples_per_type]
        else:
            selected_indices = list(range(min(num_samples_per_type, len(dataset))))
    else:
        selected_indices = list(range(min(num_samples_per_type, len(dataset))))
    
    batch_size = len(selected_indices)
    
    if batch_size == 0:
        print("  Warning: No samples found in validation dataset")
        return
    
    # Load selected samples from dataset
    batch_data = {}
    for idx in selected_indices:
        sample = dataset[idx]
        for key, value in sample.items():
            if key not in batch_data:
                batch_data[key] = []
            batch_data[key].append(value)
    
    # Convert lists to tensors
    batch = {}
    for key, values in batch_data.items():
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    
    batch = move_batch_to_device(batch, device_obj, non_blocking=False)
    
    # Extract embeddings and latents
    text_emb = batch.get("text_emb", None)
    pov_emb = batch.get("pov_emb", None)
    target_latents = batch.get("latent", None)
    
    if target_latents is None:
        return
    
    # Flatten embeddings if needed and validate dimensions
    # Only save embeddings that actually exist (don't save zero tensors for missing ones)
    if text_emb is not None:
        if text_emb.dim() > 2:
            text_emb = text_emb.flatten(start_dim=1)
        # Validate text_emb has correct shape [B, 384]
        if text_emb.shape[1] != 384:
            print(f"  Warning: text_emb has wrong dimension ({text_emb.shape[1]}), expected 384. Skipping text_emb save.")
            text_emb = None
        # Don't save if it's all zeros (missing condition)
        elif text_emb.abs().max().item() < 1e-6:
            text_emb = None
    
    if pov_emb is not None:
        if pov_emb.dim() > 2:
            pov_emb = pov_emb.flatten(start_dim=1)
        # Validate pov_emb has correct shape [B, 512]
        if pov_emb.shape[1] != 512:
            print(f"  Warning: pov_emb has wrong dimension ({pov_emb.shape[1]}), expected 512. Skipping pov_emb save.")
            pov_emb = None
        # Don't save if it's all zeros (missing condition)
        elif pov_emb.abs().max().item() < 1e-6:
            pov_emb = None
    
    # Decode target latents to RGB
    with torch.no_grad():
        target_output = {"latent": target_latents}
        target_rgb = latents2rgb(model, target_output, warning_prefix="Decoder for target latents")
    
    if target_rgb is None:
        return
    
    # Convert to numpy for saving
    target_np = (target_rgb.cpu().numpy() * 255.0).astype(np.uint8)
    
    # Determine which conditions are configured in the experiment
    dataset_outputs = config.get("dataset", {}).get("outputs", {})
    use_text_emb = "text_emb" in dataset_outputs
    use_pov_emb = "pov_emb" in dataset_outputs
    
    # Save per-sample structure
    for i in range(batch_size):
        sample_dir = conditioned_dir / f"sample_{i}"
        conditions_dir = sample_dir / "conditions"
        target_dir = sample_dir / "target"
        
        conditions_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save target image
        target_img = Image.fromarray(target_np[i].transpose(1, 2, 0))
        target_img.save(target_dir / "target.png")
        
        # Save embeddings (per-sample, keep batch dimension for consistency)
        # Only save if they exist (don't save zero tensors for missing conditions)
        if text_emb is not None and text_emb.shape[1] == 384:
            torch.save(text_emb[i:i+1].cpu(), conditions_dir / "text_embedding.pt")
        if pov_emb is not None and pov_emb.shape[1] == 512:
            torch.save(pov_emb[i:i+1].cpu(), conditions_dir / "pov_embedding.pt")
        
        # Save original conditions (only for conditions actually used in the experiment)
        idx = selected_indices[i] if i < len(selected_indices) else i
        row = dataset.df.iloc[idx]
        
        # Save graph_text if text_emb is used in the experiment
        if use_text_emb:
            graph_text_path = row.get("graph_text_path", "")
            if graph_text_path:
                # Resolve path relative to manifest directory (same logic as dataset)
                if hasattr(dataset, 'manifest_dir'):
                    resolved_path = dataset.manifest_dir / graph_text_path
                    if not resolved_path.exists():
                        resolved_path = dataset.manifest_dir.parent / graph_text_path
                else:
                    resolved_path = Path(graph_text_path)
                
                if resolved_path.exists():
                    try:
                        with open(resolved_path, 'r') as f:
                            graph_text = f.read()
                        with open(conditions_dir / "graph_text.txt", 'w') as f:
                            f.write(graph_text)
                    except Exception:
                        pass
        
        # Save POV image if pov_emb is used in the experiment
        if use_pov_emb:
            pov_path = row.get("pov_path", "")
            if pov_path:
                # Resolve path relative to manifest directory (same logic as dataset)
                if hasattr(dataset, 'manifest_dir'):
                    resolved_path = dataset.manifest_dir / pov_path
                    if not resolved_path.exists():
                        resolved_path = dataset.manifest_dir.parent / pov_path
                else:
                    resolved_path = Path(pov_path)
                
                if resolved_path.exists():
                    try:
                        pov_img = Image.open(resolved_path)
                        pov_img.save(conditions_dir / "pov.png")
                    except Exception:
                        pass
    
    # Collect type information for each selected sample
    sample_types = []
    if hasattr(dataset, 'df') and 'type' in dataset.df.columns:
        for idx in selected_indices:
            row = dataset.df.iloc[idx]
            sample_type = str(row.get('type', '')).lower().strip()
            sample_types.append(sample_type)
    else:
        sample_types = ['unknown'] * batch_size
    
    # Save global metadata
    metadata = {
        "batch_size": batch_size,
        "selected_indices": selected_indices,
        "sample_types": sample_types,
        "num_samples_per_type": num_samples_per_type,
        "has_text_emb": text_emb is not None,
        "has_pov_emb": pov_emb is not None,
        "use_text_emb": use_text_emb,
        "use_pov_emb": use_pov_emb,
    }
    with open(conditioned_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved targets and conditions to {conditioned_dir}")


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=16, exp_name=None, guidance_scale=1.0, cfg_dropout_rate=0.0, config=None):
    """Generate and save sample images.
    
    Generates:
    - Unconditioned samples: 4x4 grid (16 samples) - saved every sample interval
    - Generated samples: conditioned generation - saved every sample interval
    
    Note: Targets and conditions should be saved once using save_targets_and_conditions()
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
    
    # Use 50 steps for DDIM sampling (DDIM is designed for fewer steps)
    ddim_steps = 50
    
    # ============================================================================
    # Part 1: Generate unconditioned samples (4x4 grid)
    # ============================================================================
    # Get sampling seed from config (default: 42 + epoch for reproducibility)
    if config is not None:
        sampling_seed_base = config.get("training", {}).get("sampling_seed_base", 42)
    else:
        sampling_seed_base = 42
    sampling_seed = sampling_seed_base + epoch
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)
    
    # CRITICAL: Use zero tensors instead of None for unconditioned sampling, matching training logic
    # This ensures embedding_proj receives proper tensors even for unconditional generation
    unconditioned_batch_size = 16
    # Get dtype from model parameters for consistency
    param_dtype = next(model.parameters()).dtype
    
    # Create zero tensors with explicit dimensions: 384 for text, 512 for POV
    dummy_text = torch.zeros((unconditioned_batch_size, 384), device=device_obj, dtype=param_dtype)
    dummy_pov = torch.zeros((unconditioned_batch_size, 512), device=device_obj, dtype=param_dtype)
    
    with torch.no_grad():
        unconditioned_output = model.sample(
            batch_size=unconditioned_batch_size,  # Keep 4x4 grid for unconditioned
            num_steps=ddim_steps,
            method="ddim",
            eta=0.0,
            guidance_scale=1.0,
            text_emb=dummy_text,
            pov_emb=dummy_pov,
            device=device_obj,
            verbose=False
        )
        
        # Decode unconditioned samples
        unconditioned_rgb = latents2rgb(model, unconditioned_output, warning_prefix="Decoder for unconditioned samples")
    
    # Save 4x4 unconditioned grid
    if unconditioned_rgb is not None:
        unconditioned_np = (unconditioned_rgb.cpu().numpy() * 255.0).astype(np.uint8)
        unconditioned_images = []
        for i in range(16):
            img = Image.fromarray(unconditioned_np[i].transpose(1, 2, 0))
            unconditioned_images.append(img)
        
        img_size = unconditioned_images[0].size[0]
        grid_n = 4  # 4x4 grid
        unconditioned_grid = Image.new('RGB', (img_size * grid_n, img_size * grid_n))
        for idx, img in enumerate(unconditioned_images):
            row = idx // grid_n
            col = idx % grid_n
            unconditioned_grid.paste(img, (col * img_size, row * img_size))
        
        # Save to unconditioned directory
        unconditioned_dir = samples_dir / "unconditioned"
        unconditioned_dir.mkdir(parents=True, exist_ok=True)
        unconditioned_path = unconditioned_dir / f"epoch_{epoch:03d}.png"
        unconditioned_grid.save(unconditioned_path)

    # ============================================================================
    # Part 2: Generate conditioned samples (saved every sample interval)
    # ============================================================================
    # Load saved targets and conditions
    conditioned_dir = samples_dir / "conditioned"
    
    if not conditioned_dir.exists():
        print("  Warning: Conditioned samples directory not found. Run save_targets_and_conditions() first.")
        return
    
    # Load metadata to get batch size
    metadata_path = conditioned_dir / "metadata.json"
    if not metadata_path.exists():
        print("  Warning: Conditioned samples metadata not found.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    batch_size = metadata["batch_size"]
    
    # Get total number of samples to generate from config (default: 16)
    # This is the TOTAL number, not per condition
    if config is not None:
        total_samples = config.get("training", {}).get("num_conditioned_samples_per_type", 16)
    else:
        total_samples = 16
    
    # Distribute samples across conditions
    # If we have fewer conditions than total_samples, use all conditions and repeat
    # If we have more conditions, use first total_samples conditions
    num_conditions_to_use = min(batch_size, total_samples)
    samples_per_condition = max(1, total_samples // num_conditions_to_use)
    # Adjust if we need to distribute remainder
    remainder = total_samples % num_conditions_to_use
    samples_per_condition_list = [samples_per_condition] * num_conditions_to_use
    for i in range(remainder):
        samples_per_condition_list[i] += 1
    
    print(f"  Generating {total_samples} total samples across {num_conditions_to_use} conditions")
    print(f"  Samples per condition: {samples_per_condition_list}")
    
    # Load embeddings (batch-level for generation)
    text_emb_list = []
    pov_emb_list = []
    for i in range(num_conditions_to_use):
        sample_dir = conditioned_dir / f"sample_{i}"
        conditions_dir = sample_dir / "conditions"
        
        text_emb_path = conditions_dir / "text_embedding.pt"
        pov_emb_path = conditions_dir / "pov_embedding.pt"
        
        if text_emb_path.exists():
            text_emb_list.append(torch.load(text_emb_path, map_location=device_obj))
        if pov_emb_path.exists():
            pov_emb_list.append(torch.load(pov_emb_path, map_location=device_obj))
    
    # Stack embeddings back to batch and validate dimensions
    # If file doesn't exist, leave as None (don't create zero tensors here)
    if text_emb_list:
        text_emb_single = torch.cat(text_emb_list, dim=0)
        # Validate text_emb has correct shape [B, 384]
        if text_emb_single.shape[1] != 384:
            print(f"  Warning: text_emb had wrong dimension ({text_emb_single.shape[1]}), expected 384. Skipping text_emb.")
            text_emb = None
        # Skip if it's all zeros (was a placeholder for missing condition)
        elif text_emb_single.abs().max().item() < 1e-6:
            text_emb = None
        else:
            # Repeat each condition according to samples_per_condition_list
            text_emb_repeated = []
            for i, num_samples in enumerate(samples_per_condition_list):
                text_emb_repeated.append(text_emb_single[i:i+1].repeat(num_samples, 1))
            text_emb = torch.cat(text_emb_repeated, dim=0)
    else:
        text_emb = None  # Don't create zero tensor - let model handle it if needed
    
    if pov_emb_list:
        pov_emb_single = torch.cat(pov_emb_list, dim=0)
        # Validate pov_emb has correct shape [B, 512]
        if pov_emb_single.shape[1] != 512:
            print(f"  Warning: pov_emb had wrong dimension ({pov_emb_single.shape[1]}), expected 512. Skipping pov_emb.")
            pov_emb = None
        # Skip if it's all zeros (was a placeholder for missing condition)
        elif pov_emb_single.abs().max().item() < 1e-6:
            pov_emb = None
        else:
            # Repeat each condition according to samples_per_condition_list
            pov_emb_repeated = []
            for i, num_samples in enumerate(samples_per_condition_list):
                pov_emb_repeated.append(pov_emb_single[i:i+1].repeat(num_samples, 1))
            pov_emb = torch.cat(pov_emb_repeated, dim=0)
    else:
        pov_emb = None  # Don't create zero tensor - let model handle it if needed
    
    # Set seed for conditioned sample generation (use same base seed for consistency)
    if config is not None:
        sampling_seed_base = config.get("training", {}).get("sampling_seed_base", 42)
    else:
        sampling_seed_base = 42
    conditioned_sampling_seed = sampling_seed_base + epoch + 1000  # Offset to ensure different from unconditioned
    torch.manual_seed(conditioned_sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conditioned_sampling_seed)
    
    # Generate conditioned samples using DDIM (50 steps)
    # Process in chunks to avoid OOM - max batch size per chunk
    max_chunk_size = 32  # Process up to 32 samples at a time
    ddim_steps = 50
    
    # Load target images for comparison grids (only for conditions we're using)
    target_images = []
    for i in range(num_conditions_to_use):
        sample_dir = conditioned_dir / f"sample_{i}"
        target_path = sample_dir / "target" / "target.png"
        if target_path.exists():
            target_images.append(Image.open(target_path))
        else:
            print(f"  Warning: Target image for sample {i} not found")
            return
    
    # Process in chunks to avoid OOM
    all_generated_images = []
    
    for chunk_start in range(0, total_samples, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, total_samples)
        chunk_size = chunk_end - chunk_start
        
        # Extract embeddings for this chunk
        chunk_text_emb = None
        chunk_pov_emb = None
        
        if text_emb is not None:
            chunk_text_emb = text_emb[chunk_start:chunk_end]
        
        if pov_emb is not None:
            chunk_pov_emb = pov_emb[chunk_start:chunk_end]
        
        print(f"  Generating samples {chunk_start}-{chunk_end-1} of {total_samples} (batch size: {chunk_size})...")
        
        with torch.no_grad():
            chunk_output = model.sample(
                batch_size=chunk_size,
                num_steps=ddim_steps,
                method="ddim",
                eta=0.0,
                guidance_scale=guidance_scale,
                text_emb=chunk_text_emb,
                pov_emb=chunk_pov_emb,
                device=device_obj,
                verbose=False
            )
            
            # Decode generated latents to RGB
            chunk_rgb = latents2rgb(model, chunk_output, warning_prefix=f"Decoder for chunk {chunk_start}-{chunk_end-1}")
        
        if chunk_rgb is None:
            print(f"  Warning: Failed to decode chunk {chunk_start}-{chunk_end-1}")
            continue
        
        # Convert chunk to images
        chunk_np = (chunk_rgb.cpu().numpy() * 255.0).astype(np.uint8)
        for i in range(chunk_size):
            generated_img = Image.fromarray(chunk_np[i].transpose(1, 2, 0))
            all_generated_images.append(generated_img)
        
        # Clear GPU memory
        del chunk_output, chunk_rgb, chunk_np
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if len(all_generated_images) == 0:
        print("  Error: No samples were generated")
        return
    
    # Save generated images - create separate top-level folders for each generated sample
    sample_idx = 0
    for condition_idx in range(num_conditions_to_use):
        num_samples_for_this_condition = samples_per_condition_list[condition_idx]
        for _ in range(num_samples_for_this_condition):
            if sample_idx < len(all_generated_images):
                # Create separate folder for each generated sample at top level
                # Numbering: sample_0 to sample_N-1 are original conditions, sample_N onwards are generated
                generated_sample_num = batch_size + sample_idx
                generated_sample_dir = conditioned_dir / f"sample_{generated_sample_num}"
                generated_sample_dir.mkdir(parents=True, exist_ok=True)
                
                generated_img = all_generated_images[sample_idx]
                generated_img.save(generated_sample_dir / f"epoch_{epoch:03d}.png")
                sample_idx += 1
    
    # Create comparison grids for easy viewing
    # Show one generated sample per condition (first sample, index 0)
    img_size = target_images[0].size[0]
    grid_n = 4  # 4 columns
    num_rows = (num_conditions_to_use + grid_n - 1) // grid_n
    
    # Create target grid
    target_grid = Image.new('RGB', (img_size * grid_n, img_size * num_rows))
    for idx, img in enumerate(target_images):
        row = idx // grid_n
        col = idx % grid_n
        target_grid.paste(img, (col * img_size, row * img_size))
    
    # Create generated grid - use first generated sample for each condition
    generated_grid = Image.new('RGB', (img_size * grid_n, img_size * num_rows))
    sample_idx = 0
    for condition_idx in range(num_conditions_to_use):
        if sample_idx < len(all_generated_images):
            generated_img = all_generated_images[sample_idx]
            row = condition_idx // grid_n
            col = condition_idx % grid_n
            generated_grid.paste(generated_img, (col * img_size, row * img_size))
            # Skip to next condition's first sample
            sample_idx += samples_per_condition_list[condition_idx]
    
    # Concatenate horizontally (side by side) for comparison
    comparison_width = img_size * grid_n * 2
    comparison_height = img_size * num_rows
    comparison_img = Image.new('RGB', (comparison_width, comparison_height))
    comparison_img.paste(target_grid, (0, 0))
    comparison_img.paste(generated_grid, (img_size * grid_n, 0))
    
    # Save comparison grid
    comparison_dir = samples_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / f"comparison_epoch_{epoch:03d}.png"
    comparison_img.save(comparison_path)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model (unified for all stages)")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--no-resume", action="store_true", help="Force start from scratch")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed from config")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    
    # Set deterministic behavior
    # Use --seed argument if provided, otherwise use config value
    training_seed = args.seed if args.seed is not None else config.get("training", {}).get("seed", None)
    if training_seed is not None:
        set_deterministic(training_seed)
        # Update config in memory so it's reflected in checkpoints/logs
        if "training" not in config:
            config["training"] = {}
        config["training"]["seed"] = training_seed
    
    # Get device
    device = get_device(config)
    
    # Get output directory
    output_dir = config.get("experiment", {}).get("save_path")
    if output_dir is None:
        output_dir = Path("outputs") / exp_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for latest checkpoint (in checkpoints subdirectory)
    checkpoint_dir = output_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    
    # Build dataset
    dataset = build_dataset(config)
    
    # Build validation dataset
    train_dataset, val_dataset = split_dataset(dataset, config["training"])
    
    # Save train/val split indices
    if hasattr(train_dataset, 'df'):
        train_indices = train_dataset.df.index.tolist()
        train_indices_path = output_dir / "train_indices.json"
        with open(train_indices_path, 'w') as f:
            json.dump(train_indices, f)
        print(f"Saved training indices ({len(train_indices)} samples) to {train_indices_path}")
    
    if val_dataset and hasattr(val_dataset, 'df'):
        val_indices = val_dataset.df.index.tolist()
        val_indices_path = output_dir / "val_indices.json"
        with open(val_indices_path, 'w') as f:
            json.dump(val_indices, f)
        print(f"Saved validation indices ({len(val_indices)} samples) to {val_indices_path}")
    
    # Save filtered train/val manifests
    if hasattr(train_dataset, 'df'):
        train_manifest_path = output_dir / "manifest_train.csv"
        train_dataset.df.to_csv(train_manifest_path, index=False)
        print(f"Saved training manifest ({len(train_dataset.df)} samples) to {train_manifest_path}")
    
    if val_dataset and hasattr(val_dataset, 'df'):
        val_manifest_path = output_dir / "manifest_val.csv"
        val_dataset.df.to_csv(val_manifest_path, index=False)
        print(f"Saved validation manifest ({len(val_dataset.df)} samples) to {val_manifest_path}")
    
    device_obj = to_device(device)
    
    # Try to load VAE metadata first (if autoencoder checkpoint is specified)
    vae_metadata = None
    ae_cfg = config.get("autoencoder") or config.get("diffusion", {}).get("autoencoder")
    if ae_cfg and isinstance(ae_cfg, dict):
        ae_checkpoint = ae_cfg.get("checkpoint")
        if ae_checkpoint:
            from training.utils import load_vae_metadata
            vae_metadata = load_vae_metadata(ae_checkpoint)
    
    # Get scale_factor from config or VAE metadata (should be part of VAE statistics)
    # Check both diffusion section and top-level config
    diffusion_cfg = config.get("diffusion", {})
    if not diffusion_cfg:
        diffusion_cfg = {}
    
    # Priority: config > VAE metadata
    scale_factor = diffusion_cfg.get("scale_factor") or config.get("scale_factor")
    
    if scale_factor is None:
        # Try VAE metadata
        if vae_metadata and vae_metadata.get("scale_factor") is not None:
            scale_factor = vae_metadata["scale_factor"]
            # Add to config for model building
            if "diffusion" in config:
                config["diffusion"]["scale_factor"] = scale_factor
            else:
                config["scale_factor"] = scale_factor
        else:
            scale_factor = 1.0
            if "diffusion" in config:
                config["diffusion"]["scale_factor"] = scale_factor
            else:
                config["scale_factor"] = scale_factor
    
    # Load latent_clamp values (priority: config > VAE metadata > defaults)
    latent_clamp_min = config.get("latent_clamp_min")
    latent_clamp_max = config.get("latent_clamp_max")
    
    if latent_clamp_min is None and vae_metadata and vae_metadata.get("latent_clamp_min") is not None:
        latent_clamp_min = vae_metadata["latent_clamp_min"]
        config["latent_clamp_min"] = latent_clamp_min
    
    if latent_clamp_max is None and vae_metadata and vae_metadata.get("latent_clamp_max") is not None:
        latent_clamp_max = vae_metadata["latent_clamp_max"]
        config["latent_clamp_max"] = latent_clamp_max
    
    # Check if we should resume or start fresh
    should_resume = not args.no_resume and latest_checkpoint.exists()
    extra_state = {}  # Initialize empty dict for non-resume case
    
    # Load checkpoint (resume) or build from config (fresh start)
    if should_resume:
        # Load checkpoint manually (before Trainer is created)
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
        
        if not training_history:
            from training.utils import load_training_history_from_csv
            training_history = load_training_history_from_csv(metrics_csv_path, start_epoch)
        
        # Check if we need to continue training beyond the checkpoint
        epochs = config["training"].get("epochs", 100)
        if start_epoch >= epochs:
            print(f"WARNING: Checkpoint is at epoch {start_epoch + 1}, but config specifies only {epochs} epochs.")
            print(f"Training is already complete. To continue training, increase 'epochs' in config.")
    else:
        # Build model from config (fresh start)
        diffusion_cfg = config.get("diffusion", {})
        
        # If no diffusion section, extract from top-level config (for ablation configs)
        if not diffusion_cfg:
            diffusion_cfg = {
                "autoencoder": config.get("autoencoder"),
                "unet": config.get("unet", {}),
                "scheduler": config.get("scheduler", {}),
                "embedding_projection": config.get("embedding_projection")
            }
        else:
            # Ensure embedding_projection is included if it exists at top level
            if "embedding_projection" not in diffusion_cfg and "embedding_projection" in config:
                diffusion_cfg["embedding_projection"] = config.get("embedding_projection")
        
        # Add scale_factor if it was calculated or provided
        if scale_factor is not None:
            diffusion_cfg["scale_factor"] = scale_factor
        
        # Add latent_clamp values if available
        if latent_clamp_min is not None:
            diffusion_cfg["latent_clamp_min"] = latent_clamp_min
        if latent_clamp_max is not None:
            diffusion_cfg["latent_clamp_max"] = latent_clamp_max
        
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
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    # CRITICAL: Ensure UNet is trainable (explicitly set requires_grad=True)
    # This overrides any frozen settings from config or checkpoint
    if hasattr(model, 'unet'):
        for param in model.unet.parameters():
            if not param.requires_grad:
                param.requires_grad = True  # Force trainable
    
    # Build data loaders
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 8)
    shuffle = config["training"].get("shuffle", True)
    
    # Weighted sampling options
    use_weighted_sampling = config["training"].get("use_weighted_sampling", False)
    use_precomputed_weights = config["training"].get("use_precomputed_weights", False)
    precomputed_weight_column = config["training"].get("precomputed_weight_column", "sample_weight")
    max_weight = config["training"].get("max_weight", None)
    
    # Auto-generate weight stats if using column-based weighting (legacy approach)
    weights_stats_path = None
    if use_weighted_sampling and not use_precomputed_weights:
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
                max_weight=max_weight,
                min_weight=config["training"].get("min_weight", 1.0),
                filters=dataset_filters  # Apply same filters as dataset
            )
    
    # Use dataset's make_dataloader to support weighted sampling
    train_loader = train_dataset.make_dataloader(
        batch_size=batch_size,
        shuffle=shuffle if not (use_weighted_sampling or use_precomputed_weights) else False,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
        persistent_workers=num_workers > 0,
        # Precomputed weights (new approach)
        use_precomputed_weights=use_precomputed_weights,
        precomputed_weight_column=precomputed_weight_column,
        # Column-based weights (legacy approach)
        use_weighted_sampling=use_weighted_sampling and not use_precomputed_weights,
        weight_column=config["training"].get("weight_column", None) or config["training"].get("column", None),
        weights_stats_path=weights_stats_path,
        use_grouped_weights=config["training"].get("use_grouped_weights", False),
        group_rare_classes=config["training"].get("group_rare_classes", False),
        class_grouping_path=config["training"].get("class_grouping_path", None),
        max_weight=max_weight,
        exclude_extremely_rare=config["training"].get("exclude_extremely_rare", False),
        min_samples_threshold=config["training"].get("min_samples_threshold", 50)
    )
    
    # Verify if weighted sampling is actually active
    from torch.utils.data import WeightedRandomSampler
    is_using_weights = hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, WeightedRandomSampler)
    print(f"\n{'='*60}")
    print(f"WEIGHTED SAMPLING STATUS")
    print(f"{'='*60}")
    print(f"  Config: use_precomputed_weights = {use_precomputed_weights}")
    print(f"  Config: precomputed_weight_column = {precomputed_weight_column}")
    print(f"  Actual: WeightedRandomSampler active = {is_using_weights}")
    if is_using_weights:
        print(f"  ✓ Precomputed weights ARE being used for training!")
        if hasattr(train_loader.sampler, 'weights'):
            weight_tensor = train_loader.sampler.weights
            print(f"  Weight stats: min={weight_tensor.min():.4f}, max={weight_tensor.max():.4f}, mean={weight_tensor.mean():.4f}")
    else:
        print(f"  ✗ Precomputed weights are NOT being used (regular random sampling)")
    print(f"{'='*60}\n")
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
    
    # Build loss function from config (uses CompositeLoss)
    loss_fn = build_loss(config)
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Verify optimizer has trainable parameters
    if hasattr(model, 'unet'):
        unet_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        if unet_trainable == 0:
            print("ERROR: UNet has no trainable parameters! Training will not work!")
    
    # Build learning rate scheduler (account for already-trained epochs when resuming)
    # Use last_epoch=-1 for fresh start, or start_epoch for resume
    last_epoch = start_epoch if start_epoch > 0 else -1
    scheduler = build_scheduler(optimizer, config, last_epoch=last_epoch)
    
    # Training settings
    epochs = config["training"].get("epochs", 500)
    use_amp = config["training"].get("use_amp", True)  # Default to True for speedup and memory efficiency
    max_grad_norm = config["training"].get("max_grad_norm", None)
    eval_interval = config["training"].get("eval_interval", 5)
    sample_interval = config["training"].get("sample_interval", 10)
    save_interval = config["training"].get("save_interval", 1)  # Default to 1 (every epoch) for backward compatibility
    use_non_uniform_sampling = config["training"].get("use_non_uniform_sampling", False)  # Default False for uniform sampling
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    gradient_accumulation_steps = config.get("training", {}).get("gradient_accumulation_steps", 1)
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device_obj,
        use_amp=use_amp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
    )
    # Store diffusion-specific config in trainer for step function access
    trainer.use_non_uniform_sampling = use_non_uniform_sampling
    
    # Determine which conditions are configured in the experiment
    dataset_outputs = config.get("dataset", {}).get("outputs", {})
    trainer.use_text_emb = "text_emb" in dataset_outputs
    trainer.use_pov_emb = "pov_emb" in dataset_outputs
    
    # Restore optimizer and scheduler state if resuming
    if should_resume and "optimizer_state" in extra_state:
        trainer.optimizer.load_state_dict(extra_state["optimizer_state"])
    if should_resume and "scheduler_state" in extra_state and scheduler is not None:
        trainer.scheduler.load_state_dict(extra_state["scheduler_state"])
    if should_resume and "scaler_state" in extra_state and trainer.scaler is not None:
        trainer.scaler.load_state_dict(extra_state["scaler_state"])
    
    # Early stopping state (restore from checkpoint if resuming)
    if should_resume:
        # Try to restore epochs_without_improvement from checkpoint, but it might not be saved
        epochs_without_improvement = extra_state.get("epochs_without_improvement", 0)
    else:
        epochs_without_improvement = 0
    
    # Get CFG dropout rate (constant, no scheduling)
    cfg_dropout_rate = config.get("training", {}).get("cfg_dropout_rate", 0.0)
    if isinstance(cfg_dropout_rate, dict):
        # Legacy dict format - use end rate as constant
        cfg_dropout_rate = cfg_dropout_rate.get("end", 0.0)
    guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
    
    # Set CFG dropout rate in trainer (constant throughout training)
    trainer.cfg_dropout_rate = cfg_dropout_rate
    
    # Print conditioning configuration for verification
    print(f"\n[CONDITIONING CONFIG]")
    print(f"  text_emb: {'ENABLED' if trainer.use_text_emb else 'DISABLED'}")
    print(f"  pov_emb:  {'ENABLED' if trainer.use_pov_emb else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_logs = trainer.train_epoch(
            dataloader=train_loader,
            loss_fn=loss_fn,
            epoch=epoch + 1,
            step_fn=diffusion_step_fn,
            collect_fn=None  # Diffusion doesn't collect latents
        )
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}", flush=True)
        
        # Validate
        val_loss = float("inf")
        val_logs = {}
        # Always evaluate at epoch 1, then according to eval_interval
        should_eval = val_loader and ((epoch + 1 == 1) or ((epoch + 1) % eval_interval == 0))
        if should_eval:
            # Get guidance_scale from config for evaluation
            guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
            
            val_loss, val_logs = trainer.eval_epoch(
                dataloader=val_loader,
                loss_fn=loss_fn,
                step_fn=diffusion_eval_step_fn,
                collect_fn=None,  # Diffusion doesn't collect latents
                limit_batches=50
            )
            
            # Check if this is the best validation loss BEFORE updating best_val_loss
            is_best = val_loss < best_val_loss
            
            # Early stopping logic
            if early_stopping_patience is not None:
                improvement = best_val_loss - val_loss
                if improvement > early_stopping_min_delta:
                    epochs_without_improvement = 0
                    if is_best:
                        best_val_loss = val_loss
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            elif is_best:
                # Update best_val_loss if not using early stopping
                best_val_loss = val_loss
        
        # Save targets and conditions once (at epoch 1)
        if val_loader and (epoch + 1 == 1):
            save_targets_and_conditions(model, val_loader, device_obj, output_dir, config, exp_name=exp_name)
        
        # Save generated samples every sample_interval epochs
        if val_loader and ((epoch + 1 == 1) or ((epoch + 1) % sample_interval == 0)):
            # Get guidance_scale from config (default 1.0 = no CFG)
            guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
            save_samples(model, val_loader, device_obj, output_dir, epoch + 1, sample_batch_size=64, exp_name=exp_name, guidance_scale=guidance_scale, cfg_dropout_rate=cfg_dropout_rate, config=config)
        
        # Save checkpoint (is_best was already determined above if validation ran)
        # Use same condition as evaluation: always at epoch 1, then according to eval_interval
        if should_eval:
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
        trainer.save_metrics_csv(metrics_csv_path, training_history)
        
        # Plot loss curves
        if len(training_history) > 0:
            try:
                df = pd.DataFrame(training_history)
                plot_loss_curves(df, output_dir, exp_name=exp_name)
            except Exception:
                pass
        
        # Save periodic checkpoint at specified interval
        should_save_periodic = (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs
        if should_save_periodic:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            periodic_checkpoint_path = checkpoint_dir / f"{exp_name}_checkpoint_epoch_{epoch + 1:03d}.pt"
            
            # Prepare extra state for model.save_checkpoint
            extra_state = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "training_history": training_history,
                "optimizer_state": trainer.optimizer.state_dict(),
            }
            
            if trainer.scheduler is not None:
                extra_state["scheduler_state"] = trainer.scheduler.state_dict()
            
            if trainer.scaler is not None:
                extra_state["scaler_state"] = trainer.scaler.state_dict()
            
            # Save periodic checkpoint
            model.save_checkpoint(periodic_checkpoint_path, include_config=True, exclude_projections=True, **extra_state)
            print(f"Saved periodic checkpoint: {periodic_checkpoint_path}")
        
        # Save checkpoint using Trainer (always saves latest for resume, and best when applicable)
        trainer.save_training_checkpoint(
            output_dir=output_dir,
            exp_name=exp_name,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            training_history=training_history,
            is_best=is_best
            )
        
        # Early stopping check
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()