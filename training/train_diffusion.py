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
import math
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
from training.plotting_utils import plot_diffusion_metrics_epochs, plot_evaluation_metrics
from training.evaluation_metrics import compute_evaluation_metrics
from models.diffusion import DiffusionModel
from models.losses.base_loss import LOSS_REGISTRY


def load_vae_metadata(ae_checkpoint_path):
    """
    Load VAE metadata JSON file and extract recommended values.
    
    Args:
        ae_checkpoint_path: Path to autoencoder checkpoint
        
    Returns:
        dict with 'scale_factor', 'latent_clamp_min', 'latent_clamp_max', or None if not found
    """
    import json
    from pathlib import Path
    
    checkpoint_path = Path(ae_checkpoint_path)
    if not checkpoint_path.exists():
        return None
    
    # Metadata file is typically in the parent directory of checkpoints/
    # e.g., /work3/.../vae_clip/checkpoints/vae_clip_checkpoint_best.pt
    # -> /work3/.../vae_clip/vae_clip_metadata.json
    checkpoint_dir = checkpoint_path.parent  # checkpoints/
    vae_dir = checkpoint_dir.parent  # vae_clip/
    
    # Try to find metadata file - could be named {exp_name}_metadata.json
    # Look for any *_metadata.json in the VAE directory
    metadata_files = list(vae_dir.glob("*_metadata.json"))
    
    if not metadata_files:
        return None
    
    # Use the first metadata file found (or could match by experiment name)
    metadata_path = metadata_files[0]
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        recommended = metadata.get("recommended_values", {})
        if recommended:
            return {
                "scale_factor": recommended.get("scale_factor"),
                "latent_clamp_min": recommended.get("latent_clamp_min"),
                "latent_clamp_max": recommended.get("latent_clamp_max")
            }
    except Exception as e:
        print(f"Warning: Failed to load VAE metadata from {metadata_path}: {e}")
    
    return None






def compute_loss(
    model, batch, latents, t, noise, cond, loss_fn, 
    use_amp=False, device_obj=None, cfg_dropout_rate=0.0, compute_eval_metrics=False, taxonomy=None
):
    """
    Compute loss using CompositeLoss from config.
    
    This function prepares preds and targets dictionaries that are passed to
    CompositeLoss, which then distributes them to each sub-loss component.
    
    Loss components supported:
    - MSELoss: preds["pred_noise"], targets["noise"]
    - SNRWeightedNoiseLoss: preds["pred_noise"], preds["scheduler"], preds["timesteps"], targets["noise"]
    
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
        cfg_dropout_rate: CFG dropout rate for conditioning
        compute_eval_metrics: Whether to compute evaluation metrics (CLIP Score, FID, mIoU) (default: False)
    
    Returns:
        (total_loss, logs_dict)
    """
    # Extract embeddings if available (for cross-attention conditioning)
    text_emb = batch.get("text_emb", None)
    pov_emb = batch.get("pov_emb", None)
    
    # Ensure embeddings are 1D (flatten if needed) - optimized: only clone if needed
    if text_emb is not None:
        if text_emb.dim() > 1:
            text_emb = text_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
    if pov_emb is not None:
        if pov_emb.dim() > 1:
            pov_emb = pov_emb.flatten(start_dim=1)  # [B, ...] -> [B, D]
    
    # Apply CFG dropout for conditioning signal C (randomly drop entire conditioning with cfg_dropout_rate probability)
    # CRITICAL: Replace None with torch.zeros_like to avoid errors in embedding_proj
    # C = [c_pov, c_graph] but CFG doesn't care about structure - it drops the entire conditioning signal
    # This teaches the model to work both with and without cross-attention conditioning
    cfg_dropped = False
    if cfg_dropout_rate > 0.0 and (text_emb is not None or pov_emb is not None):
        if torch.rand(1, device=device_obj).item() < cfg_dropout_rate:
            # Replace with zero tensors instead of None to avoid embedding_proj errors
            # If embedding_proj exists, it requires at least one non-None input
            if text_emb is not None:
                text_emb = torch.zeros_like(text_emb)
            elif pov_emb is not None:
                # If text_emb was None but pov_emb exists, create zero text_emb matching pov_emb batch size
                text_emb = torch.zeros((pov_emb.shape[0], 384), device=pov_emb.device, dtype=pov_emb.dtype)
            
            if pov_emb is not None:
                pov_emb = torch.zeros_like(pov_emb)
            elif text_emb is not None:
                # If pov_emb was None but text_emb exists, create zero pov_emb matching text_emb batch size
                pov_emb = torch.zeros((text_emb.shape[0], 512), device=text_emb.device, dtype=text_emb.dtype)
            
            cfg_dropped = True
    
    # CRITICAL: Ensure at least one embedding is not None if embedding_proj exists
    # This prevents embedding_proj from raising ValueError("At least one of text_emb or pov_emb must be provided")
    if hasattr(model, 'embedding_proj') and model.embedding_proj is not None:
        if text_emb is None and pov_emb is None:
            # Both are None - create zero tensors with appropriate batch size from latents
            batch_size = latents.shape[0]
            text_emb = torch.zeros((batch_size, 384), device=device_obj, dtype=latents.dtype)
            pov_emb = torch.zeros((batch_size, 512), device=device_obj, dtype=latents.dtype)
        elif text_emb is None and pov_emb is not None:
            # text_emb is None but pov_emb exists - create zero text_emb
            text_emb = torch.zeros((pov_emb.shape[0], 384), device=pov_emb.device, dtype=pov_emb.dtype)
        elif pov_emb is None and text_emb is not None:
            # pov_emb is None but text_emb exists - create zero pov_emb
            pov_emb = torch.zeros((text_emb.shape[0], 512), device=text_emb.device, dtype=text_emb.dtype)
    
    # Forward pass through model
    outputs = model(latents, t, cond=cond, noise=noise, text_emb=text_emb, pov_emb=pov_emb)
    
    # Note: Evaluation metrics computation removed from compute_loss for speed
    # Metrics are computed separately in eval_epoch when needed
    
    # Prepare preds dict for loss computation
    preds = {
        "pred_noise": outputs["pred_noise"],
        "scheduler": model.scheduler,
        "timesteps": t,
    }
    
    # Prepare targets dict
    targets = {
        "noise": noise,
    }
    
    # Compute loss using CompositeLoss
    if use_amp and device_obj.type == "cuda":
        with torch.amp.autocast('cuda'):
            total_loss, logs = loss_fn(preds, targets)
    else:
        total_loss, logs = loss_fn(preds, targets)
    
    return total_loss, logs


def train_epoch(
    model, dataloader, scheduler, loss_fn, 
    optimizer, device, epoch, use_amp=False, max_grad_norm=None, use_non_uniform_sampling=False, cfg_dropout_rate=0.0, gradient_accumulation_steps=1
):
    """Train for one epoch using CompositeLoss."""
    print(f"[TRAIN] Starting training epoch {epoch}...")
    start_time = time.time()
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    device_obj = to_device(device)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device_obj)
        
        # Verify CLIP projections are being used on first batch of first epoch (optimized: only check once)
        if epoch == 1 and batch_idx == 0:
            print("[TRAIN] Verifying CLIP projections on first batch...")
            from models.components.embedding_projection import CLIPEmbeddingToSpatial
            if hasattr(model, 'embedding_proj') and model.embedding_proj is not None:
                if isinstance(model.embedding_proj, CLIPEmbeddingToSpatial):
                    text_emb = batch.get("text_emb", None)
                    pov_emb = batch.get("pov_emb", None)
                    if text_emb is not None and pov_emb is not None:
                        # Flatten if needed
                        if text_emb.dim() > 1:
                            text_emb = text_emb.flatten(start_dim=1)
                        if pov_emb.dim() > 1:
                            pov_emb = pov_emb.flatten(start_dim=1)
                        
                        # Check that CLIP projections are actually being called
                        with torch.no_grad():
                            try:
                                spatial_features = model.embedding_proj(text_emb[:1], pov_emb[:1])
                                print(f"[TRAIN] [CLIP] ✓ CLIP projections verified")
                                print(f"  Input text_emb shape: {text_emb[:1].shape}, pov_emb shape: {pov_emb[:1].shape}")
                                print(f"  Output spatial features shape: {spatial_features.shape}")
                            except Exception as e:
                                print(f"[TRAIN] [CLIP] ✗ ERROR: CLIP projections failed: {e}")
        
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
        
        # No type-based conditioning - only using control signals (text_emb, pov_emb)
        cond = None
        
        # Compute loss
        # Scale loss by 1/gradient_accumulation_steps to maintain effective learning rate
        loss_scale = 1.0 / gradient_accumulation_steps
        
        # Don't compute eval metrics during training (too expensive)
        # They're computed during evaluation instead
        
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    use_amp, device_obj, cfg_dropout_rate, compute_eval_metrics=False
                )
                # Scale loss for gradient accumulation
                total_loss_val = total_loss_val * loss_scale
            
            scaler = getattr(train_epoch, '_scaler', None)
            if scaler is None:
                scaler = create_grad_scaler(use_amp, device_obj)
                train_epoch._scaler = scaler
            
            if scaler:
                scaler.scale(total_loss_val).backward()
            else:
                total_loss_val.backward()
            
            # Only step optimizer and zero gradients every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if hasattr(model, 'update_ema'):
                    model.update_ema()
            else:
                # Still need to zero grad on first iteration if not already done
                if batch_idx == 0:
                    optimizer.zero_grad()
        else:
            total_loss_val, logs = compute_loss(
                model, batch, latents, t, noise, cond, loss_fn,
                use_amp, device_obj, cfg_dropout_rate, compute_eval_metrics=False
            )
            # Scale loss for gradient accumulation
            total_loss_val = total_loss_val * loss_scale
            
            total_loss_val.backward()
            
            # Only step optimizer and zero gradients every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if hasattr(model, 'update_ema'):
                    model.update_ema()
            else:
                # Still need to zero grad on first iteration if not already done
                if batch_idx == 0:
                    optimizer.zero_grad()
        
        batch_size = latents.shape[0]
        # For logging, use unscaled loss (multiply back by accumulation_steps since we scaled it down)
        # Note: total_loss_val was scaled by 1/gradient_accumulation_steps, so we multiply back
        loss_val = total_loss_val.detach().item() * gradient_accumulation_steps
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
    
    elapsed_time = time.time() - start_time
    print(f"[TRAIN] Epoch {epoch} completed in {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    
    return avg_loss, avg_logs


def eval_epoch(
    model, dataloader, scheduler, loss_fn, 
    device, use_amp=False, taxonomy=None, compute_eval_metrics=True, guidance_scale=1.0, limit_val_batches=50
):
    """Evaluate for one epoch using CompositeLoss."""
    print(f"[EVAL] Starting evaluation epoch (compute_eval_metrics={compute_eval_metrics})...")
    start_time = time.time()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    device_obj = to_device(device)
    
    # Determine the actual number of batches to process
    total_batches = len(dataloader)
    num_batches = min(limit_val_batches, total_batches) if limit_val_batches is not None else total_batches
    print(f"[EVAL] Processing {num_batches}/{total_batches} batches (limit={limit_val_batches})")
    
    loss_compute_time = 0.0
    metrics_compute_time = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=num_batches)):
            # Break early if we've reached the limit
            if limit_val_batches is not None and batch_idx >= limit_val_batches:
                break
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
            
            # No type-based conditioning - only using control signals (text_emb, pov_emb)
            cond = None
            
            batch_start_time = time.time()
            if use_amp and device_obj.type == "cuda":
                with torch.amp.autocast('cuda'):
                    total_loss_val, logs = compute_loss(
                        model, batch, latents, t, noise, cond, loss_fn,
                        use_amp, device_obj, cfg_dropout_rate=0.0, 
                        compute_eval_metrics=False, taxonomy=taxonomy  # Never compute metrics in compute_loss
                    )
            else:
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    use_amp, device_obj, cfg_dropout_rate=0.0,
                    compute_eval_metrics=False, taxonomy=taxonomy  # Never compute metrics in compute_loss
                )
            
            batch_size = latents.shape[0]
            loss_val = total_loss_val.item()
            total_loss += loss_val * batch_size
            total_samples += batch_size
            loss_compute_time += time.time() - batch_start_time
            
            # Compute evaluation metrics if requested (only on first batch to avoid overhead)
            if compute_eval_metrics and batch_idx == 0:
                metrics_start_time = time.time()
                print(f"[EVAL] [METRICS] Computing metrics on batch {batch_idx}...", flush=True)
                try:
                    with torch.no_grad():
                        # Generate conditioned samples for evaluation (like in save_samples)
                        # For efficiency, only compute on a small subset (reduced to 4 for speed)
                        eval_batch_size = min(4, batch_size)
                        print(f"[EVAL] [METRICS] Using eval_batch_size={eval_batch_size} for metrics computation", flush=True)
                        
                        # Get conditioning from batch
                        print(f"[EVAL] [METRICS] Extracting conditioning from batch...", flush=True)
                        eval_text_emb = batch.get("text_emb", None)
                        eval_pov_emb = batch.get("pov_emb", None)
                        
                        # CRITICAL: Replace None with torch.zeros_like to avoid embedding_proj errors
                        if eval_text_emb is not None:
                            eval_text_emb = eval_text_emb[:eval_batch_size]
                            if eval_text_emb.dim() > 1:
                                eval_text_emb = eval_text_emb.flatten(start_dim=1)
                            print(f"[EVAL] [METRICS] Text embeddings shape: {eval_text_emb.shape}", flush=True)
                        elif eval_pov_emb is not None:
                            # Create zero tensor matching pov_emb shape for text_emb
                            eval_text_emb = torch.zeros_like(eval_pov_emb[:eval_batch_size])
                            if eval_text_emb.dim() > 1:
                                eval_text_emb = eval_text_emb.flatten(start_dim=1)
                            print(f"[EVAL] [METRICS] Text embeddings missing, using zeros with shape: {eval_text_emb.shape}", flush=True)
                        
                        if eval_pov_emb is not None:
                            eval_pov_emb = eval_pov_emb[:eval_batch_size]
                            if eval_pov_emb.dim() > 1:
                                eval_pov_emb = eval_pov_emb.flatten(start_dim=1)
                            print(f"[EVAL] [METRICS] POV embeddings shape: {eval_pov_emb.shape}", flush=True)
                        elif eval_text_emb is not None:
                            # Create zero tensor matching text_emb shape for pov_emb
                            eval_pov_emb = torch.zeros_like(eval_text_emb)
                            print(f"[EVAL] [METRICS] POV embeddings missing, using zeros with shape: {eval_pov_emb.shape}", flush=True)
                        
                        # Get target latents for comparison
                        target_latents = latents[:eval_batch_size]
                        
                        # Extract the latent shape from target latents to ensure generated latents match
                        current_latent_shape = target_latents.shape[1:]  # Get (Channels, Height, Width)
                        print(f"[EVAL] [METRICS] Target latent shape: {target_latents.shape}, using shape {current_latent_shape} for generation", flush=True)
                        
                        # CRITICAL: Generate completely new images using FULL sampling process
                        # This performs the complete DDIM reverse process: noise -> denoised image
                        # We do NOT use single-step denoised latents from the training forward pass!
                        # model.sample() starts from random noise and performs num_steps denoising steps
                        num_steps = model.scheduler.num_steps
                        
                        print(f"[EVAL] [METRICS] Starting DDIM sampling: {eval_batch_size} samples, 50 steps, shape {current_latent_shape}...", flush=True)
                        sample_start_time = time.time()
                        
                        conditioned_output = model.sample(
                            batch_size=eval_batch_size,
                            latent_shape=current_latent_shape,  # Force matching resolution with dataset
                            num_steps=50,  # Reduced from 100 to 50 for faster evaluation
                            method="ddim",
                            eta=0.0,
                            cond=None,
                            guidance_scale=guidance_scale,
                            text_emb=eval_text_emb,
                            pov_emb=eval_pov_emb,
                            device=device_obj,
                            verbose=False
                        )
                        
                        sample_time = time.time() - sample_start_time
                        print(f"[EVAL] [METRICS] Sampling completed in {sample_time:.2f}s. Generated latents shape: {conditioned_output.get('latent', 'N/A').shape if 'latent' in conditioned_output else 'N/A'}", flush=True)
                        
                        # Verify that generated latents are different from target latents
                        if "latent" in conditioned_output:
                            gen_latents = conditioned_output["latent"]
                            latent_diff = torch.abs(gen_latents - target_latents).mean().item()
                            if latent_diff < 0.01:
                                raise RuntimeError(
                                    f"BUG: Generated latents are identical to target latents (diff={latent_diff:.6f})! "
                                    f"This suggests model.sample() is not working correctly or is using target latents."
                                )
                            print(f"  [DEBUG] Latent comparison: mean abs diff={latent_diff:.6f}")
                        
                        # Decode generated samples
                        print(f"[EVAL] [METRICS] Decoding generated samples...", flush=True)
                        print(f"[EVAL] [METRICS] Latent shape: {conditioned_output['latent'].shape if 'latent' in conditioned_output else 'N/A'}", flush=True)
                        decode_start_time = time.time()
                        # Note: model.sample() may return rgb already normalized to [0, 1]
                        if "rgb" in conditioned_output:
                            print(f"[EVAL] [METRICS] Using RGB from sample output directly", flush=True)
                            pred_images = conditioned_output["rgb"].clone()
                            # Ensure in [0, 1] range
                            if pred_images.min() < -0.1:  # Likely in [-1, 1] range
                                pred_images = (pred_images + 1.0) / 2.0
                            pred_images = torch.clamp(pred_images, 0.0, 1.0)
                        else:
                            print(f"[EVAL] [METRICS] Decoding latents through decoder...", flush=True)
                            pred_decoded = model.decoder({"latent": conditioned_output["latent"]})
                            pred_images = pred_decoded.get("rgb", None)
                            if pred_images is not None:
                                # Decoder typically outputs in [-1, 1] range
                                if pred_images.min() < -0.1:
                                    pred_images = (pred_images + 1.0) / 2.0
                                pred_images = torch.clamp(pred_images, 0.0, 1.0)
                        
                        # Check and log actual decoder output size
                        if pred_images is not None:
                            print(f"[EVAL] [METRICS] Decoder output shape: {pred_images.shape}", flush=True)
                            # Only resize if sizes don't match (shouldn't normally be needed)
                            if pred_images.shape[-1] != 256 or pred_images.shape[-2] != 256:
                                print(f"[EVAL] [METRICS] WARNING: Decoder output size {pred_images.shape[-2]}x{pred_images.shape[-1]} != expected 256x256", flush=True)
                                print(f"[EVAL] [METRICS] Resizing pred images to 256x256", flush=True)
                                pred_images = torch.nn.functional.interpolate(
                                    pred_images, 
                                    size=(256, 256), 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                        
                        decode_time = time.time() - decode_start_time
                        print(f"[EVAL] [METRICS] Generated samples decoded in {decode_time:.2f}s", flush=True)
                        
                        # Decode target latents to get ground truth images
                        print(f"[EVAL] [METRICS] Decoding target latents...", flush=True)
                        print(f"[EVAL] [METRICS] Target latent shape: {target_latents.shape}", flush=True)
                        gt_decode_start_time = time.time()
                        # These are the actual ground truth latents from the dataset
                        gt_decoded = model.decoder({"latent": target_latents})
                        gt_images = gt_decoded.get("rgb", None)
                        if gt_images is not None:
                            print(f"[EVAL] [METRICS] GT decoder output shape: {gt_images.shape}", flush=True)
                            # Decoder typically outputs in [-1, 1] range
                            if gt_images.min() < -0.1:
                                gt_images = (gt_images + 1.0) / 2.0
                            gt_images = torch.clamp(gt_images, 0.0, 1.0)
                            
                            # Only resize if sizes don't match (shouldn't normally be needed)
                            if gt_images.shape[-1] != 256 or gt_images.shape[-2] != 256:
                                print(f"[EVAL] [METRICS] WARNING: GT decoder output size {gt_images.shape[-2]}x{gt_images.shape[-1]} != expected 256x256", flush=True)
                                print(f"[EVAL] [METRICS] Resizing GT images to 256x256", flush=True)
                                gt_images = torch.nn.functional.interpolate(
                                    gt_images, 
                                    size=(256, 256), 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                        
                        gt_decode_time = time.time() - gt_decode_start_time
                        print(f"[EVAL] [METRICS] Target latents decoded in {gt_decode_time:.2f}s", flush=True)
                        
                        # CRITICAL: Verify we have different images
                        if pred_images is not None and gt_images is not None:
                            # Check if images are identical (would indicate a bug)
                            if torch.allclose(pred_images, gt_images, atol=1e-5):
                                raise RuntimeError(
                                    "BUG: Generated images and target images are identical! "
                                    "This should never happen. Check if images are being swapped."
                                )
                        
                        if pred_images is not None and gt_images is not None:
                            # Sanity check: compute pixel-wise MSE to verify images are different
                            pixel_mse = torch.nn.functional.mse_loss(pred_images, gt_images).item()
                            
                            # Debug: print image statistics
                            pred_mean = pred_images.mean().item()
                            pred_std = pred_images.std().item()
                            pred_min = pred_images.min().item()
                            pred_max = pred_images.max().item()
                            gt_mean = gt_images.mean().item()
                            gt_std = gt_images.std().item()
                            gt_min = gt_images.min().item()
                            gt_max = gt_images.max().item()
                            
                            # Print debug info
                            print(f"  [DEBUG] Image comparison:")
                            print(f"    Pred: mean={pred_mean:.3f}, std={pred_std:.3f}, range=[{pred_min:.3f}, {pred_max:.3f}]")
                            print(f"    GT:   mean={gt_mean:.3f}, std={gt_std:.3f}, range=[{gt_min:.3f}, {gt_max:.3f}]")
                            print(f"    Pixel MSE: {pixel_mse:.6f}")
                            
                            # If images are too similar (MSE < 0.01), something is wrong
                            if pixel_mse < 0.01:
                                warnings.warn(
                                    f"WARNING: Generated and target images are very similar (MSE={pixel_mse:.6f}). "
                                    f"This suggests a bug - images may be swapped or identical. "
                                    f"Pred stats: mean={pred_mean:.3f}, std={pred_std:.3f}, "
                                    f"GT stats: mean={gt_mean:.3f}, std={gt_std:.3f}"
                                )
                            
                            # Compute evaluation metrics between conditioned samples and targets
                            # IMPORTANT: pred_images = GENERATED (from model.sample()), gt_images = TARGETS (from dataset)
                            # Verify order is correct before computing metrics
                            print(f"[EVAL] [METRICS] Computing evaluation metrics (CLIP, mIoU)...", flush=True)
                            print(f"[EVAL] [METRICS]   pred_images shape: {pred_images.shape}, mean: {pred_images.mean().item():.3f}", flush=True)
                            print(f"[EVAL] [METRICS]   gt_images shape: {gt_images.shape}, mean: {gt_images.mean().item():.3f}", flush=True)
                            
                            metrics_comp_start_time = time.time()
                            print(f"[EVAL] [METRICS] Calling compute_evaluation_metrics()...", flush=True)
                            # CRITICAL: Ensure we're passing them in the correct order
                            # pred_images = generated (should be noise-like if model is untrained)
                            # gt_images = targets (should be real floor plans)
                            # Compute evaluation metrics (FID disabled for speed - requires 100+ samples)
                            try:
                                eval_metrics = compute_evaluation_metrics(
                                    pred_images,  # GENERATED images (from model.sample())
                                    gt_images,    # TARGET images (from dataset)
                                    eval_text_emb, eval_pov_emb,
                                    taxonomy=taxonomy, device=device_obj,
                                    compute_clip=True, compute_fid=False, compute_miou=True  # FID disabled - too few samples
                                )
                                print(f"[EVAL] [METRICS] compute_evaluation_metrics() returned successfully", flush=True)
                            except Exception as e:
                                print(f"[EVAL] [METRICS] ERROR in compute_evaluation_metrics(): {e}", flush=True)
                                raise
                            
                            metrics_comp_time = time.time() - metrics_comp_start_time
                            print(f"[EVAL] [METRICS] Metrics computation completed in {metrics_comp_time:.2f}s", flush=True)
                            
                            # Add pixel MSE to metrics for debugging
                            eval_metrics["pixel_mse"] = pixel_mse
                            
                            # Print metric values for debugging
                            print(f"[EVAL] [METRICS] Computed metrics:", flush=True)
                            for k, v in eval_metrics.items():
                                print(f"[EVAL] [METRICS]   {k}: {v:.6f}", flush=True)
                            
                            metrics_compute_time = time.time() - metrics_start_time
                            print(f"[EVAL] [METRICS] Total metrics computation time: {metrics_compute_time:.2f}s", flush=True)
                            
                            # Add to logs (weighted by eval batch size)
                            for k, v in eval_metrics.items():
                                if k not in log_dict:
                                    log_dict[k] = 0.0
                                log_dict[k] += v * eval_batch_size
                except Exception as e:
                    # Silently fail if evaluation metrics computation failed
                    import warnings
                    import traceback
                    metrics_compute_time = time.time() - metrics_start_time if 'metrics_start_time' in locals() else 0.0
                    print(f"[EVAL] [METRICS] Metrics computation failed after {metrics_compute_time:.2f}s: {e}", flush=True)
                    print(f"[EVAL] [METRICS] Traceback:", flush=True)
                    print(traceback.format_exc(), flush=True)
                    warnings.warn(f"Evaluation metrics computation failed: {e}")
            
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
    
    elapsed_time = time.time() - start_time
    print(f"[EVAL] Evaluation completed in {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)", flush=True)
    print(f"[EVAL] Breakdown: Loss={loss_compute_time:.2f}s ({loss_compute_time/elapsed_time*100:.1f}%)", end="", flush=True)
    if compute_eval_metrics:
        print(f", Metrics={metrics_compute_time:.2f}s ({metrics_compute_time/elapsed_time*100:.1f}%)", flush=True)
    else:
        print(flush=True)
    
    return avg_loss, avg_logs


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=16, exp_name=None, guidance_scale=1.0, cfg_dropout_rate=0.0):
    """Generate and save sample images.
    
    Generates:
    - Unconditioned samples: 4x4 grid (16 samples)
    - Targets vs Generated comparison: side-by-side comparison from validation batch
    """
    print(f"[SAMPLES] Starting sample generation for epoch {epoch}...")
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
    print(f"  [SAMPLING] Generating 16 unconditioned samples (4x4 grid) using DDIM ({ddim_steps} steps)...")
    
    sampling_seed = 42 + epoch
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)
    
    # CRITICAL: Use zero tensors instead of None for unconditioned sampling, matching training logic
    # This ensures embedding_proj receives proper tensors even for unconditional generation
    unconditioned_batch_size = 16
    if hasattr(model, 'embedding_proj') and model.embedding_proj is not None:
        # Try to infer embedding dimensions from batch if available, otherwise use defaults
        text_dim = 384  # Default CLIP text embedding dimension
        pov_dim = 512   # Default POV embedding dimension
        
        # Try to get dimensions from batch
        if "text_emb" in batch:
            sample_text_emb = batch["text_emb"]
            if isinstance(sample_text_emb, torch.Tensor):
                if sample_text_emb.dim() > 1:
                    text_dim = sample_text_emb.flatten(start_dim=1).shape[1]
                else:
                    text_dim = sample_text_emb.shape[0] if sample_text_emb.dim() == 1 else 384
        
        if "pov_emb" in batch:
            sample_pov_emb = batch["pov_emb"]
            if isinstance(sample_pov_emb, torch.Tensor):
                if sample_pov_emb.dim() > 1:
                    pov_dim = sample_pov_emb.flatten(start_dim=1).shape[1]
                else:
                    pov_dim = sample_pov_emb.shape[0] if sample_pov_emb.dim() == 1 else 512
        
        # Get dtype from model parameters for consistency
        param_dtype = next(model.parameters()).dtype
        
        # Create zero tensors matching the expected dimensions
        unconditioned_text_emb = torch.zeros((unconditioned_batch_size, text_dim), device=device_obj, dtype=param_dtype)
        unconditioned_pov_emb = torch.zeros((unconditioned_batch_size, pov_dim), device=device_obj, dtype=param_dtype)
        print(f"  [SAMPLING] Using zero tensors for unconditioned embeddings: text_emb={unconditioned_text_emb.shape}, pov_emb={unconditioned_pov_emb.shape}")
    else:
        # No embedding_proj, can use None
        unconditioned_text_emb = None
        unconditioned_pov_emb = None
    
    with torch.no_grad():
        unconditioned_output = model.sample(
            batch_size=unconditioned_batch_size,  # Keep 4x4 grid for unconditioned
            num_steps=ddim_steps,
            method="ddim",
            eta=0.0,
            cond=None,
            guidance_scale=1.0,
            text_emb=unconditioned_text_emb,
            pov_emb=unconditioned_pov_emb,
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
        
        epoch_prefix = f"{exp_name}_epoch_{epoch:03d}" if exp_name else f"epoch_{epoch:03d}"
        unconditioned_path = samples_dir / f"{epoch_prefix}_unconditioned_4x4.png"
        unconditioned_grid.save(unconditioned_path)
        print(f"  Saved 16 unconditioned samples (4x4 grid) to {unconditioned_path}")

    # ============================================================================
    # Part 2: Targets vs Generated comparison (4x4 = 16 samples)
    # ============================================================================
    # Get dataset to find rooms and scenes (different conditioning structures)
    dataset = val_loader.dataset
    
    # Find samples from the validation dataset
    # Handle both mixed-type and single-type datasets
    room_indices = []
    scene_indices = []
    
    # Check if dataset has 'type' column
    if hasattr(dataset, 'df') and 'type' in dataset.df.columns:
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]
            sample_type = str(row.get('type', '')).lower().strip()
            if sample_type == 'room' and len(room_indices) < 8:
                room_indices.append(idx)
            elif sample_type == 'scene' and len(scene_indices) < 8:
                scene_indices.append(idx)
            # For mixed-type datasets, stop when we have 8 of each (16 total)
            # For single-type datasets, continue until we have 16 samples
            if len(room_indices) >= 8 and len(scene_indices) >= 8:
                break
            # For single-type datasets, collect 16 samples of the available type
            if (len(room_indices) > 0 and len(scene_indices) == 0 and len(room_indices) >= 16):
                break
            if (len(scene_indices) > 0 and len(room_indices) == 0 and len(scene_indices) >= 16):
                break
    else:
        # Fallback: use first 16 samples if type column not available
        room_indices = list(range(min(16, len(dataset))))
        scene_indices = []
    
    # Handle single-type vs mixed-type datasets
    if len(room_indices) == 0 or len(scene_indices) == 0:
        # Single-type dataset - use 16 samples of the available type
        available_indices = room_indices if len(room_indices) > 0 else scene_indices
        selected_indices = available_indices[:16] if len(available_indices) >= 16 else available_indices
        dataset_type = "rooms" if len(room_indices) > 0 else "scenes"
        print(f"  Type-filtered dataset detected: Using {len(selected_indices)} samples of type '{dataset_type}'")
    else:
        # Mixed-type dataset - use 8 of each type (16 total)
        selected_indices = room_indices + scene_indices
        print(f"  Mixed-type dataset: Using {len(room_indices)} rooms and {len(scene_indices)} scenes")
    
    batch_size = len(selected_indices)
    
    if batch_size == 0:
        print("  Warning: No samples found in validation dataset for comparison")
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
    
    # Print selection summary (only if both types are present)
    if len(room_indices) > 0 and len(scene_indices) > 0:
        print(f"  Selected {len(room_indices)} rooms and {len(scene_indices)} scenes for comparison")
    
    # Extract embeddings (control signals only - no type-based conditioning)
    text_emb = batch.get("text_emb", None)
    pov_emb = batch.get("pov_emb", None)
    target_latents = batch.get("latent", None)
    
    # No type-based conditioning - only using control signals (text_emb, pov_emb)
    cond = None
    
    # CRITICAL: Replace None with torch.zeros_like to avoid embedding_proj errors
    # Ensure embeddings are 1D (flatten if needed)
    if text_emb is not None:
        if text_emb.dim() > 1:
            text_emb = text_emb.flatten(start_dim=1)
    elif pov_emb is not None:
        # Create zero tensor matching pov_emb shape for text_emb
        text_emb = torch.zeros_like(pov_emb)
        if text_emb.dim() > 1:
            text_emb = text_emb.flatten(start_dim=1)
    
    if pov_emb is not None:
        if pov_emb.dim() > 1:
            pov_emb = pov_emb.flatten(start_dim=1)
    elif text_emb is not None:
        # Create zero tensor matching text_emb shape for pov_emb
        pov_emb = torch.zeros_like(text_emb)
        if pov_emb.dim() > 1:
            pov_emb = pov_emb.flatten(start_dim=1)
    
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
    
    # Generate conditioned samples using DDIM (50 steps)
    ddim_steps = 50
    print(f"  [SAMPLING] Generating {batch_size} conditioned samples using DDIM ({ddim_steps} steps)...")
    
    with torch.no_grad():
        conditioned_output = model.sample(
            batch_size=batch_size,
            num_steps=ddim_steps,
            method="ddim",
            eta=0.0,
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
        grid_n = 4  # 4 columns
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
        # Print summary based on dataset type
        if len(room_indices) > 0 and len(scene_indices) > 0:
            print(f"  Saved target vs generated comparison ({batch_size} samples: {len(room_indices)} rooms + {len(scene_indices)} scenes) to {comparison_path}")
        else:
            dataset_type = "rooms" if len(room_indices) > 0 else "scenes"
            print(f"  Saved target vs generated comparison ({batch_size} samples of type '{dataset_type}') to {comparison_path}")
        
        # Also save generated samples only
        samples_path = samples_dir / f"{epoch_prefix}_generated.png"
        generated_grid.save(samples_path)
        print(f"  Saved generated samples ({batch_size} samples) to {samples_path}")
        
        # ============================================================================
        # Part 3: Save individual samples with conditioning information
        # ============================================================================
        data_dir = output_dir / "sample_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = data_dir / f"{epoch_prefix}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get paths from dataset
        graph_texts = []
        layout_paths = []
        pov_paths = []
        
        for i in range(batch_size):
            idx = selected_indices[i] if i < len(selected_indices) else i
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
        
        # Save control signal embeddings (no type-based conditioning)
        if text_emb is not None:
            torch.save(text_emb.cpu(), images_dir / "text_embeddings.pt")
        if pov_emb is not None:
            torch.save(pov_emb.cpu(), images_dir / "pov_embeddings.pt")
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "batch_size": batch_size,
            "room_indices": room_indices,
            "scene_indices": scene_indices,
            "selected_indices": selected_indices,
            "layout_paths": layout_paths,
            "pov_paths": pov_paths,
            "graph_texts": graph_texts,
            "has_text_emb": text_emb is not None,
            "has_pov_emb": pov_emb is not None,
            "guidance_scale": guidance_scale,
            "cfg_dropout_rate": cfg_dropout_rate,
        }
        with open(images_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved individual samples with conditioning to {images_dir}")


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
    print("[DATASET] Building dataset...")
    dataset = build_dataset(config)
    print(f"[DATASET] Dataset built: {len(dataset)} samples")
    
    # Build validation dataset
    print("[DATASET] Splitting dataset into train/val...")
    train_dataset, val_dataset = split_dataset(dataset, config["training"])
    print(f"[DATASET] Train: {len(train_dataset)} samples, Val: {len(val_dataset) if val_dataset else 0} samples")
    
    device_obj = to_device(device)
    
    # Try to load VAE metadata first (if autoencoder checkpoint is specified)
    print("[VAE_METADATA] Checking for VAE metadata file...")
    vae_metadata = None
    ae_cfg = config.get("autoencoder") or config.get("diffusion", {}).get("autoencoder")
    if ae_cfg and isinstance(ae_cfg, dict):
        ae_checkpoint = ae_cfg.get("checkpoint")
        if ae_checkpoint:
            vae_metadata = load_vae_metadata(ae_checkpoint)
            if vae_metadata:
                print(f"[VAE_METADATA] Found VAE metadata file")
                if vae_metadata.get("scale_factor"):
                    print(f"  scale_factor: {vae_metadata['scale_factor']:.6f}")
                if vae_metadata.get("latent_clamp_min") is not None:
                    print(f"  latent_clamp_min: {vae_metadata['latent_clamp_min']:.6f}")
                if vae_metadata.get("latent_clamp_max") is not None:
                    print(f"  latent_clamp_max: {vae_metadata['latent_clamp_max']:.6f}")
            else:
                print("[VAE_METADATA] No VAE metadata file found")
    else:
        print("[VAE_METADATA] No autoencoder checkpoint specified")
    
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
            print(f"\n[SCALE_FACTOR] Using scale_factor from VAE metadata: {scale_factor:.6f}")
            # Add to config for model building
            if "diffusion" in config:
                config["diffusion"]["scale_factor"] = scale_factor
            else:
                config["scale_factor"] = scale_factor
        else:
            print("\n[SCALE_FACTOR] WARNING: scale_factor not found in config or VAE metadata")
            print("  Using default scale_factor=1.0 (no scaling)")
            print("  Note: scale_factor should be calculated as part of VAE statistics")
            scale_factor = 1.0
            if "diffusion" in config:
                config["diffusion"]["scale_factor"] = scale_factor
            else:
                config["scale_factor"] = scale_factor
    else:
        print(f"\n[SCALE_FACTOR] Using scale_factor from config: {scale_factor}")
    
    # Load latent_clamp values (priority: config > VAE metadata > defaults)
    latent_clamp_min = config.get("latent_clamp_min")
    latent_clamp_max = config.get("latent_clamp_max")
    
    if latent_clamp_min is None and vae_metadata and vae_metadata.get("latent_clamp_min") is not None:
        latent_clamp_min = vae_metadata["latent_clamp_min"]
        config["latent_clamp_min"] = latent_clamp_min
        print(f"Using latent_clamp_min from VAE metadata: {latent_clamp_min:.6f}")
    
    if latent_clamp_max is None and vae_metadata and vae_metadata.get("latent_clamp_max") is not None:
        latent_clamp_max = vae_metadata["latent_clamp_max"]
        config["latent_clamp_max"] = latent_clamp_max
        print(f"Using latent_clamp_max from VAE metadata: {latent_clamp_max:.6f}")
    
    # Check if we should resume or start fresh
    should_resume = not args.no_resume and latest_checkpoint.exists()
    
    # Load checkpoint (Stage 1, Stage 2, or resume)
    print("[MODEL] Loading model...")
    stage1_checkpoint = config.get("diffusion", {}).get("stage1_checkpoint")
    stage2_checkpoint = config.get("diffusion", {}).get("stage2_checkpoint")
    
    if stage2_checkpoint and not should_resume:
        print(f"[MODEL] Loading Stage 2 checkpoint from: {stage2_checkpoint}")
        model, _ = DiffusionModel.load_checkpoint(
            stage2_checkpoint,
            map_location=device,
            return_extra=True,
            config=config
        )
        model = model.to(device_obj)
        print("[MODEL] Stage 2 checkpoint loaded successfully")
    elif stage1_checkpoint and not should_resume:
        print(f"[MODEL] Loading Stage 1 checkpoint from: {stage1_checkpoint}")
        model, _ = DiffusionModel.load_checkpoint(
            stage1_checkpoint,
            map_location=device,
            return_extra=True,
            config=config
        )
        model = model.to(device_obj)
        print("[MODEL] Stage 1 checkpoint loaded successfully")
    elif should_resume:
        print(f"[MODEL] Found latest checkpoint: {latest_checkpoint}")
        print("[MODEL] Resuming training...")
        
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
                print("[MODEL] Loading training history from CSV...")
                df = pd.read_csv(metrics_csv_path)
                df_filtered = df[df['epoch'] < (start_epoch + 1)]
                training_history = df_filtered.to_dict('records')
                print(f"[MODEL] Loaded {len(training_history)} epochs from CSV file")
            except Exception as e:
                print(f"[MODEL] Warning: Could not load metrics from CSV: {e}")
        
        print(f"[MODEL] Resuming from epoch {start_epoch + 1}")
        print(f"[MODEL] Best validation loss so far: {best_val_loss:.6f}")
        
        # Check if we need to continue training beyond the checkpoint
        epochs = config["training"].get("epochs", 100)
        if start_epoch >= epochs:
            print(f"[MODEL] WARNING: Checkpoint is at epoch {start_epoch + 1}, but config specifies only {epochs} epochs.")
            print(f"[MODEL] Training is already complete. To continue training, increase 'epochs' in config.")
        else:
            remaining_epochs = epochs - start_epoch
            print(f"[MODEL] Will continue training for {remaining_epochs} more epochs (until epoch {epochs})")
    else:
        # Build model from config (fresh start)
        print("[MODEL] Building model from config...")
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
    
    # Verify CLIP projections are being used if configured
    if hasattr(model, 'embedding_proj') and model.embedding_proj is not None:
        from models.components.embedding_projection import CLIPEmbeddingToSpatial
        if isinstance(model.embedding_proj, CLIPEmbeddingToSpatial):
            print("\n" + "="*60)
            print("CLIP Projections Verification")
            print("="*60)
            
            # Check if CLIP projections exist
            if hasattr(model.embedding_proj, 'clip_projections') and model.embedding_proj.clip_projections is not None:
                clip_proj = model.embedding_proj.clip_projections
                print("✓ CLIPEmbeddingToSpatial is using CLIP projections")
                print(f"  - Projection dim: {clip_proj.projection_dim}")
                print(f"  - Text dim: {clip_proj.text_dim}")
                print(f"  - POV dim: {clip_proj.pov_dim}")
                print(f"  - Spatial alignment: {getattr(clip_proj, 'spatial_alignment', False)}")
                
                # Verify projection layers exist
                has_text = hasattr(clip_proj, 'text_proj') and clip_proj.text_proj is not None
                has_pov = hasattr(clip_proj, 'pov_proj') and clip_proj.pov_proj is not None
                print(f"  - Text projection layer: {'✓' if has_text else '✗'}")
                print(f"  - POV projection layer: {'✓' if has_pov else '✗'}")
                
                if not (has_text and has_pov):
                    print("  ✗ WARNING: CLIP projections missing required layers!")
                else:
                    # Test forward pass with dummy data to verify it works
                    try:
                        dummy_text = torch.randn(1, clip_proj.text_dim, device=device_obj)
                        dummy_pov = torch.randn(1, clip_proj.pov_dim, device=device_obj)
                        with torch.no_grad():
                            test_output = model.embedding_proj(dummy_text, dummy_pov)
                        print(f"  ✓ Forward pass test successful: output shape {test_output.shape}")
                    except Exception as e:
                        print(f"  ✗ ERROR: Forward pass test failed: {e}")
            else:
                print("✗ WARNING: CLIPEmbeddingToSpatial does not have clip_projections!")
        else:
            print("\n" + "="*60)
            print("Embedding Projection Info")
            print("="*60)
            print(f"Using {type(model.embedding_proj).__name__} (not CLIPEmbeddingToSpatial)")
    else:
        print("\n" + "="*60)
        print("Embedding Projection Info")
        print("="*60)
        print("No embedding projection configured")
    
    print("="*60 + "\n")
    
    # Keep decoder frozen - only UNet is trained
    if hasattr(model, 'decoder'):
        print("[MODEL] Freezing decoder - only UNet will be trained...")
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("[MODEL] Decoder frozen")
    
    # CRITICAL: Ensure UNet is trainable (explicitly set requires_grad=True)
    # This overrides any frozen settings from config or checkpoint
    if hasattr(model, 'unet'):
        print("[MODEL] Ensuring UNet is trainable...")
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
            print(f"[MODEL] WARNING: Found {frozen_tensors} frozen UNet parameter tensors ({frozen_params:,} params) - setting them to trainable!")
        total_params = trainable_params + frozen_params
        print(f"[MODEL] UNet parameters: {total_params:,} total, {total_params:,} trainable")
    
    # Build data loaders
    print("[DATALOADER] Building data loaders...")
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 8)
    shuffle = config["training"].get("shuffle", True)
    use_weighted_sampling = config["training"].get("use_weighted_sampling", False)
    print(f"[DATALOADER] Batch size: {batch_size}, Workers: {num_workers}, Shuffle: {shuffle}, Weighted sampling: {use_weighted_sampling}")
    
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
    print("[LOSS] Building loss function from config...")
    loss_fn = build_loss(config)
    print(f"[LOSS] Loss type: {type(loss_fn).__name__}")
    if hasattr(loss_fn, 'losses'):
        print(f"[LOSS] Loss components: {[type(l).__name__ for l in loss_fn.losses]}")
    
    # Build optimizer
    print("[OPTIMIZER] Building optimizer...")
    optimizer = build_optimizer(model, config)
    
    # Verify optimizer has trainable parameters
    total_optimizer_params = sum(len(group['params']) for group in optimizer.param_groups)
    total_trainable_model_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"[OPTIMIZER] Parameter groups: {len(optimizer.param_groups)}")
    print(f"[OPTIMIZER] Total trainable parameters: {total_trainable_model_params:,}")
    
    # Count UNet parameters specifically
    if hasattr(model, 'unet'):
        unet_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        unet_total = sum(p.numel() for p in model.unet.parameters())
        print(f"[OPTIMIZER] UNet parameters: {unet_trainable:,} trainable / {unet_total:,} total")
        if unet_trainable == 0:
            print("[OPTIMIZER] ERROR: UNet has no trainable parameters! Training will not work!")
    
    # Build learning rate scheduler (account for already-trained epochs when resuming)
    # Use last_epoch=-1 for fresh start, or start_epoch for resume
    print("[SCHEDULER] Building learning rate scheduler...")
    last_epoch = start_epoch if start_epoch > 0 else -1
    scheduler = build_scheduler(optimizer, config, last_epoch=last_epoch)
    print(f"[SCHEDULER] Initial learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Training settings
    epochs = config["training"].get("epochs", 100)
    use_amp = config["training"].get("use_amp", False)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    eval_interval = config["training"].get("eval_interval", 5)
    sample_interval = config["training"].get("sample_interval", 10)
    metrics_interval = config["training"].get("metrics_interval", 100)  # Compute full metrics every N epochs
    use_non_uniform_sampling = config["training"].get("use_non_uniform_sampling", False)  # Default False for uniform sampling
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    
    # Early stopping state
    epochs_without_improvement = 0
    
    print(f"\nStarting training...")
    print(f"  Total epochs: {epochs}")
    if start_epoch > 0:
        print(f"  Starting from epoch: {start_epoch + 1} (resumed from checkpoint)")
        print(f"  Remaining epochs: {epochs - start_epoch}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Non-uniform timestep sampling: {use_non_uniform_sampling}")
    print(f"  Evaluation interval: every {eval_interval} epochs")
    print(f"  Sample generation interval: every {sample_interval} epochs")
    print(f"  Metrics computation interval: every {metrics_interval} epochs")
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
        gradient_accumulation_steps = config.get("training", {}).get("gradient_accumulation_steps", 1)
        
        train_loss, train_logs = train_epoch(
            model, train_loader, scheduler, loss_fn,
            optimizer, device_obj, epoch + 1, use_amp=use_amp, max_grad_norm=max_grad_norm,
            use_non_uniform_sampling=use_non_uniform_sampling, cfg_dropout_rate=cfg_dropout_rate,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Print current CFG dropout rate if using schedule
        if isinstance(cfg_dropout_config, dict):
            print(f"  Current CFG dropout rate: {cfg_dropout_rate:.4f}", flush=True)
        print(f"Train Loss: {train_loss:.6f}", flush=True)
        for k, v in train_logs.items():
            print(f"  {k}: {v:.6f}", flush=True)
        
        # Validate
        print(f"[EPOCH] Starting validation for epoch {epoch + 1}...", flush=True)
        val_loss = float("inf")
        val_logs = {}
        # Always evaluate at epoch 1, then according to eval_interval
        should_eval = val_loader and ((epoch + 1 == 1) or ((epoch + 1) % eval_interval == 0))
        # Compute full metrics (CLIP, mIoU, etc.) only at epoch 1 and every metrics_interval epochs
        should_compute_metrics = should_eval and ((epoch + 1 == 1) or ((epoch + 1) % metrics_interval == 0))
        if should_eval:
            # Try to get taxonomy from dataset if available
            taxonomy = None
            if hasattr(val_loader.dataset, 'taxonomy'):
                taxonomy = val_loader.dataset.taxonomy
            elif hasattr(train_loader.dataset, 'taxonomy'):
                taxonomy = train_loader.dataset.taxonomy
            
            # Get guidance_scale from config for evaluation
            guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
            
            val_loss, val_logs = eval_epoch(
                model, val_loader, scheduler, loss_fn,
                device_obj, use_amp=use_amp, taxonomy=taxonomy, compute_eval_metrics=should_compute_metrics,
                guidance_scale=guidance_scale, limit_val_batches=50
            )
            print(f"[EPOCH] Validation completed. Val Loss: {val_loss:.6f}", flush=True)
            for k, v in val_logs.items():
                print(f"  {k}: {v:.6f}", flush=True)
            
            # Check if this is the best validation loss BEFORE updating best_val_loss
            is_best = val_loss < best_val_loss
            
            # Early stopping logic
            if early_stopping_patience is not None:
                improvement = best_val_loss - val_loss
                if improvement > early_stopping_min_delta:
                    epochs_without_improvement = 0
                    if is_best:
                        best_val_loss = val_loss
                    print(f"[EPOCH] Improvement: {improvement:.6f} (new best: {best_val_loss:.6f})", flush=True)
                else:
                    epochs_without_improvement += 1
                    print(f"[EPOCH] No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs", flush=True)
            elif is_best:
                # Update best_val_loss if not using early stopping
                best_val_loss = val_loss
                print(f"[EPOCH] New best validation loss: {best_val_loss:.6f}", flush=True)
        
        # Save samples
        # Always save at epoch 1, then every sample_interval epochs
        if val_loader and ((epoch + 1 == 1) or ((epoch + 1) % sample_interval == 0)):
            print(f"[EPOCH] Saving samples for epoch {epoch + 1}...", flush=True)
            # Get guidance_scale from config (default 1.0 = no CFG)
            guidance_scale = config.get("training", {}).get("guidance_scale", 1.0)
            save_samples(model, val_loader, device_obj, output_dir, epoch + 1, sample_batch_size=64, exp_name=exp_name, guidance_scale=guidance_scale, cfg_dropout_rate=cfg_dropout_rate)
            print(f"[EPOCH] Samples saved", flush=True)
        
        # Save checkpoint (is_best was already determined above if validation ran)
        # Use same condition as evaluation: always at epoch 1, then according to eval_interval
        print(f"[EPOCH] Saving checkpoint for epoch {epoch + 1}...", flush=True)
        if should_eval:
            # is_best already determined above
            pass
        else:
            # No validation this epoch, so not best
            is_best = False
        
        # Record history
        print(f"[EPOCH] Recording training history...", flush=True)
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
        print(f"[EPOCH] Saving metrics to CSV...", flush=True)
        save_metrics_csv(training_history, metrics_csv_path)
        print(f"[EPOCH] Metrics CSV saved", flush=True)
        
        # Plot metrics with loss breakdown
        if len(training_history) > 0:
            print(f"[EPOCH] Plotting metrics...", flush=True)
            try:
                df = pd.DataFrame(training_history)
                plot_diffusion_metrics_epochs(df, output_dir, exp_name=exp_name)
                # Also create dedicated evaluation metrics plot if metrics exist
                plot_evaluation_metrics(df, output_dir, exp_name=exp_name)
                print(f"[EPOCH] Metrics plots saved", flush=True)
            except Exception as e:
                print(f"[EPOCH] Warning: Could not plot metrics: {e}", flush=True)
        
        # Save checkpoint
        print(f"[EPOCH] Writing checkpoint files...", flush=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
        model.save_checkpoint(
            checkpoint_path,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            training_history=training_history
        )
        print(f"[EPOCH] Saved latest checkpoint: {checkpoint_path}", flush=True)
        
        if is_best:
            best_checkpoint_path = checkpoint_dir / f"{exp_name}_checkpoint_best.pt"
            model.save_checkpoint(
                best_checkpoint_path,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                training_history=training_history
            )
            print(f"[EPOCH] Saved best checkpoint (val_loss={best_val_loss:.6f}): {best_checkpoint_path}", flush=True)
        
        print(f"[EPOCH] Epoch {epoch + 1} completed successfully", flush=True)
        
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
