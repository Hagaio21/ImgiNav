#!/usr/bin/env python3
"""
Stage 2: Discriminator Training Pipeline

This script orchestrates the full discriminator training pipeline:
1. Load a diffusion model checkpoint
2. Generate N samples (latents) from the model
3. Get N real samples from dataset (must be pre-embedded latents)
4. Train a discriminator on these latents
5. Train the diffusion model with discriminator loss
6. Optionally iterate this process (adversarial training)

This replaces the old Stage 2 which wasn't working well.

Note: No on-the-fly encoding/decoding is performed. The dataset must provide
pre-embedded latents via 'latent_path' in the manifest. Decoding only occurs
during sampling (generating new samples for visualization).
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml
import sys
import os
import json
from datetime import datetime

from training.utils import (
    load_config,
    set_deterministic,
    get_device,
    build_dataset,
    split_dataset,
    to_device,
    build_optimizer,
    build_loss,
    build_scheduler,
    create_grad_scaler,
    save_metrics_csv,
    move_batch_to_device,
)
from models.diffusion import DiffusionModel
from models.autoencoder import Autoencoder
from models.components.discriminator import LatentDiscriminator
from models.datasets.datasets import ManifestDataset
from torch.utils.data import DataLoader
from common.utils import is_augmented_path

# Import training functions
from training.train_discriminator import train_discriminator, load_latents
from training.train_diffusion import (
    train_epoch,
    eval_epoch,
    save_samples,
    compute_loss,
)
from training.plotting_utils import plot_discriminator_metrics, plot_diffusion_metrics, plot_overall_iteration_metrics
from torchvision.utils import save_image
import math


def generate_fake_latents(
    model,
    num_samples,
    batch_size=32,
    device="cuda",
    seed=None,
    use_single_step=True
):
    """
    Generate fake latents from diffusion model.
    
    Args:
        model: Diffusion model (loaded and on device)
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to use
        seed: Random seed (optional)
        use_single_step: If True, use single-step predictions (matches training distribution).
                        If False, use full DDPM sampling (cleaner but distribution mismatch).
    
    Returns:
        Tensor of fake latents [N, C, H, W]
    """
    device_obj = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    model.eval()
    all_latents = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    if use_single_step:
        # Use single-step predictions to match what discriminator sees during training
        print(f"Generating {num_samples} fake latents using single-step predictions (matches training distribution)...")
        
        # Infer latent shape from decoder config
        latent_ch = model.decoder._init_kwargs.get('latent_channels', 4)
        up_steps = model.decoder._init_kwargs.get('upsampling_steps', 4)
        spatial_res = 512 // (2 ** up_steps)
        latent_shape = (latent_ch, spatial_res, spatial_res)
        
        num_steps = model.scheduler.num_steps
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating fake latents"):
                # Calculate remaining samples needed
                samples_generated = sum(l.shape[0] for l in all_latents)
                current_batch_size = min(batch_size, num_samples - samples_generated)
                
                # Start with clean latents (random noise that will be denoised)
                # In practice, we'll add noise and then predict, matching training
                dummy = torch.zeros((current_batch_size, *latent_shape), device=device_obj)
                clean_latents = model.scheduler.randn_like(dummy)
                
                # Sample random timesteps (same as during training)
                t = torch.randint(0, num_steps, (current_batch_size,), device=device_obj)
                
                # Add noise to get noisy latents
                noise = model.scheduler.randn_like(clean_latents)
                result = model.scheduler.add_noise(clean_latents, noise, t, return_scaled_noise=True)
                noisy_latents, _ = result
                
                # Predict noise using model
                pred_noise = model.unet(noisy_latents, t, cond=None)
                
                # Compute predicted clean latents (single-step prediction)
                # This matches what happens during diffusion training
                alpha_bars = model.scheduler.alpha_bars.to(device_obj)
                alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
                
                # Predict x0 from noisy latents: x0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
                pred_latents = (noisy_latents - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp(min=1e-8)
                pred_latents = torch.clamp(pred_latents, -10.0, 10.0)
                
                all_latents.append(pred_latents.cpu())
    else:
        # Use full DDPM sampling (original approach)
        num_steps = model.scheduler.num_steps
        
        print(f"Generating {num_samples} fake latents using DDPM ({num_steps} steps from scheduler)...")
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating fake latents"):
                # Calculate remaining samples needed
                samples_generated = sum(l.shape[0] for l in all_latents)
                current_batch_size = min(batch_size, num_samples - samples_generated)
                
                # Generate samples using DDPM (stochastic, better quality and diversity)
                sample_output = model.sample(
                    batch_size=current_batch_size,
                    num_steps=num_steps,
                    method="ddpm",
                    eta=1.0,
                    device=device_obj,
                    verbose=False
                )
                
                # Get latents
                if "latent" in sample_output:
                    latents = sample_output["latent"]
                else:
                    raise ValueError("Model should return latents, not images")
                
                all_latents.append(latents.cpu())
    
    fake_latents = torch.cat(all_latents, dim=0)[:num_samples]
    print(f"Generated {len(fake_latents)} fake latents: {fake_latents.shape}")
    return fake_latents


def get_real_latents(
    manifest_path,
    autoencoder_checkpoint,
    num_samples,
    batch_size=32,
    device="cuda",
    seed=42
):
    """
    Get real latents from dataset. Uses pre-embedded latents if available, otherwise encodes RGB images.
    This is a one-time operation at the beginning of each iteration (not during training steps).
    
    Args:
        manifest_path: Path to manifest CSV (preferred: contains 'latent_path' column)
        autoencoder_checkpoint: Path to autoencoder checkpoint (needed if encoding RGB)
        num_samples: Number of real samples to get
        batch_size: Batch size for encoding (if needed)
        device: Device to use
        seed: Random seed (different seed per iteration = different subset)
    
    Returns:
        Tensor of real latents [N, C, H, W] (clean latents representing viable layouts)
    """
    device_obj = torch.device(device)
    manifest_path = Path(manifest_path)
    
    # Load manifest and filter for real (non-augmented) images
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} total samples")
    
    # Filter non-empty layouts
    if "is_empty" in df.columns:
        df = df[df["is_empty"] == False].copy()
    
    # Filter for non-augmented images
    if "is_augmented" in df.columns:
        df_real = df[df["is_augmented"] == False].copy()
        print(f"Found {len(df_real)} non-augmented images")
    else:
        # Check path patterns
        layout_col = "layout_path" if "layout_path" in df.columns else "path"
        df = df.dropna(subset=[layout_col])
        if layout_col in df.columns:
            df["_is_augmented"] = df[layout_col].apply(is_augmented_path)
            df_real = df[df["_is_augmented"] == False].copy()
            df_real = df_real.drop(columns=["_is_augmented"])
            print(f"Found {len(df_real)} non-augmented images (by path pattern)")
        else:
            df_real = df.copy()
            print(f"Warning: Cannot identify augmented images, using all {len(df_real)} images")
    
    # Ensure we have enough samples
    if len(df_real) < num_samples:
        print(f"Warning: Only {len(df_real)} real images available, using all of them")
        num_samples = len(df_real)
    
    # Randomly select num_samples (different subset each iteration due to seed)
    if len(df_real) > num_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(df_real), size=num_samples, replace=False)
        df_real = df_real.iloc[indices].reset_index(drop=True)
    
    print(f"Selected {len(df_real)} real images")
    
    # Check if pre-embedded latents are available (preferred, faster)
    has_latent_path = "latent_path" in df_real.columns
    if has_latent_path:
        # Check if all selected samples have valid latent paths
        df_real = df_real.dropna(subset=["latent_path"])
        if len(df_real) < num_samples:
            print(f"Warning: Only {len(df_real)} samples have valid latent_path, using all of them")
            num_samples = len(df_real)
    
    if has_latent_path and len(df_real) > 0:
        # Load pre-embedded latents directly (preferred, faster)
        print(f"Loading {len(df_real)} pre-embedded latents from manifest...")
        all_latents = []
        manifest_dir = manifest_path.parent
        
        for idx, row in tqdm(df_real.iterrows(), total=len(df_real), desc="Loading latents"):
            latent_path = Path(row["latent_path"])
            if not latent_path.is_absolute():
                latent_path = manifest_dir / latent_path
            
            if not latent_path.exists():
                print(f"Warning: Latent file not found: {latent_path}, will encode from RGB instead")
                has_latent_path = False
                break
            
            latent = torch.load(latent_path, map_location="cpu")
            # Handle different tensor shapes (ensure it's [C, H, W])
            if latent.dim() == 4:
                latent = latent.squeeze(0)  # Remove batch dimension if present
            all_latents.append(latent)
        
        if has_latent_path:  # All latents loaded successfully
            real_latents = torch.stack(all_latents, dim=0)[:num_samples]
            print(f"Loaded {len(real_latents)} pre-embedded latents: {real_latents.shape}")
            return real_latents
    
    # Fallback: encode RGB images at the beginning of iteration (one-time operation, not during training)
    print(f"Pre-embedded latents not available, encoding RGB images at iteration start...")
    # Load autoencoder
    print(f"Loading autoencoder from {autoencoder_checkpoint}")
    autoencoder = Autoencoder.load_checkpoint(autoencoder_checkpoint, map_location=device)
    autoencoder = autoencoder.to(device_obj)
    autoencoder.eval()
    
    # Create temporary manifest
    temp_manifest = manifest_path.parent / f"temp_real_manifest_{seed}.csv"
    df_real.to_csv(temp_manifest, index=False)
    
    # Create dataset
    dataset = ManifestDataset(
        manifest=str(temp_manifest),
        outputs={"rgb": "layout_path"},
        return_path=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device_obj.type == "cuda"
    )
    
    all_latents = []
    print(f"Encoding {len(df_real)} real images (one-time operation at iteration start)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding real images"):
            rgb = batch["rgb"].to(device_obj)
            
            # Encode - encoder expects tensor directly, not dict
            encoder_out = autoencoder.encoder(rgb)
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
            
            all_latents.append(latents.cpu())
    
    # Clean up temp manifest
    temp_manifest.unlink()
    
    real_latents = torch.cat(all_latents, dim=0)[:num_samples]
    print(f"Encoded {len(real_latents)} real latents: {real_latents.shape}")
    return real_latents


def save_samples_with_discriminator(
    model,
    discriminator,
    output_dir,
    step,
    sample_batch_size=64,
    exp_name=None,
    num_best=16,
    num_worst=16
):
    """
    Generate samples, evaluate with discriminator, and save best/worst samples.
    
    Args:
        model: Diffusion model
        discriminator: Discriminator model
        output_dir: Output directory
        step: Current training step
        sample_batch_size: Number of samples to generate
        exp_name: Experiment name
        num_best: Number of best (high discriminator score) samples to save
        num_worst: Number of worst (low discriminator score) samples to save
    """
    model.eval()
    discriminator.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = next(model.parameters()).device
    
    # Save current RNG state
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = None
    if torch.cuda.is_available():
        cuda_rng_states = torch.cuda.get_rng_state_all()
    
    sampling_seed = 42 + step
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)
    
    num_steps_scheduler = model.scheduler.num_steps
    print(f"  Generating {sample_batch_size} samples using DDPM ({num_steps_scheduler} steps)...")
    
    with torch.no_grad():
        # Generate samples
        sample_output = model.sample(
            batch_size=sample_batch_size,
            num_steps=num_steps_scheduler,
            method="ddpm",
            eta=1.0,
            device=device_obj,
            verbose=False
        )
        
        # Get latents
        if "latent" in sample_output:
            sample_latents = sample_output["latent"]
        else:
            raise ValueError("Model should return latents")
        
        # Evaluate with discriminator
        discriminator_scores = discriminator(sample_latents)  # [B, 1]
        discriminator_scores = discriminator_scores.squeeze(1)  # [B]
        
        # Decode to images
        decoded = model.decoder({"latent": sample_latents})
        samples = (decoded["rgb"] + 1.0) / 2.0  # Normalize to [0, 1]
        
        # Get indices sorted by discriminator score (high to low)
        sorted_indices = torch.argsort(discriminator_scores, descending=True)
        
        # Get best and worst samples
        best_indices = sorted_indices[:num_best]
        worst_indices = sorted_indices[-num_worst:]
        
        best_samples = samples[best_indices]
        worst_samples = samples[worst_indices]
        best_scores = discriminator_scores[best_indices].cpu().numpy()
        worst_scores = discriminator_scores[worst_indices].cpu().numpy()
        
        # Save best samples
        best_grid_n = int(math.sqrt(num_best))
        best_path = samples_dir / (f"{exp_name}_step_{step:06d}_best_samples.png" if exp_name else f"step_{step:06d}_best_samples.png")
        save_image(best_samples, best_path, nrow=best_grid_n, normalize=False)
        
        # Save worst samples
        worst_grid_n = int(math.sqrt(num_worst))
        worst_path = samples_dir / (f"{exp_name}_step_{step:06d}_worst_samples.png" if exp_name else f"step_{step:06d}_worst_samples.png")
        save_image(worst_samples, worst_path, nrow=worst_grid_n, normalize=False)
        
        # Save all samples grid
        all_grid_n = int(math.sqrt(sample_batch_size))
        all_path = samples_dir / (f"{exp_name}_step_{step:06d}_all_samples.png" if exp_name else f"step_{step:06d}_all_samples.png")
        save_image(samples, all_path, nrow=all_grid_n, normalize=False)
        
        # Save metrics (convert all to Python native types for JSON serialization)
        metrics = {
            "step": int(step),
            "mean_discriminator_score": float(discriminator_scores.mean().item()),
            "std_discriminator_score": float(discriminator_scores.std().item()),
            "min_discriminator_score": float(discriminator_scores.min().item()),
            "max_discriminator_score": float(discriminator_scores.max().item()),
            "median_discriminator_score": float(discriminator_scores.median().item()),
            "best_mean_score": float(best_scores.mean()),
            "worst_mean_score": float(worst_scores.mean()),
            "num_samples": int(sample_batch_size)
        }
        
        # Save metrics to JSON
        import json
        metrics_path = samples_dir / (f"{exp_name}_step_{step:06d}_sample_metrics.json" if exp_name else f"step_{step:06d}_sample_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  Saved samples:")
        print(f"    Best {num_best} samples (mean score: {best_scores.mean():.4f}) -> {best_path.name}")
        print(f"    Worst {num_worst} samples (mean score: {worst_scores.mean():.4f}) -> {worst_path.name}")
        print(f"    All {sample_batch_size} samples (mean score: {metrics['mean_discriminator_score']:.4f}) -> {all_path.name}")
        print(f"    Metrics -> {metrics_path.name}")
    
    # Restore RNG state
    torch.set_rng_state(cpu_rng_state)
    if torch.cuda.is_available() and cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    
    return metrics


def train_discriminator_iteration_steps(
    real_latents,
    fake_latents,
    output_dir,
    iteration,
    discriminator_config=None,
    max_steps=50000,  # Large default for convergence
    batch_size=64,
    learning_rate=0.0002,
    eval_interval=500,
    early_stopping_patience=10,  # Number of eval intervals without improvement
    early_stopping_min_delta=0.0001,  # Minimum change to count as improvement
    device="cuda",
    seed=42
):
    """
    Train discriminator for one iteration using steps-based training.
    
    Returns:
        Path to best discriminator checkpoint
    """
    import torch.nn as nn
    
    set_deterministic(seed)
    device_obj = torch.device(device)
    iteration_dir = output_dir / f"discriminator_iter_{iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    
    # Save latents for this iteration
    real_path = iteration_dir / "real_latents.pt"
    fake_path = iteration_dir / "fake_latents.pt"
    torch.save(real_latents, real_path)
    torch.save(fake_latents, fake_path)
    
    print(f"\n{'='*60}")
    print(f"Training Discriminator - Iteration {iteration} (Steps-based)")
    print(f"{'='*60}")
    print(f"  Max steps: {max_steps} (train until convergence)")
    print(f"  Batch size: {batch_size}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Early stopping patience: {early_stopping_patience} evaluations")
    print(f"  Early stopping min delta: {early_stopping_min_delta}")
    
    # Load latents
    real_latents = real_latents.to(device_obj)
    fake_latents = fake_latents.to(device_obj)
    
    # Create labels
    real_labels = torch.ones(len(real_latents), 1, device=device_obj)
    fake_labels = torch.zeros(len(fake_latents), 1, device=device_obj)
    
    # Combine datasets
    all_latents = torch.cat([real_latents, fake_latents], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)
    
    # Split train/val (80/20)
    split_idx = int(0.8 * len(all_latents))
    train_latents = all_latents[:split_idx]
    train_labels = all_labels[:split_idx]
    val_latents = all_latents[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"Train: {len(train_latents)}, Val: {len(val_latents)}")
    
    # Build discriminator
    if discriminator_config:
        discriminator = LatentDiscriminator.from_config(discriminator_config)
    else:
        latent_channels = real_latents.shape[1]
        discriminator = LatentDiscriminator(
            latent_channels=latent_channels,
            base_channels=64,
            num_layers=4
        )
    discriminator = discriminator.to(device_obj)
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Check for resume
    discriminator_checkpoint_path = iteration_dir / "discriminator_checkpoint_latest.pt"
    start_step = 0
    avg_val_loss = float("inf")
    val_acc = 0.0
    best_val_loss = float("inf")
    history = []
    
    if discriminator_checkpoint_path.exists():
        print(f"  Found existing discriminator checkpoint, loading to resume...")
        checkpoint_data = torch.load(discriminator_checkpoint_path, map_location=device_obj)
        if "step" in checkpoint_data:
            start_step = checkpoint_data["step"]
            discriminator.load_state_dict(checkpoint_data["state_dict"])
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            if "history" in checkpoint_data:
                history = checkpoint_data["history"]
            if "best_val_loss" in checkpoint_data:
                best_val_loss = checkpoint_data["best_val_loss"]
            if "val_loss" in checkpoint_data:
                avg_val_loss = checkpoint_data["val_loss"]
            if "val_acc" in checkpoint_data:
                val_acc = checkpoint_data["val_acc"]
            print(f"  Resumed from step {start_step}")
            if avg_val_loss != float("inf"):
                print(f"  Last val_loss: {avg_val_loss:.4f}, val_acc: {val_acc:.4f}")
    
    # Training loop
    step = start_step
    num_train_batches = (len(train_latents) + batch_size - 1) // batch_size
    steps_without_improvement = 0
    best_step = start_step
    
    # Create infinite iterator over training data
    train_indices = torch.randperm(len(train_latents), device=device_obj)
    
    pbar = tqdm(total=max_steps, initial=start_step, desc="Training discriminator")
    
    while step < max_steps:
        discriminator.train()
        
        # Get batch
        batch_start = (step % num_train_batches) * batch_size
        batch_end = min(batch_start + batch_size, len(train_latents))
        batch_indices = train_indices[batch_start:batch_end]
        
        batch_latents = train_latents[batch_indices]
        batch_labels = train_labels[batch_indices]
        
        # Forward
        optimizer.zero_grad()
        scores = discriminator(batch_latents)
        loss = criterion(scores, batch_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        step += 1
        pbar.update(1)
        
        # Evaluate periodically
        if step % eval_interval == 0 or step == max_steps:
            # Train metrics (on current batch)
            train_loss = loss.item()
            predictions = (scores > 0.5).float()
            train_acc = (predictions == batch_labels).sum().item() / len(batch_labels)
            
            # Validation
            discriminator.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_tp = 0
            val_fp = 0
            val_fn = 0
            
            with torch.no_grad():
                num_val_batches = (len(val_latents) + batch_size - 1) // batch_size
                for val_batch_idx in range(num_val_batches):
                    val_start = val_batch_idx * batch_size
                    val_end = min(val_start + batch_size, len(val_latents))
                    
                    val_batch_latents = val_latents[val_start:val_end]
                    val_batch_labels = val_labels[val_start:val_end]
                    
                    val_scores = discriminator(val_batch_latents)
                    val_loss_batch = criterion(val_scores, val_batch_labels)
                    
                    val_loss += val_loss_batch.item()
                    val_predictions = (val_scores > 0.5).float()
                    val_correct += (val_predictions == val_batch_labels).sum().item()
                    val_total += len(val_batch_labels)
                    
                    # Compute confusion matrix for F1 score
                    val_tp += ((val_predictions == 1) & (val_batch_labels == 1)).sum().item()
                    val_fp += ((val_predictions == 1) & (val_batch_labels == 0)).sum().item()
                    val_fn += ((val_predictions == 0) & (val_batch_labels == 1)).sum().item()
            
            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float("inf")
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            if num_val_batches > 0:
                scheduler.step(avg_val_loss)
            
            print(f"\nStep {step}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Calculate F1 score properly using precision and recall
            # For binary classification: F1 = 2 * (precision * recall) / (precision + recall)
            # Where precision = TP / (TP + FP), recall = TP / (TP + FN)
            
            # Train F1
            train_tp = ((predictions == 1) & (batch_labels == 1)).sum().item()
            train_fp = ((predictions == 1) & (batch_labels == 0)).sum().item()
            train_fn = ((predictions == 0) & (batch_labels == 1)).sum().item()
            train_precision = train_tp / (train_tp + train_fp + 1e-8)
            train_recall = train_tp / (train_tp + train_fn + 1e-8)
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
            
            # Validation F1
            val_precision = val_tp / (val_tp + val_fp + 1e-8)
            val_recall = val_tp / (val_tp + val_fn + 1e-8)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
            
            history.append({
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "steps_without_improvement": steps_without_improvement
            })
            
            # Check for improvement
            improvement = best_val_loss - avg_val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = avg_val_loss
                best_step = step
                steps_without_improvement = 0
                
                losses_checkpoint_dir = Path("models/losses/checkpoints")
                losses_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save best checkpoint
                discriminator.save_checkpoint(
                    losses_checkpoint_dir / "discriminator_best.pt",
                    step=step,
                    val_loss=avg_val_loss,
                    val_acc=val_acc
                )
                
                # Also save to iteration directory for resume
                discriminator.save_checkpoint(
                    iteration_dir / "discriminator_checkpoint_latest.pt",
                    step=step,
                    val_loss=avg_val_loss,
                    val_acc=val_acc,
                    optimizer_state_dict=optimizer.state_dict(),
                    history=history,
                    best_val_loss=best_val_loss
                )
                print(f"  ✓ Improved! Saved best checkpoint (val_loss={best_val_loss:.4f}, val_acc={val_acc:.4f})")
            else:
                steps_without_improvement += 1
                print(f"  No improvement for {steps_without_improvement}/{early_stopping_patience} evaluations")
            
            # Save latest checkpoint after evaluation (for resume capability)
            discriminator.save_checkpoint(
                iteration_dir / "discriminator_checkpoint_latest.pt",
                step=step,
                val_loss=avg_val_loss,
                val_acc=val_acc,
                optimizer_state_dict=optimizer.state_dict(),
                history=history,
                best_val_loss=best_val_loss
            )
            
            # Early stopping check
            if steps_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered!")
                print(f"  No improvement for {steps_without_improvement} evaluations")
                print(f"  Best validation loss: {best_val_loss:.4f} at step {best_step}")
                print(f"  Current step: {step}")
                print(f"{'='*60}")
                break
            
            # Reshuffle training data periodically
            if step % num_train_batches == 0:
                train_indices = torch.randperm(len(train_latents), device=device_obj)
    
    pbar.close()
    
    # If we never evaluated, do a final evaluation now
    if avg_val_loss == float("inf"):
        print(f"\nFinal evaluation at step {step}...")
        discriminator.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            num_val_batches = (len(val_latents) + batch_size - 1) // batch_size
            for val_batch_idx in range(num_val_batches):
                val_start = val_batch_idx * batch_size
                val_end = min(val_start + batch_size, len(val_latents))
                
                val_batch_latents = val_latents[val_start:val_end]
                val_batch_labels = val_labels[val_start:val_end]
                
                val_scores = discriminator(val_batch_latents)
                val_loss_batch = criterion(val_scores, val_batch_labels)
                
                val_loss += val_loss_batch.item()
                val_predictions = (val_scores > 0.5).float()
                val_correct += (val_predictions == val_batch_labels).sum().item()
                val_total += len(val_batch_labels)
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"  Final Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Save final checkpoint
    losses_checkpoint_dir = Path("models/losses/checkpoints")
    losses_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    discriminator.save_checkpoint(
        losses_checkpoint_dir / "discriminator_final.pt",
        step=step,
        val_loss=avg_val_loss,
        val_acc=val_acc
    )
    
    # Also save to iteration directory for resume
    discriminator.save_checkpoint(
        iteration_dir / "discriminator_checkpoint_latest.pt",
        step=step,
        val_loss=avg_val_loss,
        val_acc=val_acc,
        optimizer_state_dict=optimizer.state_dict(),
        history=history
    )
    
    # Save history and plot
    df = pd.DataFrame(history)
    csv_path = iteration_dir / "discriminator_history.csv"
    df.to_csv(csv_path, index=False)
    
    # Plot metrics
    if len(history) > 0:
        plot_discriminator_metrics(df, iteration_dir, iteration, exp_name="discriminator")
    
    print(f"\nDiscriminator training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    
    # Return path to best checkpoint
    best_checkpoint = losses_checkpoint_dir / "discriminator_best.pt"
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Discriminator checkpoint not found at {best_checkpoint}")
    
    return best_checkpoint


def train_diffusion_with_discriminator_steps(
    model,
    config,
    discriminator_checkpoint,
    train_loader,
    val_loader,
    output_dir,
    iteration,
    max_steps=100000,  # Large default for convergence
    eval_interval_steps=1000,  # Evaluate every N steps
    sample_interval_steps=5000,  # Sample every N steps
    early_stopping_patience=10,  # Number of evaluations without improvement
    early_stopping_min_delta=0.0001,
    device="cuda",
    use_amp=False,
    max_grad_norm=None
):
    """
    Train diffusion model with discriminator loss for one iteration.
    
    Returns:
        Updated model, best_val_loss
    """
    device_obj = to_device(device)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    
    # Load discriminator
    print(f"\nLoading discriminator from {discriminator_checkpoint}")
    discriminator_checkpoint_data = torch.load(discriminator_checkpoint, map_location=device_obj)
    discriminator_config = discriminator_checkpoint_data.get("config")
    
    if discriminator_config:
        discriminator = LatentDiscriminator.from_config(discriminator_config)
    else:
        # Infer from latents
        sample_latent = next(iter(train_loader))["latent"][0:1]
        latent_channels = sample_latent.shape[1]
        discriminator = LatentDiscriminator(
            latent_channels=latent_channels,
            base_channels=64,
            num_layers=4
        )
    
    discriminator.load_state_dict(discriminator_checkpoint_data["state_dict"])
    discriminator = discriminator.to(device_obj)
    discriminator.eval()  # Freeze during training
    for param in discriminator.parameters():
        param.requires_grad = False
    
    print(f"  Discriminator loaded")
    
    # Build loss function (should include discriminator loss)
    loss_fn = build_loss(config)
    print(f"  Loss type: {type(loss_fn).__name__}")
    if hasattr(loss_fn, 'losses'):
        print(f"  Loss components: {[type(l).__name__ for l in loss_fn.losses]}")
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Build scheduler
    scheduler = build_scheduler(optimizer, config, last_epoch=-1)
    
    # Training loop
    best_val_loss = float("inf")
    training_history = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_csv_path = output_dir / f"{exp_name}_metrics_iter_{iteration}.csv"
    
    # Check for resume
    latest_checkpoint_path = checkpoint_dir / f"{exp_name}_iter_{iteration}_checkpoint_latest.pt"
    start_step = 0
    if latest_checkpoint_path.exists():
        print(f"  Found existing checkpoint, loading to resume...")
        checkpoint_data = torch.load(latest_checkpoint_path, map_location=device_obj)
        if "step" in checkpoint_data:
            start_step = checkpoint_data["step"]
            model.load_state_dict(checkpoint_data["state_dict"])
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            if "training_history" in checkpoint_data:
                training_history = checkpoint_data["training_history"]
            if "best_val_loss" in checkpoint_data:
                best_val_loss = checkpoint_data["best_val_loss"]
            print(f"  Resumed from step {start_step}")
    
    print(f"\n{'='*60}")
    print(f"Training Diffusion Model with Discriminator - Iteration {iteration} (Steps-based)")
    print(f"{'='*60}")
    print(f"  Max steps: {max_steps} (train until convergence)")
    print(f"  Start step: {start_step}")
    print(f"  Batch size: {config['training'].get('batch_size', 32)}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Eval interval: {eval_interval_steps} steps")
    print(f"  Sample interval: {sample_interval_steps} steps")
    print(f"  Early stopping patience: {early_stopping_patience} evaluations")
    
    # Steps-based training
    step = start_step
    evals_without_improvement = 0
    best_step = start_step
    train_iter = iter(train_loader)
    
    pbar = tqdm(total=max_steps, initial=start_step, desc="Training diffusion")
    
    while step < max_steps:
        model.train()
        
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = move_batch_to_device(batch, device_obj)
        
        # Get latents (must be pre-embedded, no on-the-fly encoding)
        latents = batch.get("latent")
        if latents is None:
            raise ValueError(
                "Dataset must provide pre-embedded 'latent' key. "
                "No on-the-fly encoding is allowed. Please ensure your dataset manifest includes 'latent_path'."
            )
        
        # Sample random timesteps
        num_steps_scheduler = model.scheduler.num_steps
        t = torch.randint(0, num_steps_scheduler, (latents.shape[0],), device=device_obj)
        noise = model.scheduler.randn_like(latents)
        cond = batch.get("cond", None)
        
        # Compute loss
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    discriminator, use_amp, device_obj
                )
            
            optimizer.zero_grad()
            scaler = getattr(train_diffusion_with_discriminator_steps, '_scaler', None)
            if scaler is None:
                scaler = create_grad_scaler(use_amp, device_obj)
                train_diffusion_with_discriminator_steps._scaler = scaler
            
            if scaler:
                scaler.scale(total_loss_val).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
        else:
            total_loss_val, logs = compute_loss(
                model, batch, latents, t, noise, cond, loss_fn,
                discriminator, use_amp, device_obj
            )
            
            optimizer.zero_grad()
            total_loss_val.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        step += 1
        pbar.update(1)
        
        # Evaluate periodically (skip if eval_interval is too small relative to max_steps to avoid overhead)
        should_eval = (step % eval_interval_steps == 0 or step == max_steps) and (eval_interval_steps > 1 or step == max_steps or step == start_step + 1)
        if should_eval:
            # Get average train metrics over recent steps
            train_loss = total_loss_val.detach().item()
            train_logs = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in logs.items()}
            
            # Validate (limit validation batches for very frequent evaluations)
            val_loss = float("inf")
            val_logs = {}
            if val_loader:
                model.eval()
                val_loss_sum = 0.0
                val_logs_sum = {}
                val_count = 0
                
                # Limit validation batches if evaluating very frequently (e.g., every step)
                max_val_batches = None
                if eval_interval_steps == 1 and step < max_steps:
                    max_val_batches = 5  # Only use 5 batches for frequent evaluations
                
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_loader):
                        if max_val_batches is not None and val_batch_idx >= max_val_batches:
                            break
                        val_batch = move_batch_to_device(val_batch, device_obj)
                        val_latents = val_batch.get("latent")
                        if val_latents is None:
                            raise ValueError(
                                "Validation dataset must provide pre-embedded 'latent' key. "
                                "No on-the-fly encoding is allowed. Please ensure your dataset manifest includes 'latent_path'."
                            )
                        
                        val_t = torch.randint(0, num_steps_scheduler, (val_latents.shape[0],), device=device_obj)
                        val_noise = model.scheduler.randn_like(val_latents)
                        val_cond = val_batch.get("cond", None)
                        
                        if use_amp and device_obj.type == "cuda":
                            with torch.amp.autocast('cuda'):
                                val_loss_batch, val_logs_batch = compute_loss(
                                    model, val_batch, val_latents, val_t, val_noise, val_cond, loss_fn,
                                    discriminator, use_amp, device_obj
                                )
                        else:
                            val_loss_batch, val_logs_batch = compute_loss(
                                model, val_batch, val_latents, val_t, val_noise, val_cond, loss_fn,
                                discriminator, use_amp, device_obj
                            )
                        
                        val_loss_sum += val_loss_batch.item()
                        for k, v in val_logs_batch.items():
                            if k not in val_logs_sum:
                                val_logs_sum[k] = 0.0
                            val_logs_sum[k] += v.item() if isinstance(v, torch.Tensor) else v
                        val_count += 1
                
                val_loss = val_loss_sum / val_count
                val_logs = {k: v / val_count for k, v in val_logs_sum.items()}
            
            print(f"\nStep {step}: Train Loss={train_loss:.6f}")
            for k, v in train_logs.items():
                print(f"  train_{k}: {v:.6f}")
            if val_loader:
                print(f"  Val Loss={val_loss:.6f}")
                for k, v in val_logs.items():
                    print(f"  val_{k}: {v:.6f}")
            
            # Evaluate discriminator on real vs generated samples for diagnostics
            if should_eval:
                discriminator.eval()
                with torch.no_grad():
                    # Get a batch of real latents from validation set
                    if val_loader:
                        real_batch = next(iter(val_loader))
                        real_batch = move_batch_to_device(real_batch, device_obj)
                        real_latents_eval = real_batch.get("latent")
                        if real_latents_eval is not None:
                            real_scores = discriminator(real_latents_eval)
                            real_mean_score = real_scores.mean().item()
                            print(f"  Discriminator on REAL samples: mean_score={real_mean_score:.4f}")
                    
                    # Generate a small batch to check discriminator scores
                    model.eval()
                    sample_batch = 32  # Fixed small batch for diagnostic purposes
                    sample_output = model.sample(
                        batch_size=sample_batch,
                        num_steps=model.scheduler.num_steps,
                        method="ddpm",
                        eta=1.0,
                        device=device_obj,
                        verbose=False
                    )
                    if "latent" in sample_output:
                        fake_latents_eval = sample_output["latent"]
                        fake_scores = discriminator(fake_latents_eval)
                        fake_mean_score = fake_scores.mean().item()
                        fake_min_score = fake_scores.min().item()
                        fake_max_score = fake_scores.max().item()
                        print(f"  Discriminator on GENERATED samples: mean_score={fake_mean_score:.4f}, min={fake_min_score:.4f}, max={fake_max_score:.4f}")
                        print(f"  Score gap (real - fake): {real_mean_score - fake_mean_score:.4f}" if val_loader and real_latents_eval is not None else "")
                    model.train()
            
            # Save samples periodically with discriminator evaluation
            # Skip sampling if interval is too small (very expensive operation)
            should_sample = (step % sample_interval_steps == 0 and step > start_step) and \
                           (sample_interval_steps > 1 or step == max_steps)
            if should_sample:
                sample_metrics = save_samples_with_discriminator(
                    model, discriminator, output_dir, step,
                    sample_batch_size=64, exp_name=f"{exp_name}_iter_{iteration}",
                    num_best=16, num_worst=16
                )
                # Add sample metrics to history (will be added to next history entry)
                # Store temporarily to add to history entry below
                current_sample_metrics = sample_metrics
            else:
                current_sample_metrics = {}
        
        # Record history with all metrics (only if we evaluated)
        if should_eval:
            history_entry = {
                "step": step,
                "iteration": iteration,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "evals_without_improvement": evals_without_improvement,
                **{f"train_{k}": v for k, v in train_logs.items()},
                **{f"val_{k}": v for k, v in val_logs.items()},
                **current_sample_metrics  # Add sample metrics if available
            }
            training_history.append(history_entry)
            
            # Save metrics to CSV and plot
            save_metrics_csv(training_history, metrics_csv_path)
            if len(training_history) > 0:
                df = pd.DataFrame(training_history)
                plot_diffusion_metrics(df, output_dir, iteration, exp_name=exp_name)
            
            # Check for improvement (only when we evaluated)
            improvement = best_val_loss - val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = val_loss
                best_step = step
                evals_without_improvement = 0
                
                best_checkpoint_path = checkpoint_dir / f"{exp_name}_iter_{iteration}_checkpoint_best.pt"
                model.save_checkpoint(
                    best_checkpoint_path,
                    step=step,
                    best_val_loss=best_val_loss,
                    training_history=training_history,
                    iteration=iteration
                )
                print(f"  ✓ Improved! Saved best checkpoint (val_loss={best_val_loss:.6f})")
            else:
                evals_without_improvement += 1
                print(f"  No improvement for {evals_without_improvement}/{early_stopping_patience} evaluations")
            
            # Save latest checkpoint after evaluation (for resume capability)
            latest_checkpoint_path = checkpoint_dir / f"{exp_name}_iter_{iteration}_checkpoint_latest.pt"
            model.save_checkpoint(
                latest_checkpoint_path,
                step=step,
                best_val_loss=best_val_loss,
                training_history=training_history,
                iteration=iteration,
                optimizer_state_dict=optimizer.state_dict()
            )
            
            # Early stopping check
            if evals_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered!")
                print(f"  No improvement for {evals_without_improvement} evaluations")
                print(f"  Best validation loss: {best_val_loss:.6f} at step {best_step}")
                print(f"  Current step: {step}")
                print(f"{'='*60}")
                break
    
    pbar.close()
    
    # Save final checkpoint if training completed without final evaluation
    if step == max_steps and (step % eval_interval_steps != 0):
        latest_checkpoint_path = checkpoint_dir / f"{exp_name}_iter_{iteration}_checkpoint_latest.pt"
        model.save_checkpoint(
            latest_checkpoint_path,
            step=step,
            best_val_loss=best_val_loss,
            training_history=training_history,
            iteration=iteration,
            optimizer_state_dict=optimizer.state_dict()
        )
    
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Discriminator Training Pipeline"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment config YAML file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of real and fake samples to generate per iteration (default: 5000)"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="Number of adversarial training iterations (default: None = train until convergence)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of iterations (safety limit, default: 10)"
    )
    parser.add_argument(
        "--iteration_convergence_patience",
        type=int,
        default=2,
        help="Number of iterations without improvement to stop (default: 2)"
    )
    parser.add_argument(
        "--iteration_convergence_min_delta",
        type=float,
        default=0.01,
        help="Minimum improvement in viability score to count as progress (default: 0.01)"
    )
    parser.add_argument(
        "--discriminator_max_steps",
        type=int,
        default=50000,
        help="Max steps to train discriminator per iteration (trains until convergence with early stopping, default: 50000)"
    )
    parser.add_argument(
        "--diffusion_max_steps",
        type=int,
        default=100000,
        help="Max steps to train diffusion model per iteration (trains until convergence with early stopping, default: 100000)"
    )
    parser.add_argument(
        "--discriminator_eval_interval",
        type=int,
        default=500,
        help="Evaluate discriminator every N steps (default: 500)"
    )
    parser.add_argument(
        "--diffusion_eval_interval",
        type=int,
        default=1000,
        help="Evaluate diffusion model every N steps (default: 1000)"
    )
    parser.add_argument(
        "--diffusion_sample_interval",
        type=int,
        default=5000,
        help="Sample from diffusion model every N steps (default: 5000)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience (number of evaluations without improvement, default: 10)"
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0001,
        help="Early stopping minimum delta (minimum change to count as improvement, default: 0.0001)"
    )
    parser.add_argument(
        "--discriminator_batch_size",
        type=int,
        default=512,
        help="Batch size for discriminator training (default: 512). For 32x32x16 latents, can use 256-1024 on 24GB+ GPUs"
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=32,
        help="Batch size for generating samples (default: 32)"
    )
    parser.add_argument(
        "--generation_steps",
        type=int,
        default=None,
        help="DEPRECATED: Number of steps is now determined by model.scheduler.num_steps (inherent to diffusion model). This argument is ignored."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        default=0,
        help="Start from this iteration (for resuming, default: 0)"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    print(f"Experiment: {exp_name}")
    
    # Set deterministic behavior
    training_seed = config.get("training", {}).get("seed", args.seed)
    set_deterministic(training_seed)
    print(f"Set deterministic mode with seed: {training_seed}")
    
    # Get device
    device = get_device(config) if "device" not in config.get("training", {}) else args.device
    device_obj = to_device(device)
    print(f"Device: {device}")
    
    # Get output directory
    output_dir = config.get("experiment", {}).get("save_path")
    if output_dir is None:
        output_dir = Path("outputs") / exp_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Resume state file path
    resume_state_path = output_dir / "resume_state.json"
    
    # Load resume state if it exists
    resume_state = None
    if resume_state_path.exists():
        try:
            with open(resume_state_path, 'r') as f:
                resume_state = json.load(f)
            print(f"\n{'='*60}")
            print(f"Found resume state file: {resume_state_path}")
            print(f"  Last completed iteration: {resume_state.get('last_completed_iteration', -1)}")
            print(f"  Best validation loss: {resume_state.get('best_val_loss', 'N/A')}")
            print(f"  Last update: {resume_state.get('last_update', 'N/A')}")
            print(f"{'='*60}")
            
            # Auto-resume from last completed iteration + 1
            if args.start_iteration == 0:  # Only auto-resume if user didn't specify
                auto_resume_iter = resume_state.get('last_completed_iteration', -1) + 1
                if auto_resume_iter > 0:
                    print(f"Auto-resuming from iteration {auto_resume_iter}")
                    args.start_iteration = auto_resume_iter
                    best_iteration_val_loss = resume_state.get('best_val_loss', float("inf"))
                    if best_iteration_val_loss != float("inf"):
                        print(f"  Best validation loss so far: {best_iteration_val_loss:.6f}")
        except Exception as e:
            print(f"Warning: Could not load resume state: {e}")
            print("Starting from scratch...")
            resume_state = None
    
    # Load diffusion model checkpoint
    stage1_checkpoint = config.get("diffusion", {}).get("stage1_checkpoint")
    if not stage1_checkpoint:
        raise ValueError("Config must specify 'diffusion.stage1_checkpoint'")
    
    # Check if we should load from a previous iteration checkpoint
    if resume_state and args.start_iteration > 0:
        # Try to load from last iteration's checkpoint
        checkpoint_dir = output_dir / "checkpoints"
        last_iter = args.start_iteration - 1
        checkpoint_path = checkpoint_dir / f"{exp_name}_iter_{last_iter}_checkpoint_latest.pt"
        if checkpoint_path.exists():
            print(f"\nLoading diffusion model from previous iteration checkpoint: {checkpoint_path}")
            model, _ = DiffusionModel.load_checkpoint(
                checkpoint_path,
                map_location=device,
                return_extra=True,
                config=config
            )
            model = model.to(device_obj)
            print("Diffusion model loaded from previous iteration checkpoint")
        else:
            # Fall back to stage1 checkpoint
            print(f"\nPrevious iteration checkpoint not found, loading from Stage 1: {stage1_checkpoint}")
            model, _ = DiffusionModel.load_checkpoint(
                stage1_checkpoint,
                map_location=device,
                return_extra=True,
                config=config
            )
            model = model.to(device_obj)
            print("Diffusion model loaded from Stage 1 checkpoint")
    else:
        print(f"\nLoading diffusion model from {stage1_checkpoint}")
        model, _ = DiffusionModel.load_checkpoint(
            stage1_checkpoint,
            map_location=device,
            return_extra=True,
            config=config
        )
        model = model.to(device_obj)
        print("Diffusion model loaded successfully")
    
    # Keep decoder frozen
    if hasattr(model, 'decoder'):
        print("Keeping decoder frozen - only UNet will be trained...")
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    # Get autoencoder checkpoint for encoding real images
    autoencoder_checkpoint = config.get("autoencoder", {}).get("checkpoint")
    if not autoencoder_checkpoint:
        raise ValueError("Config must specify 'autoencoder.checkpoint'")
    
    # Get manifest path
    manifest_path = config.get("dataset", {}).get("manifest")
    if not manifest_path:
        raise ValueError("Config must specify 'dataset.manifest'")
    
    # Get discriminator config
    discriminator_config = config.get("discriminator", {}).get("config")
    
    # Main training loop
    print(f"\n{'='*80}")
    print(f"Starting Stage 2 Discriminator Training Pipeline")
    print(f"{'='*80}")
    if args.num_iterations is None:
        print(f"  Iterations: Train until convergence (max: {args.max_iterations})")
        print(f"  Convergence patience: {args.iteration_convergence_patience} iterations")
        print(f"  Convergence min delta: {args.iteration_convergence_min_delta}")
    else:
        print(f"  Iterations: {args.num_iterations}")
    print(f"  Samples per iteration: {args.num_samples}")
    print(f"  Discriminator batch size: {args.discriminator_batch_size}")
    print(f"  Discriminator max steps: {args.discriminator_max_steps} (with early stopping)")
    print(f"  Diffusion max steps: {args.diffusion_max_steps} (with early stopping)")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  Early stopping min delta: {args.early_stopping_min_delta}")
    print(f"{'='*80}\n")
    
    # Build dataset for diffusion training
    print("Building dataset for diffusion training...")
    dataset = build_dataset(config)
    train_dataset, val_dataset = split_dataset(dataset, config["training"])
    
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 8)
    shuffle = config["training"].get("shuffle", True)
    
    train_loader = train_dataset.make_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
        persistent_workers=num_workers > 0,
        use_weighted_sampling=False
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
            use_weighted_sampling=False
        )
        print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)}")
    
    use_amp = config["training"].get("use_amp", False)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    eval_interval = config["training"].get("eval_interval", 5)
    sample_interval = config["training"].get("sample_interval", 10)
    
    # Convergence tracking for outer loop
    if resume_state and 'best_val_loss' in resume_state and resume_state.get('best_val_loss') is not None:
        best_iteration_val_loss = resume_state.get('best_val_loss', float("inf"))
    else:
        best_iteration_val_loss = float("inf")
    iterations_without_improvement = 0
    max_iterations = args.max_iterations if args.num_iterations is None else args.num_iterations
    iteration_history = []
    
    # Helper function to save resume state
    def save_resume_state(iteration, val_loss, status="in_progress"):
        """Save current training state to resume_state.json"""
        state = {
            "last_completed_iteration": iteration if status == "completed" else iteration - 1,
            "current_iteration": iteration,
            "status": status,  # "in_progress", "completed", "converged", "finished"
            "best_val_loss": float(best_iteration_val_loss) if best_iteration_val_loss != float("inf") else None,
            "last_update": datetime.now().isoformat(),
            "num_iterations": args.num_iterations,
            "max_iterations": max_iterations
        }
        try:
            with open(resume_state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save resume state: {e}")
    
    # Iterative adversarial training
    iteration = args.start_iteration
    while iteration < max_iterations:
        if args.num_iterations is not None:
            print(f"\n{'#'*80}")
            print(f"ITERATION {iteration + 1}/{args.num_iterations}")
            print(f"{'#'*80}")
        else:
            print(f"\n{'#'*80}")
            print(f"ITERATION {iteration + 1}/{max_iterations} (convergence-based)")
            print(f"{'#'*80}")
        
        # Step 1: Generate fake latents from current model
        step_label = f"[Step 1]" if args.num_iterations is None else f"[Step 1/{args.num_iterations}]"
        print(f"\n{step_label} Generating fake latents...")
        fake_latents = generate_fake_latents(
            model,
            num_samples=args.num_samples,
            batch_size=args.generation_batch_size,
            device=device,
            seed=training_seed + iteration,
            use_single_step=False  # Use full sampling to get actual generated layouts (good or bad)
        )
        
        # Step 2: Get real latents from dataset
        step_label = f"[Step 2]" if args.num_iterations is None else f"[Step 2/{args.num_iterations}]"
        print(f"\n{step_label} Getting real latents from dataset...")
        real_latents = get_real_latents(
            manifest_path=manifest_path,
            autoencoder_checkpoint=autoencoder_checkpoint,
            num_samples=args.num_samples,
            batch_size=args.generation_batch_size,
            device=device,
            seed=training_seed + iteration
        )
        
        # Step 3: Train discriminator (steps-based with early stopping)
        step_label = f"[Step 3]" if args.num_iterations is None else f"[Step 3/{args.num_iterations}]"
        print(f"\n{step_label} Training discriminator...")
        save_resume_state(iteration, best_iteration_val_loss if best_iteration_val_loss != float("inf") else 0.0, "in_progress")
        discriminator_checkpoint = train_discriminator_iteration_steps(
            real_latents=real_latents,
            fake_latents=fake_latents,
            output_dir=output_dir,
            iteration=iteration,
            discriminator_config=discriminator_config,
            max_steps=args.discriminator_max_steps,
            batch_size=args.discriminator_batch_size,
            learning_rate=0.0002,
            eval_interval=args.discriminator_eval_interval,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            device=device,
            seed=training_seed + iteration
        )
        
        # Step 4: Train diffusion model with discriminator loss (steps-based with early stopping)
        step_label = f"[Step 4]" if args.num_iterations is None else f"[Step 4/{args.num_iterations}]"
        print(f"\n{step_label} Training diffusion model with discriminator...")
        model, iteration_val_loss = train_diffusion_with_discriminator_steps(
            model=model,
            config=config,
            discriminator_checkpoint=discriminator_checkpoint,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            iteration=iteration,
            max_steps=args.diffusion_max_steps,
            eval_interval_steps=args.diffusion_eval_interval,
            sample_interval_steps=args.diffusion_sample_interval,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            device=device,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm
        )
        
        # Track iteration metrics for convergence
        iteration_history.append({
            "iteration": iteration,
            "val_loss": iteration_val_loss
        })
        
        # Save resume state after iteration completes
        save_resume_state(iteration, iteration_val_loss, "completed")
        
        # Create/update overall plot across all iterations
        plot_overall_iteration_metrics(output_dir, exp_name=exp_name)
        
        # Create/update iterative refinement metrics
        from training.plotting_utils import plot_iterative_refinement_metrics
        plot_iterative_refinement_metrics(output_dir, exp_name=exp_name)
        
        # Check for improvement across iterations (for convergence-based training)
        if args.num_iterations is None:
            # Initialize on first iteration
            if iteration == args.start_iteration:
                best_iteration_val_loss = iteration_val_loss
                iterations_without_improvement = 0
                print(f"\n✓ Iteration {iteration + 1} (baseline): val_loss={iteration_val_loss:.6f}")
            else:
                improvement = best_iteration_val_loss - iteration_val_loss
                if improvement > args.iteration_convergence_min_delta:
                    best_iteration_val_loss = iteration_val_loss
                    iterations_without_improvement = 0
                    print(f"\n✓ Iteration {iteration + 1} improved! (val_loss: {iteration_val_loss:.6f})")
                else:
                    iterations_without_improvement += 1
                    print(f"\n  Iteration {iteration + 1}: No improvement ({iterations_without_improvement}/{args.iteration_convergence_patience})")
                    print(f"    Current val_loss: {iteration_val_loss:.6f}")
                    print(f"    Best val_loss: {best_iteration_val_loss:.6f}")
                    print(f"    Improvement needed: {args.iteration_convergence_min_delta:.6f}")
            
            # Check convergence
            if iterations_without_improvement >= args.iteration_convergence_patience:
                print(f"\n{'='*80}")
                print(f"Adversarial training converged!")
                print(f"  No improvement for {iterations_without_improvement} iterations")
                print(f"  Best validation loss: {best_iteration_val_loss:.6f} at iteration {iteration - iterations_without_improvement + 1}")
                print(f"  Total iterations: {iteration + 1}")
                print(f"{'='*80}")
                save_resume_state(iteration, iteration_val_loss, "converged")
                break
        
        iteration += 1
        
        if args.num_iterations is not None:
            print(f"\nIteration {iteration} complete! Best val loss: {iteration_val_loss:.6f}")
        else:
            print(f"\nIteration {iteration} complete! Best val loss: {best_iteration_val_loss:.6f}")
    
    # Final resume state update
    save_resume_state(iteration - 1, best_iteration_val_loss if args.num_iterations is None else iteration_val_loss, "finished")
    
    # Create final overall plot
    plot_overall_iteration_metrics(output_dir, exp_name=exp_name)
    
    # Create final iterative refinement metrics
    from training.plotting_utils import plot_iterative_refinement_metrics
    plot_iterative_refinement_metrics(output_dir, exp_name=exp_name)
    
    print(f"\n{'='*80}")
    print("Stage 2 Discriminator Training Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Resume state saved to: {resume_state_path}")


if __name__ == "__main__":
    main()

