#!/usr/bin/env python3
"""
Debug script for diffusion training pipeline sanity checks.

Performs three sanity checks:
1. VAE Round Trip: Encode and decode a batch, check latent statistics
2. Noise Schedule: Visualize forward diffusion process at different timesteps
3. Overfit Test: Train on a single batch for 100 iterations to verify loss decreases
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from training.utils import (
    load_config,
    set_deterministic,
    get_device,
    build_optimizer,
    build_loss,
    build_dataset,
    to_device,
    move_batch_to_device,
)
from models.diffusion import DiffusionModel


def vae_round_trip_test(model, dataloader, device_obj, output_path):
    """Test 1: VAE Round Trip - encode and decode a batch."""
    print("\n" + "="*60)
    print("Test 1: VAE Round Trip")
    print("="*60)
    
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch, device_obj)
    
    # Get RGB images (either from batch or decode latents)
    if "rgb" in batch:
        original_images = batch["rgb"]
    elif "latent" in batch:
        # Decode latents to get RGB
        with torch.no_grad():
            decoded = model.decoder({"latent": batch["latent"]})
            original_images = decoded.get("rgb", None)
            if original_images is None:
                raise ValueError("Decoder did not return RGB")
            # Normalize from [-1, 1] to [0, 1]
            if original_images.min() < -0.1:
                original_images = (original_images + 1.0) / 2.0
            original_images = torch.clamp(original_images, 0.0, 1.0)
    else:
        raise ValueError("Batch must contain either 'rgb' or 'latent'")
    
    print(f"Original images shape: {original_images.shape}")
    
    # Encode to latents
    with torch.no_grad():
        if model._has_encoder:
            encoder_out = model.encoder(original_images)
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
        else:
            raise ValueError("Model does not have encoder - cannot perform VAE round trip")
    
    print(f"Latents shape: {latents.shape}")
    print(f"Latents mean: {latents.mean().item():.6f}")
    print(f"Latents std: {latents.std().item():.6f}")
    
    # Check if std is close to 1.0
    latent_std = latents.std().item()
    if abs(latent_std - 1.0) < 0.1:
        print(f"✓ Latent std ({latent_std:.6f}) is close to 1.0")
    else:
        print(f"⚠ WARNING: Latent std ({latent_std:.6f}) is not close to 1.0 (expected ~1.0)")
    
    # Decode back to RGB
    with torch.no_grad():
        decoded = model.decoder({"latent": latents})
        reconstructed_images = decoded.get("rgb", None)
        if reconstructed_images is None:
            raise ValueError("Decoder did not return RGB")
        # Normalize from [-1, 1] to [0, 1]
        if reconstructed_images.min() < -0.1:
            reconstructed_images = (reconstructed_images + 1.0) / 2.0
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    
    print(f"Reconstructed images shape: {reconstructed_images.shape}")
    
    # Compute reconstruction error
    mse = torch.nn.functional.mse_loss(original_images, reconstructed_images).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Save comparison image
    batch_size = min(8, original_images.shape[0])
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 2, 4))
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # Original
        orig_img = original_images[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')
        
        # Reconstructed
        recon_img = reconstructed_images[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved VAE round trip comparison to {output_path}")


def noise_schedule_test(model, dataloader, device_obj, output_path):
    """Test 2: Noise Schedule - visualize forward process at different timesteps."""
    print("\n" + "="*60)
    print("Test 2: Noise Schedule")
    print("="*60)
    
    model.eval()
    
    # Get a single image
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch, device_obj)
    
    # Get latents
    if "latent" in batch:
        latents = batch["latent"][:1]  # Take first sample
    elif "rgb" in batch and model._has_encoder:
        with torch.no_grad():
            encoder_out = model.encoder(batch["rgb"][:1])
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
    else:
        raise ValueError("Batch must contain either 'latent' or 'rgb'")
    
    print(f"Original latent shape: {latents.shape}")
    
    # Test timesteps
    test_timesteps = [0, 250, 500, 999]
    num_steps = model.scheduler.num_steps
    print(f"Total diffusion steps: {num_steps}")
    
    # Decode original
    with torch.no_grad():
        decoded = model.decoder({"latent": latents})
        original_rgb = decoded.get("rgb", None)
        if original_rgb is None:
            raise ValueError("Decoder did not return RGB")
        if original_rgb.min() < -0.1:
            original_rgb = (original_rgb + 1.0) / 2.0
        original_rgb = torch.clamp(original_rgb, 0.0, 1.0)
    
    # Apply noise at different timesteps
    noisy_images = []
    timestep_labels = []
    
    for t_val in test_timesteps:
        if t_val >= num_steps:
            t_val = num_steps - 1
        
        t = torch.tensor([t_val], device=device_obj)
        noise = model.scheduler.randn_like(latents)
        
        # Apply forward process: q(x_t | x_0)
        with torch.no_grad():
            noisy_latent = model.scheduler.add_noise(latents, noise, t)
        
        # Decode noisy latent
        with torch.no_grad():
            decoded = model.decoder({"latent": noisy_latent})
            noisy_rgb = decoded.get("rgb", None)
            if noisy_rgb is None:
                raise ValueError("Decoder did not return RGB")
            if noisy_rgb.min() < -0.1:
                noisy_rgb = (noisy_rgb + 1.0) / 2.0
            noisy_rgb = torch.clamp(noisy_rgb, 0.0, 1.0)
        
        noisy_images.append(noisy_rgb.cpu())
        timestep_labels.append(f"t={t_val}")
        # Get alpha_bar value for this timestep
        alpha_bar_val = model.scheduler.alpha_bars[t_val].item() if t_val < len(model.scheduler.alpha_bars) else 0.0
        print(f"  t={t_val}: alpha_bar = {alpha_bar_val:.6f}")
    
    # Save visualization
    fig, axes = plt.subplots(1, len(test_timesteps) + 1, figsize=((len(test_timesteps) + 1) * 2, 2))
    
    # Original
    orig_img = original_rgb[0].cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(orig_img)
    axes[0].set_title("Original (t=0)")
    axes[0].axis('off')
    
    # Noisy versions
    for i, (noisy_img, label) in enumerate(zip(noisy_images, timestep_labels)):
        img = noisy_img[0].numpy().transpose(1, 2, 0)
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(label)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved noise schedule visualization to {output_path}")


def overfit_test(model, dataloader, device_obj, loss_fn, optimizer):
    """Test 3: Overfit Test - train on a single batch for 100 iterations."""
    print("\n" + "="*60)
    print("Test 3: Overfit Test")
    print("="*60)
    
    model.train()
    
    # Get a single batch and keep it
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch, device_obj)
    
    print(f"Training on batch of size: {batch.get('latent', batch.get('rgb')).shape[0]}")
    
    losses = []
    
    for iteration in range(100):
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
                raise ValueError("Batch must contain either 'latent' or 'rgb'")
        
        # Sample random timesteps
        num_steps = model.scheduler.num_steps
        t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
        noise = model.scheduler.randn_like(latents)
        
        # Get embeddings
        text_emb = batch.get("text_emb", None)
        pov_emb = batch.get("pov_emb", None)
        
        # Flatten embeddings if needed
        if text_emb is not None and text_emb.dim() > 1:
            text_emb = text_emb.flatten(start_dim=1)
        if pov_emb is not None and pov_emb.dim() > 1:
            pov_emb = pov_emb.flatten(start_dim=1)
        
        # Forward pass
        optimizer.zero_grad()
        
        outputs = model(latents, t, cond=None, noise=noise, text_emb=text_emb, pov_emb=pov_emb)
        
        # Compute loss
        preds = {
            "pred_noise": outputs["pred_noise"],
            "scheduler": model.scheduler,
            "timesteps": t,
        }
        targets = {
            "noise": noise,
        }
        
        loss, logs = loss_fn(preds, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update EMA if exists
        if hasattr(model, 'update_ema'):
            model.update_ema()
        
        loss_val = loss.item()
        losses.append(loss_val)
        
        # Print every 10 steps
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1:3d}/100: Loss = {loss_val:.6f}")
    
    # Check if loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nLoss reduction: {initial_loss:.6f} -> {final_loss:.6f} ({reduction:.1f}% reduction)")
    
    if final_loss < initial_loss * 0.5:
        print(f"✓ Loss decreased significantly (final < 50% of initial)")
    elif final_loss < initial_loss:
        print(f"⚠ Loss decreased but not significantly (final < initial)")
    else:
        print(f"✗ WARNING: Loss did not decrease! This may indicate a training issue.")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Overfit Test: Loss vs Iteration")
    plt.grid(True)
    plt.savefig("debug_overfit_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curve to debug_overfit_loss.png")


def main():
    parser = argparse.ArgumentParser(description="Debug diffusion training pipeline")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Set deterministic behavior
    training_seed = config.get("training", {}).get("seed", 42)
    set_deterministic(training_seed)
    print(f"Set deterministic mode with seed: {training_seed}")
    
    # Get device
    device = get_device(config)
    device_obj = to_device(device)
    print(f"Device: {device}")
    
    # Build dataset
    print("\n[DATASET] Building dataset...")
    dataset = build_dataset(config)
    print(f"[DATASET] Dataset built: {len(dataset)} samples")
    
    # Create dataloader
    batch_size = config.get("training", {}).get("batch_size", 32)
    dataloader = dataset.make_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for debugging to avoid multiprocessing issues
        pin_memory=device_obj.type == "cuda",
    )
    
    # Build model from config
    print("\n[MODEL] Building model from config...")
    diffusion_cfg = config.get("diffusion", {})
    if not diffusion_cfg:
        diffusion_cfg = {
            "autoencoder": config.get("autoencoder"),
            "unet": config.get("unet", {}),
            "scheduler": config.get("scheduler", {}),
            "embedding_projection": config.get("embedding_projection")
        }
    
    # Add scale_factor if available
    scale_factor = diffusion_cfg.get("scale_factor") or config.get("scale_factor")
    if scale_factor is not None:
        diffusion_cfg["scale_factor"] = scale_factor
    
    # Add latent_clamp values if available
    latent_clamp_min = config.get("latent_clamp_min")
    latent_clamp_max = config.get("latent_clamp_max")
    if latent_clamp_min is not None:
        diffusion_cfg["latent_clamp_min"] = latent_clamp_min
    if latent_clamp_max is not None:
        diffusion_cfg["latent_clamp_max"] = latent_clamp_max
    
    model = DiffusionModel(**diffusion_cfg)
    model = model.to(device_obj)
    print("[MODEL] Model built")
    
    # Run Test 1: VAE Round Trip
    vae_round_trip_test(model, dataloader, device_obj, "debug_vae_reconstruction.png")
    
    # Run Test 2: Noise Schedule
    noise_schedule_test(model, dataloader, device_obj, "debug_forward_process.png")
    
    # Run Test 3: Overfit Test
    print("\n[OVERFIT] Setting up optimizer and loss...")
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)
    overfit_test(model, dataloader, device_obj, loss_fn, optimizer)
    
    print("\n" + "="*60)
    print("All debug tests completed!")
    print("="*60)
    print("Generated files:")
    print("  - debug_vae_reconstruction.png")
    print("  - debug_forward_process.png")
    print("  - debug_overfit_loss.png")


if __name__ == "__main__":
    main()

