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
    create_grad_scaler,
)
from training.train_diffusion import compute_loss, load_vae_metadata
from models.diffusion import DiffusionModel
from models.autoencoder import Autoencoder


def vae_round_trip_test(vae_model, dataloader, device_obj, output_path):
    """Test 1: VAE Round Trip - encode and decode a batch using VAE separately."""
    print("\n" + "="*60)
    print("Test 1: VAE Round Trip (Separate VAE Test)")
    print("="*60)
    
    vae_model.eval()
    
    # Get a batch with RGB images
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch, device_obj)
    
    # Get RGB images from batch (we need RGB for VAE round trip)
    # Dataset provides RGB in [-1, 1] range (normalized by dataset transforms)
    if "rgb" in batch:
        original_images_raw = batch["rgb"]
        print(f"Using RGB images from batch: {original_images_raw.shape}")
        print(f"  Original images range: [{original_images_raw.min().item():.3f}, {original_images_raw.max().item():.3f}]")
    elif "latent" in batch:
        # If we only have latents, decode them first to get RGB
        print("Only latents available in batch, decoding to get RGB...")
        with torch.no_grad():
            decoded = vae_model.decoder({"latent": batch["latent"]})
            original_images_raw = decoded.get("rgb", None)
            if original_images_raw is None:
                raise ValueError("Decoder did not return RGB")
        print(f"Decoded RGB images: {original_images_raw.shape}")
        print(f"  Decoded images range: [{original_images_raw.min().item():.3f}, {original_images_raw.max().item():.3f}]")
    else:
        raise ValueError("Batch must contain either 'rgb' or 'latent' for VAE round trip test")
    
    # VAE Round Trip: RGB -> Encode -> Latents -> Decode -> RGB
    # VAE expects/returns images in [-1, 1] range (tanh activation)
    print("\nPerforming VAE round trip: RGB -> Encode -> Latents -> Decode -> RGB")
    
    # Encode RGB to latents (VAE encoder expects [-1, 1] range)
    with torch.no_grad():
        encoder_out = vae_model.encode(original_images_raw)
        if "latent" in encoder_out:
            latents = encoder_out["latent"]
        elif "mu" in encoder_out:
            latents = encoder_out["mu"]
        else:
            raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
    
    print(f"Encoded latents shape: {latents.shape}")
    print(f"Latents mean: {latents.mean().item():.6f}")
    print(f"Latents std: {latents.std().item():.6f}")
    
    # Check if std is close to 1.0
    latent_std = latents.std().item()
    if abs(latent_std - 1.0) < 0.1:
        print(f"✓ Latent std ({latent_std:.6f}) is close to 1.0")
    else:
        print(f"⚠ WARNING: Latent std ({latent_std:.6f}) is not close to 1.0 (expected ~1.0)")
    
    # Decode latents back to RGB (VAE decoder outputs [-1, 1] range)
    with torch.no_grad():
        decoded = vae_model.decode({"latent": latents})
        reconstructed_images_raw = decoded.get("rgb", None)
        if reconstructed_images_raw is None:
            raise ValueError("Decoder did not return RGB")
    
    print(f"Reconstructed images shape: {reconstructed_images_raw.shape}")
    print(f"  Reconstructed images range: [{reconstructed_images_raw.min().item():.3f}, {reconstructed_images_raw.max().item():.3f}]")
    
    # Compute reconstruction error in native [-1, 1] range (no extra normalization)
    mse = torch.nn.functional.mse_loss(original_images_raw, reconstructed_images_raw).item()
    print(f"Reconstruction MSE (in [-1, 1] range): {mse:.6f}")
    
    # Normalize both to [0, 1] only for visualization
    # Convert from [-1, 1] to [0, 1] for matplotlib
    original_images = (original_images_raw + 1.0) / 2.0
    original_images = torch.clamp(original_images, 0.0, 1.0)
    reconstructed_images = (reconstructed_images_raw + 1.0) / 2.0
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    
    if mse < 0.01:
        print(f"✓ Excellent reconstruction (MSE < 0.01)")
    elif mse < 0.05:
        print(f"✓ Good reconstruction (MSE < 0.05)")
    else:
        print(f"⚠ WARNING: High reconstruction error (MSE >= 0.05)")
    
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
    
    # Get latents (prefer pre-encoded from dataset)
    if "latent" in batch:
        latents = batch["latent"][:1]  # Take first sample
        print(f"Using pre-encoded latents from dataset: {latents.shape}")
    elif "rgb" in batch and model._has_encoder:
        with torch.no_grad():
            encoder_out = model.encoder(batch["rgb"][:1])
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
        print(f"Encoded RGB to latents: {latents.shape}")
    else:
        raise ValueError("Batch must contain either 'latent' (pre-encoded) or 'rgb' (with encoder)")
    
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


def overfit_test(model, dataloader, device_obj, loss_fn, optimizer, config):
    """Test 3: Overfit Test - train on a single batch for 100 iterations using same logic as training."""
    print("\n" + "="*60)
    print("Test 3: Overfit Test")
    print("="*60)
    
    model.train()
    
    # Get a single batch and keep it (same as training)
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch, device_obj)
    
    batch_size = batch.get("latent", batch.get("rgb", torch.empty(1))).shape[0]
    print(f"Training on batch of size: {batch_size}")
    
    # Get training settings from config (same as training script)
    use_amp = config.get("training", {}).get("use_amp", False)
    max_grad_norm = config.get("training", {}).get("max_grad_norm", None)
    gradient_accumulation_steps = config.get("training", {}).get("gradient_accumulation_steps", 1)
    use_non_uniform_sampling = config.get("training", {}).get("use_non_uniform_sampling", False)
    cfg_dropout_rate = 0.0  # No CFG dropout for overfit test
    
    print(f"Training settings:")
    print(f"  Mixed precision (AMP): {use_amp}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Non-uniform sampling: {use_non_uniform_sampling}")
    
    # Create grad scaler if using AMP
    scaler = None
    if use_amp and device_obj.type == "cuda":
        scaler = create_grad_scaler(use_amp, device_obj)
    
    losses = []
    
    # Initialize optimizer (zero gradients)
    optimizer.zero_grad()
    
    for iteration in range(100):
        # Get latents (same logic as train_epoch)
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
        
        # Sample random timesteps (same logic as train_epoch)
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
        
        # No type-based conditioning - only using control signals (text_emb, pov_emb)
        cond = None
        
        # Compute loss using same function as training (handles CFG dropout, embeddings, etc.)
        loss_scale = 1.0 / gradient_accumulation_steps
        
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    use_amp, device_obj, cfg_dropout_rate, compute_eval_metrics=False
                )
                # Scale loss for gradient accumulation
                total_loss_val = total_loss_val * loss_scale
            
            if scaler:
                scaler.scale(total_loss_val).backward()
            else:
                total_loss_val.backward()
            
            # Step optimizer every gradient_accumulation_steps
            if (iteration + 1) % gradient_accumulation_steps == 0:
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
            total_loss_val, logs = compute_loss(
                model, batch, latents, t, noise, cond, loss_fn,
                use_amp, device_obj, cfg_dropout_rate, compute_eval_metrics=False
            )
            # Scale loss for gradient accumulation
            total_loss_val = total_loss_val * loss_scale
            
            total_loss_val.backward()
            
            # Step optimizer every gradient_accumulation_steps
            if (iteration + 1) % gradient_accumulation_steps == 0:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if hasattr(model, 'update_ema'):
                    model.update_ema()
        
        # For logging, use unscaled loss (multiply back by accumulation_steps)
        loss_val = total_loss_val.detach().item() * gradient_accumulation_steps
        losses.append(loss_val)
        
        # Print every 10 steps
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1:3d}/100: Loss = {loss_val:.6f}")
    
    # Ensure gradients are zeroed at the end
    optimizer.zero_grad()
    
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
    
    # Load VAE metadata (same as training script)
    print("\n[VAE_METADATA] Checking for VAE metadata file...")
    vae_metadata = None
    ae_cfg = config.get("autoencoder") or config.get("diffusion", {}).get("autoencoder")
    if ae_cfg and isinstance(ae_cfg, dict):
        ae_checkpoint = ae_cfg.get("checkpoint")
        if ae_checkpoint:
            # Use same load_vae_metadata function as training script
            checkpoint_path = Path(ae_checkpoint)
            if checkpoint_path.exists():
                checkpoint_dir = checkpoint_path.parent
                vae_dir = checkpoint_dir.parent
                metadata_files = list(vae_dir.glob("*_metadata.json"))
                if metadata_files:
                    import json
                    metadata_path = metadata_files[0]
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        recommended = metadata.get("recommended_values", {})
                        if recommended:
                            vae_metadata = {
                                "scale_factor": recommended.get("scale_factor"),
                                "latent_clamp_min": recommended.get("latent_clamp_min"),
                                "latent_clamp_max": recommended.get("latent_clamp_max")
                            }
                            print(f"[VAE_METADATA] Found VAE metadata file")
                            if vae_metadata.get("scale_factor"):
                                print(f"  scale_factor: {vae_metadata['scale_factor']:.6f}")
                    except Exception as e:
                        print(f"[VAE_METADATA] Warning: Failed to load VAE metadata: {e}")
    
    # Build model from config (same logic as training script)
    print("\n[MODEL] Building model from config...")
    diffusion_cfg = config.get("diffusion", {})
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
    
    # Get scale_factor from config or VAE metadata (priority: config > VAE metadata)
    scale_factor = diffusion_cfg.get("scale_factor") or config.get("scale_factor")
    if scale_factor is None:
        if vae_metadata and vae_metadata.get("scale_factor") is not None:
            scale_factor = vae_metadata["scale_factor"]
            print(f"[SCALE_FACTOR] Using scale_factor from VAE metadata: {scale_factor:.6f}")
        else:
            print("[SCALE_FACTOR] WARNING: scale_factor not found, using default 1.0")
            scale_factor = 1.0
    else:
        print(f"[SCALE_FACTOR] Using scale_factor from config: {scale_factor}")
    
    if scale_factor is not None:
        diffusion_cfg["scale_factor"] = scale_factor
    
    # Add latent_clamp values if available (priority: config > VAE metadata)
    latent_clamp_min = config.get("latent_clamp_min")
    latent_clamp_max = config.get("latent_clamp_max")
    if latent_clamp_min is None and vae_metadata and vae_metadata.get("latent_clamp_min") is not None:
        latent_clamp_min = vae_metadata["latent_clamp_min"]
        print(f"[LATENT_CLAMP] Using latent_clamp_min from VAE metadata: {latent_clamp_min:.6f}")
    if latent_clamp_max is None and vae_metadata and vae_metadata.get("latent_clamp_max") is not None:
        latent_clamp_max = vae_metadata["latent_clamp_max"]
        print(f"[LATENT_CLAMP] Using latent_clamp_max from VAE metadata: {latent_clamp_max:.6f}")
    
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
    print("[MODEL] Model built")
    
    # Run Test 1: VAE Round Trip (separate VAE test)
    # Load VAE separately from diffusion model
    print("\n[VAE] Loading VAE separately for round trip test...")
    vae_model = None
    ae_cfg = config.get("autoencoder") or config.get("diffusion", {}).get("autoencoder")
    if ae_cfg and isinstance(ae_cfg, dict):
        ae_checkpoint = ae_cfg.get("checkpoint")
        if ae_checkpoint:
            print(f"Loading VAE from checkpoint: {ae_checkpoint}")
            vae_model = Autoencoder.load_checkpoint(ae_checkpoint, map_location=device)
            vae_model = vae_model.to(device_obj)
            vae_model.eval()
            print("[VAE] VAE loaded successfully")
        else:
            print("[VAE] WARNING: No VAE checkpoint specified, skipping VAE round trip test")
    else:
        print("[VAE] WARNING: No autoencoder config found, skipping VAE round trip test")
    
    if vae_model is not None:
        vae_round_trip_test(vae_model, dataloader, device_obj, "debug_vae_reconstruction.png")
    else:
        print("[VAE] Skipping VAE round trip test (VAE not available)")
    
    # Run Test 2: Noise Schedule
    noise_schedule_test(model, dataloader, device_obj, "debug_forward_process.png")
    
    # Run Test 3: Overfit Test
    print("\n[OVERFIT] Setting up optimizer and loss...")
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)
    overfit_test(model, dataloader, device_obj, loss_fn, optimizer, config)
    
    print("\n" + "="*60)
    print("All debug tests completed!")
    print("="*60)
    print("Generated files:")
    print("  - debug_vae_reconstruction.png")
    print("  - debug_forward_process.png")
    print("  - debug_overfit_loss.png")


if __name__ == "__main__":
    main()

