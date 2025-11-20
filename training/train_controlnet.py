#!/usr/bin/env python3
"""
Training script for ControlNet models.
ControlNet uses a frozen base UNet from a pretrained diffusion model and trains only the adapter.

Usage:
    python training/train_controlnet.py --config experiments/controlnet/config.yaml
"""

import argparse
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import math
import yaml
from PIL import Image

# Add project root to path
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
from models.diffusion import DiffusionModel
from models.components.controlnet import ControlNet
from torchvision.utils import save_image, make_grid


def train_epoch(model, controlnet, dataloader, scheduler, loss_fn, optimizer, device, epoch, use_amp=False, max_grad_norm=None):
    """Train ControlNet for one epoch."""
    controlnet.train()  # Only adapter is trainable, base_unet is frozen
    total_loss = 0.0
    total_samples = 0
    
    device_obj = to_device(device)
    
    scaler = create_grad_scaler(use_amp, device_obj)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device_obj)
        
        # Get latents (required for diffusion training)
        latents = batch.get("latent")
        if latents is None:
            raise ValueError("Dataset must provide 'latent' key (pre-embedded latents required)")
        
        # Get conditioning inputs
        # Dataset should provide embeddings via outputs config:
        # outputs: {text_emb: graph_embedding, pov_emb: pov_embedding}
        text_emb = batch.get("text_emb")  # Text embeddings (from graph_embedding column)
        pov_emb = batch.get("pov_emb")   # POV embeddings (from pov_embedding column)
        
        if text_emb is None:
            # Try alternative key names
            text_emb = batch.get("graph_emb") or batch.get("graph_embedding")
        if pov_emb is None:
            pov_emb = batch.get("pov_embedding")
        
        if text_emb is None or pov_emb is None:
            raise ValueError(f"Dataset must provide text and POV embeddings. Got keys: {list(batch.keys())}")
        
        # Sample random timesteps uniformly (SNR weighting will be applied in loss)
        num_steps = model.scheduler.num_steps
        t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
        
        # Sample standard normal noise N(0,1)
        noise = model.scheduler.randn_like(latents)
        
        # Add noise to latents using the diffusion schedule
        noisy_latents = model.scheduler.add_noise(latents, noise, t)
        
        # Forward pass with ControlNet
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                # ControlNet replaces UNet in the forward pass
                pred_noise = controlnet(noisy_latents, t, text_emb, pov_emb)
                
                # SNR-weighted loss: w = snr / (1 + snr) where snr = alpha_bar / (1 - alpha_bar)
                alpha_bars = model.scheduler.alpha_bars.to(device_obj)
                alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
                snr = alpha_bar / (1 - alpha_bar + 1e-8)  # Signal-to-noise ratio
                w = snr / (1 + snr)  # SNR weighting
                
                # Compute weighted MSE loss
                loss = ((pred_noise - noise).pow(2) * w).mean()
            
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # ControlNet replaces UNet in the forward pass
            pred_noise = controlnet(noisy_latents, t, text_emb, pov_emb)
            
            # SNR-weighted loss
            alpha_bars = model.scheduler.alpha_bars.to(device_obj)
            alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
            snr = alpha_bar / (1 - alpha_bar + 1e-8)
            w = snr / (1 + snr)
            
            loss = ((pred_noise - noise).pow(2) * w).mean()
            
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item()
        total_samples += latents.shape[0]
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/total_samples:.4f}"})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, {"mse_loss": avg_loss}


def eval_epoch(model, controlnet, dataloader, scheduler, loss_fn, device, use_amp=False):
    """Evaluate ControlNet for one epoch."""
    controlnet.eval()
    total_loss = 0.0
    total_samples = 0
    
    device_obj = to_device(device)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval")
        for batch in pbar:
            batch = move_batch_to_device(batch, device_obj)
            
            latents = batch.get("latent")
            if latents is None:
                continue
            
            text_emb = batch.get("text_emb")
            pov_emb = batch.get("pov_emb")
            
            if text_emb is None or pov_emb is None:
                continue
            
            num_steps = model.scheduler.num_steps
            t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
            noise = model.scheduler.randn_like(latents)
            noisy_latents = model.scheduler.add_noise(latents, noise, t)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred_noise = controlnet(noisy_latents, t, text_emb, pov_emb)
                    
                    alpha_bars = model.scheduler.alpha_bars.to(device_obj)
                    alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
                    snr = alpha_bar / (1 - alpha_bar + 1e-8)
                    w = snr / (1 + snr)
                    
                    loss = ((pred_noise - noise).pow(2) * w).mean()
            else:
                pred_noise = controlnet(noisy_latents, t, text_emb, pov_emb)
                
                alpha_bars = model.scheduler.alpha_bars.to(device_obj)
                alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
                snr = alpha_bar / (1 - alpha_bar + 1e-8)
                w = snr / (1 + snr)
                
                loss = ((pred_noise - noise).pow(2) * w).mean()
            
            total_loss += loss.item()
            total_samples += latents.shape[0]
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/total_samples:.4f}"})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, {"mse_loss": avg_loss}


def save_samples(model, controlnet, val_loader, device, output_dir, epoch, sample_batch_size=4, exp_name=None):
    """Generate and save sample images using ControlNet, with target vs generated comparison.
    
    Saves:
    - Target and generated images (individual and grids)
    - Graph text files
    - POV images (if available)
    - Conditions, targets, and generated latents as .pt files
    """
    model.eval()
    controlnet.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = to_device(device)
    
    # Get a batch for sampling
    try:
        batch_iter = iter(val_loader)
        batch = next(batch_iter)
        batch = move_batch_to_device(batch, device_obj, non_blocking=False)
        
        # Limit batch size
        batch_size = min(sample_batch_size, batch["latent"].shape[0])
        batch = {k: v[:batch_size] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get dataset to access paths
        dataset = val_loader.dataset
        # For validation, dataloader is typically sequential, so we can track which samples we're using
        # We'll use a simple approach: get the first batch_size samples
        # In practice, validation should be deterministic, so this should work
        batch_indices = list(range(batch_size))
        
        text_emb = batch.get("text_emb")
        pov_emb = batch.get("pov_emb")
        target_latents = batch.get("latent")  # Target latents from dataset
        
        if text_emb is None or pov_emb is None:
            print("  Warning: Cannot generate samples without text_emb and pov_emb")
            return
        
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
        
        # Generate samples using DDIM (faster for testing)
        num_steps = 20  # Fewer steps for faster sampling
        print(f"  Generating {batch_size} samples using ControlNet with DDIM ({num_steps} steps)...")
        
        # Use epoch-based seed for sampling to show diversity across epochs
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = None
        if torch.cuda.is_available():
            cuda_rng_states = torch.cuda.get_rng_state_all()
        
        sampling_seed = 42 + epoch  # Different seed per epoch
        torch.manual_seed(sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed)
        
        with torch.no_grad():
            # Use model's sample method but we need to replace UNet with ControlNet
            # For now, we'll do manual sampling
            latent_shape = batch["latent"].shape[1:]  # [C, H, W]
            latents = model.scheduler.randn_like(torch.zeros((batch_size, *latent_shape), device=device_obj))
            
            # DDIM sampling schedule
            step_size = model.scheduler.num_steps // num_steps
            timesteps = torch.arange(0, model.scheduler.num_steps, step_size, device=device_obj).long()
            
            alpha_bars = model.scheduler.alpha_bars.to(device_obj)
            
            for i, t in enumerate(timesteps):
                t_batch = t.expand(batch_size)
                
                # Predict noise using ControlNet
                pred_noise = controlnet(latents, t_batch, text_emb, pov_emb)
                
                # DDIM step
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                if i > 0:
                    alpha_bar_prev = alpha_bars[timesteps[i-1]].view(-1, 1, 1, 1)
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device_obj, dtype=alpha_bar_t.dtype).view(-1, 1, 1, 1)
                
                # Predict x0 and update
                pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
                
                latents = alpha_bar_prev.sqrt() * pred_x0 + (1 - alpha_bar_prev).sqrt() * pred_noise
            
            # Decode generated latents to RGB
            decoded = model.decoder({"latent": latents})
            if "rgb" in decoded:
                generated_rgb = (decoded["rgb"] + 1.0) / 2.0
                generated_rgb = torch.clamp(generated_rgb, 0.0, 1.0)
            else:
                print("  Warning: Decoder did not produce RGB output")
                return
        
        # Restore original RNG state to maintain training determinism
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available() and cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        
        # Save raw data: conditions, targets, and generated latents
        data_dir = output_dir / "sample_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        epoch_prefix = f"{exp_name}_epoch_{epoch:03d}" if exp_name else f"epoch_{epoch:03d}"
        
        # Save individual images and graph text
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
        
        # Save target images individually
        if target_rgb is not None:
            for i in range(batch_size):
                save_image(target_rgb[i], images_dir / f"sample_{i:03d}_target.png", normalize=False)
        
        # Save generated images individually
        for i in range(batch_size):
            save_image(generated_rgb[i], images_dir / f"sample_{i:03d}_generated.png", normalize=False)
        
        print(f"  Saved {batch_size} individual images and graph texts to {images_dir}")
        
        # Save metadata (paths and graph texts)
        metadata = {
            "layout_paths": layout_paths,
            "pov_paths": pov_paths,
            "graph_texts": graph_texts,
        }
        with open(images_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save conditioning embeddings
        torch.save({
            "graph_embeddings": text_emb.cpu(),  # Graph embeddings (conditions)
            "pov_embeddings": pov_emb.cpu(),     # POV embeddings (conditions)
        }, data_dir / f"{epoch_prefix}_conditions.pt")
        print(f"  Saved conditioning embeddings to {data_dir / f'{epoch_prefix}_conditions.pt'}")
        
        # Save target latents
        torch.save({
            "target_latents": target_latents.cpu(),  # Target latents from dataset
        }, data_dir / f"{epoch_prefix}_targets.pt")
        print(f"  Saved target latents to {data_dir / f'{epoch_prefix}_targets.pt'}")
        
        # Save generated latents
        torch.save({
            "generated_latents": latents.cpu(),  # Generated latents from ControlNet
        }, data_dir / f"{epoch_prefix}_generated.pt")
        print(f"  Saved generated latents to {data_dir / f'{epoch_prefix}_generated.pt'}")
        
        # Create side-by-side comparison: target (left) | generated (right)
        grid_n = int(math.sqrt(batch_size))
        if grid_n * grid_n < batch_size:
            grid_n += 1
        
        # Create target grid (n×n)
        if target_rgb is not None:
            target_grid = make_grid(target_rgb, nrow=grid_n, padding=2, normalize=False)
            # Create generated grid (n×n)
            generated_grid = make_grid(generated_rgb, nrow=grid_n, padding=2, normalize=False)
            # Concatenate horizontally (side by side)
            combined_grid = torch.cat([target_grid, generated_grid], dim=2)  # Concatenate along width
            
            # Save combined comparison: n×n target | n×n generated (side by side)
            comparison_path = samples_dir / (f"{exp_name}_epoch_{epoch:03d}_comparison.png" if exp_name else f"epoch_{epoch:03d}_comparison.png")
            save_image(combined_grid, comparison_path, normalize=False)
            print(f"  Saved target vs generated comparison to {comparison_path}")
        
        # Also save generated samples only (for quick viewing)
        samples_path = samples_dir / (f"{exp_name}_epoch_{epoch:03d}_samples.png" if exp_name else f"epoch_{epoch:03d}_samples.png")
        save_image(generated_rgb, samples_path, nrow=grid_n, normalize=False)
        print(f"  Saved generated samples to {samples_path}")
    
    except StopIteration:
        return


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detected if not specified)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from latest checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up paths
    exp_config = config.get("experiment", {})
    exp_name = exp_config.get("name", "controlnet")
    save_path = Path(exp_config.get("save_path", "experiments/controlnet"))
    save_path.mkdir(parents=True, exist_ok=True)
    
    output_dir = save_path / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    latest_checkpoint = output_dir / f"{exp_name}_checkpoint_latest.pt"
    best_checkpoint = output_dir / f"{exp_name}_checkpoint_best.pt"
    metrics_csv_path = output_dir / "metrics.csv"
    
    # Set deterministic
    seed = config["training"].get("seed", 42)
    set_deterministic(seed)
    
    # Get device
    if args.device is not None:
        device = args.device
    else:
        device = get_device(config)
    print(f"Using device: {device}")
    
    # Build dataset
    dataset_config = config["dataset"]
    dataset = build_dataset(config)
    train_dataset, val_dataset = split_dataset(dataset, config["training"])
    
    device_obj = to_device(device)
    
    # Load pretrained diffusion model
    diffusion_config = config.get("diffusion", {})
    diffusion_checkpoint = diffusion_config.get("checkpoint")
    if not diffusion_checkpoint:
        raise ValueError("ControlNet training requires a pretrained diffusion model checkpoint. Set 'diffusion.checkpoint' in config.")
    
    print(f"\nLoading pretrained diffusion model from: {diffusion_checkpoint}")
    diffusion_model = DiffusionModel.load_checkpoint(
        diffusion_checkpoint,
        map_location=device_obj
    )
    diffusion_model = diffusion_model.to(device_obj)
    diffusion_model.eval()
    
    # Freeze the base UNet
    for param in diffusion_model.unet.parameters():
        param.requires_grad = False
    
    print("✓ Diffusion model loaded and base UNet frozen")
    
    # Build ControlNet
    controlnet_config = config.get("controlnet", {})
    unet_config = diffusion_model.unet.to_config()
    
    # Get adapter config
    adapter_config = controlnet_config.get("adapter", {})
    if not adapter_config:
        # Default adapter config
        base_channels = unet_config.get("base_channels", 64)
        depth = unet_config.get("depth", 4)
        adapter_config = {
            "text_dim": 768,  # Default text embedding dimension
            "pov_dim": 256,   # Default POV embedding dimension
            "base_channels": base_channels,
            "depth": depth
        }
    
    controlnet_config = {
        "base_unet": unet_config,
        "adapter": adapter_config,
        "fuse_mode": controlnet_config.get("fuse_mode", "add")
    }
    
    # Create ControlNet
    controlnet = ControlNet.from_config(controlnet_config)
    controlnet = controlnet.to(device_obj)
    
    # Copy weights from trained UNet to ControlNet's base_unet
    print("\nCopying weights from trained UNet to ControlNet's base_unet...")
    controlnet.base_unet.load_state_dict(diffusion_model.unet.state_dict(), strict=False)
    
    # Freeze all base_unet parameters
    for param in controlnet.base_unet.parameters():
        param.requires_grad = False
    controlnet.base_unet.freeze_blocks(["downs", "ups", "bottleneck", "time_mlp", "final"])
    
    print("✓ ControlNet created and base_unet frozen")
    
    # Enable debug logging for control features (optional, can be disabled)
    controlnet._debug_control_features = config.get("debug_control_features", False)
    if controlnet._debug_control_features:
        print("  Debug logging enabled for control features")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in controlnet.parameters())
    print(f"ControlNet parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Build data loaders
    train_split = config["training"].get("train_split", 0.8)
    if train_split < 1.0 and val_dataset is not None:
        val_loader = val_dataset.make_dataloader(
            batch_size=config["training"].get("batch_size", 16),
            shuffle=False,
            num_workers=config["training"].get("num_workers", 4)
        )
        print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)}")
    else:
        val_loader = None
    
    train_loader = train_dataset.make_dataloader(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True),
        num_workers=config["training"].get("num_workers", 4),
        use_weighted_sampling=config["training"].get("use_weighted_sampling", False),
        group_rare_classes=config["training"].get("group_rare_classes", False),
        class_grouping_path=config["training"].get("class_grouping_path", None)
    )
    print(f"Training dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    
    # Check if we should resume
    should_resume = not args.no_resume and latest_checkpoint.exists()
    start_epoch = 1
    best_val_loss = float("inf")
    training_history = []
    
    if should_resume:
        print(f"\nFound latest checkpoint: {latest_checkpoint}")
        print("Resuming training...")
        
        checkpoint = torch.load(latest_checkpoint, map_location=device_obj)
        controlnet.load_state_dict(checkpoint.get("controlnet_state_dict", checkpoint.get("state_dict")))
        
        if "optimizer_state_dict" in checkpoint:
            # Will be loaded below with optimizer
            pass
        
        start_epoch = checkpoint.get("epoch", 1)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        training_history = checkpoint.get("training_history", [])
        
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best validation loss so far: {best_val_loss:.6f}")
    
    # Build optimizer (only for trainable parameters - the adapter)
    # build_optimizer expects (model, config) where model has parameter_groups() method
    optimizer = build_optimizer(controlnet, config)
    
    if should_resume and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Build loss function
    loss_fn = build_loss(config)
    
    # Build learning rate scheduler
    lr_scheduler = build_scheduler(optimizer, config["training"])
    
    # Training loop
    use_amp = config["training"].get("use_amp", False)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    epochs = config["training"].get("epochs", 100)
    eval_interval = config["training"].get("eval_interval", 5)
    sample_interval = config["training"].get("sample_interval", 10)
    save_interval = config["training"].get("save_interval", 10)
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Training on {len(train_dataset)} samples")
    if val_dataset:
        print(f"  Validation on {len(val_dataset)} samples")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Sample interval: {sample_interval}")
    print(f"  Save interval: {save_interval}")
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        avg_loss, logs = train_epoch(
            diffusion_model, controlnet, train_loader, diffusion_model.scheduler,
            loss_fn, optimizer, device_obj, epoch, use_amp, max_grad_norm
        )
        
        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step()
        
        # Logging
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            **logs
        }
        
        # Validation
        if val_loader and (epoch % eval_interval == 0 or epoch == epochs):
            val_loss, val_logs = eval_epoch(
                diffusion_model, controlnet, val_loader, diffusion_model.scheduler,
                loss_fn, device_obj, use_amp
            )
            log_entry["val_loss"] = val_loss
            log_entry.update({f"val_{k}": v for k, v in val_logs.items()})
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\n✓ New best validation loss: {best_val_loss:.6f}")
                torch.save({
                    "controlnet_state_dict": controlnet.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }, best_checkpoint)
        
        # Save samples
        if (epoch % sample_interval == 0 or epoch == epochs) and val_loader:
            save_samples(
                diffusion_model, controlnet, val_loader, device_obj,
                output_dir, epoch, sample_batch_size=4, exp_name=exp_name
            )
        
        # Save checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
            training_history.append(log_entry)
            torch.save({
                "controlnet_state_dict": controlnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "training_history": training_history,
            }, latest_checkpoint)
            
            # Save metrics to CSV
            save_metrics_csv(training_history, metrics_csv_path)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_loss:.6f}")
        if "val_loss" in log_entry:
            print(f"  Val Loss: {log_entry['val_loss']:.6f}")
        if lr_scheduler:
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()

