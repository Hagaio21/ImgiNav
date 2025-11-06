#!/usr/bin/env python3
"""
Training script for ControlNet models.
ControlNet uses a frozen base UNet from a pretrained diffusion model and trains only the adapter.

Usage:
    python training/train_controlnet.py --config experiments/controlnet/config.yaml
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import math
import yaml

from training.utils import (
    load_config,
    set_deterministic,
    get_device,
    build_optimizer,
    build_loss,
    build_dataset,
    build_scheduler,
)
from models.diffusion import DiffusionModel
from models.components.controlnet import ControlNet
from torchvision.utils import save_image


def train_epoch(model, controlnet, dataloader, scheduler, loss_fn, optimizer, device, epoch, use_amp=False, max_grad_norm=None):
    """Train ControlNet for one epoch."""
    controlnet.train()  # Only adapter is trainable, base_unet is frozen
    total_loss = 0.0
    total_samples = 0
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    non_blocking = device_obj.type == "cuda"
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
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
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    non_blocking = device_obj.type == "cuda"
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval")
        for batch in pbar:
            batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
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
    """Generate and save sample images using ControlNet."""
    model.eval()
    controlnet.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    # Get a batch for sampling
    try:
        batch_iter = iter(val_loader)
        batch = next(batch_iter)
        batch = {k: v.to(device_obj, non_blocking=False) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Limit batch size
        batch_size = min(sample_batch_size, batch["latent"].shape[0])
        batch = {k: v[:batch_size] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        text_emb = batch.get("text_emb")
        pov_emb = batch.get("pov_emb")
        
        if text_emb is None or pov_emb is None:
            print("  Warning: Cannot generate samples without text_emb and pov_emb")
            return
        
        # Generate samples using DDIM (faster for testing)
        num_steps = 20  # Fewer steps for faster sampling
        print(f"  Generating {batch_size} samples using ControlNet with DDIM ({num_steps} steps)...")
        
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
            
            # Decode to RGB
            decoded = model.decoder({"latent": latents})
            if "rgb" in decoded:
                samples = (decoded["rgb"] + 1.0) / 2.0
                samples = torch.clamp(samples, 0.0, 1.0)
            else:
                print("  Warning: Decoder did not produce RGB output")
                return
        
        # Save samples
        grid_n = int(math.sqrt(batch_size))
        grid_path = samples_dir / (f"{exp_name}_epoch_{epoch:03d}_samples.png" if exp_name else f"epoch_{epoch:03d}_samples.png")
        save_image(samples, grid_path, nrow=grid_n, normalize=False)
        print(f"  Saved samples to {samples_dir}")
    
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
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Build dataset
    dataset_config = config["dataset"]
    train_split = config["training"].get("train_split", 0.8)
    split_seed = config["training"].get("split_seed", 42)
    
    dataset = build_dataset(dataset_config)
    if train_split < 1.0:
        train_dataset, val_dataset = dataset.split(train_split=train_split, random_seed=split_seed)
    else:
        train_dataset = dataset
        val_dataset = None
    
    # Prepare device object
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
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
    
    # Count parameters
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in controlnet.parameters())
    print(f"ControlNet parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Build data loaders
    if train_split < 1.0:
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
    trainable_params_list = [p for p in controlnet.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        trainable_params_list,
        config["training"]
    )
    
    if should_resume and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Build loss function
    loss_fn = build_loss(config.get("training", {}).get("loss", {}))
    
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
            df = pd.DataFrame(training_history)
            df.to_csv(metrics_csv_path, index=False)
        
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

