#!/usr/bin/env python3
"""
Unified training script for diffusion models (Stage 1, Stage 2, Stage 3).
Uses CompositeLoss from config to combine noise, semantic, and discriminator losses.
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import math

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
from models.losses.base_loss import LOSS_REGISTRY
from models.components.discriminator import LatentDiscriminator
from torchvision.utils import save_image

# Import memorization check utilities (optional)
try:
    from training.memorization_utils import (
        load_training_samples,
        generate_samples,
        check_memorization
    )
    MEMORIZATION_CHECK_AVAILABLE = True
except ImportError as e:
    MEMORIZATION_CHECK_AVAILABLE = False
    print(f"Warning: Could not import memorization check utilities: {e}")


def compute_loss(
    model, batch, latents, t, noise, cond, loss_fn, 
    discriminator=None, use_amp=False, device_obj=None
):
    """
    Compute loss using CompositeLoss from config.
    
    This function prepares preds and targets dictionaries that are passed to
    CompositeLoss, which then distributes them to each sub-loss component.
    
    Each loss component expects specific keys:
    - SNRWeightedNoiseLoss: 
        preds["pred_noise"], preds["scheduler"], preds["timesteps"]
        targets["noise"]
    - SemanticLoss:
        preds["decoded_rgb"], preds["decoded_segmentation"]
        targets["rgb"], targets["segmentation"]
    - DiscriminatorLoss:
        preds["latent"], preds["discriminator"]
        (no targets needed)
    
    Args:
        model: Diffusion model
        batch: Batch dictionary
        latents: Latent tensors [B, C, H, W]
        t: Timesteps [B]
        noise: Noise tensor [B, C, H, W]
        cond: Conditioning (optional)
        loss_fn: CompositeLoss built from config
        discriminator: Optional discriminator for adversarial loss
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
        "pred_noise": outputs["pred_noise"],  # For SNRWeightedNoiseLoss
        "scheduler": model.scheduler,          # For SNRWeightedNoiseLoss
        "timesteps": t,                        # For SNRWeightedNoiseLoss
        "latent": latents,                     # For DiscriminatorLoss
    }
    
    # Add discriminator if available (for DiscriminatorLoss)
    if discriminator is not None:
        preds["discriminator"] = discriminator
    
    # Decode latents if semantic losses are needed
    # Check if any loss component needs decoded outputs
    needs_decoding = False
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    SemanticLossClass = LOSS_REGISTRY.get("SemanticLoss")
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        # Check if any sub-loss needs decoded outputs
        for sub_loss in loss_fn.losses:
            if SemanticLossClass and isinstance(sub_loss, SemanticLossClass):
                needs_decoding = True
                break
    
    if needs_decoding and "rgb" in batch and "segmentation" in batch:
        decoded = model.decoder({"latent": latents})
        preds["decoded_rgb"] = decoded.get("rgb")              # For SemanticLoss (perceptual)
        preds["decoded_segmentation"] = decoded.get("segmentation")  # For SemanticLoss (segmentation)
    
    # Prepare targets dict
    # All loss components will receive this dict, but only use the keys they need
    targets = {
        "noise": noise,  # For SNRWeightedNoiseLoss
    }
    
    # Add RGB and segmentation if available (for SemanticLoss)
    if "rgb" in batch:
        targets["rgb"] = batch["rgb"]  # For SemanticLoss (perceptual)
    if "segmentation" in batch:
        targets["segmentation"] = batch["segmentation"]  # For SemanticLoss (segmentation)
    
    # Compute loss using CompositeLoss
    if use_amp and device_obj.type == "cuda":
        with torch.amp.autocast('cuda'):
            total_loss, logs = loss_fn(preds, targets)
    else:
        total_loss, logs = loss_fn(preds, targets)
    
    return total_loss, logs


def train_epoch(
    model, dataloader, scheduler, loss_fn, discriminator, 
    optimizer, device, epoch, use_amp=False, max_grad_norm=None
):
    """Train for one epoch using CompositeLoss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    non_blocking = device_obj.type == "cuda"
    
    # Keep decoder frozen - only UNet is trained
    if hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
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
        t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
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
            scaler = getattr(train_epoch, '_scaler', None)
            if scaler is None and use_amp:
                scaler = torch.amp.GradScaler('cuda')
                train_epoch._scaler = scaler
            
            scaler.scale(total_loss_val).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
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
    
    return avg_loss, avg_logs


def eval_epoch(
    model, dataloader, scheduler, loss_fn, discriminator, 
    device, use_amp=False
):
    """Evaluate for one epoch using CompositeLoss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    non_blocking = device_obj.type == "cuda"
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
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
            cond = batch.get("cond", None)
            
            if use_amp and device_obj.type == "cuda":
                with torch.amp.autocast('cuda'):
                    total_loss_val, logs = compute_loss(
                        model, batch, latents, t, noise, cond, loss_fn,
                        discriminator, use_amp, device_obj
                    )
            else:
                total_loss_val, logs = compute_loss(
                    model, batch, latents, t, noise, cond, loss_fn,
                    discriminator, use_amp, device_obj
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


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=64, exp_name=None):
    """Generate and save sample images."""
    model.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    try:
        batch_iter = iter(val_loader)
        batch = next(batch_iter)
    except StopIteration:
        return
    
    num_steps = model.scheduler.num_steps
    ddim_steps = 100  # Use 100 steps for DDIM sampling
    print(f"  Generating {sample_batch_size} samples using DDIM ({ddim_steps} steps)...")
    
    with torch.no_grad():
        sample_output = model.sample(
            batch_size=sample_batch_size,
            num_steps=ddim_steps,
            method="ddim",
            eta=0.0,
            device=device_obj,
            verbose=False
        )
    
    if "rgb" in sample_output:
        samples = sample_output["rgb"]
    else:
        decoded = model.decoder({"latent": sample_output["latent"]})
        samples = (decoded["rgb"] + 1.0) / 2.0
    
    grid_n = int(math.sqrt(sample_batch_size))  # 8x8 grid for 64 samples
    grid_path = samples_dir / (f"{exp_name}_epoch_{epoch:03d}_samples.png" if exp_name else f"epoch_{epoch:03d}_samples.png")
    save_image(samples, grid_path, nrow=grid_n, normalize=False)
    print(f"  Saved samples to {samples_dir}")


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
    
    # Check for latest checkpoint
    latest_checkpoint = output_dir / f"{exp_name}_checkpoint_latest.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    
    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(config)
    
    # Build validation dataset
    train_split = config["training"].get("train_split", 0.8)
    split_seed = config["training"].get("split_seed", 42)
    
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
        model = DiffusionModel(**diffusion_cfg)
        model = model.to(device_obj)
    
    # Keep decoder frozen - only UNet is trained
    if hasattr(model, 'decoder'):
        print("Keeping decoder frozen - only UNet will be trained...")
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    # Build data loaders
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 8)
    shuffle = config["training"].get("shuffle", True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda"
    )
    
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device_obj.type == "cuda"
        )
    
    # Load discriminator (if specified for Stage 3)
    discriminator = None
    discriminator_path = config.get("discriminator", {}).get("checkpoint")
    if discriminator_path:
        print(f"Loading discriminator from {discriminator_path}")
        discriminator_checkpoint = torch.load(discriminator_path, map_location=device_obj)
        discriminator_config = discriminator_checkpoint.get("config")
        if discriminator_config:
            discriminator = LatentDiscriminator.from_config(discriminator_config)
        else:
            # Fallback: infer from latents
            sample_latent = next(iter(train_loader))["latent"][0:1]
            latent_channels = sample_latent.shape[1]
            discriminator = LatentDiscriminator(
                latent_channels=latent_channels,
                base_channels=64,
                num_layers=4
            )
        
        discriminator.load_state_dict(discriminator_checkpoint["state_dict"])
        discriminator = discriminator.to(device_obj)
        discriminator.eval()  # Freeze discriminator during training
        for param in discriminator.parameters():
            param.requires_grad = False
        
        print(f"  Discriminator loaded")
    else:
        print("  No discriminator configured")
    
    # Build loss function from config (uses CompositeLoss)
    print("Building loss function from config...")
    loss_fn = build_loss(config)
    print(f"  Loss type: {type(loss_fn).__name__}")
    if hasattr(loss_fn, 'losses'):
        print(f"  Loss components: {[type(l).__name__ for l in loss_fn.losses]}")
    
    # Build optimizer
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    
    # Build learning rate scheduler (account for already-trained epochs when resuming)
    scheduler = build_scheduler(optimizer, config, last_epoch=start_epoch)
    
    # Training settings
    epochs = config["training"].get("epochs", 100)
    use_amp = config["training"].get("use_amp", False)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    eval_interval = config["training"].get("eval_interval", 5)
    sample_interval = config["training"].get("sample_interval", 10)
    
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Max grad norm: {max_grad_norm}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_logs = train_epoch(
            model, train_loader, scheduler, loss_fn, discriminator,
            optimizer, device_obj, epoch + 1, use_amp=use_amp, max_grad_norm=max_grad_norm
        )
        
        print(f"Train Loss: {train_loss:.6f}")
        for k, v in train_logs.items():
            print(f"  {k}: {v:.6f}")
        
        # Validate
        val_loss = float("inf")
        val_logs = {}
        if val_loader and (epoch + 1) % eval_interval == 0:
            val_loss, val_logs = eval_epoch(
                model, val_loader, scheduler, loss_fn, discriminator,
                device_obj, use_amp=use_amp
            )
            print(f"Val Loss: {val_loss:.6f}")
            for k, v in val_logs.items():
                print(f"  {k}: {v:.6f}")
        
        # Save samples
        if val_loader and (epoch + 1) % sample_interval == 0:
            save_samples(model, val_loader, device_obj, output_dir, epoch + 1, sample_batch_size=64, exp_name=exp_name)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
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
        df = pd.DataFrame(training_history)
        df.to_csv(metrics_csv_path, index=False)
        
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
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
