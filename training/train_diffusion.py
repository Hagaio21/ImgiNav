#!/usr/bin/env python3
"""
Training script for diffusion models.
Supports automatic resume from checkpoint and epochs_target for LR scheduling.
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
from torchvision.utils import save_image, make_grid


def train_epoch(model, dataloader, scheduler, loss_fn, optimizer, device, epoch, use_amp=False, max_grad_norm=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    non_blocking = device_obj.type == "cuda"
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Diffusion training: need latents and timesteps
        # Dataset should provide "latent" key (from latent_path column if pre-embedded)
        # or "rgb" key (if encoding on-the-fly)
        latents = batch.get("latent")
        if latents is None:
            # If no latent, assume RGB images and we'll encode (model should have encoder)
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
        
        # Sample random timesteps
        num_steps = model.scheduler.num_steps
        t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Conditioning (optional)
        cond = batch.get("cond", None)
        
        # Forward pass
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(latents, t, cond=cond, noise=noise)
                loss, logs = loss_fn(outputs, {"noise": noise})
            
            optimizer.zero_grad()
            scaler = getattr(train_epoch, '_scaler', None)
            if scaler is None and use_amp:
                scaler = torch.cuda.amp.GradScaler()
                train_epoch._scaler = scaler
            
            scaler.scale(loss).backward()
            # Gradient clipping for stability (especially with cosine schedulers)
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(latents, t, cond=cond, noise=noise)
            loss, logs = loss_fn(outputs, {"noise": noise})
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability (especially with cosine schedulers)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        batch_size = latents.shape[0]
        loss_val = loss.detach().item()
        total_loss += loss_val * batch_size
        total_samples += batch_size
        
        for k, v in logs.items():
            if k not in log_dict:
                log_dict[k] = 0.0
            log_dict[k] += v.detach().item() * batch_size
        
        pbar.set_postfix({"loss": loss_val, **{k: v/total_samples for k, v in log_dict.items()}})
    
    avg_loss = total_loss / total_samples
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    return avg_loss, avg_logs


def eval_epoch(model, dataloader, scheduler, loss_fn, device, use_amp=False):
    """Evaluate for one epoch."""
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
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
            
            num_steps = model.scheduler.num_steps
            t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
            noise = torch.randn_like(latents)
            cond = batch.get("cond", None)
            
            if use_amp and device_obj.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(latents, t, cond=cond, noise=noise)
                    loss, logs = loss_fn(outputs, {"noise": noise})
            else:
                outputs = model(latents, t, cond=cond, noise=noise)
                loss, logs = loss_fn(outputs, {"noise": noise})
            
            batch_size = latents.shape[0]
            loss_val = loss.item()
            total_loss += loss_val * batch_size
            total_samples += batch_size
            
            for k, v in logs.items():
                if k not in log_dict:
                    log_dict[k] = 0.0
                log_dict[k] += v.item() * batch_size
    
    avg_loss = total_loss / total_samples
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    return avg_loss, avg_logs


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=4, exp_name=None):
    """Generate and save sample images using DDPM sampling."""
    model.eval()
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
    except StopIteration:
        return
    
    # Generate samples using DDPM (full stochastic sampling)
    num_steps = model.scheduler.num_steps
    print(f"  Generating {sample_batch_size} samples using DDPM ({num_steps} steps)...")
    with torch.no_grad():
        sample_output = model.sample(
            batch_size=sample_batch_size,
            num_steps=num_steps,  # Full DDPM sampling
            method="ddpm",
            eta=0.0,
            device=device_obj,
            verbose=False
        )
        
        if "rgb" in sample_output:
            samples = sample_output["rgb"]  # Already in [0, 1]
        else:
            # Decode latents to RGB if needed
            decoded = model.decoder({"latent": sample_output["latent"]})
            samples = (decoded["rgb"] + 1.0) / 2.0
    
    # Save samples
    grid_n = int(math.sqrt(sample_batch_size))
    grid_path = samples_dir / (f"{exp_name}_epoch_{epoch:03d}_samples.png" if exp_name else f"epoch_{epoch:03d}_samples.png")
    save_image(samples, grid_path, nrow=grid_n, normalize=False)
    print(f"  Saved samples to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model from experiment config")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists (default: auto-resume)")
    parser.add_argument("--no-resume", action="store_true", help="Force start from scratch (ignore existing checkpoints)")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    print(f"Experiment: {exp_name}")
    
    # Set deterministic behavior if seed is provided
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
    
    # Check for latest checkpoint (automatic resume unless --no-resume is specified)
    latest_checkpoint = output_dir / f"{exp_name}_checkpoint_latest.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    
    # CSV file path for metrics (defined early so we can load from it if needed)
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    
    should_resume = not args.no_resume and latest_checkpoint.exists()
    if should_resume:
        print(f"\nFound latest checkpoint: {latest_checkpoint}")
        print("Resuming training...")
        
        # Load checkpoint with extra state, using current config (not saved config)
        # This ensures we have the correct autoencoder checkpoint path
        model, extra_state = DiffusionModel.load_checkpoint(
            latest_checkpoint, 
            map_location=device,
            return_extra=True,
            config=config  # Use current config file, not saved config
        )
        model = model.to(device)
        
        # Restore training state
        start_epoch = extra_state.get("epoch", 1) - 1  # epoch in checkpoint is 1-indexed
        best_val_loss = extra_state.get("best_val_loss", float("inf"))
        training_history = extra_state.get("training_history", [])
        
        # If checkpoint doesn't have training_history, try loading from CSV
        if not training_history and metrics_csv_path.exists():
            try:
                df = pd.read_csv(metrics_csv_path)
                # Filter out epochs >= start_epoch + 1 to avoid duplicates
                # (start_epoch is 0-indexed, CSV epochs are 1-indexed)
                df_filtered = df[df['epoch'] < (start_epoch + 1)]
                training_history = df_filtered.to_dict('records')
                print(f"  Loaded {len(training_history)} epochs from CSV file (filtered to epochs < {start_epoch + 1})")
            except Exception as e:
                print(f"  Warning: Could not load metrics from CSV: {e}")
        
        print(f"  Resuming from epoch {start_epoch + 1}")
        print(f"  Best validation loss so far: {best_val_loss:.6f}")
        if training_history:
            print(f"  Loaded {len(training_history)} previous epochs from history")
    else:
        # Build model from scratch
        print("Building model...")
        # DiffusionModel.from_config automatically extracts and builds from config
        model = DiffusionModel.from_config(config)
        
        # Convert device string to device object
        if isinstance(device, str):
            device_obj = torch.device(device)
        else:
            device_obj = device
        
        model = model.to(device_obj)
        
        # Enable cudnn benchmark
        if device_obj.type == "cuda":
            torch.backends.cudnn.benchmark = True
            print("Enabled cudnn.benchmark for faster convolutions")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(config)
    
    # Build validation dataset
    train_split = config["training"].get("train_split", 0.8)
    split_seed = config["training"].get("split_seed", 42)
    
    if train_split < 1.0:
        train_dataset, val_dataset = dataset.split(train_split=train_split, random_seed=split_seed)
        val_loader = val_dataset.make_dataloader(
            batch_size=config["training"].get("batch_size", 16),
            shuffle=False,
            num_workers=config["training"].get("num_workers", 4)
        )
        print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)}")
    else:
        train_dataset = dataset
        val_loader = None
    
    train_loader = train_dataset.make_dataloader(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True),
        num_workers=config["training"].get("num_workers", 4),
        use_weighted_sampling=config["training"].get("use_weighted_sampling", False)
    )
    print(f"Train dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    
    # Build loss function
    print("Building loss function...")
    loss_fn = build_loss(config)
    
    # Build optimizer
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    if latest_checkpoint.exists():
        optimizer_state = extra_state.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            print("  Loaded optimizer state from checkpoint")
    
    # Build scheduler (uses epochs_target from config)
    scheduler = build_scheduler(optimizer, config)
    if scheduler:
        if latest_checkpoint.exists():
            scheduler_state = extra_state.get("scheduler_state")
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
                print("  Loaded scheduler state from checkpoint")
        scheduler_type = config.get('training', {}).get('scheduler', {}).get('type', 'cosine')
        epochs_target = config.get('experiment', {}).get('epochs_target', 'N/A')
        print(f"  Using {scheduler_type} scheduler with epochs_target={epochs_target}")
    else:
        print("  No scheduler configured")
    
    # Enable mixed precision
    use_amp = config.get("training", {}).get("use_amp", True)
    if use_amp and isinstance(device, str) and device == "cuda":
        print("Using mixed precision training (FP16)")
        train_epoch._scaler = torch.cuda.amp.GradScaler()
    elif not use_amp:
        print("Mixed precision training disabled (use_amp: false)")
    
    # Training configuration
    epochs_target = config.get("experiment", {}).get("epochs_target", 1000)
    epochs_to_train = config["training"].get("epochs", epochs_target)  # Additional epochs to train
    save_interval = config["training"].get("save_interval", 10)
    eval_interval = config["training"].get("eval_interval", 1)
    sample_interval = config["training"].get("sample_interval", 10)
    keep_checkpoints = config["training"].get("keep_checkpoints", None)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    
    # Calculate end epoch: start_epoch + additional epochs to train
    end_epoch = start_epoch + epochs_to_train
    
    # Early stopping
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    early_stopping_restore_best = config["training"].get("early_stopping_restore_best", True)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs target (for scheduler): {epochs_target}")
    print(f"  Additional epochs to train: {epochs_to_train}")
    print(f"  Starting from epoch: {start_epoch + 1}")
    print(f"  Will train until epoch: {end_epoch}")
    print(f"  Save interval: every {save_interval} epoch(s)")
    if val_loader:
        print(f"  Evaluation: every {eval_interval} epoch(s)")
    print(f"  Sample interval: every {sample_interval} epoch(s) (DDPM)")
    if max_grad_norm is not None:
        print(f"  Gradient clipping: max_norm={max_grad_norm}")
    if keep_checkpoints:
        print(f"  Keeping only last {keep_checkpoints} checkpoints")
    if early_stopping_patience:
        print(f"  Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    
    # Training loop
    epochs_without_improvement = 0
    checkpoint_files = []
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    for epoch in range(start_epoch, end_epoch):
        # Training
        avg_loss, avg_logs = train_epoch(model, train_loader, scheduler, loss_fn, optimizer, device, epoch + 1, use_amp=use_amp, max_grad_norm=max_grad_norm)
        
        print(f"Epoch {epoch + 1}/{end_epoch} - Train Loss: {avg_loss:.6f}")
        for k, v in avg_logs.items():
            print(f"  Train {k}: {v:.6f}")
        
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            **{f"train_{k}": float(v) for k, v in avg_logs.items()}
        }
        
        # Evaluation
        if val_loader and (epoch + 1) % eval_interval == 0:
            val_loss, val_logs = eval_epoch(model, val_loader, scheduler, loss_fn, device, use_amp=use_amp)
            print(f"  Val Loss: {val_loss:.6f}")
            for k, v in val_logs.items():
                print(f"  Val {k}: {v:.6f}")
            
            epoch_log["val_loss"] = float(val_loss)
            epoch_log.update({f"val_{k}": float(v) for k, v in val_logs.items()})
            
            # Track best validation loss
            improvement = best_val_loss - val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f"  New best validation loss: {best_val_loss:.6f}")
                
                # Save best checkpoint (model state only)
                best_path = output_dir / f"{exp_name}_checkpoint_best.pt"
                model.save_checkpoint(best_path, include_config=True)
                print(f"  Saved best checkpoint")
            else:
                epochs_without_improvement += 1
                if early_stopping_patience:
                    print(f"  No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
            
            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered!")
                if early_stopping_restore_best:
                    best_path = output_dir / f"{exp_name}_checkpoint_best.pt"
                    if best_path.exists():
                        model = DiffusionModel.load_checkpoint(best_path, map_location=device_obj)
                        model = model.to(device_obj)
                        print(f"  Restored best checkpoint")
                break
        
        # Generate samples
        # Every epoch for first 5 epochs, then every sample_interval epochs
        should_sample = (
            (epoch + 1) <= 5 or  # First 5 epochs: every epoch
            (epoch + 1) % sample_interval == 0  # After that: every sample_interval epochs
        )
        if should_sample and val_loader:
            save_samples(model, val_loader, device, output_dir, epoch + 1, 
                       sample_batch_size=64, exp_name=exp_name)  # 8x8 grid
        
        training_history.append(epoch_log)
        
        # Save metrics CSV
        df = pd.DataFrame(training_history)
        df.to_csv(metrics_csv_path, index=False)
        
        # Save checkpoint at interval (model state only - includes all nested components)
        should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == end_epoch
        if should_save:
            checkpoint_path = output_dir / f"{exp_name}_checkpoint_epoch_{epoch + 1:03d}.pt"
            model.save_checkpoint(checkpoint_path, include_config=True)
            checkpoint_files.append(checkpoint_path)
        
        # Always save latest checkpoint (for resume - includes optimizer/scheduler state)
        latest_path = output_dir / f"{exp_name}_checkpoint_latest.pt"
        model.save_checkpoint(latest_path, include_config=True,
                            epoch=epoch + 1, best_val_loss=best_val_loss,
                            optimizer_state=optimizer.state_dict(),
                            scheduler_state=scheduler.state_dict() if scheduler else None)
        
        # Clean up old checkpoints
        if keep_checkpoints and len(checkpoint_files) > keep_checkpoints:
            for old_checkpoint in checkpoint_files[:-keep_checkpoints]:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            checkpoint_files = checkpoint_files[-keep_checkpoints:]
    
    print(f"\nTraining complete!")
    print(f"  Checkpoints: {output_dir}/{exp_name}_checkpoint_*.pt")
    print(f"  Metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()

