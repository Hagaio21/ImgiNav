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
    to_device,
    move_batch_to_device,
    split_dataset,
    create_grad_scaler,
    save_metrics_csv,
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
    - LatentStructuralLoss:
        preds["pred_noise"], preds["scheduler"], preds["timesteps"], preds["noisy_latent"]
        targets["latent"]
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
        "pred_noise": outputs["pred_noise"],      # For SNRWeightedNoiseLoss
        "scheduler": model.scheduler,            # For SNRWeightedNoiseLoss
        "timesteps": t,                          # For SNRWeightedNoiseLoss
        "latent": latents,                       # For DiscriminatorLoss
        "noisy_latent": outputs.get("noisy_latent"),  # For LatentStructuralLoss
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
        "noise": noise,      # For SNRWeightedNoiseLoss
        "latent": latents,   # For LatentStructuralLoss (ground-truth clean latents)
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
    
    device_obj = to_device(device)
    
    # Keep decoder frozen - only UNet is trained
    if hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = False
    
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
    
    device_obj = to_device(device)
    
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
    
    device_obj = to_device(device)
    
    try:
        batch_iter = iter(val_loader)
        batch = next(batch_iter)
    except StopIteration:
        return
    
    num_steps = model.scheduler.num_steps
    print(f"  Generating {sample_batch_size} samples using DDPM ({num_steps} steps)...")
    
    with torch.no_grad():
        sample_output = model.sample(
            batch_size=sample_batch_size,
            num_steps=num_steps,  # Use full DDPM schedule
            method="ddpm",
            eta=1.0,  # eta=1.0 for full DDPM (stochastic)
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
    use_weighted_sampling = config["training"].get("use_weighted_sampling", False)
    
    # Auto-generate weight stats if needed
    weights_stats_path = None
    if use_weighted_sampling:
        weight_column = config["training"].get("column", None)
        if weight_column:
            # Get manifest path from dataset config
            manifest_path = Path(config["dataset"]["manifest"])
            
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
                min_weight=config["training"].get("min_weight", 1.0)
            )
    
    # Use dataset's make_dataloader to support weighted sampling
    train_loader = train_dataset.make_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
        persistent_workers=num_workers > 0,
        use_weighted_sampling=use_weighted_sampling,
        weight_column=config["training"].get("column", None),
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
    # Use last_epoch=-1 for fresh start, or start_epoch for resume
    last_epoch = start_epoch if start_epoch > 0 else -1
    scheduler = build_scheduler(optimizer, config, last_epoch=last_epoch)
    
    # Training settings
    epochs = config["training"].get("epochs", 100)
    use_amp = config["training"].get("use_amp", False)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    eval_interval = config["training"].get("eval_interval", 5)
    sample_interval = config["training"].get("sample_interval", 10)
    
    # Memorization testing settings (support for both light and heavy)
    memorization_check_interval = config["training"].get("memorization_check_interval", None)
    memorization_num_generate = config["training"].get("memorization_num_generate", 100)
    memorization_num_training = config["training"].get("memorization_num_training", 1000)
    
    # Light memorization testing (more frequent, fewer samples)
    memorization_light_interval = config["training"].get("memorization_light_interval", None)
    memorization_light_num_generate = config["training"].get("memorization_light_num_generate", 50)
    memorization_light_num_training = config["training"].get("memorization_light_num_training", 1000)
    
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Max grad norm: {max_grad_norm}")
    if memorization_light_interval:
        print(f"  Light memorization check: every {memorization_light_interval} epochs ({memorization_light_num_generate} samples vs {memorization_light_num_training} training)")
    if memorization_check_interval:
        print(f"  Heavy memorization check: every {memorization_check_interval} epochs ({memorization_num_generate} samples vs {memorization_num_training} training)")
    
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
        
        # Memorization checks (light and/or heavy)
        should_run_light = MEMORIZATION_CHECK_AVAILABLE and memorization_light_interval and (epoch + 1) % memorization_light_interval == 0
        should_run_heavy = MEMORIZATION_CHECK_AVAILABLE and memorization_check_interval and (epoch + 1) % memorization_check_interval == 0
        
        # Run light memorization check (more frequent, fewer samples)
        if should_run_light and not should_run_heavy:  # Don't run both on the same epoch
            print(f"\n{'='*60}")
            print(f"Running LIGHT memorization check (epoch {epoch + 1})...")
            print(f"{'='*60}")
            try:
                # Load training samples
                training_samples = load_training_samples(
                    train_dataset,
                    num_samples=memorization_light_num_training,
                    device=device_obj,
                    load_rgb=False  # Only load latents for memory efficiency
                )
                
                # Generate samples
                generated_samples = generate_samples(
                    model,
                    num_samples=memorization_light_num_generate,
                    batch_size=min(batch_size, 16),  # Smaller batch for generation
                    device=device_obj,
                    method="ddpm"  # Use DDPM for correct sampling
                )
                
                # Run memorization check
                memorization_dir = output_dir / "memorization_checks" / f"light_epoch_{epoch + 1:03d}"
                results = check_memorization(
                    model,
                    training_samples,
                    generated_samples,
                    memorization_dir,
                    latent_perturbation_std=0.0,
                    run_perturbation_test=False,
                    method="ddpm",
                    device=device_obj
                )
                
                print(f"  Light memorization check complete. Results saved to: {memorization_dir}")
                if 'latent_l2_distances' in results:
                    mean_l2 = results['latent_l2_distances'].mean().item()
                    print(f"  Mean L2 distance to nearest training sample: {mean_l2:.4f}")
            except Exception as e:
                print(f"  Warning: Light memorization check failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Run heavy memorization check (less frequent, more samples, full dataset)
        if should_run_heavy:
            print(f"\n{'='*60}")
            print(f"Running HEAVY memorization check (epoch {epoch + 1})...")
            print(f"{'='*60}")
            try:
                # Clear CUDA cache before heavy check to free up memory
                if device_obj.type == "cuda":
                    torch.cuda.empty_cache()
                
                # Load training samples (use full dataset if num_training is very large)
                training_samples = load_training_samples(
                    train_dataset,
                    num_samples=memorization_num_training if memorization_num_training < 999999 else None,
                    device=device_obj,
                    load_rgb=False  # Only load latents for memory efficiency
                )
                
                # Generate samples
                generated_samples = generate_samples(
                    model,
                    num_samples=memorization_num_generate,
                    batch_size=min(batch_size, 16),  # Smaller batch for generation
                    device=device_obj,
                    method="ddpm"  # Use DDPM for correct sampling
                )
                
                # Run memorization check with smaller batch sizes for memory efficiency
                memorization_dir = output_dir / "memorization_checks" / f"heavy_epoch_{epoch + 1:03d}"
                results = check_memorization(
                    model,
                    training_samples,
                    generated_samples,
                    memorization_dir,
                    latent_perturbation_std=0.0,
                    run_perturbation_test=False,
                    method="ddpm",
                    device=device_obj,
                    gen_batch_size=16,  # Smaller batch for generated samples
                    train_batch_size=500  # Smaller batch for training samples
                )
                
                print(f"  Heavy memorization check complete. Results saved to: {memorization_dir}")
                if 'latent_l2_distances' in results:
                    mean_l2 = results['latent_l2_distances'].mean().item()
                    print(f"  Mean L2 distance to nearest training sample: {mean_l2:.4f}")
            except Exception as e:
                print(f"  Warning: Heavy memorization check failed: {e}")
                import traceback
                traceback.print_exc()
        
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
        save_metrics_csv(training_history, metrics_csv_path)
        
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
