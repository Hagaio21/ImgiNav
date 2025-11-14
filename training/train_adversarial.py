#!/usr/bin/env python3
"""
Adversarial training pipeline for diffusion models.

Iteratively:
1. Generates fake latents from current diffusion model (full T-step sampling with no_grad)
2. Trains discriminator on real vs fake latents
3. Fine-tunes diffusion model with discriminator loss using gradient reconnection
4. Repeats for configurable number of iterations
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
from training.plotting_utils import (
    plot_discriminator_metrics,
    plot_diffusion_metrics,
    plot_overall_iteration_metrics,
    plot_iterative_refinement_metrics,
)
from training.train_discriminator import train_discriminator, load_latents
from models.diffusion import DiffusionModel
from models.losses.base_loss import LOSS_REGISTRY
from models.components.discriminator import LatentDiscriminator
from torchvision.utils import save_image


def generate_fake_latents(
    model, output_dir, iteration, num_samples, num_steps, method, batch_size, device_obj, seed=None
):
    """
    Generate fake latents from current diffusion model using full T-step sampling.
    
    Args:
        model: DiffusionModel (current checkpoint)
        output_dir: Output directory
        iteration: Current iteration number
        num_samples: Number of fake latents to generate
        num_steps: Number of sampling steps
        method: Sampling method ("ddim" or "ddpm")
        batch_size: Batch size for generation
        device_obj: Device object
        seed: Optional random seed
    
    Returns:
        Path to saved fake latents directory
    """
    output_dir = Path(output_dir)
    fake_latents_dir = output_dir / f"fake_latents_iter_{iteration}"
    fake_latents_dir.mkdir(parents=True, exist_ok=True)
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    model.eval()
    all_latents = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} fake latents (iteration {iteration})...")
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating fakes"):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Sample from model
            sample_output = model.sample(
                batch_size=current_batch_size,
                num_steps=num_steps,
                method=method,
                eta=0.0 if method == "ddim" else 1.0,
                device=device_obj,
                verbose=False
            )
            
            # Get latents
            if "latent" in sample_output:
                latents = sample_output["latent"]
            else:
                raise ValueError("Model should return latents, not images")
            
            # Save individual latents
            for i in range(latents.shape[0]):
                idx = batch_idx * batch_size + i
                latent_path = fake_latents_dir / f"fake_latent_{idx:05d}.pt"
                torch.save(latents[i].cpu(), latent_path)
            
            all_latents.append(latents.cpu())
    
    # Save all latents as single file
    all_latents_tensor = torch.cat(all_latents, dim=0)
    all_latents_path = fake_latents_dir / "fake_latents_all.pt"
    torch.save(all_latents_tensor, all_latents_path)
    print(f"Saved all fake latents to: {all_latents_path}")
    
    return all_latents_path


def train_discriminator_iteration(
    real_latents_path, fake_latents_path, output_dir, iteration, discriminator_config,
    epochs, batch_size, learning_rate, device, seed, initial_checkpoint=None
):
    """
    Train discriminator for one iteration.
    
    Note: By default, trains a NEW discriminator from scratch each iteration.
    To resume from previous iteration's discriminator, provide initial_checkpoint.
    
    Args:
        real_latents_path: Path to real latents
        fake_latents_path: Path to fake latents
        output_dir: Output directory
        iteration: Current iteration number
        discriminator_config: Discriminator config dict
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device string
        seed: Random seed
        initial_checkpoint: Optional path to previous discriminator checkpoint to resume from
    
    Returns:
        (discriminator_model, checkpoint_path)
    """
    output_dir = Path(output_dir)
    disc_output_dir = output_dir / f"discriminator_iter_{iteration}"
    
    print(f"\n{'='*60}")
    print(f"Training Discriminator - Iteration {iteration}")
    print(f"{'='*60}")
    
    if initial_checkpoint and Path(initial_checkpoint).exists():
        print(f"Resuming discriminator from: {initial_checkpoint}")
        # Load previous discriminator as starting point
        prev_checkpoint = torch.load(initial_checkpoint, map_location=device)
        if discriminator_config:
            discriminator = LatentDiscriminator.from_config(discriminator_config)
        else:
            config = prev_checkpoint.get("config")
            if config:
                discriminator = LatentDiscriminator.from_config(config)
            else:
                # Infer from latents
                sample_latent = load_latents(real_latents_path, device)[0:1]
                latent_channels = sample_latent.shape[1]
                discriminator = LatentDiscriminator(
                    latent_channels=latent_channels,
                    base_channels=64,
                    num_layers=4
                )
        discriminator.load_state_dict(prev_checkpoint["state_dict"])
        print(f"  Loaded discriminator from previous iteration")
    else:
        print(f"Training NEW discriminator from scratch")
        discriminator = None
    
    # Train discriminator using existing function
    # Note: train_discriminator always trains from scratch, so if we want to resume,
    # we'd need to modify it or create a wrapper. For now, we train fresh each time.
    train_discriminator(
        real_latents_path=real_latents_path,
        fake_latents_path=fake_latents_path,
        output_dir=disc_output_dir,
        config=discriminator_config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed
    )
    
    # Load the best checkpoint (check multiple possible locations)
    checkpoint_path = Path("models/losses/checkpoints") / "discriminator_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = disc_output_dir / "discriminator_best.pt"
    if not checkpoint_path.exists():
        # Try final checkpoint
        checkpoint_path = Path("models/losses/checkpoints") / "discriminator_final.pt"
    if not checkpoint_path.exists():
        checkpoint_path = disc_output_dir / "discriminator_final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Discriminator checkpoint not found. Checked: models/losses/checkpoints/ and {disc_output_dir}/")
    
    # Load discriminator
    discriminator_checkpoint = torch.load(checkpoint_path, map_location=device)
    if discriminator_config:
        discriminator = LatentDiscriminator.from_config(discriminator_config)
    else:
        # Infer from checkpoint
        config = discriminator_checkpoint.get("config")
        if config:
            discriminator = LatentDiscriminator.from_config(config)
        else:
            # Fallback: load a sample to infer shape
            sample_latent = load_latents(real_latents_path, device)[0:1]
            latent_channels = sample_latent.shape[1]
            discriminator = LatentDiscriminator(
                latent_channels=latent_channels,
                base_channels=64,
                num_layers=4
            )
    
    discriminator.load_state_dict(discriminator_checkpoint["state_dict"])
    discriminator = discriminator.to(to_device(device))
    discriminator.eval()
    for param in discriminator.parameters():
        param.requires_grad = False
    
    # Save with iteration suffix
    iter_checkpoint_path = output_dir / "checkpoints" / f"discriminator_iter_{iteration}_best.pt"
    iter_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(discriminator_checkpoint, iter_checkpoint_path)
    
    print(f"Discriminator checkpoint saved to: {iter_checkpoint_path}")
    
    return discriminator, iter_checkpoint_path


def compute_loss_with_reconnection(
    model, batch, latents, t, noise, cond, loss_fn, 
    discriminator=None, use_amp=False, device_obj=None,
    fake_sampling_steps=50, fake_sampling_method="ddim"
):
    """
    Compute loss with gradient reconnection for discriminator loss.
    
    Similar to compute_loss() but:
    - Generates x_fake with no_grad (full sampling)
    - Adds x_fake, model, cond to preds dict for DiscriminatorLossWithReconnection
    
    Args:
        model: Diffusion model
        batch: Batch dictionary
        latents: Latent tensors [B, C, H, W]
        t: Timesteps [B]
        noise: Noise tensor [B, C, H, W]
        cond: Conditioning (optional)
        loss_fn: CompositeLoss built from config
        discriminator: Discriminator model
        use_amp: Whether to use mixed precision
        device_obj: Device object
        fake_sampling_steps: Number of steps for full sampling
        fake_sampling_method: Sampling method ("ddim" or "ddpm")
    
    Returns:
        (total_loss, logs_dict)
    """
    # Forward pass through model (for standard losses)
    outputs = model(latents, t, cond=cond, noise=noise)
    
    # Prepare preds dict for loss computation
    pred_latent = outputs.get("pred_latent")
    
    preds = {
        "pred_noise": outputs["pred_noise"],
        "pred_latent": pred_latent,
        "scheduler": model.scheduler,
        "timesteps": t,
        "noisy_latent": outputs.get("noisy_latent"),
    }
    
    # Generate fully sampled fakes for discriminator loss (if discriminator available)
    if discriminator is not None:
        preds["discriminator"] = discriminator
        
        # Sample with no_grad to get detached x_fake
        with torch.no_grad():
            model.eval()
            batch_size = latents.shape[0]
            latent_shape = latents.shape[1:]
            
            # Generate fully sampled fake latents
            sample_output = model.sample(
                batch_size=batch_size,
                latent_shape=latent_shape,
                cond=cond,
                num_steps=fake_sampling_steps,
                method=fake_sampling_method,
                eta=0.0 if fake_sampling_method == "ddim" else 1.0,
                device=device_obj,
                verbose=False
            )
            
            x_fake = sample_output.get("latent")  # Fully sampled fake latents (detached)
            model.train()  # Restore training mode
        
        # Add x_fake and model to preds for gradient reconnection
        preds["x_fake"] = x_fake
        preds["model"] = model
        preds["cond"] = cond
    
    # Decode latents if semantic losses are needed
    needs_decoding = False
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    SemanticLossClass = LOSS_REGISTRY.get("SemanticLoss")
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        for sub_loss in loss_fn.losses:
            if SemanticLossClass and isinstance(sub_loss, SemanticLossClass):
                needs_decoding = True
                break
    
    if needs_decoding and "rgb" in batch and "segmentation" in batch:
        decoded = model.decoder({"latent": latents})
        preds["decoded_rgb"] = decoded.get("rgb")
        preds["decoded_segmentation"] = decoded.get("segmentation")
    
    # Prepare targets dict
    targets = {
        "noise": noise,
        "latent": latents,
    }
    
    if "rgb" in batch:
        targets["rgb"] = batch["rgb"]
    if "segmentation" in batch:
        targets["segmentation"] = batch["segmentation"]
    
    # Compute loss using CompositeLoss
    if use_amp and device_obj.type == "cuda":
        with torch.amp.autocast('cuda'):
            total_loss, logs = loss_fn(preds, targets)
    else:
        total_loss, logs = loss_fn(preds, targets)
    
    return total_loss, logs


def train_epoch_with_reconnection(
    model, dataloader, scheduler, loss_fn, discriminator, 
    optimizer, device, epoch, use_amp=False, max_grad_norm=None, 
    use_non_uniform_sampling=False, fake_sampling_steps=50, fake_sampling_method="ddim"
):
    """Train for one epoch using CompositeLoss with gradient reconnection."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    device_obj = to_device(device)
    train_epoch_with_reconnection._use_non_uniform_sampling = use_non_uniform_sampling
    
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
        use_non_uniform_sampling = getattr(train_epoch_with_reconnection, '_use_non_uniform_sampling', False)
        if use_non_uniform_sampling:
            probs = torch.exp(-torch.linspace(0, 2, num_steps, device=device_obj))
            probs = probs / probs.sum()
            t = torch.multinomial(probs, latents.shape[0], replacement=True)
        else:
            t = torch.randint(0, num_steps, (latents.shape[0],), device=device_obj)
        noise = model.scheduler.randn_like(latents)
        cond = batch.get("cond", None)
        
        # Compute loss with reconnection
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                total_loss_val, logs = compute_loss_with_reconnection(
                    model, batch, latents, t, noise, cond, loss_fn,
                    discriminator, use_amp, device_obj,
                    fake_sampling_steps=fake_sampling_steps,
                    fake_sampling_method=fake_sampling_method
                )
            
            optimizer.zero_grad()
            scaler = getattr(train_epoch_with_reconnection, '_scaler', None)
            if scaler is None:
                from training.utils import create_grad_scaler
                scaler = create_grad_scaler(use_amp, device_obj)
                train_epoch_with_reconnection._scaler = scaler
            
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
            total_loss_val, logs = compute_loss_with_reconnection(
                model, batch, latents, t, noise, cond, loss_fn,
                discriminator, use_amp, device_obj,
                fake_sampling_steps=fake_sampling_steps,
                fake_sampling_method=fake_sampling_method
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


def eval_epoch_with_reconnection(
    model, dataloader, scheduler, loss_fn, discriminator, 
    device, use_amp=False, fake_sampling_steps=50, fake_sampling_method="ddim"
):
    """Evaluate for one epoch using CompositeLoss with gradient reconnection."""
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
                    total_loss_val, logs = compute_loss_with_reconnection(
                        model, batch, latents, t, noise, cond, loss_fn,
                        discriminator, use_amp, device_obj,
                        fake_sampling_steps=fake_sampling_steps,
                        fake_sampling_method=fake_sampling_method
                    )
            else:
                total_loss_val, logs = compute_loss_with_reconnection(
                    model, batch, latents, t, noise, cond, loss_fn,
                    discriminator, use_amp, device_obj,
                    fake_sampling_steps=fake_sampling_steps,
                    fake_sampling_method=fake_sampling_method
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


def train_diffusion_iteration(
    model, train_loader, val_loader, loss_fn, discriminator, optimizer, scheduler,
    device, iteration, epochs, use_amp=False, max_grad_norm=None,
    use_non_uniform_sampling=False, eval_interval=5, sample_interval=10,
    output_dir=None, exp_name=None, fake_sampling_steps=50, fake_sampling_method="ddim"
):
    """
    Fine-tune diffusion model for one iteration with discriminator loss.
    
    Args:
        model: DiffusionModel
        train_loader: Training dataloader
        val_loader: Validation dataloader (optional)
        loss_fn: CompositeLoss with DiscriminatorLossWithReconnection
        discriminator: Trained discriminator
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device
        iteration: Current iteration number
        epochs: Number of epochs
        use_amp: Use mixed precision
        max_grad_norm: Gradient clipping
        use_non_uniform_sampling: Non-uniform timestep sampling
        eval_interval: Eval every N epochs
        sample_interval: Sample every N epochs
        output_dir: Output directory
        exp_name: Experiment name
        fake_sampling_steps: Steps for fake generation
        fake_sampling_method: Method for fake generation
    
    Returns:
        (model, metrics_history)
    """
    output_dir = Path(output_dir) if output_dir else None
    device_obj = to_device(device)
    
    best_val_loss = float("inf")
    training_history = []
    start_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Diffusion Model - Iteration {iteration}")
    print(f"{'='*60}")
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_logs = train_epoch_with_reconnection(
            model, train_loader, scheduler, loss_fn, discriminator,
            optimizer, device, epoch + 1, use_amp=use_amp, max_grad_norm=max_grad_norm,
            use_non_uniform_sampling=use_non_uniform_sampling,
            fake_sampling_steps=fake_sampling_steps,
            fake_sampling_method=fake_sampling_method
        )
        
        print(f"Train Loss: {train_loss:.6f}")
        for k, v in train_logs.items():
            print(f"  {k}: {v:.6f}")
        
        # Validate
        val_loss = float("inf")
        val_logs = {}
        if val_loader and (epoch + 1) % eval_interval == 0:
            val_loss, val_logs = eval_epoch_with_reconnection(
                model, val_loader, scheduler, loss_fn, discriminator,
                device, use_amp=use_amp,
                fake_sampling_steps=fake_sampling_steps,
                fake_sampling_method=fake_sampling_method
            )
            print(f"Val Loss: {val_loss:.6f}")
            for k, v in val_logs.items():
                print(f"  {k}: {v:.6f}")
        
        # Record history
        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_logs.items()},
            **{f"val_{k}": v for k, v in val_logs.items()}
        }
        training_history.append(history_entry)
        
        # Save checkpoint
        if output_dir and exp_name:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            checkpoint_path = checkpoint_dir / f"{exp_name}_iter_{iteration}_checkpoint_latest.pt"
            model.save_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                training_history=training_history
            )
            
            if is_best:
                best_checkpoint_path = checkpoint_dir / f"{exp_name}_iter_{iteration}_checkpoint_best.pt"
                model.save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                    training_history=training_history
                )
                print(f"  Saved best checkpoint (val_loss={best_val_loss:.6f})")
    
    return model, training_history


def main():
    parser = argparse.ArgumentParser(description="Adversarial training pipeline for diffusion models")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "adversarial")
    print(f"Experiment: {exp_name}")
    
    # Set deterministic behavior
    training_seed = config.get("training", {}).get("seed", 42)
    set_deterministic(training_seed)
    print(f"Set deterministic mode with seed: {training_seed}")
    
    # Get device
    device = get_device(config)
    print(f"Device: {device}")
    device_obj = to_device(device)
    
    # Get output directory
    output_dir = config.get("experiment", {}).get("save_path")
    if output_dir is None:
        output_dir = Path("outputs") / exp_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get adversarial config
    adversarial_cfg = config.get("adversarial", {})
    num_iterations = adversarial_cfg.get("num_iterations", 3)
    print(f"Number of iterations: {num_iterations}")
    
    # Get fake sampling config
    fake_sampling_cfg = adversarial_cfg.get("fake_sampling", {})
    num_fake_samples = fake_sampling_cfg.get("num_samples", 10000)
    fake_sampling_steps = fake_sampling_cfg.get("num_steps", 50)
    fake_sampling_method = fake_sampling_cfg.get("method", "ddim")
    fake_batch_size = fake_sampling_cfg.get("batch_size", 32)
    
    # Number of real latents should match fake latents
    num_real_samples = num_fake_samples
    
    # Get discriminator config
    disc_cfg = adversarial_cfg.get("discriminator", {})
    disc_epochs = disc_cfg.get("epochs", 50)
    disc_batch_size = disc_cfg.get("batch_size", 64)
    disc_lr = disc_cfg.get("learning_rate", 0.0002)
    discriminator_config = disc_cfg.get("config", None)
    
    # Get diffusion finetune config
    finetune_cfg = adversarial_cfg.get("diffusion_finetune", {})
    finetune_epochs = finetune_cfg.get("epochs", 10)
    finetune_batch_size = finetune_cfg.get("batch_size", 32)
    finetune_lr = finetune_cfg.get("learning_rate", 1e-5)
    use_amp = finetune_cfg.get("use_amp", False)
    max_grad_norm = finetune_cfg.get("max_grad_norm", None)
    use_non_uniform_sampling = finetune_cfg.get("use_non_uniform_sampling", False)
    eval_interval = finetune_cfg.get("eval_interval", 5)
    sample_interval = finetune_cfg.get("sample_interval", 10)
    
    # Build dataset (needed for both real latents extraction and fine-tuning)
    print("\nBuilding dataset...")
    dataset = build_dataset(config)
    
    # Get real latents path (from dataset or config)
    real_latents_path = adversarial_cfg.get("real_latents_path", None)
    if real_latents_path is None or (real_latents_path and not Path(real_latents_path).exists()):
        # Generate real latents from dataset if path doesn't exist
        print(f"\nReal latents path not found or not provided. Generating from dataset...")
        dataset_cfg = config.get("dataset", {})
        
        # Check if dataset has pre-embedded latents
        # The dataset outputs key is "latent" (from "latent_path" column)
        outputs = dataset_cfg.get("outputs", {})
        if "latent" not in outputs:
            raise ValueError(
                "real_latents_path must be provided in adversarial config, OR "
                "dataset must have 'latent' in outputs (e.g., outputs: {latent: latent_path}) to extract real latents"
            )
        
        # Extract real latents from dataset
        print("Extracting real latents from dataset...")
        print(f"Will extract {num_real_samples} real latents (matching number of fake latents)")
        
        real_latents_list = []
        
        # Create dataloader to extract latents
        temp_loader = dataset.make_dataloader(
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=device_obj.type == "cuda",
            persistent_workers=False,
            use_weighted_sampling=False
        )
        
        print(f"Extracting {num_real_samples} real latents from dataset...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Extracting latents")):
                if len(real_latents_list) >= num_real_samples:
                    break
                
                # Dataset returns "latent" key (from outputs config)
                latents = batch.get("latent")
                if latents is None:
                    print(f"Warning: No 'latent' key in batch. Available keys: {list(batch.keys())}")
                    continue
                
                # Filter for non-augmented if possible (check batch metadata)
                batch_size = latents.shape[0]
                for i in range(min(batch_size, num_real_samples - len(real_latents_list))):
                    real_latents_list.append(latents[i].cpu())
        
        if len(real_latents_list) == 0:
            raise ValueError("No latents found in dataset. Check dataset configuration.")
        
        # Save real latents
        real_latents_tensor = torch.stack(real_latents_list[:num_real_samples])
        real_latents_path = output_dir / "real_latents_all.pt"
        torch.save(real_latents_tensor, real_latents_path)
        print(f"Saved {len(real_latents_tensor)} real latents to: {real_latents_path}")
        real_latents_path = str(real_latents_path)  # Convert to string for consistency
    else:
        real_latents_path = str(Path(real_latents_path))
        if not Path(real_latents_path).exists():
            raise FileNotFoundError(f"Real latents path does not exist: {real_latents_path}")
    
    # Load initial diffusion checkpoint
    initial_checkpoint = config.get("diffusion", {}).get("checkpoint") or config.get("diffusion", {}).get("initial_checkpoint")
    if not initial_checkpoint:
        raise ValueError("Initial diffusion checkpoint must be provided in config")
    
    initial_checkpoint = Path(initial_checkpoint)
    if not initial_checkpoint.exists():
        raise FileNotFoundError(
            f"Initial diffusion checkpoint not found: {initial_checkpoint}\n"
            f"Please ensure the improved model has been trained and the checkpoint exists.\n"
            f"Expected path format: /work3/s233249/ImgiNav/experiments/diffusion/ablation/capacity_unet*_d*_improved/checkpoints/diffusion_ablation_capacity_unet*_d*_improved_checkpoint_best.pt"
        )
    
    print(f"\nLoading initial diffusion checkpoint from: {initial_checkpoint}")
    model, _ = DiffusionModel.load_checkpoint(
        str(initial_checkpoint),
        map_location=device,
        return_extra=True,
        config=config
    )
    model = model.to(device_obj)
    print("Initial checkpoint loaded successfully")
    
    # Split dataset for fine-tuning (already built above)
    print("\nSplitting dataset for fine-tuning...")
    train_dataset, val_dataset = split_dataset(dataset, config.get("training", {}))
    
    # Build data loaders
    train_loader = train_dataset.make_dataloader(
        batch_size=finetune_batch_size,
        shuffle=True,
        num_workers=config.get("training", {}).get("num_workers", 8),
        pin_memory=device_obj.type == "cuda",
        persistent_workers=config.get("training", {}).get("num_workers", 8) > 0,
        use_weighted_sampling=False
    )
    print(f"Training dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    
    val_loader = None
    if val_dataset:
        val_loader = val_dataset.make_dataloader(
            batch_size=finetune_batch_size,
            shuffle=False,
            num_workers=config.get("training", {}).get("num_workers", 8),
            pin_memory=device_obj.type == "cuda",
            persistent_workers=config.get("training", {}).get("num_workers", 8) > 0,
            use_weighted_sampling=False
        )
        print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)}")
    
    # Build loss function (with DiscriminatorLossWithReconnection)
    print("\nBuilding loss function...")
    loss_config = finetune_cfg.get("loss", config.get("loss"))
    loss_fn = build_loss({"loss": loss_config})
    print(f"  Loss type: {type(loss_fn).__name__}")
    if hasattr(loss_fn, 'losses'):
        print(f"  Loss components: {[type(l).__name__ for l in loss_fn.losses]}")
    
    # Main adversarial loop
    print(f"\n{'='*60}")
    print(f"Starting Adversarial Training Pipeline")
    print(f"{'='*60}")
    
    overall_metrics = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Step 1: Generate fake latents
        print(f"\n[Step 1/{num_iterations}] Generating fake latents...")
        fake_latents_path = generate_fake_latents(
            model=model,
            output_dir=output_dir,
            iteration=iteration,
            num_samples=num_fake_samples,
            num_steps=fake_sampling_steps,
            method=fake_sampling_method,
            batch_size=fake_batch_size,
            device_obj=device_obj,
            seed=training_seed + iteration
        )
        
        # Step 2: Train discriminator
        print(f"\n[Step 2/{num_iterations}] Training discriminator...")
        
        # Optionally resume from previous iteration's discriminator
        initial_disc_checkpoint = None
        if iteration > 0:
            prev_checkpoint_path = output_dir / "checkpoints" / f"discriminator_iter_{iteration-1}_best.pt"
            if prev_checkpoint_path.exists():
                initial_disc_checkpoint = prev_checkpoint_path
                print(f"  Found previous discriminator checkpoint: {initial_disc_checkpoint}")
                print(f"  Note: Currently training NEW discriminator from scratch.")
                print(f"  To resume from previous, modify train_discriminator_iteration()")
        
        discriminator, disc_checkpoint_path = train_discriminator_iteration(
            real_latents_path=real_latents_path,
            fake_latents_path=fake_latents_path,
            output_dir=output_dir,
            iteration=iteration,
            discriminator_config=discriminator_config,
            epochs=disc_epochs,
            batch_size=disc_batch_size,
            learning_rate=disc_lr,
            device=device,
            seed=training_seed + iteration,
            initial_checkpoint=initial_disc_checkpoint
        )
        
        # Load discriminator history for plotting
        disc_history_path = output_dir / f"discriminator_iter_{iteration}" / "discriminator_history.csv"
        if disc_history_path.exists():
            disc_history_df = pd.read_csv(disc_history_path)
            plot_discriminator_metrics(disc_history_df, output_dir, iteration, exp_name)
        
        # Step 3: Fine-tune diffusion with discriminator loss
        print(f"\n[Step 3/{num_iterations}] Fine-tuning diffusion model...")
        
        # Build optimizer and scheduler for this iteration
        # Create a temporary config dict with training section for build_optimizer
        temp_config = config.copy()
        temp_config["training"] = temp_config.get("training", {}).copy()
        temp_config["training"]["learning_rate"] = finetune_lr
        temp_config["training"]["optimizer"] = finetune_cfg.get("optimizer", temp_config["training"].get("optimizer", "AdamW"))
        temp_config["training"]["weight_decay"] = finetune_cfg.get("weight_decay", temp_config["training"].get("weight_decay", 0.0))
        
        optimizer = build_optimizer(model, temp_config)
        
        scheduler = build_scheduler(optimizer, finetune_cfg.get("scheduler", config.get("scheduler", {})), last_epoch=-1)
        
        model, training_history = train_diffusion_iteration(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            discriminator=discriminator,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            iteration=iteration,
            epochs=finetune_epochs,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            use_non_uniform_sampling=use_non_uniform_sampling,
            eval_interval=eval_interval,
            sample_interval=sample_interval,
            output_dir=output_dir,
            exp_name=exp_name,
            fake_sampling_steps=fake_sampling_steps,
            fake_sampling_method=fake_sampling_method
        )
        
        # Save metrics
        metrics_path = output_dir / f"{exp_name}_metrics_iter_{iteration}.csv"
        save_metrics_csv(training_history, metrics_path)
        
        # Plot diffusion metrics
        if len(training_history) > 0:
            training_df = pd.DataFrame(training_history)
            plot_diffusion_metrics(training_df, output_dir, iteration, exp_name)
        
        # Record overall metrics
        if len(training_history) > 0:
            best_entry = min(training_history, key=lambda x: x.get("val_loss", float("inf")))
            best_entry["iteration"] = iteration
            overall_metrics.append(best_entry)
    
    # Generate overall plots
    print(f"\n{'='*60}")
    print("Generating overall metrics plots...")
    print(f"{'='*60}")
    
    if len(overall_metrics) > 0:
        overall_df = pd.DataFrame(overall_metrics)
        overall_csv_path = output_dir / f"{exp_name}_overall_metrics.csv"
        overall_df.to_csv(overall_csv_path, index=False)
        print(f"Saved overall metrics to: {overall_csv_path}")
    
    plot_overall_iteration_metrics(output_dir, exp_name)
    plot_iterative_refinement_metrics(output_dir, exp_name)
    
    print(f"\n{'='*60}")
    print("Adversarial Training Pipeline Completed!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

