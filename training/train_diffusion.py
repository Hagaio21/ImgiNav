#!/usr/bin/env python3
"""
LatentDiffusion training script.
"""

import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.diffusion import LatentDiffusion
from models.losses.custom_loss import DiffusionLoss, VGGPerceptualLoss
from models.datasets import build_dataloaders
from training.training_utils import (
    load_config, setup_device, setup_experiment_directories, 
    save_experiment_config, create_progress_bar, save_model_checkpoint,
    TrainingLogger
)


def build_loss_function(loss_cfg):
    """Build loss function from config."""
    loss_type = loss_cfg.get("type", "diffusion").lower()
    
    if loss_type in ("diffusion", "mse", "hybrid"):
        # Handle VGG loss if specified
        vgg_loss_fn = None
        if loss_cfg.get("lambda_vgg", 0) > 0:
            vgg_loss_fn = VGGPerceptualLoss()
        
        return DiffusionLoss(
            lambda_mse=loss_cfg.get("lambda_mse", 1.0),
            lambda_vgg=loss_cfg.get("lambda_vgg", 0.0),
            vgg_loss_fn=vgg_loss_fn,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_optimizer(model, training_cfg):
    """Build optimizer from config (only train backbone)."""
    lr = training_cfg.get("lr", 1e-4)
    opt_type = training_cfg.get("optimizer", {}).get("type", "adam").lower()
    weight_decay = training_cfg.get("optimizer", {}).get("weight_decay", 0.0)
    
    # Only train the backbone (UNet), not the autoencoder
    params = model.backbone.parameters()
    
    if opt_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def train_epoch(model, train_loader, loss_fn, optimizer, epoch, device, training_logger, training_cfg):
    """Train for one epoch."""
    model.train()
    training_logger.start_epoch(epoch)
    
    with create_progress_bar(train_loader, epoch, training_cfg.get("epochs", 10)) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Compute loss
            loss_result = loss_fn(outputs)
            
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
                metrics = loss_result[1] if len(loss_result) > 1 else {}
            else:
                loss = loss_result
                metrics = {}
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if training_cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), training_cfg["grad_clip"])
            
            optimizer.step()
            
            # Log batch metrics
            metrics["loss"] = loss.item()
            training_logger.log_batch(metrics, batch_idx, "train")
            
            # Update progress bar
            if training_logger.should_log(batch_idx):
                epoch_metrics = training_logger.get_epoch_metrics("train")
                pbar.set_postfix_str(training_logger.format_metric_string(epoch_metrics))
            
            # Generate samples
            if training_logger.should_sample(batch_idx):
                generate_samples(model, device, epoch, batch_idx, training_cfg.get("output_dir", "outputs"))
    
    return training_logger.get_epoch_metrics("train")


def evaluate(model, val_loader, loss_fn, device, training_logger):
    """Evaluate model on validation set."""
    model.eval()
    training_logger.start_epoch(training_logger.current_epoch)  # Reset for validation
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            loss_result = loss_fn(outputs)
            
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
                metrics = loss_result[1] if len(loss_result) > 1 else {}
            else:
                loss = loss_result
                metrics = {}
            
            # Log batch metrics
            metrics["loss"] = loss.item()
            training_logger.log_batch(metrics, 0, "val")  # Use 0 as step for validation
    
    return training_logger.get_epoch_metrics("val")


def generate_samples(model, device, epoch, step, output_dir):
    """Generate sample images using diffusion sampling."""
    model.eval()
    with torch.no_grad():
        # Generate samples using DDIM sampling
        samples = model.sample(
            batch_size=4,
            image=True,  # Return decoded images
            num_steps=50,  # Fast sampling for training
            device=device,
            verbose=False
        )
        
        # Save samples
        samples_dir = Path(output_dir) / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Normalize to [0, 1] if needed
        if samples.min() < 0:
            samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        save_image(samples, samples_dir / f"epoch_{epoch}_step_{step}.png", nrow=2)


def main():
    parser = argparse.ArgumentParser(description="Train LatentDiffusion")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    device = setup_device(config.get("dataset", {}).get("seed", 42))
    output_dir, ckpt_dir = setup_experiment_directories(config)
    save_experiment_config(config, output_dir)
    
    # Build model
    model = LatentDiffusion.from_config(config, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Backbone parameters: {sum(p.numel() for p in model.backbone.parameters()):,}")
    
    # Build loss function
    loss_cfg = config["training"].get("loss", {"type": "diffusion"})
    loss_fn = build_loss_function(loss_cfg)
    
    # Build optimizer
    optimizer = build_optimizer(model, config["training"])
    
    # Build dataloaders
    train_loader, val_loader = build_dataloaders(config["dataset"])
    
    # Setup training logger
    training_cfg = config["training"]
    training_logger = TrainingLogger(
        output_dir=output_dir,
        log_interval=training_cfg.get("log_interval", 10),
        sample_interval=training_cfg.get("sample_interval", 100),
        eval_interval=training_cfg.get("eval_interval", 1)
    )
    
    # Training loop
    epochs = training_cfg.get("epochs", 10)
    best_loss = float('inf')
    
    print(f"Starting training for {epochs} epochs on {device}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, epoch, device, training_logger, training_cfg)
        
        # Evaluate
        val_metrics = None
        if training_logger.should_validate(epoch) and val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn, device, training_logger)
            
            # Save best model
            val_loss = val_metrics.get("loss", float('inf'))
            if val_loss < best_loss:
                best_loss = val_loss
                save_model_checkpoint(model.backbone, os.path.join(ckpt_dir, "unet_best.pt"), 
                                    {"epoch": epoch, "val_loss": val_loss})
        
        # Log epoch summary and generate plots
        training_logger.log_epoch_summary(epoch, train_metrics, val_metrics)
        
        # Save latest model
        save_model_checkpoint(model.backbone, os.path.join(ckpt_dir, "unet_latest.pt"), 
                            {"epoch": epoch})
    
    print("Training complete!")


if __name__ == "__main__":
    main()
