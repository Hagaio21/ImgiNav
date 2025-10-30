#!/usr/bin/env python3
"""
AutoEncoder/VAE training script.
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

from models.autoencoder import AutoEncoder
from models.losses.custom_loss import StandardVAELoss, SegmentationVAELoss
from models.datasets import build_dataloaders
from training.training_utils import (
    load_config, setup_device, setup_experiment_directories, 
    save_experiment_config, create_progress_bar, save_model_checkpoint,
    TrainingLogger
)

def build_loss_function(loss_cfg):
    """Build loss function from config."""
    loss_map = {
        "standard": StandardVAELoss,
        "vae": StandardVAELoss,
        "segmentation": SegmentationVAELoss,
    }
    loss_type = loss_cfg.get("type", "standard").lower()
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss: {loss_type}")
    
    LossClass = loss_map[loss_type]
    
    # Special handling for SegmentationVAELoss
    if loss_type == "segmentation":
        from common.taxonomy import load_valid_colors
        id_to_color, valid_ids = load_valid_colors(
            loss_cfg["taxonomy_path"],
            include_background=loss_cfg.get("include_background", True)
        )
        return LossClass(
            id_to_color=id_to_color,
            kl_weight=loss_cfg.get("kl_weight", 1e-6),
            lambda_seg=loss_cfg.get("lambda_seg", 1.0),
            lambda_mse=loss_cfg.get("lambda_mse", 1.0),
        )
    
    # Standard instantiation
    return LossClass(**{k: v for k, v in loss_cfg.items() if k != "type"})

def build_optimizer(model, training_cfg):
    """Build optimizer from config."""
    lr = training_cfg.get("lr", 1e-4)
    opt_type = training_cfg.get("optimizer", {}).get("type", "adam").lower()
    weight_decay = training_cfg.get("optimizer", {}).get("weight_decay", 0.0)
    
    if opt_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
            if isinstance(outputs, dict):
                loss_result = loss_fn(outputs)
            else:
                # Legacy mode - outputs is just reconstruction
                loss_result = loss_fn(batch, outputs)
            
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
                metrics = loss_result[1] if len(loss_result) > 1 else {}
            else:
                loss = loss_result
                metrics = {}
            
            # Backward pass
            loss.backward()
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
                generate_samples(model, device, epoch, batch_idx, training_cfg.get("output_dir", "outputs"), batch)
    
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
            if isinstance(outputs, dict):
                loss_result = loss_fn(outputs)
            else:
                # Legacy mode
                loss_result = loss_fn(batch, outputs)
            
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


def generate_samples(model, device, epoch, step, output_dir, sample_batch):
    """Generate sample images by encoding and decoding input images."""
    model.eval()
    with torch.no_grad():
        # Extract input from batch - MUST be provided
        if isinstance(sample_batch, dict):
            x = sample_batch["layout"]  # Must have layout key
        else:
            x = sample_batch  # Must be tensor
        
        x = x[:4]  # Take first 4 samples
        
        # Run through autoencoder
        output = model(x)
        if isinstance(output, dict):
            recon = output["recon"]
        else:
            recon = output
        
        # Create comparison: original | reconstructed
        comparison = torch.cat([x, recon], dim=0)
        
        # Save samples
        samples_dir = Path(output_dir) / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Normalize to [0, 1] if needed
        if comparison.min() < 0:
            comparison = (comparison + 1) / 2
        comparison = torch.clamp(comparison, 0, 1)
        
        save_image(comparison, samples_dir / f"epoch_{epoch}_step_{step}.png", nrow=4)


def main():
    parser = argparse.ArgumentParser(description="Train AutoEncoder/VAE")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    device = setup_device(config.get("dataset", {}).get("seed", 42))
    output_dir, ckpt_dir = setup_experiment_directories(config)
    save_experiment_config(config, output_dir)
    
    # Build model
    model_cfg = config["model"]
    model = AutoEncoder.from_config(model_cfg).to(device)
    print(f"Model type: {model_cfg.get('type', 'vae')}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build loss function
    loss_cfg = config["training"].get("loss", {"type": "standard"})
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
                save_model_checkpoint(model, os.path.join(ckpt_dir, "ae_best.pt"), 
                                    {"epoch": epoch, "val_loss": val_loss})
        
        # Log epoch summary and generate plots
        training_logger.log_epoch_summary(epoch, train_metrics, val_metrics)
        
        # Save latest model
        save_model_checkpoint(model, os.path.join(ckpt_dir, "ae_latest.pt"), 
                            {"epoch": epoch})
    
    print("Training complete!")


if __name__ == "__main__":
    main()
