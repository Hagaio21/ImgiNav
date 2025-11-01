#!/usr/bin/env python3
"""
Training script for autoencoder experiments.
Loads experiment config, builds model, dataset, loss, and runs training.
"""

import torch
import torch.nn.functional as F
import yaml
import sys
import math
import warnings
from pathlib import Path
from tqdm import tqdm
import argparse
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import pandas as pd

# Suppress torchvision.io extension warning (we use PIL, not torchvision.io)
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.utils import (
    set_deterministic,
    load_config,
    build_model,
    build_dataset,
    build_loss,
    build_optimizer,
    get_device,
)


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    # Convert device string to device object if needed
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    # Use non_blocking transfer for faster GPU transfers when pin_memory is enabled
    non_blocking = device_obj.type == "cuda"
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move batch to device (non_blocking if using CUDA with pin_memory)
        batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(batch["rgb"])
                loss, logs = loss_fn(outputs, batch)
            
            # Backward pass with gradient scaling (scaler created once before loop)
            optimizer.zero_grad()
            # Create scaler if needed (should be passed in, but handle here for now)
            scaler = getattr(train_epoch, '_scaler', None)
            if scaler is None and use_amp:
                scaler = torch.cuda.amp.GradScaler()
                train_epoch._scaler = scaler
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            outputs = model(batch["rgb"])
            loss, logs = loss_fn(outputs, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Accumulate stats (detach before item() to avoid blocking)
        batch_size = batch["rgb"].shape[0]
        loss_val = loss.detach().item()
        total_loss += loss_val * batch_size
        total_samples += batch_size
        
        # Update logs (detach before item())
        for k, v in logs.items():
            if k not in log_dict:
                log_dict[k] = 0.0
            log_dict[k] += v.detach().item() * batch_size
        
        # Update progress bar
        pbar.set_postfix({"loss": loss_val, **{k: v/total_samples for k, v in log_dict.items()}})
    
    avg_loss = total_loss / total_samples
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    return avg_loss, avg_logs


def eval_epoch(model, dataloader, loss_fn, device, use_amp=False):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    # Convert device string to device object if needed
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    # Use non_blocking transfer for faster GPU transfers when pin_memory is enabled
    non_blocking = device_obj.type == "cuda"
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            # Move batch to device (non_blocking if using CUDA with pin_memory)
            batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if use_amp and device_obj.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(batch["rgb"])
                    loss, logs = loss_fn(outputs, batch)
            else:
                outputs = model(batch["rgb"])
                loss, logs = loss_fn(outputs, batch)
            
            # Accumulate stats (detach before item())
            batch_size = batch["rgb"].shape[0]
            loss_val = loss.detach().item()
            total_loss += loss_val * batch_size
            total_samples += batch_size
            
            # Update logs (detach before item())
            for k, v in logs.items():
                if k not in log_dict:
                    log_dict[k] = 0.0
                log_dict[k] += v.detach().item() * batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    return avg_loss, avg_logs


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=8, target_size=256):
    """Save sample images from validation set."""
    model.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert device string to device object if needed
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    # Get one batch for visualization
    non_blocking = device_obj.type == "cuda"
    batch_iter = iter(val_loader)
    batch = next(batch_iter)
    batch = {k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Limit batch size for visualization and compute grid size
    if isinstance(batch.get("rgb"), torch.Tensor):
        batch_size = min(batch["rgb"].shape[0], sample_batch_size)
        batch = {k: v[:batch_size] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Compute n for n x n grid
        grid_n = int(math.sqrt(batch_size))
        if grid_n * grid_n < batch_size:
            grid_n += 1
    
    with torch.no_grad():
        outputs = model(batch["rgb"])
    
    # Save RGB input and reconstruction as two side-by-side grids
    if "rgb" in batch and "rgb" in outputs:
        input_rgb = batch["rgb"]  # Already in [-1, 1] range (from dataset normalization)
        pred_rgb = outputs["rgb"]  # Output from tanh is in [-1, 1]
        
        # Denormalize both from [-1, 1] to [0, 1] for visualization
        # Both input and prediction are in [-1, 1], convert to [0, 1] for display
        input_rgb = (input_rgb + 1) / 2.0
        pred_rgb = (pred_rgb + 1) / 2.0
        
        # Don't clamp - preserve the actual output range
        # Clamping can make images look faded if the model hasn't learned full range yet
        
        # Resize if needed
        if input_rgb.shape[-1] != target_size:
            input_rgb = F.interpolate(input_rgb, size=(target_size, target_size), mode='bilinear', align_corners=False)
            pred_rgb = F.interpolate(pred_rgb, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Create two complete grids side by side: original (left) and reconstruction (right)
        # Create original grid (n×n)
        orig_grid = make_grid(input_rgb, nrow=grid_n, padding=2, normalize=False)
        # Create reconstruction grid (n×n)  
        recon_grid = make_grid(pred_rgb, nrow=grid_n, padding=2, normalize=False)
        
        # Concatenate horizontally (side by side)
        # Both grids are (C, H, W), concatenate along width dimension
        combined_grid = torch.cat([orig_grid, recon_grid], dim=2)  # Concatenate along width
        
        # Save combined grid: n×n original | n×n reconstruction (side by side)
        grid_path = samples_dir / f"epoch_{epoch:03d}_comparison.png"
        save_image(combined_grid, grid_path, normalize=False)
    
    # Save segmentation if available
    if "segmentation" in outputs:
        pred_seg = outputs["segmentation"]  # [B, C, H, W]
        # Get predicted classes
        pred_classes = torch.argmax(pred_seg, dim=1)  # [B, H, W]
        
        # Resize if needed
        if pred_classes.shape[-1] != target_size:
            pred_classes = F.interpolate(
                pred_classes.unsqueeze(1).float(),
                size=(target_size, target_size),
                mode='nearest'
            ).squeeze(1).long()
        
        # Normalize to [0, 1] for visualization (simple colormap)
        max_class = pred_classes.max().float().clamp(min=1)
        pred_classes_vis = (pred_classes.float() / max_class)
        pred_classes_vis = pred_classes_vis.unsqueeze(1).repeat(1, 3, 1, 1)
        
        seg_grid = torch.stack([pred_classes_vis[i] for i in range(batch_size)])
        seg_path = samples_dir / f"epoch_{epoch:03d}_segmentation.png"
        save_image(seg_grid, seg_path, nrow=grid_n, padding=2, normalize=False)
    
    print(f"  Saved samples to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder from experiment config")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    
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
    
    # Get device from config or default
    device = get_device(config)
    print(f"Device: {device}")
    
    # Get output directory from config
    output_dir = config.get("experiment", {}).get("save_path")
    if output_dir is None:
        # Default: outputs/experiment_name
        output_dir = Path("outputs") / exp_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get phase directory for shared metrics/samples (if phase is specified)
    phase = config.get("experiment", {}).get("phase", None)
    phase_dir = None
    if phase:
        # Phase folder: outputs/phase_name (shared across all experiments in phase)
        phase_dir = Path("outputs") / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        print(f"Phase directory: {phase_dir} (for shared metrics/samples)")
    
    # Build components
    print("Building model...")
    model = build_model(config)
    
    # Convert device string to device object
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    model = model.to(device_obj)
    
    # Enable cudnn benchmark for faster convolutions (optimizes for input sizes)
    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn.benchmark for faster convolutions")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Building dataset...")
    dataset = build_dataset(config)
    
    # Build validation dataset
    val_dataset = None
    val_loader = None
    
    # Check if validation dataset is explicitly provided
    if "validation" in config and "dataset" in config["validation"]:
        val_cfg = config["validation"]["dataset"]
        val_dataset = ManifestDataset(**val_cfg)
        val_loader = val_dataset.make_dataloader(
            batch_size=config["validation"].get("batch_size", config["training"]["batch_size"]),
            shuffle=False,
            num_workers=config["training"].get("num_workers", 4)
        )
        print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)} (from config)")
        # Use full dataset for training if validation is explicitly provided
        train_dataset = dataset
    else:
        # Auto-split dataset using train_split from config
        train_split = config["training"].get("train_split", 0.8)
        split_seed = config["training"].get("split_seed", 42)
        
        if train_split < 1.0:
            train_dataset, val_dataset = dataset.split(train_split=train_split, random_seed=split_seed)
            val_loader = val_dataset.make_dataloader(
                batch_size=config["validation"].get("batch_size", config["training"]["batch_size"]) if "validation" in config else config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["training"].get("num_workers", 4)
            )
            print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)} (auto-split {int((1-train_split)*100)}%)")
        else:
            train_dataset = dataset
    
    train_loader = train_dataset.make_dataloader(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True),
        num_workers=config["training"].get("num_workers", 4)
    )
    print(f"Train dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    
    print("Building loss function...")
    loss_fn = build_loss(config)
    
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    
    # Enable mixed precision training by default (can be disabled in config)
    use_amp = config.get("training", {}).get("use_amp", True)  # Default to True for speedup
    if use_amp and device_obj.type == "cuda":
        print("Using mixed precision training (FP16)")
        # Create scaler once for the training function
        train_epoch._scaler = torch.amp.GradScaler('cuda')
    elif not use_amp:
        print("Mixed precision training disabled (use_amp: false)")
    
    # Training configuration (all from config)
    num_epochs = config["training"]["epochs"]
    save_interval = config["training"].get("save_interval", 1)
    eval_interval = config["training"].get("eval_interval", 1)
    sample_interval = config["training"].get("sample_interval", 5)
    keep_checkpoints = config["training"].get("keep_checkpoints", None)
    
    # Early stopping configuration
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    early_stopping_restore_best = config["training"].get("early_stopping_restore_best", True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Save interval: every {save_interval} epoch(s)")
    if val_loader:
        print(f"  Evaluation: every epoch")
    print(f"  Sample interval: every {sample_interval} epoch(s)")
    if keep_checkpoints:
        print(f"  Keeping only last {keep_checkpoints} checkpoints")
    if early_stopping_patience:
        print(f"  Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        if early_stopping_restore_best:
            print(f"  Will restore best checkpoint on early stop")
    
    best_val_loss = float("inf")
    checkpoint_files = []
    epochs_without_improvement = 0
    training_history = []
    
    # CSV file path for metrics (in experiment folder)
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    
    # Also save metrics to phase folder if phase is specified (for analysis)
    phase_metrics_path = None
    if phase_dir:
        phase_metrics_path = phase_dir / f"{exp_name}_metrics.csv"
    
    for epoch in range(num_epochs):
        # Training
        avg_loss, avg_logs = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch + 1, use_amp=use_amp)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_loss:.6f}")
        for k, v in avg_logs.items():
            print(f"  Train {k}: {v:.6f}")
        
        # Record training history
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            **{f"train_{k}": float(v) for k, v in avg_logs.items()}
        }
        
        # Evaluation (run every epoch if validation set exists)
        if val_loader:
            val_loss, val_logs = eval_epoch(model, val_loader, loss_fn, device, use_amp=use_amp)
            print(f"  Val Loss: {val_loss:.6f}")
            for k, v in val_logs.items():
                print(f"  Val {k}: {v:.6f}")
            epoch_log["val_loss"] = float(val_loss)
            epoch_log.update({f"val_{k}": float(v) for k, v in val_logs.items()})
            
            # Track and save best validation loss immediately
            improvement = best_val_loss - val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0  # Reset counter on improvement
                print(f"  New best validation loss: {best_val_loss:.6f} (improvement: {improvement:.6f})")
                
                # Save best checkpoint immediately (always updated when best is found)
                best_path = output_dir / f"{exp_name}_checkpoint_best.pt"
                model.save_checkpoint(best_path, include_config=True)
                print(f"  Saved best checkpoint (val_loss: {best_val_loss:.6f})")
            else:
                epochs_without_improvement += 1
                if early_stopping_patience:
                    print(f"  No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
            
            # Early stopping check
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered!")
                print(f"  No improvement for {epochs_without_improvement} epochs")
                print(f"  Best validation loss: {best_val_loss:.6f}")
                
                # Restore best checkpoint if requested
                if early_stopping_restore_best:
                    best_path = output_dir / f"{exp_name}_checkpoint_best.pt"
                    if best_path.exists():
                        print(f"  Restoring best checkpoint from {best_path}")
                        model = Autoencoder.load_checkpoint(best_path, map_location=device_obj)
                        model = model.to(device_obj)
                        print(f"  Model restored to best checkpoint")
                
                # Break out of training loop
                break
        
        # Save samples at specified interval (use validation set if available, else training set)
        if (epoch + 1) % sample_interval == 0:
            loader_to_use = val_loader if val_loader else train_loader
            # Save samples to experiment folder
            save_samples(model, loader_to_use, device, output_dir, epoch + 1, sample_batch_size=32)
            
            # Also save smaller samples to phase folder if phase is specified (for analysis)
            if phase_dir:
                save_samples(model, loader_to_use, device, phase_dir, epoch + 1, sample_batch_size=8)
        
        training_history.append(epoch_log)
        
        # Save metrics CSV (overwrite with all epochs so far)
        df = pd.DataFrame(training_history)
        df.to_csv(metrics_csv_path, index=False)
        
        # Also save to phase folder if phase is specified (for analysis)
        if phase_metrics_path:
            df.to_csv(phase_metrics_path, index=False)
        
        # Save checkpoint at specified interval
        should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs
        if should_save:
            checkpoint_path = output_dir / f"{exp_name}_checkpoint_epoch_{epoch + 1:03d}.pt"
            # Save checkpoint with config inside (via save_checkpoint method)
            model.save_checkpoint(checkpoint_path, include_config=True)
            checkpoint_files.append(checkpoint_path)
        
        # Always save latest checkpoint
        latest_path = output_dir / f"{exp_name}_checkpoint_latest.pt"
        model.save_checkpoint(latest_path, include_config=True)
        
        # Clean up old checkpoints if keeping only N
        if keep_checkpoints and len(checkpoint_files) > keep_checkpoints:
            # Remove oldest checkpoint files
            for old_checkpoint in checkpoint_files[:-keep_checkpoints]:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            checkpoint_files = checkpoint_files[-keep_checkpoints:]
    
    print(f"\nTraining complete!")
    print(f"  Checkpoints (with config): {output_dir}/{exp_name}_checkpoint_*.pt")
    print(f"  Metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()

