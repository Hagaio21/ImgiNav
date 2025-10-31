#!/usr/bin/env python3
"""
Training script for autoencoder experiments.
Loads experiment config, builds model, dataset, loss, and runs training.
"""

import torch
import torch.nn.functional as F
import yaml
import json
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from models.losses.base_loss import LOSS_REGISTRY


def load_config(config_path: Path):
    """Load experiment config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    """Build autoencoder from config."""
    ae_cfg = config["autoencoder"]
    model = Autoencoder.from_config(ae_cfg)
    return model


def build_dataset(config):
    """Build dataset from config."""
    ds_cfg = config["dataset"]
    dataset = ManifestDataset(**ds_cfg)
    return dataset


def build_loss(config):
    """Build loss function from config."""
    loss_cfg = config["training"]["loss"]
    loss_fn = LOSS_REGISTRY[loss_cfg["type"]].from_config(loss_cfg)
    return loss_fn


def build_optimizer(model, config):
    """Build optimizer from config using model's parameter_groups for trainable params."""
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    optimizer_type = config["training"].get("optimizer", "AdamW").lower()
    
    # Get parameter groups from model (respects frozen params and per-module LRs)
    param_groups = model.parameter_groups()
    
    # If no groups returned (all frozen or no per-module LRs), use trainable params with base LR
    if not param_groups:
        trainable_params = list(model.trainable_parameters())
        if not trainable_params:
            raise ValueError("No trainable parameters found in model!")
        param_groups = [{"params": trainable_params, "lr": lr}]
    else:
        # Set default LR for groups that don't have one specified
        for group in param_groups:
            if "lr" not in group:
                group["lr"] = lr
    
    # Add weight_decay to all groups
    for group in param_groups:
        group["weight_decay"] = weight_decay
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_type == "sgd":
        for group in param_groups:
            group["momentum"] = 0.9
        optimizer = torch.optim.SGD(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch["rgb"])
        
        # Compute loss
        loss, logs = loss_fn(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate stats
        batch_size = batch["rgb"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update logs
        for k, v in logs.items():
            if k not in log_dict:
                log_dict[k] = 0.0
            log_dict[k] += v.item() * batch_size
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item(), **{k: v/total_samples for k, v in log_dict.items()}})
    
    avg_loss = total_loss / total_samples
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    return avg_loss, avg_logs


def eval_epoch(model, dataloader, loss_fn, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch["rgb"])
            
            # Compute loss
            loss, logs = loss_fn(outputs, batch)
            
            # Accumulate stats
            batch_size = batch["rgb"].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update logs
            for k, v in logs.items():
                if k not in log_dict:
                    log_dict[k] = 0.0
                log_dict[k] += v.item() * batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_logs = {k: v / total_samples for k, v in log_dict.items()}
    
    return avg_loss, avg_logs


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=32, target_size=256):
    """Save sample images from validation set."""
    model.eval()
    samples_dir = output_dir / "samples" / f"epoch_{epoch:03d}"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get one batch for visualization
    batch_iter = iter(val_loader)
    batch = next(batch_iter)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Limit batch size for visualization
    if isinstance(batch.get("rgb"), torch.Tensor):
        batch_size = min(batch["rgb"].shape[0], sample_batch_size)
        batch = {k: v[:batch_size] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(batch["rgb"])
    
    # Save RGB input and reconstruction
    if "rgb" in batch and "rgb" in outputs:
        input_rgb = batch["rgb"]
        pred_rgb = outputs["rgb"]
        
        # Denormalize from [-1, 1] or [0, 1] to [0, 1]
        if pred_rgb.min() < 0:
            pred_rgb = (pred_rgb + 1) / 2  # tanh output: [-1, 1] -> [0, 1]
        pred_rgb = torch.clamp(pred_rgb, 0, 1)
        
        # Handle input normalization (assumes [0, 1])
        input_rgb = torch.clamp(input_rgb, 0, 1)
        
        # Resize if needed
        if input_rgb.shape[-1] != target_size:
            input_rgb = F.interpolate(input_rgb, size=(target_size, target_size), mode='bilinear', align_corners=False)
            pred_rgb = F.interpolate(pred_rgb, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Create grid: [input, pred, input, pred, ...]
        images = []
        for i in range(batch_size):
            images.append(input_rgb[i])
            images.append(pred_rgb[i])
        
        grid = torch.stack(images)
        grid_path = samples_dir / "rgb_input_pred.png"
        save_image(grid, grid_path, nrow=2, padding=2, normalize=False)
    
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
        pred_classes_vis = (pred_classes.float() / pred_classes.max().float().clamp(min=1)) * 255
        pred_classes_vis = pred_classes_vis.unsqueeze(1).repeat(1, 3, 1, 1) / 255.0
        
        grid_path = samples_dir / "segmentation_pred.png"
        save_image(pred_classes_vis, grid_path, nrow=4, padding=2, normalize=False)
    
    print(f"  Saved samples to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder from experiment config")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    print(f"Experiment: {exp_name}")
    
    # Get device from config or default
    device = config.get("training", {}).get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    # Build components
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Building dataset...")
    dataset = build_dataset(config)
    train_loader = dataset.make_dataloader(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True),
        num_workers=config["training"].get("num_workers", 4)
    )
    print(f"Train dataset size: {len(dataset)}, Batches: {len(train_loader)}")
    
    # Build validation dataset if provided
    val_dataset = None
    val_loader = None
    if "validation" in config:
        val_cfg = config["validation"].get("dataset", {})
        if val_cfg:
            val_dataset = ManifestDataset(**val_cfg)
            val_loader = val_dataset.make_dataloader(
                batch_size=config["validation"].get("batch_size", config["training"]["batch_size"]),
                shuffle=False,
                num_workers=config["training"].get("num_workers", 4)
            )
            print(f"Validation dataset size: {len(val_dataset)}, Batches: {len(val_loader)}")
    
    print("Building loss function...")
    loss_fn = build_loss(config)
    
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    training_history = []
    if args.resume and args.resume.exists():
        print(f"Resuming from checkpoint: {args.resume}")
        # Load model using class method (includes config)
        model = Autoencoder.load_checkpoint(args.resume, map_location=device)
        model = model.to(device)
        
        # Try to load training history from logs (handle both with/without exp name)
        checkpoint_name = Path(args.resume).stem
        if "_checkpoint_" in checkpoint_name:
            logs_name = checkpoint_name.replace("_checkpoint_", "_logs_")
        else:
            logs_name = checkpoint_name.replace("checkpoint_", "logs_")
        logs_path = Path(args.resume).parent / f"{logs_name}.json"
        if logs_path.exists():
            with open(logs_path, "r") as f:
                latest_log = json.load(f)
            start_epoch = latest_log.get("epoch", 0)
            print(f"Resuming from epoch {start_epoch} (from logs)")
            
            # Try to load full training history if available (extract exp name from checkpoint)
            if "_checkpoint_" in checkpoint_name:
                exp_name_from_checkpoint = checkpoint_name.split("_checkpoint_")[0]
                history_path = Path(args.resume).parent / f"{exp_name_from_checkpoint}_training_history.json"
            else:
                history_path = Path(args.resume).parent / "training_history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    training_history = json.load(f)
                    print(f"Loaded training history ({len(training_history)} epochs)")
        else:
            # Extract epoch number from checkpoint filename
            checkpoint_name = Path(args.resume).stem
            if "epoch_" in checkpoint_name:
                try:
                    epoch_num = int(checkpoint_name.split("epoch_")[1].split(".")[0])
                    start_epoch = epoch_num
                    print(f"Resuming from epoch {start_epoch} (from filename)")
                except:
                    print("Warning: Could not determine epoch from checkpoint, starting from epoch 0")
            else:
                print("Warning: Could not determine epoch, optimizer state not resumed")
        
        # Note: Optimizer state is not saved/restored. Training will restart optimizer from scratch.
    
    # Training configuration (all from config)
    num_epochs = config["training"]["epochs"]
    save_interval = config["training"].get("save_interval", 1)
    eval_interval = config["training"].get("eval_interval", 1)
    sample_interval = config["training"].get("sample_interval", 5)
    keep_checkpoints = config["training"].get("keep_checkpoints", None)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Save interval: every {save_interval} epoch(s)")
    print(f"  Eval interval: every {eval_interval} epoch(s)")
    print(f"  Sample interval: every {sample_interval} epoch(s)")
    if keep_checkpoints:
        print(f"  Keeping only last {keep_checkpoints} checkpoints")
    
    best_val_loss = float("inf")
    checkpoint_files = []
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        avg_loss, avg_logs = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch + 1)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_loss:.6f}")
        for k, v in avg_logs.items():
            print(f"  Train {k}: {v:.6f}")
        
        # Record training history
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            **{f"train_{k}": float(v) for k, v in avg_logs.items()}
        }
        
        # Evaluation
        if val_loader and (epoch + 1) % eval_interval == 0:
            val_loss, val_logs = eval_epoch(model, val_loader, loss_fn, device)
            print(f"  Val Loss: {val_loss:.6f}")
            for k, v in val_logs.items():
                print(f"  Val {k}: {v:.6f}")
            epoch_log["val_loss"] = float(val_loss)
            epoch_log.update({f"val_{k}": float(v) for k, v in val_logs.items()})
            
            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best validation loss: {best_val_loss:.6f}")
        
        # Save samples at specified interval (use validation set if available, else training set)
        if (epoch + 1) % sample_interval == 0:
            loader_to_use = val_loader if val_loader else train_loader
            save_samples(model, loader_to_use, device, output_dir, epoch + 1, sample_batch_size=32)
        
        training_history.append(epoch_log)
        
        # Save checkpoint and logs at specified interval
        should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs
        if should_save:
            checkpoint_path = output_dir / f"{exp_name}_checkpoint_epoch_{epoch + 1:03d}.pt"
            # Save checkpoint with config inside (via save_checkpoint method)
            model.save_checkpoint(checkpoint_path, include_config=True)
            checkpoint_files.append(checkpoint_path)
            
            # Save logs (loss dict and eval dicts) as JSON
            logs_path = output_dir / f"{exp_name}_logs_epoch_{epoch + 1:03d}.json"
            with open(logs_path, "w") as f:
                json.dump(epoch_log, f, indent=2)
            
            # Save best checkpoint if validation improved
            if val_loader and "val_loss" in epoch_log and epoch_log["val_loss"] == best_val_loss:
                best_path = output_dir / f"{exp_name}_checkpoint_best.pt"
                model.save_checkpoint(best_path, include_config=True)
                best_logs_path = output_dir / f"{exp_name}_logs_best.json"
                with open(best_logs_path, "w") as f:
                    json.dump(epoch_log, f, indent=2)
                print(f"  Saved best checkpoint and logs")
        
        # Always save latest checkpoint and logs
        latest_path = output_dir / f"{exp_name}_checkpoint_latest.pt"
        model.save_checkpoint(latest_path, include_config=True)
        latest_logs_path = output_dir / f"{exp_name}_logs_latest.json"
        with open(latest_logs_path, "w") as f:
            json.dump(epoch_log, f, indent=2)
        
        # Clean up old checkpoints and logs if keeping only N
        if keep_checkpoints and len(checkpoint_files) > keep_checkpoints:
            # Remove oldest checkpoint and log files
            for old_checkpoint in checkpoint_files[:-keep_checkpoints]:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                # Handle both old format (without exp name) and new format (with exp name)
                if "_checkpoint_" in old_checkpoint.name:
                    old_logs = old_checkpoint.parent / old_checkpoint.name.replace("_checkpoint_", "_logs_")
                else:
                    old_logs = old_checkpoint.parent / old_checkpoint.name.replace("checkpoint_", "logs_")
                if old_logs.exists():
                    old_logs.unlink()
            checkpoint_files = checkpoint_files[-keep_checkpoints:]
    
    # Save full training history (all loss and eval dicts) to JSON
    history_path = output_dir / f"{exp_name}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining complete!")
    print(f"  Checkpoints (with config): {output_dir}/{exp_name}_checkpoint_*.pt")
    print(f"  Logs (loss and eval dicts): {output_dir}/{exp_name}_logs_*.json")
    print(f"  Full training history: {history_path}")


if __name__ == "__main__":
    main()

