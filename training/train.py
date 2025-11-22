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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
    to_device,
    move_batch_to_device,
    create_grad_scaler,
    save_metrics_csv,
)


def compute_latent_statistics(all_latents):
    """
    Compute statistics over all collected latents.
    
    Args:
        all_latents: List of latent tensors [B, C, H, W] or [B, C]
    
    Returns:
        Dictionary with statistics keys for logging
    """
    if not all_latents or len(all_latents) == 0:
        return {}
    
    # Concatenate all latents
    all_latents_tensor = torch.cat(all_latents, dim=0)
    
    # Flatten for global statistics
    latent_flat = all_latents_tensor.reshape(all_latents_tensor.shape[0], -1)
    
    # Compute global statistics
    latent_mean = latent_flat.mean().item()
    latent_std = latent_flat.std().item()
    latent_min = latent_flat.min().item()
    latent_max = latent_flat.max().item()
    
    stats = {
        "LatentStats_Mean": latent_mean,
        "LatentStats_Std": latent_std,
        "LatentStats_Min": latent_min,
        "LatentStats_Max": latent_max,
    }
    
    # Compute per-channel statistics if spatial dimensions exist
    if all_latents_tensor.ndim == 4:  # [B, C, H, W]
        B, C, H, W = all_latents_tensor.shape
        # Per-channel mean and std
        per_channel_mean = all_latents_tensor.mean(dim=(0, 2, 3)).cpu().numpy()  # [C]
        per_channel_std = all_latents_tensor.std(dim=(0, 2, 3)).cpu().numpy()  # [C]
        # Per-channel min/max
        latents_reshaped = all_latents_tensor.permute(1, 0, 2, 3).reshape(C, -1)  # [C, B*H*W]
        per_channel_min = latents_reshaped.min(dim=1)[0].cpu().numpy()  # [C]
        per_channel_max = latents_reshaped.max(dim=1)[0].cpu().numpy()  # [C]
        
        # Store per-channel stats as JSON strings for CSV compatibility
        import json
        stats["LatentStats_MeanPerCh"] = json.dumps(per_channel_mean.tolist())
        stats["LatentStats_StdPerCh"] = json.dumps(per_channel_std.tolist())
        stats["LatentStats_MinPerCh"] = json.dumps(per_channel_min.tolist())
        stats["LatentStats_MaxPerCh"] = json.dumps(per_channel_max.tolist())
    
    return stats


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, use_amp=False, collect_latents=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    device_obj = to_device(device)
    
    # Collect latents for statistics (if VAE and requested)
    all_latents = [] if collect_latents else None
    
    # Initialize all expected class loss keys at the start of epoch
    # This ensures all classes appear in logs even if they never appear in any batch
    from models.losses.base_loss import LOSS_REGISTRY
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    ClassWeightedMSELossClass = LOSS_REGISTRY.get("ClassWeightedMSELoss")
    
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        for sub_loss in loss_fn.losses:
            if ClassWeightedMSELossClass and isinstance(sub_loss, ClassWeightedMSELossClass):
                # Get all expected class names from the loss
                if hasattr(sub_loss, 'class_idx_to_name'):
                    key = sub_loss.key if hasattr(sub_loss, 'key') else 'rgb'
                    for class_name in sub_loss.class_idx_to_name.values():
                        log_key = f"MSE_{key}_{class_name}"
                        log_dict[log_key] = 0.0
                    # Also initialize unknown
                    log_dict[f"MSE_{key}_unknown"] = 0.0
    
    # Initialize scaler before loop if using AMP
    scaler = None
    if use_amp and device_obj.type == "cuda":
        scaler = getattr(train_epoch, '_scaler', None)
        if scaler is None:
            scaler = create_grad_scaler(use_amp, device_obj)
            train_epoch._scaler = scaler
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move batch to device (non_blocking if using CUDA with pin_memory)
        batch = move_batch_to_device(batch, device_obj)
        
        # Forward pass with mixed precision
        if use_amp and device_obj.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(batch["rgb"])
                loss, logs = loss_fn(outputs, batch)
                
                # Collect latents for statistics (VAE: mu, AE: latent)
                if collect_latents and all_latents is not None:
                    if "mu" in outputs:
                        all_latents.append(outputs["mu"].detach().cpu())
                    elif "latent" in outputs:
                        all_latents.append(outputs["latent"].detach().cpu())
            
            # Ensure loss is a tensor (exit autocast first)
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss must be a tensor, got {type(loss)}: {loss}")
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            
            if scaler:
                # Scale loss and backward - must be done in sequence
                # Ensure loss is on the correct device
                if loss.device != device_obj:
                    loss = loss.to(device_obj)
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                # Step optimizer with scaler (this will check for infs)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            # Forward pass
            outputs = model(batch["rgb"])
            loss, logs = loss_fn(outputs, batch)
            
            # Collect latents for statistics (VAE: mu, AE: latent)
            if collect_latents and all_latents is not None:
                if "mu" in outputs:
                    all_latents.append(outputs["mu"].detach().cpu())
                elif "latent" in outputs:
                    all_latents.append(outputs["latent"].detach().cpu())
            
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
    
    # Ensure all expected class keys are present (defensive check)
    # This handles edge cases where initialization might have been skipped
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        for sub_loss in loss_fn.losses:
            if ClassWeightedMSELossClass and isinstance(sub_loss, ClassWeightedMSELossClass):
                if hasattr(sub_loss, 'class_idx_to_name'):
                    key = sub_loss.key if hasattr(sub_loss, 'key') else 'rgb'
                    for class_name in sub_loss.class_idx_to_name.values():
                        log_key = f"MSE_{key}_{class_name}"
                        if log_key not in avg_logs:
                            avg_logs[log_key] = 0.0
                    if f"MSE_{key}_unknown" not in avg_logs:
                        avg_logs[f"MSE_{key}_unknown"] = 0.0
    
    # Compute latent statistics if collected
    if collect_latents and all_latents and len(all_latents) > 0:
        latent_stats = compute_latent_statistics(all_latents)
        avg_logs.update(latent_stats)
    
    return avg_loss, avg_logs


def eval_epoch(model, dataloader, loss_fn, device, use_amp=False, collect_latents=False):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    log_dict = {}
    
    device_obj = to_device(device)
    
    # Collect latents for statistics (if VAE and requested)
    all_latents = [] if collect_latents else None
    
    # Initialize all expected class loss keys at the start of epoch
    # This ensures all classes appear in logs even if they never appear in any batch
    from models.losses.base_loss import LOSS_REGISTRY
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    ClassWeightedMSELossClass = LOSS_REGISTRY.get("ClassWeightedMSELoss")
    
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        for sub_loss in loss_fn.losses:
            if ClassWeightedMSELossClass and isinstance(sub_loss, ClassWeightedMSELossClass):
                # Get all expected class names from the loss
                if hasattr(sub_loss, 'class_idx_to_name'):
                    key = sub_loss.key if hasattr(sub_loss, 'key') else 'rgb'
                    for class_name in sub_loss.class_idx_to_name.values():
                        log_key = f"MSE_{key}_{class_name}"
                        log_dict[log_key] = 0.0
                    # Also initialize unknown
                    log_dict[f"MSE_{key}_unknown"] = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            # Move batch to device (non_blocking if using CUDA with pin_memory)
            batch = move_batch_to_device(batch, device_obj)
            
            # Forward pass with mixed precision
            if use_amp and device_obj.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(batch["rgb"])
                    loss, logs = loss_fn(outputs, batch)
                    
                    # Collect latents for statistics (VAE: mu, AE: latent)
                    if collect_latents and all_latents is not None:
                        if "mu" in outputs:
                            all_latents.append(outputs["mu"].detach().cpu())
                        elif "latent" in outputs:
                            all_latents.append(outputs["latent"].detach().cpu())
            else:
                outputs = model(batch["rgb"])
                loss, logs = loss_fn(outputs, batch)
                
                # Collect latents for statistics (VAE: mu, AE: latent)
                if collect_latents and all_latents is not None:
                    if "mu" in outputs:
                        all_latents.append(outputs["mu"].detach().cpu())
                    elif "latent" in outputs:
                        all_latents.append(outputs["latent"].detach().cpu())
            
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
    
    # Ensure all expected class keys are present (defensive check)
    # This handles edge cases where initialization might have been skipped
    from models.losses.base_loss import LOSS_REGISTRY
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    ClassWeightedMSELossClass = LOSS_REGISTRY.get("ClassWeightedMSELoss")
    
    if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
        for sub_loss in loss_fn.losses:
            if ClassWeightedMSELossClass and isinstance(sub_loss, ClassWeightedMSELossClass):
                if hasattr(sub_loss, 'class_idx_to_name'):
                    key = sub_loss.key if hasattr(sub_loss, 'key') else 'rgb'
                    for class_name in sub_loss.class_idx_to_name.values():
                        log_key = f"MSE_{key}_{class_name}"
                        if log_key not in avg_logs:
                            avg_logs[log_key] = 0.0
                    if f"MSE_{key}_unknown" not in avg_logs:
                        avg_logs[f"MSE_{key}_unknown"] = 0.0
    
    # Compute latent statistics if collected
    if collect_latents and all_latents and len(all_latents) > 0:
        latent_stats = compute_latent_statistics(all_latents)
        avg_logs.update(latent_stats)
    
    return avg_loss, avg_logs


def save_samples(model, val_loader, device, output_dir, epoch, sample_batch_size=8, target_size=256, exp_name=None):
    """Save sample images from validation set."""
    model.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = to_device(device)
    
    # Get one batch for visualization
    batch_iter = iter(val_loader)
    batch = next(batch_iter)
    batch = move_batch_to_device(batch, device_obj)
    
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
        # Include experiment name in filename if provided (for phase folder to avoid overwrites)
        if exp_name:
            grid_path = samples_dir / f"{exp_name}_epoch_{epoch:03d}_comparison.png"
        else:
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
        # Include experiment name in filename if provided
        if exp_name:
            seg_path = samples_dir / f"{exp_name}_epoch_{epoch:03d}_segmentation.png"
        else:
            seg_path = samples_dir / f"epoch_{epoch:03d}_segmentation.png"
        save_image(seg_grid, seg_path, nrow=grid_n, padding=2, normalize=False)
    
    print(f"  Saved samples to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder from experiment config")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--checkpoint", type=Path, default=None, 
                       help="Path to checkpoint to resume from (overrides automatic latest checkpoint detection)")
    
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
    
    # Check for checkpoint to resume from
    # Priority: 1) --checkpoint argument, 2) latest checkpoint in output_dir
    checkpoint_to_resume = None
    if args.checkpoint:
        checkpoint_to_resume = Path(args.checkpoint)
        if not checkpoint_to_resume.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_to_resume}")
        print(f"Using checkpoint from argument: {checkpoint_to_resume}")
    else:
        latest_checkpoint = output_dir / f"{exp_name}_checkpoint_latest.pt"
        if latest_checkpoint.exists():
            checkpoint_to_resume = latest_checkpoint
            print(f"Found latest checkpoint: {latest_checkpoint}")
    
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    extra_state = {}
    
    # CSV file path for metrics (defined early so we can load from it if needed)
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    
    device_obj = to_device(device)
    
    should_resume = checkpoint_to_resume is not None
    if should_resume:
        print(f"\nResuming training from checkpoint: {checkpoint_to_resume}")
        
        # Load checkpoint with extra state (uses saved config from checkpoint)
        model, extra_state = Autoencoder.load_checkpoint(
            checkpoint_to_resume,
            map_location=device_obj,
            return_extra=True,
            config=None  # Use saved config from checkpoint
        )
        model = model.to(device_obj)
        
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
        # Build components
        print("Building model...")
        model = build_model(config)
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
    
    # Auto-generate weight stats if needed
    weights_stats_path = None
    use_weighted_sampling = config["training"].get("use_weighted_sampling", False)
    if use_weighted_sampling:
        weight_column = config["training"].get("column", None)
        if weight_column:
            # Get manifest path from dataset config
            manifest_path = Path(config["dataset"]["manifest"])
            
            # Get filters from dataset config to apply before computing weights
            # This ensures weights are computed on the same filtered dataset used for training
            dataset_filters = config["dataset"].get("filters", None)
            
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
                min_weight=config["training"].get("min_weight", 1.0),
                filters=dataset_filters  # Apply same filters as dataset
            )
    
    train_loader = train_dataset.make_dataloader(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True),
        num_workers=config["training"].get("num_workers", 4),
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
    print(f"Train dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    
    print("Building loss function...")
    loss_fn = build_loss(config)
    
    # Connect CLIP loss projections to model if using CLIP loss
    if hasattr(model, 'clip_projections') and model.clip_projections is not None:
        from models.losses.base_loss import LOSS_REGISTRY
        CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
        CLIPLossClass = LOSS_REGISTRY.get("CLIPLoss")
        if CompositeLossClass and isinstance(loss_fn, CompositeLossClass):
            for sub_loss in loss_fn.losses:
                if CLIPLossClass and isinstance(sub_loss, CLIPLossClass):
                    # Connect model's projections to CLIP loss
                    sub_loss.set_projections(model.clip_projections)
                    print("  Connected CLIP projections from model to CLIP loss")
    
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    if should_resume:
        optimizer_state = extra_state.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            print("  Loaded optimizer state from checkpoint")
    
    # Enable mixed precision training by default (can be disabled in config)
    use_amp = config.get("training", {}).get("use_amp", True)  # Default to True for speedup
    if use_amp and device_obj.type == "cuda":
        print("Using mixed precision training (FP16)")
        # Create scaler once for the training function
        train_epoch._scaler = create_grad_scaler(use_amp, device_obj)
    elif not use_amp:
        print("Mixed precision training disabled (use_amp: false)")
    
    # Training configuration (all from config)
    epochs_to_train = config["training"]["epochs"]  # Additional epochs to train
    
    # Calculate end epoch: start_epoch + additional epochs to train
    end_epoch = start_epoch + epochs_to_train
    save_interval = config["training"].get("save_interval", 1)
    eval_interval = config["training"].get("eval_interval", 1)
    sample_interval = config["training"].get("sample_interval", 5)
    keep_checkpoints = config["training"].get("keep_checkpoints", None)
    
    # Early stopping configuration
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
    early_stopping_restore_best = config["training"].get("early_stopping_restore_best", True)
    
    print(f"\nTraining configuration:")
    print(f"  Additional epochs to train: {epochs_to_train}")
    print(f"  Starting from epoch: {start_epoch + 1}")
    print(f"  Will train until epoch: {end_epoch}")
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
    
    checkpoint_files = []
    epochs_without_improvement = 0
    
    # Also save metrics to phase folder if phase is specified (for analysis)
    phase_metrics_path = None
    if phase_dir:
        phase_metrics_path = phase_dir / f"{exp_name}_metrics.csv"
    
    def extract_normalizer_stats(model):
        """Extract normalizer parameters from encoder and decoder if they exist (for backward compatibility)."""
        stats = {}
        if hasattr(model.encoder, 'latent_normalizer') and model.encoder.latent_normalizer is not None:
            enc_norm = model.encoder.latent_normalizer
            shift_enc = enc_norm.shift.data.detach().cpu()
            scale_enc = torch.exp(enc_norm.log_scale.data.detach().cpu().clamp(min=-10, max=10))
            # Compute mean/std across channels and spatial dimensions
            stats['enc_shift_mean'] = float(shift_enc.mean().item())
            stats['enc_shift_std'] = float(shift_enc.std().item())
            stats['enc_scale_mean'] = float(scale_enc.mean().item())
            stats['enc_scale_std'] = float(scale_enc.std().item())
            # Per-channel values (stored as JSON string for CSV compatibility)
            stats['enc_shift_per_ch'] = json.dumps(shift_enc.squeeze().tolist())
            stats['enc_scale_per_ch'] = json.dumps(scale_enc.squeeze().tolist())
        
        if hasattr(model.decoder, 'latent_denormalizer') and model.decoder.latent_denormalizer is not None:
            dec_norm = model.decoder.latent_denormalizer
            shift_dec = dec_norm.shift.data.detach().cpu()
            scale_dec = torch.exp(dec_norm.log_scale.data.detach().cpu().clamp(min=-10, max=10))
            # Compute mean/std across channels and spatial dimensions
            stats['dec_shift_mean'] = float(shift_dec.mean().item())
            stats['dec_shift_std'] = float(shift_dec.std().item())
            stats['dec_scale_mean'] = float(scale_dec.mean().item())
            stats['dec_scale_std'] = float(scale_dec.std().item())
            # Per-channel values (stored as JSON string for CSV compatibility)
            stats['dec_shift_per_ch'] = json.dumps(shift_dec.squeeze().tolist())
            stats['dec_scale_per_ch'] = json.dumps(scale_dec.squeeze().tolist())
            
            # Compute differences (should converge to zero)
            if 'enc_shift_mean' in stats:
                stats['shift_diff_mean'] = abs(stats['enc_shift_mean'] - stats['dec_shift_mean'])
                stats['scale_diff_mean'] = abs(stats['enc_scale_mean'] - stats['dec_scale_mean'])
        
        return stats
    
    def _plot_normalizer_convergence(df, output_dir, exp_name):
        """Plot normalizer parameter convergence from training history."""
        sns.set_style("darkgrid")
        # Check if normalizer columns exist
        required_cols = ['enc_shift_mean', 'enc_scale_mean', 'dec_shift_mean', 'dec_scale_mean']
        if not all(col in df.columns for col in required_cols):
            return  # No normalizer data to plot
        
        epochs = df['epoch'].values
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Normalizer Parameter Convergence - {exp_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Shift convergence (mean values) - ENCODER and DECODER together
        ax = axes[0, 0]
        ax.plot(epochs, df['enc_shift_mean'], label='Encoder shift', linewidth=2, marker='o', markersize=3, alpha=0.8, color='blue')
        ax.plot(epochs, df['dec_shift_mean'], label='Decoder shift', linewidth=2, marker='s', markersize=3, alpha=0.8, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Shift (mean)')
        ax.set_title('Shift Parameter: Encoder vs Decoder (should converge)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Scale convergence (mean values) - ENCODER and DECODER together
        ax = axes[0, 1]
        ax.plot(epochs, df['enc_scale_mean'], label='Encoder scale', linewidth=2, marker='o', markersize=3, alpha=0.8, color='green')
        ax.plot(epochs, df['dec_scale_mean'], label='Decoder scale', linewidth=2, marker='s', markersize=3, alpha=0.8, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Scale (mean)')
        ax.set_title('Scale Parameter: Encoder vs Decoder (should converge)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Difference convergence (should go to zero)
        ax = axes[1, 0]
        if 'shift_diff_mean' in df.columns:
            ax.plot(epochs, df['shift_diff_mean'], label='Shift difference', linewidth=2, color='red', marker='o', markersize=3)
        if 'scale_diff_mean' in df.columns:
            ax.plot(epochs, df['scale_diff_mean'], label='Scale difference', linewidth=2, color='blue', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Absolute Difference')
        ax.set_title('Parameter Difference (should → 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(df) > 1 and df['shift_diff_mean'].max() > 0:
            ax.set_yscale('log')  # Log scale to see convergence better
        
        # Plot 4: Actual latent statistics (mean and std should be ~0 and ~1)
        ax = axes[1, 1]
        
        # Check for latent statistics from standardization loss first (more accurate)
        if 'train_LatentStd_MeanVal' in df.columns:
            ax.plot(epochs, df['train_LatentStd_MeanVal'], label='Latent mean (train)', linewidth=2, color='purple', marker='o', markersize=3)
            if 'val_LatentStd_MeanVal' in df.columns:
                ax.plot(epochs, df['val_LatentStd_MeanVal'], label='Latent mean (val)', linewidth=2, color='purple', marker='o', markersize=3, linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Target: 0')
            
            ax2 = ax.twinx()
            ax2.plot(epochs, df['train_LatentStd_StdVal'], label='Latent std (train)', linewidth=2, color='brown', marker='s', markersize=3)
            if 'val_LatentStd_StdVal' in df.columns:
                ax2.plot(epochs, df['val_LatentStd_StdVal'], label='Latent std (val)', linewidth=2, color='brown', marker='s', markersize=3, linestyle='--', alpha=0.7)
            ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Target: 1')
            ax2.set_ylabel('Latent Std', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')
            ax2.legend(loc='upper right')
        elif 'latent_mean' in df.columns:
            # Fallback to direct measurement if available
            ax.plot(epochs, df['latent_mean'], label='Latent mean', linewidth=2, color='purple', marker='o', markersize=3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Target: 0')
            if 'latent_std' in df.columns:
                ax2 = ax.twinx()
                ax2.plot(epochs, df['latent_std'], label='Latent std', linewidth=2, color='brown', marker='s', markersize=3)
                ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Target: 1')
                ax2.set_ylabel('Latent Std', color='brown')
                ax2.tick_params(axis='y', labelcolor='brown')
                ax2.legend(loc='upper right')
        else:
            # No latent statistics available
            ax.text(0.5, 0.5, 'No latent statistics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Latent Statistics (not available)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Latent Mean', color='purple')
        ax.set_title('Actual Latent Statistics (should be ~0 mean, ~1 std)')
        ax.tick_params(axis='y', labelcolor='purple')
        if 'train_LatentStd_MeanVal' in df.columns or 'latent_mean' in df.columns:
            ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'{exp_name}_normalizer_convergence.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved normalizer convergence plot: {plot_path}")
    
    def _plot_latent_statistics(df, output_dir, exp_name):
        """Plot latent statistics from standardization loss."""
        sns.set_style("darkgrid")
        # Check if latent statistics columns exist (from LatentStandardizationLoss)
        has_train_stats = 'train_LatentStd_MeanVal' in df.columns
        has_val_stats = 'val_LatentStd_MeanVal' in df.columns
        
        if not has_train_stats and not has_val_stats:
            return  # No latent statistics to plot
        
        epochs = df['epoch'].values
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Latent Statistics - {exp_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Latent Mean over time
        ax = axes[0, 0]
        if has_train_stats:
            ax.plot(epochs, df['train_LatentStd_MeanVal'], label='Train mean', linewidth=2, marker='o', markersize=3, color='blue')
        if has_val_stats:
            ax.plot(epochs, df['val_LatentStd_MeanVal'], label='Val mean', linewidth=2, marker='s', markersize=3, color='red', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Target: 0')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Latent Mean')
        ax.set_title('Latent Mean (should → 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Latent Std over time
        ax = axes[0, 1]
        if has_train_stats:
            ax.plot(epochs, df['train_LatentStd_StdVal'], label='Train std', linewidth=2, marker='o', markersize=3, color='green')
        if has_val_stats:
            ax.plot(epochs, df['val_LatentStd_StdVal'], label='Val std', linewidth=2, marker='s', markersize=3, color='orange', linestyle='--')
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Target: 1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Latent Std')
        ax.set_title('Latent Std (should → 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean Loss component
        ax = axes[1, 0]
        if has_train_stats:
            ax.plot(epochs, df['train_LatentStd_Mean'], label='Train mean loss', linewidth=2, marker='o', markersize=3, color='purple')
        if has_val_stats:
            ax.plot(epochs, df['val_LatentStd_Mean'], label='Val mean loss', linewidth=2, marker='s', markersize=3, color='cyan', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Loss Component')
        ax.set_title('Mean Deviation Loss (should → 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(df) > 1 and has_train_stats:
            max_val = df['train_LatentStd_Mean'].max()
            if max_val > 0:
                ax.set_yscale('log')
        
        # Plot 4: Std Loss component
        ax = axes[1, 1]
        if has_train_stats:
            ax.plot(epochs, df['train_LatentStd_Std'], label='Train std loss', linewidth=2, marker='o', markersize=3, color='brown')
        if has_val_stats:
            ax.plot(epochs, df['val_LatentStd_Std'], label='Val std loss', linewidth=2, marker='s', markersize=3, color='pink', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Std Loss Component')
        ax.set_title('Std Deviation Loss (should → 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(df) > 1 and has_train_stats:
            max_val = df['train_LatentStd_Std'].max()
            if max_val > 0:
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'{exp_name}_latent_statistics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved latent statistics plot: {plot_path}")
    
    def _plot_per_class_losses(df, output_dir, exp_name):
        """Plot per-class MSE losses from ClassWeightedMSELoss with class colors."""
        sns.set_style("darkgrid")
        # Find all per-class loss columns (format: train_MSE_rgb_<class_name> or val_MSE_rgb_<class_name>)
        per_class_cols = [col for col in df.columns if 'MSE_rgb_' in col and col != 'train_MSE_rgb' and col != 'val_MSE_rgb']
        
        if not per_class_cols:
            return  # No per-class losses to plot
        
        # Extract class names and split train/val
        train_cols = [col for col in per_class_cols if col.startswith('train_')]
        val_cols = [col for col in per_class_cols if col.startswith('val_')]
        
        if not train_cols and not val_cols:
            return
        
        # Load class colors from rgb_to_class.yaml
        import yaml
        from pathlib import Path
        rgb_config_path = Path(__file__).parent.parent / "models" / "losses" / "rgb_to_class.yaml"
        class_colors = {}
        with open(rgb_config_path, "r") as f:
            config = yaml.safe_load(f)
            for name, entry in config.items():
                rgb = entry["rgb"]
                # Convert RGB [0-255] to matplotlib color [0-1]
                class_colors[name] = tuple(c / 255.0 for c in rgb)
        
        epochs = df['epoch'].values
        
        # Get unique class names (remove train_/val_ prefix and MSE_rgb_ prefix)
        class_names = set()
        for col in train_cols + val_cols:
            # Extract class name: train_MSE_rgb_bed -> bed
            parts = col.split('_')
            if len(parts) >= 4:
                class_name = '_'.join(parts[3:])  # Handle multi-word class names
                class_names.add(class_name)
        
        class_names = sorted(class_names)
        
        # Create a single plot with all classes (overlaid)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Per-Class MSE Losses - {exp_name}', fontsize=16, fontweight='bold')
        
        # Plot each class with its RGB color
        for class_name in class_names:
            train_col = f'train_MSE_rgb_{class_name}'
            val_col = f'val_MSE_rgb_{class_name}'
            
            # Get color for this class (default to gray if not found)
            color = class_colors.get(class_name, (0.5, 0.5, 0.5))
            
            # Use slightly darker color for validation
            val_color = tuple(max(0, c * 0.7) for c in color)
            
            label_name = class_name.replace("_", " ").title()
            
            if train_col in df.columns:
                ax.plot(epochs, df[train_col], label=f'{label_name} (train)', 
                       linewidth=2, marker='o', markersize=2, color=color, alpha=0.8)
            if val_col in df.columns:
                ax.plot(epochs, df[val_col], label=f'{label_name} (val)', 
                       linewidth=2, marker='s', markersize=2, color=val_color, 
                       linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'{exp_name}_per_class_losses.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved per-class losses plot: {plot_path}")
    
    def _plot_kld_loss(df, output_dir, exp_name):
        """Plot KLD (KL Divergence) loss from KLDLoss."""
        sns.set_style("darkgrid")
        # Check if KLD columns exist
        has_train_kld = 'train_KLD' in df.columns
        has_val_kld = 'val_KLD' in df.columns
        
        if not has_train_kld and not has_val_kld:
            return  # No KLD loss to plot
        
        epochs = df['epoch'].values
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'KL Divergence Loss - {exp_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: KLD Loss over time
        ax = axes[0]
        if has_train_kld:
            ax.plot(epochs, df['train_KLD'], label='Train KLD', linewidth=2, marker='o', markersize=3, color='blue')
        if has_val_kld:
            ax.plot(epochs, df['val_KLD'], label='Val KLD', linewidth=2, marker='s', markersize=3, color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KLD Loss')
        ax.set_title('KL Divergence Loss (should decrease or stabilize)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(df) > 1 and has_train_kld:
            max_val = df['train_KLD'].max()
            min_val = df['train_KLD'].min()
            if max_val > 0 and min_val > 0 and max_val / min_val > 10:
                ax.set_yscale('log')
        
        # Plot 2: KLD Loss trend (smoothed)
        ax = axes[1]
        if has_train_kld:
            # Compute moving average for smoother trend
            window = min(5, len(df) // 4) if len(df) > 4 else 1
            if window > 1:
                train_smooth = df['train_KLD'].rolling(window=window, center=True).mean()
                ax.plot(epochs, df['train_KLD'], label='Train KLD (raw)', linewidth=1, alpha=0.3, color='blue')
                ax.plot(epochs, train_smooth, label=f'Train KLD (MA-{window})', linewidth=2, marker='o', markersize=3, color='blue')
            else:
                ax.plot(epochs, df['train_KLD'], label='Train KLD', linewidth=2, marker='o', markersize=3, color='blue')
        if has_val_kld:
            window = min(5, len(df) // 4) if len(df) > 4 else 1
            if window > 1:
                val_smooth = df['val_KLD'].rolling(window=window, center=True).mean()
                ax.plot(epochs, df['val_KLD'], label='Val KLD (raw)', linewidth=1, alpha=0.3, color='red', linestyle='--')
                ax.plot(epochs, val_smooth, label=f'Val KLD (MA-{window})', linewidth=2, marker='s', markersize=3, color='red', linestyle='--')
            else:
                ax.plot(epochs, df['val_KLD'], label='Val KLD', linewidth=2, marker='s', markersize=3, color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KLD Loss')
        ax.set_title('KLD Loss Trend (smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(df) > 1 and has_train_kld:
            max_val = df['train_KLD'].max()
            min_val = df['train_KLD'].min()
            if max_val > 0 and min_val > 0 and max_val / min_val > 10:
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'{exp_name}_kld_loss.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved KLD loss plot: {plot_path}")
    
    def _plot_loss_curves(df, output_dir, exp_name):
        """Plot main training and validation loss curves."""
        sns.set_style("darkgrid")
        # Check if loss columns exist
        has_train_loss = 'train_loss' in df.columns
        has_val_loss = 'val_loss' in df.columns
        
        if not has_train_loss:
            return  # No loss data to plot
        
        epochs = df['epoch'].values
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Training and Validation Loss - {exp_name}', fontsize=16, fontweight='bold')
        
        # Plot training loss
        if has_train_loss:
            ax.plot(epochs, df['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3, color='blue', alpha=0.8)
        
        # Plot validation loss if available
        if has_val_loss:
            ax.plot(epochs, df['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3, color='red', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if loss values span multiple orders of magnitude
        if has_train_loss and len(df) > 1:
            max_loss = df['train_loss'].max()
            min_loss = df['train_loss'].min()
            if max_loss > 0 and min_loss > 0 and max_loss / min_loss > 10:
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'{exp_name}_loss_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved loss curves plot: {plot_path}")
    
    # Check if model is VAE (variational encoder) to enable latent statistics collection
    is_vae = hasattr(model.encoder, 'variational') and model.encoder.variational
    
    for epoch in range(start_epoch, end_epoch):
        # Training
        avg_loss, avg_logs = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch + 1, use_amp=use_amp, collect_latents=is_vae)
        
        print(f"Epoch {epoch + 1}/{end_epoch} - Train Loss: {avg_loss:.6f}")
        for k, v in avg_logs.items():
            if isinstance(v, str):
                # Handle string values (e.g., JSON-encoded per-channel stats)
                print(f"  Train {k}: {v}")
            else:
                print(f"  Train {k}: {v:.6f}")
        
        # Record training history
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            **{f"train_{k}": (v if isinstance(v, str) else float(v)) for k, v in avg_logs.items()}
        }
        
        # Extract normalizer statistics if available (for backward compatibility with old models)
        # Note: This is only relevant if normalization layers are still present
        norm_stats = extract_normalizer_stats(model)
        if norm_stats:
            epoch_log.update(norm_stats)
            # Print normalizer convergence info
            if 'shift_diff_mean' in norm_stats:
                print(f"  Normalizer convergence:")
                print(f"    Shift diff: {norm_stats['shift_diff_mean']:.6f} (enc: {norm_stats.get('enc_shift_mean', 0):.6f}, dec: {norm_stats.get('dec_shift_mean', 0):.6f})")
                print(f"    Scale diff: {norm_stats['scale_diff_mean']:.6f} (enc: {norm_stats.get('enc_scale_mean', 0):.6f}, dec: {norm_stats.get('dec_scale_mean', 0):.6f})")
        
        # Evaluation (run every epoch if validation set exists)
        if val_loader:
            val_loss, val_logs = eval_epoch(model, val_loader, loss_fn, device, use_amp=use_amp, collect_latents=is_vae)
            print(f"  Val Loss: {val_loss:.6f}")
            for k, v in val_logs.items():
                if isinstance(v, str):
                    # Handle string values (e.g., JSON-encoded per-channel stats)
                    print(f"  Val {k}: {v}")
                else:
                    print(f"  Val {k}: {v:.6f}")
            epoch_log["val_loss"] = float(val_loss)
            epoch_log.update({f"val_{k}": (v if isinstance(v, str) else float(v)) for k, v in val_logs.items()})
            
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
            # Include experiment name in filename to avoid overwrites
            if phase_dir:
                save_samples(model, loader_to_use, device, phase_dir, epoch + 1, 
                           sample_batch_size=8, exp_name=exp_name)
        
        training_history.append(epoch_log)
        
        # Save metrics CSV (overwrite with all epochs so far)
        save_metrics_csv(training_history, metrics_csv_path, phase_metrics_path)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(training_history)
        
        # Plot main loss curves (always plot if training loss exists)
        _plot_loss_curves(df, output_dir, exp_name)
        
        # Plot normalizer convergence if available (overwrites same file each time)
        if norm_stats:
            _plot_normalizer_convergence(df, output_dir, exp_name)
        
        # Plot latent statistics if available (from LatentStandardizationLoss)
        _plot_latent_statistics(df, output_dir, exp_name)
        
        # Plot per-class losses if available (from ClassWeightedMSELoss)
        _plot_per_class_losses(df, output_dir, exp_name)
        
        # Plot KLD loss if available (from KLDLoss)
        _plot_kld_loss(df, output_dir, exp_name)
        
        # Save checkpoint at specified interval
        should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == end_epoch
        if should_save:
            checkpoint_path = output_dir / f"{exp_name}_checkpoint_epoch_{epoch + 1:03d}.pt"
            # Save checkpoint with config inside (via save_checkpoint method)
            model.save_checkpoint(checkpoint_path, include_config=True)
            checkpoint_files.append(checkpoint_path)
        
        # Always save latest checkpoint (for resume - includes optimizer state)
        latest_path = output_dir / f"{exp_name}_checkpoint_latest.pt"
        model.save_checkpoint(latest_path, include_config=True,
                            epoch=epoch + 1, best_val_loss=best_val_loss,
                            optimizer_state=optimizer.state_dict(),
                            training_history=training_history)
        
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

