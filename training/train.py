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
import json

# Suppress torchvision.io extension warning (we use PIL, not torchvision.io)
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.engine import Trainer
from training.utils import (
    set_deterministic,
    load_config,
    build_model,
    build_dataset,
    build_loss,
    build_optimizer,
    build_scheduler,
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


# Step functions for Trainer
def ae_step_fn(model, batch, batch_idx, loss_fn, trainer):
    """Step function for autoencoder training - computes loss only (Trainer handles backward/step)."""
    from models.components.dataflow import DataFlow
    
    # Wrap batch in DataFlow for tracking
    batch_dataflow = DataFlow(batch, source_component="Dataset")
    
    # Forward pass with AMP if enabled
    if trainer.use_amp:
        with torch.amp.autocast('cuda'):
            outputs = model(batch_dataflow.get("rgb", batch["rgb"]))
            # Loss functions work with DataFlow since it's dict-like
            loss, logs = loss_fn(outputs, batch_dataflow)
    else:
        outputs = model(batch_dataflow.get("rgb", batch["rgb"]))
        # Loss functions work with DataFlow since it's dict-like
        loss, logs = loss_fn(outputs, batch_dataflow)
    
    # Return outputs for latent collection (convert DataFlow to dict if needed)
    if isinstance(outputs, DataFlow):
        outputs_dict = outputs.to_dict()
    else:
        outputs_dict = outputs
    return loss, logs, outputs_dict


def ae_eval_step_fn(model, batch, batch_idx, loss_fn, trainer):
    """Step function for autoencoder evaluation - computes loss only."""
    from models.components.dataflow import DataFlow
    
    # Wrap batch in DataFlow for tracking
    batch_dataflow = DataFlow(batch, source_component="Dataset")
    
    # Forward pass with AMP if enabled
    if trainer.use_amp:
        with torch.amp.autocast('cuda'):
            outputs = model(batch_dataflow.get("rgb", batch["rgb"]))
            # Loss functions work with DataFlow since it's dict-like
            loss, logs = loss_fn(outputs, batch_dataflow)
    else:
        outputs = model(batch_dataflow.get("rgb", batch["rgb"]))
        # Loss functions work with DataFlow since it's dict-like
        loss, logs = loss_fn(outputs, batch_dataflow)
    
    # Return outputs for latent collection (convert DataFlow to dict if needed)
    if isinstance(outputs, DataFlow):
        outputs_dict = outputs.to_dict()
    else:
        outputs_dict = outputs
    return loss, logs, outputs_dict


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
        
        input_rgb = (input_rgb + 1) / 2.0
        pred_rgb = (pred_rgb + 1) / 2.0
        
        # Resize if needed
        if input_rgb.shape[-1] != target_size:
            input_rgb = F.interpolate(input_rgb, size=(target_size, target_size), mode='bilinear', align_corners=False)
            pred_rgb = F.interpolate(pred_rgb, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        orig_grid = make_grid(input_rgb, nrow=grid_n, padding=2, normalize=False)
        recon_grid = make_grid(pred_rgb, nrow=grid_n, padding=2, normalize=False)
        combined_grid = torch.cat([orig_grid, recon_grid], dim=2)
        if exp_name:
            grid_path = samples_dir / f"{exp_name}_epoch_{epoch:03d}_comparison.png"
        else:
            grid_path = samples_dir / f"epoch_{epoch:03d}_comparison.png"
        save_image(combined_grid, grid_path, normalize=False)


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder from experiment config")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--checkpoint", type=Path, default=None, 
                       help="Path to checkpoint to resume from (overrides automatic latest checkpoint detection)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    
    # Set deterministic behavior if seed is provided
    training_seed = config.get("training", {}).get("seed", None)
    if training_seed is not None:
        set_deterministic(training_seed)
    
    # Get device from config or default
    device = get_device(config)
    
    # Get output directory from config
    output_dir = config.get("experiment", {}).get("save_path")
    if output_dir is None:
        # Default: outputs/experiment_name
        output_dir = Path("outputs") / exp_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Check for checkpoint to resume from
    # Priority: 1) --checkpoint argument, 2) latest checkpoint in output_dir
    checkpoint_to_resume = None
    if args.checkpoint:
        checkpoint_to_resume = Path(args.checkpoint)
        if not checkpoint_to_resume.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_to_resume}")
    else:
        # Check in checkpoints folder first, then fallback to root (for backward compatibility)
        checkpoint_dir_temp = output_dir / "checkpoints"
        latest_checkpoint = checkpoint_dir_temp / f"{exp_name}_checkpoint_latest.pt"
        if not latest_checkpoint.exists():
            latest_checkpoint = output_dir / f"{exp_name}_checkpoint_latest.pt"
        if latest_checkpoint.exists():
            checkpoint_to_resume = latest_checkpoint
    
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    extra_state = {}
    
    # CSV file path for metrics (defined early so we can load from it if needed)
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    
    device_obj = to_device(device)
    
    should_resume = checkpoint_to_resume is not None
    if should_resume:
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
        if not training_history:
            from training.utils import load_training_history_from_csv
            training_history = load_training_history_from_csv(metrics_csv_path, start_epoch)
    else:
        # Build components
        model = build_model(config)
        model = model.to(device_obj)
    
    # Enable cudnn benchmark for faster convolutions (optimizes for input sizes)
    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    dataset = build_dataset(config)
    
    # Get weighted sampling config
    training_cfg = config.get("training", {})
    use_precomputed_weights = training_cfg.get("use_precomputed_weights", False)
    precomputed_weight_column = training_cfg.get("precomputed_weight_column", "sample_weight")
    max_weight = training_cfg.get("max_weight", None)
    
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
        # Use full dataset for training if validation is explicitly provided
        train_dataset = dataset
    else:
        # Auto-split dataset using train_split from config
        train_split = config["training"].get("train_split", 0.8)
        split_seed = config["training"].get("split_seed", 42)
        
        if train_split < 1.0:
            # Calculate val_ratio from train_split (remaining goes to val, test=0)
            val_ratio = 1.0 - train_split
            train_dataset, val_dataset, _ = dataset.split(train_ratio=train_split, val_ratio=val_ratio, test_ratio=0.0, seed=split_seed)
            val_loader = val_dataset.make_dataloader(
                batch_size=config["validation"].get("batch_size", config["training"]["batch_size"]) if "validation" in config else config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["training"].get("num_workers", 4)
            )
        else:
            train_dataset = dataset
    
    # Create train dataloader with optional precomputed weights
    train_loader = train_dataset.make_dataloader(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True) if not use_precomputed_weights else False,
        num_workers=config["training"].get("num_workers", 4),
        use_precomputed_weights=use_precomputed_weights,
        precomputed_weight_column=precomputed_weight_column,
        max_weight=max_weight,
    )
    
    loss_fn = build_loss(config)
    
    from models.losses.base_loss import LOSS_REGISTRY
    CompositeLossClass = LOSS_REGISTRY.get("CompositeLoss")
    CLIPLossClass = LOSS_REGISTRY.get("CLIPLoss")
    
    if not hasattr(model, 'clip_projection') or model.clip_projection is None:
        raise RuntimeError("Model does not have clip_projection! Check autoencoder config has clip_projection section.")
    
    if not isinstance(loss_fn, CompositeLossClass):
        raise RuntimeError("Loss function is not CompositeLoss! Cannot connect CLIP projections.")
    
    clip_loss_found = False
    for sub_loss in loss_fn.losses:
        if isinstance(sub_loss, CLIPLossClass):
            clip_loss_found = True
            sub_loss.set_projections(model.clip_projection)
            if sub_loss.projections is None:
                raise RuntimeError("CLIP loss projections are None after set_projections!")
            if sub_loss.projections is not model.clip_projection:
                raise RuntimeError("CLIP loss projections are not the same instance as model.clip_projection!")
            proj_params = list(model.clip_projection.parameters())
            trainable_proj_params = [p for p in proj_params if p.requires_grad]
    
    if not clip_loss_found:
        raise RuntimeError("CLIP loss not found in CompositeLoss! Check loss config has CLIPLoss component.")
    
    optimizer = build_optimizer(model, config)
    if should_resume:
        optimizer_state = extra_state.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
    
    # Build scheduler if configured
    scheduler = build_scheduler(optimizer, config.get("training", {}).get("scheduler", None))
    if should_resume and scheduler:
        scheduler_state = extra_state.get("scheduler_state")
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    
    # Enable mixed precision training by default (can be disabled in config)
    use_amp = config.get("training", {}).get("use_amp", True)  # Default to True for speedup
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        gradient_accumulation_steps=1,  # AE doesn't use gradient accumulation
        max_grad_norm=None,  # Can be added to config if needed
    )
    
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
    
    # Create checkpoints directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_files = []
    epochs_without_improvement = 0
    
    from training.plotting_utils import plot_loss_curves
    
    # Check if model is VAE (variational encoder) to enable latent statistics collection
    is_vae = hasattr(model.encoder, 'variational') and model.encoder.variational
    
    # Create latent collection function if needed
    all_latents_train = [] if is_vae else None
    all_latents_val = [] if is_vae else None
    
    def make_collect_fn(all_latents_list):
        """Create a collect function for latent statistics."""
        if all_latents_list is None:
            return None
        def collect_fn(model, batch, outputs):
            if "mu" in outputs:
                all_latents_list.append(outputs["mu"].detach().cpu())
            elif "latent" in outputs:
                all_latents_list.append(outputs["latent"].detach().cpu())
        return collect_fn
    
    for epoch in range(start_epoch, end_epoch):
        # Reset latent collections for this epoch
        if is_vae:
            all_latents_train.clear()
            all_latents_val.clear()
        
        # Training
        avg_loss, avg_logs = trainer.train_epoch(
            dataloader=train_loader,
            loss_fn=loss_fn,
            epoch=epoch + 1,
            step_fn=ae_step_fn,
            collect_fn=make_collect_fn(all_latents_train)
        )
        
        # Compute latent statistics if collected
        if is_vae and all_latents_train and len(all_latents_train) > 0:
            latent_stats = compute_latent_statistics(all_latents_train)
            avg_logs.update(latent_stats)
        
        print(f"Epoch {epoch + 1}/{end_epoch} - Train Loss: {avg_loss:.6f}")
        
        # Record training history
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            **{f"train_{k}": (v if isinstance(v, str) else float(v)) for k, v in avg_logs.items()}
        }
        
        # Track if this epoch is the best (for checkpoint saving)
        is_best_this_epoch = False
        
        # Evaluation (run every epoch if validation set exists)
        if val_loader:
            val_loss, val_logs = trainer.eval_epoch(
                dataloader=val_loader,
                loss_fn=loss_fn,
                step_fn=ae_eval_step_fn,
                collect_fn=make_collect_fn(all_latents_val)
            )
            
            # Compute latent statistics if collected
            if is_vae and all_latents_val and len(all_latents_val) > 0:
                latent_stats = compute_latent_statistics(all_latents_val)
                val_logs.update(latent_stats)
            print(f"  Val Loss: {val_loss:.6f}")
            epoch_log["val_loss"] = float(val_loss)
            epoch_log.update({f"val_{k}": (v if isinstance(v, str) else float(v)) for k, v in val_logs.items()})
            
            # Track and save best validation loss immediately
            improvement = best_val_loss - val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0  # Reset counter on improvement
                is_best_this_epoch = True
                
                # Save best checkpoint immediately using Trainer (always updated when best is found)
                trainer.save_training_checkpoint(
                    output_dir=output_dir,
                    exp_name=exp_name,
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                    training_history=training_history,
                    is_best=True
                )
                
                # Save VAE metadata with latent statistics if available
                if is_vae and val_logs:
                    # Extract latent statistics from validation logs
                    latent_stats = {k: v for k, v in val_logs.items() if k.startswith("LatentStats_")}
                    if latent_stats:
                        from training.utils import save_vae_metadata
                        save_vae_metadata(output_dir, exp_name, latent_stats)
            else:
                epochs_without_improvement += 1
            
            # Early stopping check
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                
                # Restore best checkpoint if requested
                if early_stopping_restore_best:
                    best_path = checkpoint_dir / f"{exp_name}_checkpoint_best.pt"
                    if not best_path.exists():
                        # Fallback to root for backward compatibility
                        best_path = output_dir / f"{exp_name}_checkpoint_best.pt"
                    if best_path.exists():
                        model = Autoencoder.load_checkpoint(best_path, map_location=device_obj)
                        model = model.to(device_obj)
                
                # Break out of training loop
                break
        
        # Save samples at specified interval (use validation set if available, else training set)
        if (epoch + 1) % sample_interval == 0:
            loader_to_use = val_loader if val_loader else train_loader
            save_samples(model, loader_to_use, device, output_dir, epoch + 1, sample_batch_size=32)
        
        training_history.append(epoch_log)
        
        # Save metrics CSV (overwrite with all epochs so far)
        trainer.save_metrics_csv(metrics_csv_path, training_history)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(training_history)
        
        # Plot loss curves (simple train/val loss only)
        plot_loss_curves(df, output_dir, exp_name=exp_name)
        
        # Determine if this is the best checkpoint (for latest checkpoint saving)
        # Best checkpoint is saved immediately when found, so latest checkpoint is_best=False
        # unless this epoch just became the best (which was already saved above)
        # Note: is_best_this_epoch is set in the validation block above
        
        # Save checkpoint at specified interval (periodic checkpoints)
        should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == end_epoch
        if should_save:
            checkpoint_path = checkpoint_dir / f"{exp_name}_checkpoint_epoch_{epoch + 1:03d}.pt"
            # Save checkpoint with config inside (via save_checkpoint method)
            model.save_checkpoint(checkpoint_path, include_config=True)
            checkpoint_files.append(checkpoint_path)
        
        # Always save latest checkpoint (for resume - includes optimizer state)
        # Note: best checkpoint is already saved above when found, so is_best=False here
        # (is_best_this_epoch is only True if validation ran and this epoch is best)
        trainer.save_training_checkpoint(
            output_dir=output_dir,
            exp_name=exp_name,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            training_history=training_history,
            is_best=is_best_this_epoch if val_loader else False
        )
        
        # Clean up old checkpoints if keeping only N
        if keep_checkpoints and len(checkpoint_files) > keep_checkpoints:
            # Remove oldest checkpoint files
            for old_checkpoint in checkpoint_files[:-keep_checkpoints]:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            checkpoint_files = checkpoint_files[-keep_checkpoints:]
    
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    
    # Save final VAE metadata if not already saved (use final validation stats)
    if is_vae and val_loader:
        final_latents = []
        final_val_loss, final_val_logs = trainer.eval_epoch(
            dataloader=val_loader,
            loss_fn=loss_fn,
            step_fn=ae_eval_step_fn,
            collect_fn=make_collect_fn(final_latents)
        )
        # Compute latent statistics if collected
        if final_latents and len(final_latents) > 0:
            latent_stats = compute_latent_statistics(final_latents)
            final_val_logs.update(latent_stats)
        
        latent_stats = {k: v for k, v in final_val_logs.items() if k.startswith("LatentStats_")}
        if latent_stats:
            # Check if metadata already exists (from best checkpoint save)
            metadata_path = output_dir / f"{exp_name}_metadata.json"
            if not metadata_path.exists():
                from training.utils import save_vae_metadata
                save_vae_metadata(output_dir, exp_name, latent_stats)


if __name__ == "__main__":
    main()