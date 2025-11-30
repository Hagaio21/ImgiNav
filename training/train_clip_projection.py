#!/usr/bin/env python3
"""
Standalone training script for CLIP projection (Experiment 3).

This script trains a CLIP projection to map:
- Textured and segmented POV embeddings
- Graph embeddings
to segmented layout latents.

The layout latents are pre-encoded using a trained VAE, and only the
CLIP projection is trained.

Usage:
    python train_clip_projection.py \
        --config experiments/diffusion/clip/experiment3_clip_projection.yaml \
        --layout-vae-checkpoint outputs/autoencoders/v2/vae_seg_256_clip/checkpoints/vae_seg_256_clip_checkpoint_best.pt \
        --layout-latents-dir dataset_v2/layouts/latents/seg_vae_seg_256_clip
"""

import torch
import torch.nn.functional as F
import yaml
import sys
import math
import warnings
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

# Suppress torchvision.io extension warning
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.components.projections import CLIPProjections
from models.datasets.datasets import ManifestDataset
from training.engine import Trainer
from training.utils import (
    set_deterministic,
    load_config,
    build_optimizer,
    build_scheduler,
    get_device,
    to_device,
    move_batch_to_device,
    create_grad_scaler,
    save_metrics_csv,
)
from models.losses.clip_loss import CLIPLoss
from models.losses.base_loss import LOSS_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_clip_projection_dataset(
    manifest: Path,
    dataset_root: Path,
    layout_latents_dir: Path,
    pov_embedding_col: str = "pov_embedding_path",
    graph_embedding_col: str = "graph_embedding_path",
    layout_latent_col: str = "layout_latent_path",
):
    """
    Create a dataset for CLIP projection training.
    
    The dataset returns:
    - layout_latent: Pre-encoded layout latent [B, C, H, W] or [B, D]
    - pov_emb: POV embedding [B, pov_dim]
    - graph_emb: Graph embedding [B, text_dim]
    
    Args:
        manifest: Path to manifest CSV
        dataset_root: Root directory of dataset
        layout_latents_dir: Directory containing pre-encoded layout latents
        pov_embedding_col: Column name for POV embeddings
        graph_embedding_col: Column name for graph embeddings
        layout_latent_col: Column name for layout latents (will be added)
    
    Returns:
        ManifestDataset instance
    """
    # Load manifest
    df = pd.read_csv(manifest)
    
    # Add layout latent paths
    def get_latent_path(row):
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        if not scene_id or not room_id:
            return ""
        
        latent_path = layout_latents_dir / f"{scene_id}_{room_id}_layout.pt"
        if latent_path.exists():
            return str(latent_path.relative_to(dataset_root))
        return ""
    
    df[layout_latent_col] = df.apply(get_latent_path, axis=1)
    
    # Filter to rows with all required embeddings
    has_all = (
        df[pov_embedding_col].notna() & (df[pov_embedding_col] != "") &
        df[graph_embedding_col].notna() & (df[graph_embedding_col] != "") &
        df[layout_latent_col].notna() & (df[layout_latent_col] != "")
    )
    df_filtered = df[has_all].copy()
    
    # Save filtered manifest temporarily
    temp_manifest = layout_latents_dir / "temp_training_manifest.csv"
    df_filtered.to_csv(temp_manifest, index=False)
    
    # Create dataset
    dataset = ManifestDataset(
        manifest=temp_manifest,
        outputs={
            "layout_latent": layout_latent_col,
            "pov_emb": pov_embedding_col,
            "graph_emb": graph_embedding_col,
        },
        transform=None,  # No transform needed - we're loading pre-computed embeddings
    )
    
    return dataset, temp_manifest


def clip_projection_step_fn(model, batch, batch_idx, loss_fn, trainer):
    """
    Step function for CLIP projection training.
    
    Args:
        model: CLIPProjections instance
        batch: Batch with layout_latent, pov_emb, graph_emb
        batch_idx: Batch index
        loss_fn: CLIPLoss instance
        trainer: Trainer instance
    
    Returns:
        (loss, logs, outputs_dict)
    """
    # Get inputs
    layout_latent = batch.get("layout_latent")  # [B, C, H, W] or [B, D]
    pov_emb = batch.get("pov_emb")  # [B, pov_dim]
    graph_emb = batch.get("graph_emb")  # [B, text_dim]
    
    # Ensure tensors are on correct device
    device = next(model.parameters()).device
    layout_latent = layout_latent.to(device)
    pov_emb = pov_emb.to(device)
    graph_emb = graph_emb.to(device)
    
    # Prepare for loss computation
    # The loss expects:
    # - preds: {"latent_features": layout_latent} - original latent features (not projected)
    # - targets: {"text_emb": graph_emb, "pov_emb": pov_emb}
    # The loss function will call model.projections() internally (via loss_fn.projections),
    # which will project the latent_features and combine text_emb/pov_emb
    preds = {"latent_features": layout_latent}
    targets = {
        "text_emb": graph_emb,
        "pov_emb": pov_emb,
    }
    
    # Compute loss (this will call model.projections() internally)
    if trainer.use_amp and device.type == "cuda":
        with torch.amp.autocast(device_type='cuda'):
            loss, logs = loss_fn(preds, targets)
    else:
        loss, logs = loss_fn(preds, targets)
    
    # For logging, we can compute projections separately (detached)
    with torch.no_grad():
        latent_proj, combined_emb = model(
            layout_latent,
            graph_emb,  # text_emb
            pov_emb,    # pov_emb
            combine_method="average"
        )
    
    # Return outputs for logging
    outputs_dict = {
        "latent_proj": latent_proj.detach(),
        "combined_emb": combined_emb.detach(),
    }
    
    return loss, logs, outputs_dict


def save_checkpoint(model, checkpoint_dir, exp_name, epoch, is_best=False, val_loss=None):
    """Save CLIP projection checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
    }
    if val_loss is not None:
        checkpoint_data['val_loss'] = val_loss
    
    if is_best:
        checkpoint_path = checkpoint_dir / f"{exp_name}_clip_projection_best.pt"
    else:
        checkpoint_path = checkpoint_dir / f"{exp_name}_clip_projection_epoch_{epoch:03d}.pt"
    
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train CLIP projection standalone")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--layout-vae-checkpoint", type=Path, default=None,
                       help="Path to layout VAE checkpoint (for latent dimension inference)")
    parser.add_argument("--layout-latents-dir", type=Path, required=True,
                       help="Directory containing pre-encoded layout latents")
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    
    # Set deterministic behavior if seed is provided
    training_seed = config.get("training", {}).get("seed", None)
    if training_seed is not None:
        set_deterministic(training_seed)
    
    # Get device
    device = get_device(config)
    device_obj = to_device(device)
    
    # Get output directory
    output_dir = config.get("experiment", {}).get("save_path")
    if output_dir is None:
        output_dir = Path("outputs") / "clip_projections" / exp_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    dataset_cfg = config.get("dataset", {})
    manifest = Path(dataset_cfg.get("manifest"))
    dataset_root = Path(dataset_cfg.get("dataset_root", manifest.parent.parent))
    
    # Infer latent dimension from a sample latent
    sample_latent_path = next(args.layout_latents_dir.glob("*.pt"), None)
    if sample_latent_path is None:
        raise ValueError(f"No latent files found in {args.layout_latents_dir}")
    
    sample_latent = torch.load(sample_latent_path, map_location="cpu", weights_only=True)
    if sample_latent.dim() == 4:
        # Spatial latent [C, H, W]
        latent_dim = sample_latent.shape[0]
    elif sample_latent.dim() == 1:
        # Global latent [D]
        latent_dim = sample_latent.shape[0]
    else:
        raise ValueError(f"Unexpected latent shape: {sample_latent.shape}")
    
    logger.info(f"Inferred latent dimension: {latent_dim} (from sample shape: {sample_latent.shape})")
    
    # Build CLIP projection
    clip_proj_cfg = config.get("clip_projection", {})
    clip_proj_cfg["latent_dim"] = latent_dim  # Set inferred latent dimension
    
    clip_projection = CLIPProjections(**clip_proj_cfg)
    clip_projection = clip_projection.to(device_obj)
    
    # Check for checkpoint to resume
    checkpoint_to_resume = None
    if args.checkpoint:
        checkpoint_to_resume = Path(args.checkpoint)
    else:
        checkpoint_dir = output_dir / "checkpoints"
        latest_checkpoint = checkpoint_dir / f"{exp_name}_clip_projection_latest.pt"
        if latest_checkpoint.exists():
            checkpoint_to_resume = latest_checkpoint
    
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    
    if checkpoint_to_resume and checkpoint_to_resume.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_to_resume}")
        payload = torch.load(checkpoint_to_resume, map_location=device_obj)
        clip_projection.load_state_dict(payload.get("state_dict", payload))
        start_epoch = payload.get("epoch", 0)
        best_val_loss = payload.get("val_loss", float("inf"))
        training_history = payload.get("training_history", [])
        logger.info(f"Resuming from epoch {start_epoch + 1}, best val loss: {best_val_loss:.6f}")
    
    # Create dataset
    dataset, temp_manifest = create_clip_projection_dataset(
        manifest=manifest,
        dataset_root=dataset_root,
        layout_latents_dir=args.layout_latents_dir,
    )
    
    # Split dataset
    train_split = config.get("training", {}).get("train_split", 0.8)
    split_seed = config.get("training", {}).get("split_seed", 42)
    
    if train_split < 1.0:
        val_ratio = 1.0 - train_split
        train_dataset, val_dataset, _ = dataset.split(
            train_ratio=train_split,
            val_ratio=val_ratio,
            test_ratio=0.0,
            seed=split_seed
        )
    else:
        train_dataset = dataset
        val_dataset = None
    
    # Create dataloaders
    train_loader = train_dataset.make_dataloader(
        batch_size=config.get("training", {}).get("batch_size", 32),
        shuffle=True,
        num_workers=config.get("training", {}).get("num_workers", 4),
    )
    
    val_loader = None
    if val_dataset:
        val_loader = val_dataset.make_dataloader(
            batch_size=config.get("validation", {}).get("batch_size", config.get("training", {}).get("batch_size", 32)),
            shuffle=False,
            num_workers=config.get("training", {}).get("num_workers", 4),
        )
    
    # Build loss function
    loss_cfg = config.get("loss", {})
    clip_loss = CLIPLoss(**loss_cfg)
    clip_loss.set_projections(clip_projection)
    
    # Build optimizer (only for CLIP projection parameters)
    optimizer = build_optimizer(clip_projection, config)
    
    # Build scheduler
    scheduler = build_scheduler(optimizer, config)
    
    # Enable mixed precision
    use_amp = config.get("training", {}).get("use_amp", True)
    
    # Create Trainer
    trainer = Trainer(
        model=clip_projection,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        gradient_accumulation_steps=1,
        max_grad_norm=None,
    )
    
    # Training configuration
    epochs_to_train = config.get("training", {}).get("epochs", 150)
    end_epoch = start_epoch + epochs_to_train
    save_interval = config.get("training", {}).get("save_interval", 1)
    eval_interval = config.get("training", {}).get("eval_interval", 1)
    
    # Early stopping
    early_stopping_patience = config.get("training", {}).get("early_stopping_patience", None)
    early_stopping_min_delta = config.get("training", {}).get("early_stopping_min_delta", 0.0)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_csv_path = output_dir / f"{exp_name}_metrics.csv"
    epochs_without_improvement = 0
    
    # Training loop
    for epoch in range(start_epoch, end_epoch):
        # Training
        avg_loss, avg_logs = trainer.train_epoch(
            dataloader=train_loader,
            loss_fn=clip_loss,
            epoch=epoch + 1,
            step_fn=clip_projection_step_fn,
        )
        
        logger.info(f"Epoch {epoch + 1}/{end_epoch} - Train Loss: {avg_loss:.6f}")
        
        # Record training history
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            **{f"train_{k}": (v if isinstance(v, str) else float(v)) for k, v in avg_logs.items()}
        }
        
        is_best_this_epoch = False
        
        # Evaluation
        if val_loader:
            val_loss, val_logs = trainer.eval_epoch(
                dataloader=val_loader,
                loss_fn=clip_loss,
                step_fn=clip_projection_step_fn,
            )
            
            logger.info(f"  Val Loss: {val_loss:.6f}")
            epoch_log["val_loss"] = float(val_loss)
            epoch_log.update({f"val_{k}": (v if isinstance(v, str) else float(v)) for k, v in val_logs.items()})
            
            # Check for improvement
            improvement = best_val_loss - val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                is_best_this_epoch = True
                
                # Save best checkpoint
                save_checkpoint(
                    clip_projection, checkpoint_dir, exp_name, epoch + 1,
                    is_best=True, val_loss=val_loss
                )
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        training_history.append(epoch_log)
        
        # Save metrics CSV
        save_metrics_csv(metrics_csv_path, training_history)
        
        # Save periodic checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == end_epoch:
            save_checkpoint(
                clip_projection, checkpoint_dir, exp_name, epoch + 1,
                is_best=False, val_loss=val_loss if val_loader else None
            )
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / f"{exp_name}_clip_projection_latest.pt"
        torch.save({
            'state_dict': clip_projection.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'training_history': training_history,
        }, latest_path)
    
    # Clean up temp manifest
    if temp_manifest.exists():
        temp_manifest.unlink()
    
    logger.info(f"Training complete. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

