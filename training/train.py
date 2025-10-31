#!/usr/bin/env python3
"""
Training script for autoencoder experiments.
Loads experiment config, builds model, dataset, loss, and runs training.
"""

import torch
import yaml
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

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
    dataset = ManifestDataset.from_config(ds_cfg)
    return dataset


def build_loss(config):
    """Build loss function from config."""
    loss_cfg = config["training"]["loss"]
    loss_fn = LOSS_REGISTRY[loss_cfg["type"]].from_config(loss_cfg)
    return loss_fn


def build_optimizer(model, config):
    """Build optimizer from config."""
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    optimizer_type = config["training"].get("optimizer", "AdamW").lower()
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
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


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder from experiment config")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to train on (default: cuda if available, else cpu)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for checkpoints and logs")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    print(f"Experiment: {exp_name}")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("outputs") / exp_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Build components
    print("Building model...")
    model = build_model(config)
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Building dataset...")
    dataset = build_dataset(config)
    train_loader = dataset.make_dataloader(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True),
        num_workers=config["training"].get("num_workers", 4)
    )
    print(f"Dataset size: {len(dataset)}, Batches: {len(train_loader)}")
    
    print("Building loss function...")
    loss_fn = build_loss(config)
    
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and args.resume.exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    num_epochs = config["training"]["epochs"]
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        avg_loss, avg_logs = train_epoch(model, train_loader, loss_fn, optimizer, args.device, epoch + 1)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")
        for k, v in avg_logs.items():
            print(f"  {k}: {v:.6f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "loss": avg_loss,
            **avg_logs
        }
        checkpoint_path = args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest
        latest_path = args.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
    
    print(f"\nTraining complete! Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()

