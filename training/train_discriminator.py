#!/usr/bin/env python3
"""
Train discriminator to distinguish between viable (real) and non-viable (fake) layout latents.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from training.utils import set_deterministic, get_device
from models.components.discriminator import LatentDiscriminator
from models.autoencoder import Autoencoder


def load_latents(path, device="cuda"):
    """Load latents from .pt file or directory of .pt files."""
    path = Path(path)
    device_obj = torch.device(device)
    
    if path.is_file() and path.suffix == ".pt":
        latents = torch.load(path, map_location=device_obj)
        return latents
    elif path.is_dir():
        # Load from directory of .pt files
        latent_files = sorted(path.glob("*.pt"))
        if not latent_files:
            raise ValueError(f"No .pt files found in {path}")
        latents_list = [torch.load(f, map_location=device_obj) for f in tqdm(latent_files, desc="Loading latents")]
        latents = torch.cat(latents_list, dim=0)
        return latents
    else:
        raise ValueError(f"Invalid path: {path}. Must be .pt file or directory of .pt files")


def train_discriminator(
    real_latents_path,
    fake_latents_path,
    output_dir,
    config=None,
    epochs=50,
    batch_size=64,
    learning_rate=0.0002,
    device="cuda",
    seed=42
):
    """
    Train discriminator on real vs fake latents.
    
    Args:
        real_latents_path: Path to real latents (.pt file or directory)
        fake_latents_path: Path to fake latents (.pt file or directory)
        output_dir: Directory to save discriminator
        config: Optional config dict for discriminator
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        seed: Random seed
    """
    set_deterministic(seed)
    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load latents
    print("Loading real latents...")
    real_latents = load_latents(real_latents_path, device)
    print(f"Real latents: {real_latents.shape}")
    
    print("Loading fake latents...")
    fake_latents = load_latents(fake_latents_path, device)
    print(f"Fake latents: {fake_latents.shape}")
    
    # Create labels
    real_labels = torch.ones(len(real_latents), 1, device=device_obj)
    fake_labels = torch.zeros(len(fake_latents), 1, device=device_obj)
    
    # Combine datasets
    all_latents = torch.cat([real_latents, fake_latents], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)
    
    # Shuffle
    indices = torch.randperm(len(all_latents), device=device_obj)
    all_latents = all_latents[indices]
    all_labels = all_labels[indices]
    
    # Split train/val (80/20)
    split_idx = int(0.8 * len(all_latents))
    train_latents = all_latents[:split_idx]
    train_labels = all_labels[:split_idx]
    val_latents = all_latents[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"Train: {len(train_latents)}, Val: {len(val_latents)}")
    
    # Build discriminator
    if config:
        discriminator = LatentDiscriminator.from_config(config)
    else:
        # Default config - infer from latents
        latent_channels = real_latents.shape[1]
        discriminator = LatentDiscriminator(
            latent_channels=latent_channels,
            base_channels=64,
            num_layers=4
        )
    discriminator = discriminator.to(device_obj)
    
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float("inf")
    history = []
    
    for epoch in range(epochs):
        # Train
        discriminator.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        num_batches = (len(train_latents) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_latents))
            
            batch_latents = train_latents[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]
            
            # Forward
            optimizer.zero_grad()
            scores = discriminator(batch_latents)
            loss = criterion(scores, batch_labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            predictions = (scores > 0.5).float()
            train_correct += (predictions == batch_labels).sum().item()
            train_total += len(batch_labels)
        
        avg_train_loss = train_loss / num_batches
        train_acc = train_correct / train_total
        
        # Validate
        discriminator.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            num_val_batches = (len(val_latents) + batch_size - 1) // batch_size
            for batch_idx in range(num_val_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(val_latents))
                
                batch_latents = val_latents[start_idx:end_idx]
                batch_labels = val_labels[start_idx:end_idx]
                
                scores = discriminator(batch_latents)
                loss = criterion(scores, batch_labels)
                
                val_loss += loss.item()
                predictions = (scores > 0.5).float()
                val_correct += (predictions == batch_labels).sum().item()
                val_total += len(batch_labels)
        
        avg_val_loss = val_loss / num_val_batches
        val_acc = val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "state_dict": discriminator.state_dict(),
                "config": discriminator.to_config(),
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            }, output_dir / "discriminator_best.pt")
            print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f}, val_acc={val_acc:.4f})")
    
    # Save final
    torch.save({
        "state_dict": discriminator.state_dict(),
        "config": discriminator.to_config(),
        "epoch": epochs,
        "val_loss": avg_val_loss,
        "val_acc": val_acc
    }, output_dir / "discriminator_final.pt")
    
    # Save history
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "discriminator_history.csv", index=False)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Discriminator saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train latent space discriminator")
    parser.add_argument("--real_latents", type=Path, required=True,
                       help="Path to real latents (.pt file or directory)")
    parser.add_argument("--fake_latents", type=Path, required=True,
                       help="Path to fake latents (.pt file or directory)")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for discriminator")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0002,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    train_discriminator(
        real_latents_path=args.real_latents,
        fake_latents_path=args.fake_latents,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

