import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from datetime import datetime
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

from dataset.datasets import LayoutDataset
from modules.autoencoder import AutoEncoder


def print_separator(char="-", length=80):
    print(char * length)


def get_loss(name: str):
    name = name.lower()
    print(f"[LOSS] Initializing loss function: {name}")
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss {name}")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def train(cfg, args):
    print_separator("=")
    print("AUTOENCODER TRAINING SESSION")
    print_separator("=")
    print(f"[TIME] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ----- Device -----
    print_separator()
    print("[DEVICE] Setting up compute device...")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"  ✓ CUDA available - using GPU")
            print(f"    GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = "mps"
            print(f"  ✓ MPS available - using Apple Silicon GPU")
        else:
            device = "cpu"
            print(f"  ⚠ No GPU available - using CPU")
    else:
        device = args.device
        print(f"  → Manually specified device: {device}")
    print(f"[DEVICE] Using device: {device}")
    print()

    # ----- Model -----
    print_separator()
    print("[MODEL] Building AutoEncoder from config...")
    print(f"  Config file: {args.config}")
    
    start_time = time.time()
    ae = AutoEncoder.from_config(cfg).to(device)
    load_time = time.time() - start_time
    
    print(f"  ✓ Model built successfully in {load_time:.2f} seconds")
    print(f"\n[MODEL] Architecture Summary:")
    print(f"  → Total parameters: {count_parameters(ae):,}")
    print(f"  → Model size: {get_model_size(ae):.2f} MB")
    print(f"  → Latent dimension: {cfg['encoder']['latent_dim']}")
    print(f"  → Input image size: {cfg['encoder']['image_size']}x{cfg['encoder']['image_size']}")
    print(f"  → Input channels: {cfg['encoder']['in_channels']}")
    print(f"  → Output channels: {cfg['decoder']['out_channels']}")
    
    # Print encoder architecture
    print(f"\n  Encoder layers:")
    for i, layer_cfg in enumerate(cfg['encoder']['layers']):
        print(f"    Layer {i+1}: {layer_cfg.get('out_channels')} channels, "
              f"stride={layer_cfg.get('stride', 1)}, "
              f"norm={layer_cfg.get('norm', 'none')}, "
              f"act={layer_cfg.get('act', 'none')}")
    print()

    # ----- Dataset -----
    print_separator()
    print("[DATASET] Loading dataset...")
    print(f"  Manifest: {args.layout_manifest}")
    print(f"  Mode: {args.layout_mode}")
    print(f"  Keep empty: {args.keep_empty}")
    print(f"  Return embeddings: False")
    
    # ----- Transform -----
    print(f"\n[TRANSFORM] Setting up data transformations...")
    print(f"  → Resize to: {args.resize}x{args.resize}")
    print(f"  → Convert to tensor")
    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor()
    ])
    
    print(f"\n[DATASET] Initializing LayoutDataset...")
    dataset = LayoutDataset(
        args.layout_manifest,
        transform=transform,
        mode=args.layout_mode,
        skip_empty=not args.keep_empty,
        return_embeddings=False
    )
    print(f"  ✓ Dataset loaded with {len(dataset)} samples")
    
    print(f"\n[DATALOADER] Creating DataLoader...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Shuffle: True")
    print(f"  Number of batches: {len(dataset) // args.batch_size}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  ✓ DataLoader ready")
    print()

    # ----- Training setup -----
    print_separator()
    print("[TRAINING] Setting up training components...")
    
    criterion = get_loss(args.loss)
    print(f"  ✓ Loss function: {args.loss}")
    
    print(f"\n[OPTIMIZER] Initializing Adam optimizer...")
    print(f"  Learning rate: {args.lr}")
    optimizer = optim.Adam(ae.parameters(), lr=args.lr)
    print(f"  ✓ Optimizer ready")
    print()

    # ----- Run directory -----
    print_separator()
    print("[OUTPUT] Setting up output directory...")
    exp_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outdir, f"{exp_id}_{args.name}")
    
    # Make sure we create a unique directory if it somehow exists
    original_run_dir = run_dir
    counter = 1
    while os.path.exists(run_dir):
        run_dir = f"{original_run_dir}_v{counter}"
        counter += 1
    
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Output directory: {run_dir}")
    print(f"  Experiment ID: {exp_id}")
    print(f"  Job name: {args.name}")
    
    # Log structure that will be created
    print(f"\n[DIRECTORY STRUCTURE]")
    print(f"  {run_dir}/")
    print(f"    ├── config_used.yml    (training configuration)")
    print(f"    ├── metrics.csv        (loss metrics)")
    print(f"    ├── best.pt           (best model checkpoint)")
    print(f"    ├── last.pt           (final model checkpoint)")
    if args.save_checkpoint_every > 0 and not args.keep_only_best:
        print(f"    ├── epoch*.pt         (periodic checkpoints)")
    if args.save_images:
        print(f"    ├── inputs_epoch*.png  (input samples)")
        print(f"    └── recons_epoch*.png  (reconstructions)")
    
    # save config + CLI args
    print(f"\n[CONFIG] Saving configuration...")
    full_cfg = {"model_cfg": cfg, "cli_args": vars(args)}
    config_path = os.path.join(run_dir, "config_used.yml")
    with open(config_path, "w") as f:
        yaml.safe_dump(full_cfg, f)
    print(f"  ✓ Config saved to: {config_path}")
    
    # metrics log
    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics = []
    print(f"  ✓ Metrics will be saved to: {metrics_path}")
    
    # Image saving info
    if args.save_images:
        print(f"\n[IMAGES] Image saving enabled")
        print(f"  Save every: {args.save_every} epochs")
    else:
        print(f"\n[IMAGES] Image saving disabled (use --save_images to enable)")
    
    # Checkpoint saving info
    print(f"\n[CHECKPOINT POLICY]")
    if args.keep_only_best:
        print(f"  → Only keeping best.pt and last.pt")
    elif args.save_checkpoint_every > 0:
        print(f"  → Saving checkpoints every {args.save_checkpoint_every} epochs")
        print(f"  → Always saving best.pt and last.pt")
    else:
        print(f"  → Only saving best.pt and last.pt (no periodic checkpoints)")
    print()

    # ----- Training loop -----
    print_separator("=")
    print("STARTING TRAINING")
    print_separator("=")
    print(f"[TRAINING] Training for {args.epochs} epochs")
    print()
    
    best_loss = float("inf")
    total_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"[EPOCH {epoch+1}/{args.epochs}] " + "="*50)
        
        ae.train()
        total_loss = 0.0
        batch_losses = []
        num_skipped = 0
        
        # Progress bar for batches
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, batch in enumerate(pbar):
            if batch["layout"] is None:
                num_skipped += 1
                continue
                
            imgs = batch["layout"].to(device)
            batch_size = imgs.size(0)
            
            # Forward pass
            optimizer.zero_grad()
            recon = ae(imgs)
            loss = criterion(recon, imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            total_loss += batch_loss * batch_size
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 
                             'avg_loss': f'{np.mean(batch_losses):.4f}'})
            
            # Verbose batch logging every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx+1}/{len(loader)}: "
                      f"Loss={batch_loss:.4f}, "
                      f"Running avg={np.mean(batch_losses):.4f}")
        
        pbar.close()
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(loader.dataset)
        
        print(f"\n[EPOCH {epoch+1} COMPLETE]")
        print(f"  → Average Loss: {avg_loss:.6f}")
        print(f"  → Min batch loss: {min(batch_losses):.6f}")
        print(f"  → Max batch loss: {max(batch_losses):.6f}")
        print(f"  → Std batch loss: {np.std(batch_losses):.6f}")
        print(f"  → Epoch time: {epoch_time:.2f} seconds")
        print(f"  → Samples/second: {len(dataset)/epoch_time:.2f}")
        if num_skipped > 0:
            print(f"  ⚠ Skipped {num_skipped} empty batches")
        
        # Check if best model
        is_best = avg_loss < best_loss
        if is_best:
            print(f"  ★ NEW BEST MODEL! (previous best: {best_loss:.6f})")
            best_loss = avg_loss
        
        # Save metrics
        metrics.append({
            "epoch": epoch+1, 
            "loss": avg_loss,
            "min_loss": min(batch_losses),
            "max_loss": max(batch_losses),
            "std_loss": np.std(batch_losses),
            "epoch_time": epoch_time
        })
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        print(f"  ✓ Metrics saved")
        
        # Save checkpoints
        print(f"\n[CHECKPOINTS]")
        
        # Always save last checkpoint
        torch.save(ae.state_dict(), os.path.join(run_dir, "last.pt"))
        print(f"  ✓ Updated last.pt")
        
        # Save best checkpoint if this is the best epoch
        if is_best:
            torch.save(ae.state_dict(), os.path.join(run_dir, "best.pt"))
            print(f"  ✓ Updated best.pt (loss={best_loss:.6f})")
        
        # Save numbered checkpoint only if specified
        if args.save_checkpoint_every > 0 and (epoch+1) % args.save_checkpoint_every == 0:
            if not args.keep_only_best:
                ckpt_path = os.path.join(run_dir, f"epoch{epoch+1}.pt")
                torch.save(ae.state_dict(), ckpt_path)
                print(f"  ✓ Saved epoch checkpoint: epoch{epoch+1}.pt")
            else:
                print(f"  → Skipped epoch checkpoint (keep_only_best=True)")
        
        # Save sample images
        if args.save_images and (epoch+1) % args.save_every == 0:
            print(f"\n[IMAGES] Saving sample reconstructions...")
            
            with torch.no_grad():
                ae.eval()
                
                # Get a fresh batch for visualization
                sample_batch = next(iter(loader))
                while sample_batch["layout"] is None:
                    sample_batch = next(iter(loader))
                
                sample_imgs = sample_batch["layout"].to(device)[:8]
                sample_recon = ae(sample_imgs)
                
                # Create side-by-side comparison: top row = input, bottom row = reconstruction
                comparison = torch.cat([sample_imgs[:4], sample_recon[:4]], dim=0)
                comparison_grid = utils.make_grid(comparison, nrow=4, normalize=False, value_range=(0, 1))
                
                # Also save separate grids
                imgs_grid = utils.make_grid(sample_imgs, nrow=4, normalize=False, value_range=(0, 1))
                recons_grid = utils.make_grid(sample_recon, nrow=4, normalize=False, value_range=(0, 1))
                
                # Save all versions
                comparison_path = os.path.join(run_dir, f"comparison_epoch{epoch+1}.png")
                input_path = os.path.join(run_dir, f"inputs_epoch{epoch+1}.png")
                recon_path = os.path.join(run_dir, f"recons_epoch{epoch+1}.png")
                
                utils.save_image(comparison_grid, comparison_path)
                utils.save_image(imgs_grid, input_path)
                utils.save_image(recons_grid, recon_path)
                
                print(f"  ✓ Saved comparison: {comparison_path} (top=input, bottom=output)")
                print(f"  ✓ Saved input samples: {input_path}")
                print(f"  ✓ Saved reconstructions: {recon_path}")
                
                # Calculate reconstruction metrics
                mse = nn.MSELoss()(sample_recon, sample_imgs).item()
                print(f"  → Sample reconstruction MSE: {mse:.6f}")
        
        # Estimated time remaining
        elapsed_time = time.time() - total_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs
        
        print(f"\n[PROGRESS] {epoch+1}/{args.epochs} epochs complete")
        print(f"  Total elapsed: {elapsed_time/60:.2f} minutes")
        print(f"  ETA: {eta/60:.2f} minutes")
        print()
    
    # ----- Training complete -----
    total_time = time.time() - total_start_time
    
    print_separator("=")
    print("TRAINING COMPLETE")
    print_separator("=")
    print(f"[SUMMARY]")
    print(f"  Total training time: {total_time/60:.2f} minutes")
    print(f"  Average epoch time: {total_time/args.epochs:.2f} seconds")
    print(f"  Final loss: {avg_loss:.6f}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Results directory: {run_dir}")
    print()
    print(f"[FILES SAVED]")
    
    # Count actual checkpoint files
    checkpoint_count = 2  # best.pt and last.pt
    if args.save_checkpoint_every > 0 and not args.keep_only_best:
        checkpoint_count += (args.epochs // args.save_checkpoint_every)
    
    print(f"  ✓ Model checkpoints: best.pt, last.pt")
    if checkpoint_count > 2:
        print(f"    + {checkpoint_count - 2} periodic checkpoint(s)")
    print(f"  ✓ Configuration: config_used.yml")
    print(f"  ✓ Training metrics: metrics.csv")
    
    if args.save_images:
        num_img_saves = (args.epochs // args.save_every)
        print(f"  ✓ Sample images: {num_img_saves} sets of reconstructions")
    print()
    print(f"[END] Training session finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("=")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AutoEncoder model with detailed logging")
    
    # manifests
    parser.add_argument("--layout_manifest", type=str, required=True,
                        help="Path to the layout manifest CSV file")
    
    # config + job name
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the model configuration YAML file")
    parser.add_argument("--name", type=str, default="job",
                        help="Name for this training job (default: job)")
    
    # training params
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer (default: 1e-3)")
    parser.add_argument("--resize", type=int, default=512,
                        help="Size to resize images to (default: 512)")
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "l1", "huber", "bce"],
                        help="Loss function to use (default: mse)")
    
    # checkpoint params
    parser.add_argument("--save_checkpoint_every", type=int, default=0,
                        help="Save checkpoint every N epochs (0 = only save best and last)")
    parser.add_argument("--keep_only_best", action="store_true",
                        help="Only keep best.pt and last.pt checkpoints")
    
    # dataset
    parser.add_argument("--layout_mode", type=str, default="all",
                        help="Layout mode for dataset (default: all)")
    parser.add_argument("--keep_empty", action="store_true",
                        help="Keep empty layouts in dataset")
    
    # device + output
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device to train on (default: auto)")
    parser.add_argument("--outdir", type=str, default="runs",
                        help="Output directory for training runs (default: runs)")
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save sample reconstruction images during training (default: True)")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save images every N epochs (default: 1)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AUTOENCODER TRAINING LAUNCHER")
    print("="*80)
    print("\n[ARGUMENTS] Parsed command line arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    print("[CONFIG] Loading model configuration...")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("  ✓ Configuration loaded successfully")
    print()
    
    # Start training
    train(cfg, args)