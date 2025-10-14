
import os
import yaml
import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchvision.utils import save_image

from modules.unet import UNet
from modules.scheduler import *
from modules.autoencoder import AutoEncoder
from modules.unified_dataset import UnifiedLayoutDataset
from modules.condition_mixer import ConcatMixer, WeightedMixer, LearnedWeightedMixer


# ---------------- Utility Functions ----------------

def load_experiment_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_experiment_config(cfg, exp_dir):
    out_path = exp_dir / "experiment_config.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Saved config to {out_path}", flush=True)


def setup_experiment_dir(exp_dir, unet_config, resume):
    exp_dir = Path(exp_dir)
    if exp_dir.exists() and not resume:
        raise ValueError(f"Experiment dir {exp_dir} already exists. Use --resume or change path.")
    exp_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "samples", "configs", "logs"]:
        (exp_dir / sub).mkdir(exist_ok=True)
    if unet_config:
        shutil.copy(unet_config, exp_dir / "configs" / Path(unet_config).name)
    return exp_dir


def save_split_files(exp_dir, train_df, val_df):
    train_file = exp_dir / "trained_on.txt"
    val_file = exp_dir / "evaluated_on.txt"
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    print(f"Saved split files: {train_file}, {val_file}", flush=True)


def save_checkpoint(exp_dir, epoch, unet, mixer, optimizer, scheduler_lr, val_loss, best_loss, training_stats, is_best=False, save_periodic=False):
    """Save checkpoints with training stats"""
    ckpt_dir = exp_dir / "checkpoints"
    
    # Latest checkpoint (full)
    latest_path = ckpt_dir / "latest.pt"
    torch.save({
        "epoch": epoch,
        "unet": unet.state_dict(),
        "mixer": mixer.state_dict(),
        "opt": optimizer.state_dict(),
        "sched": scheduler_lr.state_dict() if scheduler_lr else None,
        "val_loss": val_loss,
        "best_loss": best_loss,
        "training_stats": training_stats
    }, latest_path)
    
    # Best checkpoint
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "unet": unet.state_dict(),
            "mixer": mixer.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": scheduler_lr.state_dict() if scheduler_lr else None,
            "val_loss": val_loss,
            "best_loss": best_loss,
            "training_stats": training_stats
        }, best_path)
        print(f"Saved best checkpoint with loss: {val_loss:.6f}", flush=True)
    
    # Periodic lightweight checkpoint
    if save_periodic:
        periodic_path = ckpt_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "unet": unet.state_dict(),
            "mixer": mixer.state_dict(),
            "val_loss": val_loss,
        }, periodic_path)
        print(f"Saved periodic checkpoint at epoch {epoch+1}", flush=True)


def save_training_stats(exp_dir, training_stats):
    """Save training statistics to JSON and generate plots"""
    stats_path = exp_dir / 'logs' / 'training_stats.json'
    
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(training_stats['epochs'], training_stats['train_loss'], label='Train Loss')
        axes[0].plot(training_stats['epochs'], training_stats['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1].plot(training_stats['epochs'], training_stats['learning_rate'], label='Learning Rate', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = exp_dir / 'logs' / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training plots to {plot_path}", flush=True)
    except ImportError:
        print("Matplotlib not available, skipping plots", flush=True)


def load_checkpoint(checkpoint_path, unet, mixer, optimizer, scheduler_lr):
    """Load checkpoint and return epoch, best_loss, training_stats"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    unet.load_state_dict(checkpoint['unet'])
    mixer.load_state_dict(checkpoint['mixer'])
    optimizer.load_state_dict(checkpoint['opt'])
    if scheduler_lr is not None and checkpoint['sched'] is not None:
        scheduler_lr.load_state_dict(checkpoint['sched'])
    
    epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float('inf'))
    training_stats = checkpoint.get('training_stats', {
        'epochs': [], 'train_loss': [], 'val_loss': [], 
        'learning_rate': [], 'timestamps': []
    })
    
    print(f"Loaded checkpoint from epoch {epoch}, best_loss: {best_loss:.6f}", flush=True)
    return epoch, best_loss, training_stats

def collate_fn(batch):
    """Handle batches where some samples have None POV"""
    return {
        'scene_id': [b['scene_id'] for b in batch],
        'room_id': [b['room_id'] for b in batch],
        'pov': torch.stack([b['pov'] for b in batch if b['pov'] is not None]) if any(b['pov'] is not None for b in batch) else None,
        'graph': torch.stack([b['graph'] for b in batch]),
        'layout': torch.stack([b['layout'] for b in batch]),
    }
# ---------------- Validation----------------

@torch.no_grad()
def validate(unet, scheduler, mixer, val_loader, device):
    """Compute validation loss WITH CONDITIONING"""
    unet.eval()
    total_loss = 0
    
    for batch in val_loader:
        latents = batch["layout"].to(device)
        cond_pov = batch["pov"].to(device) if batch["pov"] is not None else None
        cond_graph = batch["graph"].to(device)
        
        B = latents.size(0)
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)
        
        # FIXED: Build conditioning
        conds = [c for c in [cond_pov, cond_graph] if c is not None]
        cond = mixer(conds)
        
        # FIXED: Use conditioning in validation
        noise_pred = unet(noisy_latents, t, cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()
    
    unet.train()
    return total_loss / len(val_loader)


# ---------------- Sampling (Manual DDPM) ----------------

@torch.no_grad()
def generate_samples(unet, scheduler, autoencoder, mixer, samples, exp_dir, epoch, device):
    """Generate samples using manual DDPM like base script"""
    unet.eval()
    autoencoder.eval()

    num_samples = len(samples)
    
    # Prepare conditioning
    cond_povs = []
    for s in samples:
        if s["pov"] is not None:
            cond_povs.append(s["pov"])
    cond_pov = torch.stack(cond_povs).to(device) if cond_povs else None
    cond_graph = torch.stack([s["graph"] for s in samples]).to(device)
    
    # Build mixed conditioning
    conds = [c for c in [cond_pov, cond_graph] if c is not None]
    cond = mixer(conds)
    
    # Target latents for comparison
    target_latents = torch.stack([s["layout"] for s in samples]).to(device)
    B, C, H, W = target_latents.shape

    # Start from pure noise
    latents = torch.randn(B, C, H, W, device=device)
    
    # FIXED: Manual DDPM sampling loop (matching base script)
    timesteps = torch.linspace(scheduler.num_steps - 1, 0, scheduler.num_steps, dtype=torch.long, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = unet(latents, t_batch, cond=cond)
        
        if t > 0:
            # Get scheduler parameters
            alpha_t = scheduler.alphas[t]
            alpha_bar_t = scheduler.alpha_bars[t]
            alpha_bar_prev = scheduler.alpha_bars[t - 1]
            beta_t = scheduler.betas[t]
            
            # Predict x_0 from current latents
            pred_x0 = (latents - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            # Compute mean of x_{t-1}
            coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            mean = coef1 * pred_x0 + coef2 * latents
            
            # Add noise (variance)
            noise = torch.randn_like(latents)
            variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
            latents = mean + torch.sqrt(variance) * noise
        else:
            # Final step: just predict x_0
            alpha_bar_t = scheduler.alpha_bars[t]
            latents = (latents - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
    
    print(f"Sampled latent stats - min: {latents.min():.4f}, max: {latents.max():.4f}, mean: {latents.mean():.4f}, std: {latents.std():.4f}", flush=True)
    
    # Compute MSE with target
    mse = F.mse_loss(latents, target_latents).item()
    print(f"MSE between generated and target latents: {mse:.6f}", flush=True)

    # Decode both generated and target
    pred_images = autoencoder.decoder(latents)
    target_images = autoencoder.decoder(target_latents)
    
    # Save side-by-side (pred on top, target on bottom)
    both = torch.cat([pred_images, target_images], dim=0)
    save_path = exp_dir / "samples" / f"epoch_{epoch+1:04d}_samples.png"
    save_image(both, save_path, nrow=B, normalize=True)
    print(f"Saved samples with targets to {save_path}", flush=True)

    unet.train()
    return mse


# ---------------- Training Loop----------------

def train_epoch(unet, scheduler, mixer, train_loader, optimizer, device, epoch):
    """Train for one epoch with conditioning"""
    unet.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        latents = batch["layout"].to(device)
        cond_pov = batch["pov"].to(device) if batch["pov"] is not None else None
        cond_graph = batch["graph"].to(device)
        
        B = latents.size(0)
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)
        
        # Build conditioning
        conds = [c for c in [cond_pov, cond_graph] if c is not None]
        cond = mixer(conds)
        
        # Predict noise with conditioning
        noise_pred = unet(noisy_latents, t, cond=cond)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", required=True)
    parser.add_argument("--room_manifest", required=True)
    parser.add_argument("--scene_manifest", required=True)
    parser.add_argument("--pov_type", required=False, default=None)
    parser.add_argument("--mixer_type", choices=["concat", "weighted", "learned"], default="concat")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.exp_config)
    exp_dir = setup_experiment_dir(
        config["experiment"]["exp_dir"], 
        config["unet"]["config_path"], 
        args.resume
    )
    
    if not args.resume:
        save_experiment_config(config, exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # --- Load models ---
    print(f"Loading U-Net from {config['unet']['config_path']}", flush=True)
    unet = UNet.from_config(config["unet"]["config_path"]).to(device)
    
    print(f"Loading {config['scheduler']['type']} with {config['scheduler']['num_steps']} steps", flush=True)
    scheduler_class = globals()[config["scheduler"]["type"]]
    scheduler = scheduler_class(num_steps=config["scheduler"]["num_steps"]).to(device)
    
    print(f"Loading Autoencoder from {config['autoencoder']['config_path']}", flush=True)
    autoencoder = AutoEncoder.from_config(config["autoencoder"]["config_path"]).to(device)
    if config["autoencoder"]["checkpoint_path"]:
        autoencoder.load_state_dict(torch.load(config["autoencoder"]["checkpoint_path"], map_location=device))
    autoencoder.eval()

    # --- Load dataset ---
    print(f"Loading unified dataset...", flush=True)
    dataset = UnifiedLayoutDataset(
            args.room_manifest, 
            args.scene_manifest, 
            use_embeddings=True,
            pov_type=args.pov_type
        )
            
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    # from modules.datasets import compute_sample_weights
    # train_weights = compute_sample_weights(dataset.df.iloc[train_dataset.indices])
    # train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )

    # Save split info
    train_df = dataset.df.iloc[train_dataset.indices]
    val_df = dataset.df.iloc[val_dataset.indices]
    save_split_files(exp_dir, train_df, val_df)

    # --- Mixer ---
    latent_shape = tuple(config["latent_shape"][-2:])  # (H, W)
    
    # Get UNet conditioning channels from config
    with open(config["unet"]["config_path"], 'r') as f:
        unet_cfg = yaml.safe_load(f)
    
    # Mixer output should match UNet's cond_channels
    cond_channels = unet_cfg["unet"]["cond_channels"]
    
    print(f"Creating {args.mixer_type} mixer (out_channels={cond_channels}, target_size={latent_shape})...", flush=True)
    
    if args.mixer_type == "concat":
        mixer = ConcatMixer(
            out_channels=cond_channels, 
            target_size=latent_shape
        ).to(device)
    elif args.mixer_type == "weighted":
        mixer = WeightedMixer(
            out_channels=cond_channels, 
            target_size=latent_shape
        ).to(device)
    else:
        mixer = LearnedWeightedMixer(
            num_conds=2, 
            out_channels=cond_channels, 
            target_size=latent_shape
        ).to(device)

    # --- Optimizer ---
    optimizer = Adam(unet.parameters(), lr=config["training"]["learning_rate"])
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])

    # --- Fixed samples for visual tracking ---
    num_fixed = min(8, len(val_dataset))
    fixed_indices = random.sample(range(len(val_dataset)), num_fixed)
    fixed_samples = [val_dataset[i] for i in fixed_indices]
    torch.save(fixed_indices, exp_dir / "fixed_indices.pt")

    # --- Resume from checkpoint ---
    start_epoch = 0
    best_loss = float("inf")
    training_stats = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'timestamps': []
    }
    
    latest_checkpoint = exp_dir / 'checkpoints' / 'latest.pt'
    if args.resume and latest_checkpoint.exists():
        start_epoch, best_loss, training_stats = load_checkpoint(
            latest_checkpoint, unet, mixer, optimizer, scheduler_lr
        )
        start_epoch += 1

    # --- Training loop ---
    print(f"\nStarting training from epoch {start_epoch+1} to {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}, Learning rate: {config['training']['learning_rate']}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("="*60)

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        # Train
        train_loss = train_epoch(unet, scheduler, mixer, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(unet, scheduler, mixer, val_loader, device)
        
        # Step scheduler
        scheduler_lr.step()
        current_lr = optimizer.param_groups[0]['lr']
        timestamp = datetime.now().isoformat()
        
        # Update training stats
        training_stats['epochs'].append(epoch + 1)
        training_stats['train_loss'].append(train_loss)
        training_stats['val_loss'].append(val_loss)
        training_stats['learning_rate'].append(current_lr)
        training_stats['timestamps'].append(timestamp)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}", flush=True)
        
        # Save checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        
        save_periodic = (epoch + 1) % config["training"].get("periodic_checkpoint_every", 10) == 0
        save_checkpoint(
            exp_dir, epoch, unet, mixer, optimizer, scheduler_lr, 
            val_loss, best_loss, training_stats, is_best, save_periodic
        )
        
        # Save training stats
        save_training_stats(exp_dir, training_stats)

        # --- Sampling ---
        if (epoch + 1) % config["training"]["sample_every"] == 0:
            print(f"Generating samples at epoch {epoch+1}...")
            generate_samples(
                unet, scheduler, autoencoder, mixer, 
                fixed_samples, exp_dir, epoch, device
            )

    print(f"\nTraining complete! Best validation loss: {best_loss:.6f}")
    print(f"Checkpoints saved in: {exp_dir / 'checkpoints'}")
    print(f"Samples saved in: {exp_dir / 'samples'}")


if __name__ == "__main__":
    main()