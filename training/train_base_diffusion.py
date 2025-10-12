import os
import yaml
import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from modules.unet import UNet
from modules.scheduler import *
from modules.autoencoder import AutoEncoder
from modules.datasets import LayoutDataset

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_experiment_config(config_path):
    """Load experiment config from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_experiment_config(config, exp_dir):
    """Save experiment config to experiment directory"""
    config_path = exp_dir / 'experiment_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved experiment config to {config_path}", flush=True)


def setup_experiment_dir(exp_dir, unet_config, resume):
    """
    Setup experiment directory structure.
    If resuming and dir exists, load existing config.
    Otherwise, create new structure and save configs.
    """
    exp_dir = Path(exp_dir)
    
    # Check if resuming
    if exp_dir.exists() and resume:
        print(f"Resuming experiment from {exp_dir}", flush=True)
        return exp_dir
    
    # Create new experiment
    if exp_dir.exists() and not resume:
        raise ValueError(
            f"Experiment directory {exp_dir} already exists. "
            "Use --resume to continue training or choose a different path."
        )
    
    # Create directory structure
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'samples').mkdir(exist_ok=True)
    (exp_dir / 'configs').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    
    # Copy U-Net config to experiment directory
    if unet_config:
        unet_config_path = Path(unet_config)
        shutil.copy(unet_config_path, exp_dir / 'configs' / unet_config_path.name)
    
    print(f"Created experiment directory: {exp_dir}", flush=True)
    return exp_dir


def load_checkpoint(checkpoint_path, unet, optimizer, scheduler_lr):
    """Load checkpoint and return epoch, best_loss, training_stats"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    unet.load_state_dict(checkpoint['unet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler_lr is not None and 'scheduler_state_dict' in checkpoint:
        scheduler_lr.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float('inf'))
    training_stats = checkpoint.get('training_stats', {'epochs': [], 'train_loss': [], 'learning_rate': [], 'timestamps': []})
    
    print(f"Loaded checkpoint from epoch {epoch}, best_loss: {best_loss:.6f}", flush=True)
    return epoch, best_loss, training_stats


def save_checkpoint(exp_dir, epoch, unet, optimizer, scheduler_lr, loss, best_loss, training_stats, is_best=False, save_periodic=False):
    """Save latest, best, and optional periodic checkpoints."""
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Full checkpoint for resuming (latest only) ---
    latest_path = ckpt_dir / 'latest.pt'
    torch.save({
        'epoch': epoch,
        'unet_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_lr.state_dict() if scheduler_lr else None,
        'loss': loss,
        'best_loss': best_loss,
        'training_stats': training_stats,
    }, latest_path)

    # --- Full checkpoint for best model ---
    if is_best:
        best_path = ckpt_dir / 'best.pt'
        torch.save({
            'epoch': epoch,
            'unet_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_lr.state_dict() if scheduler_lr else None,
            'loss': loss,
            'best_loss': best_loss,
            'training_stats': training_stats,
        }, best_path)
        print(f"Saved best checkpoint with loss: {loss:.6f}", flush=True)
    
    # --- Lightweight periodic checkpoint (model only) ---
    if save_periodic:
        periodic_path = ckpt_dir / f'epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'unet_state_dict': unet.state_dict(),
            'loss': loss,
        }, periodic_path)
        print(f"Saved periodic checkpoint (model only) at epoch {epoch+1}", flush=True)

def save_training_stats(exp_dir, training_stats):
    """Save training statistics to JSON and generate plots"""
    stats_path = exp_dir / 'logs' / 'training_stats.json'
    
    # Save JSON
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    # Generate plots if matplotlib available
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


@torch.no_grad()
def validate(unet, scheduler, dataloader, device):
    """Compute validation loss"""
    unet.eval()
    total_loss = 0
    
    for batch in dataloader:
        if batch is None:
            continue
        
        latents = batch['layout'].to(device)
        B = latents.shape[0]
        
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)
        
        noise_pred = unet(noisy_latents, t, cond=None)
        loss = F.mse_loss(noise_pred, noise)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    unet.train()
    return avg_loss

@torch.no_grad()
def generate_samples(unet, scheduler, autoencoder, exp_dir, epoch, num_samples, latent_shape, device):
    """Generate and save sample images"""
    # show progress on same noise


    unet.eval()
    autoencoder.eval()
    
    # Sample latents
    fixed_latents_path = exp_dir / "fixed_latents.pt"
    if fixed_latents_path.exists():
        latents = torch.load(fixed_latents_path).to(device)
    else:
        # set seed for reproducibility
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        latents = torch.randn(num_samples, *latent_shape, device=device)
        torch.save(latents.cpu(), fixed_latents_path)
    
    # Denoise
    timesteps = torch.linspace(scheduler.num_steps - 1, 0, scheduler.num_steps, dtype=torch.long, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = unet(latents, t_batch, cond=None)
        
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
    
    # Print stats AFTER denoising loop completes
    print(f"Sampled latent stats - min: {latents.min():.4f}, max: {latents.max():.4f}, mean: {latents.mean():.4f}, std: {latents.std():.4f}", flush=True)
    
    # Decode
    images = autoencoder.decoder(latents)
    
    # Save images
    from torchvision.utils import save_image
    sample_path = exp_dir / 'samples' / f'epoch_{epoch+1}.png'
    save_image(images, sample_path, nrow=int(num_samples**0.5), normalize=True)
    print(f"Saved samples to {sample_path}", flush=True)
    
    unet.train()


def train_epoch(unet, scheduler, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    unet.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        if batch is None:
            continue
        
        latents = batch['layout'].to(device)
        B = latents.shape[0]
        
        # Add noise
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)
        
        # Predict noise
        noise_pred = unet(noisy_latents, t, cond=None)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    
    # Experiment config (contains all parameters)
    parser.add_argument('--exp_config', type=str, required=True,
                        help='Path to experiment config YAML file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from existing experiment')
    
    args = parser.parse_args()
    
    # Load experiment config
    config = load_experiment_config(args.exp_config)
    
    # Extract config parameters
    exp_dir = Path(config['experiment']['exp_dir'])
    unet_config = config['unet']['config_path']
    scheduler_type = config['scheduler']['type']
    scheduler_steps = config['scheduler']['num_steps']
    autoencoder_config = config['autoencoder']['config_path']
    autoencoder_checkpoint = config['autoencoder']['checkpoint_path']
    dataset_manifest = config['dataset']['manifest_path']
    latent_shape = tuple(config['latent_shape'])  # (C, H, W)
    
    # Training parameters
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    sample_every = config['training']['sample_every']
    num_samples = config['training']['num_samples']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(exp_dir, unet_config, args.resume)
    
    # Save experiment config to exp_dir
    if not args.resume:
        save_experiment_config(config, exp_dir)
    
    # Load U-Net
    print(f"Loading U-Net from {unet_config}", flush=True)
    unet = UNet.from_config(unet_config)
    unet.to(device)
    
    # Load Scheduler
    print(f"Loading {scheduler_type} with {scheduler_steps} steps", flush=True)
    scheduler_class = globals()[scheduler_type]
    scheduler = scheduler_class(num_steps=scheduler_steps)
    scheduler.to(device)
    # Load Autoencoder (for sampling only)
    print(f"Loading Autoencoder from {autoencoder_config}", flush=True)
    autoencoder = AutoEncoder.from_config(autoencoder_config)
    if autoencoder_checkpoint:
        autoencoder.load_state_dict(torch.load(autoencoder_checkpoint, map_location=device))
    autoencoder.to(device)
    autoencoder.eval()
    
    # Load Dataset
    print(f"Loading dataset from {dataset_manifest}", flush=True)
    dataset = LayoutDataset(
        dataset_manifest,
        transform=None,
        mode="all",
        skip_empty=True,
        return_embeddings=True
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    from modules.datasets import collate_skip_none

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_skip_none)
    
    # Optimizer and scheduler
    optimizer = Adam(unet.parameters(), lr=learning_rate)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_loss = float('inf')
    training_stats = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'timestamps': []
    }
    
    latest_checkpoint = exp_dir / 'checkpoints' / 'latest.pt'
    if args.resume and latest_checkpoint.exists():
        start_epoch, best_loss, training_stats = load_checkpoint(latest_checkpoint, unet, optimizer, scheduler_lr)
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch+1} to {num_epochs}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Dataset size: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
    print("="*60)
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        avg_train_loss = train_epoch(unet, scheduler, train_loader, optimizer, device, epoch)
        scheduler_lr.step()
        avg_val_loss = validate(unet, scheduler, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        timestamp = datetime.now().isoformat()
        
        # Update training stats
        training_stats['epochs'].append(epoch + 1)
        training_stats['train_loss'].append(avg_train_loss)
        training_stats['val_loss'].append(avg_val_loss)
        training_stats['learning_rate'].append(current_lr)
        training_stats['timestamps'].append(timestamp)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.6f}, LR: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss

        # Save periodic lightweight checkpoint every N epochs
        save_periodic = (epoch + 1) % 10 == 0  # or whatever interval you want

        save_checkpoint(exp_dir, epoch, unet, optimizer, scheduler_lr, avg_val_loss, 
                        best_loss, training_stats, is_best, save_periodic)

        # Save training stats
        save_training_stats(exp_dir, training_stats)
        
        # Generate samples
        if (epoch + 1) % sample_every == 0:
            print(f"Generating samples at epoch {epoch+1}...")
            generate_samples(unet, scheduler, autoencoder, exp_dir, epoch, num_samples, latent_shape, device)
    
    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved in: {exp_dir / 'checkpoints'}")
    print(f"Samples saved in: {exp_dir / 'samples'}")


if __name__ == '__main__':
    main()