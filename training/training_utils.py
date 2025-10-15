"""
Training utilities for diffusion model training.
Shared functions for experiment management, checkpointing, and visualization.
"""

import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
import torch.nn.functional as F


# ============================================================================
# Configuration Management
# ============================================================================

def load_experiment_config(config_path):
    """Load experiment config from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_experiment_config(config, exp_dir):
    """Save experiment config to experiment directory"""
    exp_dir = Path(exp_dir)
    config_path = exp_dir / 'experiment_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved experiment config to {config_path}", flush=True)


def setup_experiment_dir(exp_dir, unet_config=None, resume=False):
    """
    Setup experiment directory structure.
    If resuming and dir exists, use existing.
    Otherwise, create new structure and copy configs.
    """
    exp_dir = Path(exp_dir)
    
    if exp_dir.exists() and not resume:
        raise ValueError(
            f"Experiment directory {exp_dir} already exists. "
            "Use --resume to continue training or choose a different path."
        )
    
    if exp_dir.exists() and resume:
        print(f"Resuming experiment from {exp_dir}", flush=True)
        return exp_dir
    
    # Create directory structure
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ['checkpoints', 'samples', 'configs', 'logs']:
        (exp_dir / subdir).mkdir(exist_ok=True)
    
    # Copy U-Net config to experiment directory
    if unet_config:
        unet_config_path = Path(unet_config)
        shutil.copy(unet_config_path, exp_dir / 'configs' / unet_config_path.name)
    
    print(f"Created experiment directory: {exp_dir}", flush=True)
    return exp_dir


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(exp_dir, epoch, state_dict, training_stats, 
                   val_loss, best_loss, is_best=False, save_periodic=False):
    """
    Save checkpoints (latest, best, and optional periodic).
    
    Args:
        exp_dir: Experiment directory
        epoch: Current epoch number
        state_dict: Dictionary with model states (unet, optimizer, etc.)
        training_stats: Training statistics dictionary
        val_loss: Current validation loss
        best_loss: Best validation loss so far
        is_best: Whether this is the best model
        save_periodic: Whether to save a periodic checkpoint
    """
    ckpt_dir = Path(exp_dir) / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'val_loss': val_loss,
        'best_loss': best_loss,
        'training_stats': training_stats,
        **state_dict
    }
    
    # Save latest checkpoint (full)
    latest_path = ckpt_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = ckpt_dir / 'best.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint with loss: {val_loss:.6f}", flush=True)
    
    # Save periodic lightweight checkpoint (model only)
    if save_periodic:
        periodic_checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            **{k: v for k, v in state_dict.items() if 'state_dict' in k or k in ['unet', 'mixer']}
        }
        periodic_path = ckpt_dir / f'epoch_{epoch+1}.pt'
        torch.save(periodic_checkpoint, periodic_path)
        print(f"Saved periodic checkpoint at epoch {epoch+1}", flush=True)


def load_checkpoint(checkpoint_path, models_dict, optimizer=None, scheduler_lr=None):
    """
    Load checkpoint and restore model states.
    
    Args:
        checkpoint_path: Path to checkpoint file
        models_dict: Dict mapping names to model objects (e.g., {'unet': unet, 'mixer': mixer})
        optimizer: Optimizer to restore (optional)
        scheduler_lr: LR scheduler to restore (optional)
    
    Returns:
        Tuple of (start_epoch, best_loss, training_stats)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model states
    for name, model in models_dict.items():
        if name in checkpoint:
            model.load_state_dict(checkpoint[name])
        elif f'{name}_state_dict' in checkpoint:
            model.load_state_dict(checkpoint[f'{name}_state_dict'])
    
    # Load optimizer state
    if optimizer and 'opt' in checkpoint:
        optimizer.load_state_dict(checkpoint['opt'])
    elif optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler_lr and 'sched' in checkpoint:
        if checkpoint['sched'] is not None:
            scheduler_lr.load_state_dict(checkpoint['sched'])
    elif scheduler_lr and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler_lr.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float('inf'))
    training_stats = checkpoint.get('training_stats', {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'learning_rate': [], 'timestamps': []
    })
    
    print(f"Loaded checkpoint from epoch {epoch}, best_loss: {best_loss:.6f}", flush=True)
    return epoch, best_loss, training_stats


# ============================================================================
# Training Statistics
# ============================================================================

def save_training_stats(exp_dir, training_stats):
    """Save training statistics to JSON and generate plots"""
    exp_dir = Path(exp_dir)
    stats_path = exp_dir / 'logs' / 'training_stats.json'
    
    # Save JSON
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    # Generate plots if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        if 'train_loss' in training_stats and training_stats['train_loss']:
            axes[0].plot(training_stats['epochs'], training_stats['train_loss'], 
                        label='Train Loss', marker='o', markersize=3)
        if 'val_loss' in training_stats and training_stats['val_loss']:
            axes[0].plot(training_stats['epochs'], training_stats['val_loss'], 
                        label='Val Loss', marker='s', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rate' in training_stats and training_stats['learning_rate']:
            axes[1].plot(training_stats['epochs'], training_stats['learning_rate'], 
                        label='Learning Rate', color='orange', marker='o', markersize=3)
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


def init_training_stats():
    """Initialize empty training statistics dictionary"""
    return {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'timestamps': []
    }


def update_training_stats(training_stats, epoch, train_loss, val_loss, learning_rate):
    """Update training statistics with current epoch data"""
    training_stats['epochs'].append(epoch + 1)
    training_stats['train_loss'].append(train_loss)
    training_stats['val_loss'].append(val_loss)
    training_stats['learning_rate'].append(learning_rate)
    training_stats['timestamps'].append(datetime.now().isoformat())
    return training_stats


# ============================================================================
# Training Loop Helpers
# ============================================================================

def train_epoch_unconditioned(unet, scheduler, dataloader, optimizer, device, epoch):
    """Train for one epoch without conditioning"""
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
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def train_epoch_conditioned(unet, scheduler, mixer, dataloader, optimizer, device, epoch):
    """Train for one epoch with conditioning"""
    unet.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
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
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_unconditioned(unet, scheduler, dataloader, device):
    """Compute validation loss without conditioning"""
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
    
    unet.train()
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_conditioned(unet, scheduler, mixer, dataloader, device):
    """Compute validation loss with conditioning"""
    unet.eval()
    total_loss = 0
    
    for batch in dataloader:
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
        
        noise_pred = unet(noisy_latents, t, cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()
    
    unet.train()
    return total_loss / len(dataloader)


# ============================================================================
# Dataset Utilities
# ============================================================================

def save_split_files(exp_dir, train_df, val_df):
    """Save train/val split information to CSV files"""
    exp_dir = Path(exp_dir)
    train_file = exp_dir / "trained_on.csv"
    val_file = exp_dir / "evaluated_on.csv"
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    print(f"Saved split files: {train_file}, {val_file}", flush=True)