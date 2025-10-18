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
import numpy as np


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

def init_training_stats():
    return {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'learning_rate': [], 'timestamps': [],
        'corr_pov': [], 'corr_graph': [], 'corr_mix': [],
        'cond_std_pov': [], 'cond_std_graph': [],
        'dropout_ratio_pov': [], 'dropout_ratio_graph': []
    }


def update_training_stats(training_stats, epoch, train_loss, val_loss, learning_rate,
                          corr_pov=None, corr_graph=None, corr_mix=None,
                          cond_std_pov=None, cond_std_graph=None,
                          dropout_ratio_pov=None, dropout_ratio_graph=None):
    """Update training statistics with current epoch data"""
    training_stats['epochs'].append(epoch + 1)
    training_stats['train_loss'].append(train_loss)
    training_stats['val_loss'].append(val_loss)
    training_stats['learning_rate'].append(learning_rate)
    training_stats['timestamps'].append(datetime.now().isoformat())

    # Optional correlation logging
    training_stats['corr_pov'].append(corr_pov if corr_pov is not None else None)
    training_stats['corr_graph'].append(corr_graph if corr_graph is not None else None)
    training_stats['corr_mix'].append(corr_mix if corr_mix is not None else None)

    training_stats['cond_std_pov'].append(cond_std_pov)
    training_stats['cond_std_graph'].append(cond_std_graph)
    training_stats['dropout_ratio_pov'].append(dropout_ratio_pov)
    training_stats['dropout_ratio_graph'].append(dropout_ratio_graph)
    return training_stats


def save_training_stats(exp_dir, training_stats):
    """Save training statistics to JSON and generate plots"""
    exp_dir = Path(exp_dir)
    stats_path = exp_dir / 'logs' / 'training_stats.json'

    # Save JSON
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        # ---------------------------------------------------------
        # Main training curves (Loss, LR, Correlation)
        # ---------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # --- Loss Plot ---
        if training_stats.get('train_loss'):
            axes[0].plot(training_stats['epochs'], training_stats['train_loss'],
                         label='Train', marker='o', ms=3)
        if training_stats.get('val_loss'):
            axes[0].plot(training_stats['epochs'], training_stats['val_loss'],
                         label='Val', marker='s', ms=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Train / Val Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Learning Rate Plot ---
        if training_stats.get('learning_rate'):
            axes[1].plot(training_stats['epochs'], training_stats['learning_rate'],
                         color='orange', marker='o', ms=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)

        # --- Correlation Plot ---
        has_corr = any(len(training_stats.get(k, [])) for k in ['corr_pov', 'corr_graph', 'corr_mix'])
        if has_corr:
            if training_stats.get('corr_pov'):
                axes[2].plot(training_stats['epochs'], training_stats['corr_pov'],
                             label='POV Corr', marker='o', ms=3)
            if training_stats.get('corr_graph'):
                axes[2].plot(training_stats['epochs'], training_stats['corr_graph'],
                             label='Graph Corr', marker='s', ms=3)
            if training_stats.get('corr_mix'):
                axes[2].plot(training_stats['epochs'], training_stats['corr_mix'],
                             label='Mix Corr', marker='^', ms=3)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Correlation')
            axes[2].set_title('Condition Correlation')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = exp_dir / 'logs' / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training plots to {plot_path}", flush=True)

        # ---------------------------------------------------------
        # Additional visual diagnostics
        # ---------------------------------------------------------

        # --- Condition Signal Strength ---
        if training_stats.get('cond_std_pov') or training_stats.get('cond_std_graph'):
            fig, ax = plt.subplots(figsize=(6, 4))
            if training_stats.get('cond_std_pov'):
                ax.plot(training_stats['epochs'], training_stats['cond_std_pov'],
                        label='POV σ', marker='o', ms=3)
            if training_stats.get('cond_std_graph'):
                ax.plot(training_stats['epochs'], training_stats['cond_std_graph'],
                        label='Graph σ', marker='s', ms=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Std')
            ax.set_title('Condition Signal Strength')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            sig_path = exp_dir / 'logs' / 'cond_signal_strength.png'
            plt.savefig(sig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved condition signal plot to {sig_path}", flush=True)

        # --- Dropout Ratio Plot ---
        if training_stats.get('dropout_ratio_pov') or training_stats.get('dropout_ratio_graph'):
            fig, ax = plt.subplots(figsize=(6, 4))
            if training_stats.get('dropout_ratio_pov'):
                ax.plot(training_stats['epochs'], training_stats['dropout_ratio_pov'],
                        label='POV Dropout', marker='o', ms=3)
            if training_stats.get('dropout_ratio_graph'):
                ax.plot(training_stats['epochs'], training_stats['dropout_ratio_graph'],
                        label='Graph Dropout', marker='s', ms=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Dropout Ratio')
            ax.set_title('Condition Dropout Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            drop_path = exp_dir / 'logs' / 'cond_dropout.png'
            plt.savefig(drop_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved condition dropout plot to {drop_path}", flush=True)

    except ImportError:
        print("Matplotlib not available, skipping plots", flush=True)



# ============================================================================
# Training Loop Helpers
# ============================================================================

def compute_condition_correlations(cond_pov, cond_graph, cond, noisy_latents):
    """Compute cosine correlations between latent and each conditioning source."""
    corr_pov_vals, corr_graph_vals, corr_mix_vals = [], [], []
    with torch.no_grad():
        l = noisy_latents.mean(dim=[2, 3])       # [B, C_lat]
        c_mix = cond.mean(dim=[2, 3])            # [B, C_cond]

        # POV
        if cond_pov is not None:
            c_pov = cond_pov.mean(dim=[2, 3]) if cond_pov.ndim == 4 else cond_pov
            min_ch = min(c_pov.size(1), l.size(1))
            corr = torch.cosine_similarity(
                c_pov[:, :min_ch], l[:, :min_ch], dim=1
            )
            corr = torch.nan_to_num(corr, nan=0.0)
            corr_pov_vals.append(corr.mean().item())

        # Graph
        c_graph = cond_graph.mean(dim=[2, 3]) if cond_graph.ndim == 4 else cond_graph
        min_ch = min(c_graph.size(1), l.size(1))
        corr_graph_vals.append(
            torch.cosine_similarity(c_graph[:, :min_ch], l[:, :min_ch], dim=1).mean().item()
        )

        # Mixed
        min_ch = min(c_mix.size(1), l.size(1))
        corr_mix_vals.append(
            torch.cosine_similarity(c_mix[:, :min_ch], l[:, :min_ch], dim=1).mean().item()
        )

    return (
        np.mean(corr_pov_vals) if corr_pov_vals else 0.0,
        np.mean(corr_graph_vals) if corr_graph_vals else 0.0,
        np.mean(corr_mix_vals) if corr_mix_vals else 0.0,
    )


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


def train_epoch_conditioned(
    unet,
    scheduler,
    mixer,
    dataloader,
    optimizer,
    device,
    epoch,
    config,
    cfg_dropout_prob=0.1,
    align_pov=None,
    align_graph=None):
    """
    Train one epoch with conditional inputs and optional alignment projection.
    Integrates alignment, dropout, normalization, scaling, and correlation logging.
    """

    unet.train()
    total_loss = 0
    corr_pov_vals, corr_graph_vals, corr_mix_vals = [], [], []
    cond_std_pov_vals, cond_std_graph_vals = [], []
    dropout_pov_events, dropout_graph_events, total_batches = 0, 0, 0

    # ---------------- Config ----------------
    cfg_train = config["training"]["cfg"]
    norm_graph = cfg_train.get("normalize_graph", True)
    norm_pov = cfg_train.get("normalize_pov", False)
    scale_graph = cfg_train.get("cond_scale_graph", 5.0)
    scale_pov = cfg_train.get("cond_scale_pov", 1.0)
    scale_mix = cfg_train.get("cond_scale_mix", 1.0)
    clip_value = cfg_train.get("cond_clip_value", None)
    log_cond_stats = cfg_train.get("log_condition_stats", True)
    corr_every = cfg_train.get("compute_corr_every", 300)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for i, batch in enumerate(pbar):
        total_batches += 1

        # --- Load and move tensors ---
        latents = batch["layout"].to(device)
        cond_pov = batch.get("pov")
        cond_graph = batch.get("graph")
        if cond_pov is not None:
            cond_pov = cond_pov.to(device)
        if cond_graph is not None:
            cond_graph = cond_graph.to(device)

        B = latents.size(0)
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)

        # --- Apply pretrained alignment (optional) ---
        with torch.no_grad():
            if align_pov is not None and cond_pov is not None:
                cond_pov = align_pov(cond_pov).detach()
            if align_graph is not None and cond_graph is not None:
                cond_graph = align_graph(cond_graph).detach()

        # --- Classifier-free dropout ---
        if torch.rand(1).item() < cfg_dropout_prob:
            if cond_pov is not None:
                cond_pov = torch.zeros_like(cond_pov)
                dropout_pov_events += 1
            if cond_graph is not None:
                cond_graph = torch.zeros_like(cond_graph)
                dropout_graph_events += 1

        # --- Normalize and scale ---
        if cond_pov is not None:
            if norm_pov:
                mean, std = cond_pov.mean(), cond_pov.std()
                cond_pov = (cond_pov - mean) / (std + 1e-5)
            cond_pov = cond_pov * scale_pov
            cond_std_pov_vals.append(cond_pov.std().item())

        if cond_graph is not None:
            if norm_graph:
                mean, std = cond_graph.mean(), cond_graph.std()
                cond_graph = (cond_graph - mean) / (std + 1e-5)
            cond_graph = cond_graph * scale_graph
            cond_std_graph_vals.append(cond_graph.std().item())

        # --- Optional clipping ---
        if clip_value is not None:
            if cond_pov is not None:
                cond_pov = torch.clamp(cond_pov, -clip_value, clip_value)
            if cond_graph is not None:
                cond_graph = torch.clamp(cond_graph, -clip_value, clip_value)

        # --- Fuse conditions ---
        cond = mixer([cond_pov, cond_graph]) * scale_mix

        # --- Correlation diagnostics ---
        corr_pov, corr_graph, corr_mix = compute_condition_correlations(
            cond_pov, cond_graph, cond, noisy_latents
        )
        corr_pov_vals.append(corr_pov)
        corr_graph_vals.append(corr_graph)
        corr_mix_vals.append(corr_mix)

        # --- Forward + Backward ---
        noise_pred = unet(noisy_latents, t, cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if log_cond_stats and (i % corr_every == 0):
            cond_mean, cond_std = cond.mean().item(), cond.std().item()
            lat_mean, lat_std = noisy_latents.mean().item(), noisy_latents.std().item()
            print(f"[epoch {epoch+1} | batch {i}] "
                  f"latent μ={lat_mean:.3f} σ={lat_std:.3f} | "
                  f"cond μ={cond_mean:.3f} σ={cond_std:.3f}", flush=True)

        pbar.set_postfix({'loss': loss.item()})

    # ---------------- Epoch metrics ----------------
    avg_loss = total_loss / len(dataloader)
    return (
        avg_loss,
        np.mean(corr_pov_vals) if corr_pov_vals else 0.0,
        np.mean(corr_graph_vals) if corr_graph_vals else 0.0,
        np.mean(corr_mix_vals) if corr_mix_vals else 0.0,
        np.mean(cond_std_pov_vals) if cond_std_pov_vals else 0.0,
        np.mean(cond_std_graph_vals) if cond_std_graph_vals else 0.0,
        dropout_pov_events / total_batches,
        dropout_graph_events / total_batches,
    )




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
def validate_conditioned(
    unet,
    scheduler,
    mixer,
    dataloader,
    device,
    config,
    align_pov=None,
    align_graph=None
):
    """
    Validation pass for conditioned latent diffusion with optional alignment.
    Mirrors train_epoch_conditioned but without optimization.
    """

    unet.eval()
    total_loss = 0
    corr_pov_vals, corr_graph_vals, corr_mix_vals = [], [], []

    # ---------------- Config ----------------
    cfg_train = config["training"]["cfg"]
    norm_graph = cfg_train.get("normalize_graph", True)
    norm_pov = cfg_train.get("normalize_pov", False)
    scale_graph = cfg_train.get("cond_scale_graph", 5.0)
    scale_pov = cfg_train.get("cond_scale_pov", 1.0)
    scale_mix = cfg_train.get("cond_scale_mix", 1.0)
    clip_value = cfg_train.get("cond_clip_value", None)

    pbar = tqdm(dataloader, desc="Validating")
    for batch in pbar:
        latents = batch["layout"].to(device)
        cond_pov = batch.get("pov")
        cond_graph = batch.get("graph")
        if cond_pov is not None:
            cond_pov = cond_pov.to(device)
        if cond_graph is not None:
            cond_graph = cond_graph.to(device)

        B = latents.size(0)
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)

        # --- Apply pretrained alignment (optional) ---
        if align_pov is not None and cond_pov is not None:
            cond_pov = align_pov(cond_pov).detach()
        if align_graph is not None and cond_graph is not None:
            cond_graph = align_graph(cond_graph).detach()

        # --- Normalize and scale ---
        if cond_pov is not None:
            if norm_pov:
                mean, std = cond_pov.mean(), cond_pov.std()
                cond_pov = (cond_pov - mean) / (std + 1e-5)
            cond_pov = cond_pov * scale_pov

        if cond_graph is not None:
            if norm_graph:
                mean, std = cond_graph.mean(), cond_graph.std()
                cond_graph = (cond_graph - mean) / (std + 1e-5)
            cond_graph = cond_graph * scale_graph

        # --- Optional clipping ---
        if clip_value is not None:
            if cond_pov is not None:
                cond_pov = torch.clamp(cond_pov, -clip_value, clip_value)
            if cond_graph is not None:
                cond_graph = torch.clamp(cond_graph, -clip_value, clip_value)

        # --- Fuse conditions ---
        cond = mixer([cond_pov, cond_graph]) * scale_mix

        # --- Correlation diagnostics ---
        corr_pov, corr_graph, corr_mix = compute_condition_correlations(
            cond_pov, cond_graph, cond, noisy_latents
        )
        corr_pov_vals.append(corr_pov)
        corr_graph_vals.append(corr_graph)
        corr_mix_vals.append(corr_mix)

        # --- Forward ---
        noise_pred = unet(noisy_latents, t, cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()
        pbar.set_postfix({'val_loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.mean(corr_pov_vals), np.mean(corr_graph_vals), np.mean(corr_mix_vals)

@torch.no_grad()
def validate_generation_quality(diffusion_model, mixer, dataloader, device, num_samples=10):
    """Validate by generating from scratch and comparing to ground truth"""
    diffusion_model.eval()
    total_metric = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        # Get conditions
        cond_pov = batch["pov"].to(device) if batch["pov"] is not None else None
        cond_graph = batch["graph"].to(device)
        ground_truth = batch["layout"].to(device)
        
        # Generate from pure noise
        conds = [c for c in [cond_pov, cond_graph] if c is not None]
        cond = mixer(conds)
        
        # Sample (this should use your sampling function)
        generated = diffusion_model.sample(cond=cond, batch_size=ground_truth.shape[0])
        
        # Compute generation metric (e.g., MSE, SSIM, etc.)
        def compute_generation_metric(pred, target):
            """Simple placeholder metric (MSE)."""
            return torch.mean((pred - target) ** 2).item()

        metric = compute_generation_metric(generated, ground_truth)
        total_metric += metric
    
    return total_metric / min(num_samples, len(dataloader))

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