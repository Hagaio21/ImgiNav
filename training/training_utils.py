"""
Training utilities for diffusion model training.
Shared functions for experiment management, checkpointing, and visualization.
"""

# --- Standard Library ---
import json
import shutil
import yaml
import random
import argparse
from pathlib import Path
from datetime import datetime

# --- Third-Party ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm


# my classes:
# --- Local Modules ---
try:
    # Import the actual schedulers from your file
    from modules.scheduler import (
        LinearScheduler,
        CosineScheduler,
        SquaredCosineScheduler,
        SigmoidScheduler,
        ExponentialScheduler,
        QuadraticScheduler
    )
    
    # This map safely links config strings to your actual classes
    SCHEDULER_MAP = {
        "LinearScheduler": LinearScheduler,
        "CosineScheduler": CosineScheduler,
        "SquaredCosineScheduler": SquaredCosineScheduler,
        "SigmoidScheduler": SigmoidScheduler,
        "ExponentialScheduler": ExponentialScheduler,
        "QuadraticScheduler": QuadraticScheduler,
    }
except ImportError:
    print("Warning: Could not import schedulers from 'modules.scheduler'.")
    SCHEDULER_MAP = {}

from modules.unet import UNet
from modules.autoencoder import AutoEncoder
from modules.unified_dataset import UnifiedLayoutDataset, collate_fn
from modules.condition_mixer import LinearConcatMixer, NonLinearConcatMixer
from modules.diffusion import LatentDiffusion
from modules.loss_diffusion import ConstructiveDiffusionLoss


def _copy_arch_configs(config: dict, exp_dir: Path):
    """Copies architecture configs (U-Net, AE) to the experiment config dir."""
    configs_save_dir = exp_dir / 'configs'
    configs_save_dir.mkdir(exist_ok=True) # Already created by setup_experiment_dir, but good to be safe

    # Copy U-Net config
    unet_config_path_str = config.get("unet", {}).get("config_path")
    if unet_config_path_str:
        unet_config_path = Path(unet_config_path_str)
        if unet_config_path.exists():
            shutil.copy(unet_config_path, configs_save_dir / unet_config_path.name)
        else:
            print(f"Warning: U-Net config not found at {unet_config_path_str}")

    # Copy Autoencoder config
    ae_config_path_str = config.get("autoencoder", {}).get("config_path")
    if ae_config_path_str:
        ae_config_path = Path(ae_config_path_str)
        if ae_config_path.exists():
            shutil.copy(ae_config_path, configs_save_dir / ae_config_path.name)
        else:
            print(f"Warning: Autoencoder config not found at {ae_config_path_str}")

def _create_experiment_readme(config: dict, main_config_path: str | Path, exp_dir: Path):
    """Creates a README.md file summarizing the experiment."""
    readme_path = exp_dir / "exp_readme.md"
    main_config_path = Path(main_config_path)

    def read_file_content(file_path_str: str | None) -> tuple[str, str]:
        """Reads config file content, returns name and content."""
        if not file_path_str:
            return "File not specified", "# Config path not found in experiment.yml\n"
        
        file_path = Path(file_path_str)
        if not file_path.exists():
            return file_path.name, f"# File not found at: {file_path_str}\n"
        
        with open(file_path, 'r') as f:
            return file_path.name, f.read()

    main_config_name, main_config_content = read_file_content(str(main_config_path))
    unet_config_name, unet_config_content = read_file_content(config.get("unet", {}).get("config_path"))
    ae_config_name, ae_config_content = read_file_content(config.get("autoencoder", {}).get("config_path"))

    with open(readme_path, 'w') as f:
        f.write(f"# {config.get('experiment', {}).get('name', 'Experiment')}\n\n")
        f.write(f"**Description:** {config.get('experiment', {}).get('description', 'No description.')}\n\n")
        f.write(f"**Experiment Directory:** `{exp_dir}`\n")
        f.write(f"**Start Time:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")
        
        f.write(f"## Main Experiment Config (`{main_config_name}`)\n\n")
        f.write(f"```yaml\n{main_config_content}\n```\n\n")
        
        f.write(f"## U-Net Config (`{unet_config_name}`)\n\n")
        f.write(f"```yaml\n{unet_config_content}\n```\n\n")
        
        f.write(f"## Autoencoder Config (`{ae_config_name}`)\n\n")
        f.write(f"```yaml\n{ae_config_content}\n```\n\n")

    print(f"Saved experiment README to {readme_path}", flush=True)


def setup_experiment(exp_config_path: str, resume_flag: bool) -> tuple[dict, Path, torch.device]:
    """Loads config, sets up directories, saves configs, sets device."""
    config = load_experiment_config(exp_config_path)
    exp_dir = setup_experiment_dir(
        config["experiment"]["exp_dir"],
        resume_flag
    )
    
    if not resume_flag:
        # Save all config files to the experiment directory for reproducibility
        save_experiment_config(config, exp_dir) # Saves main config as 'experiment_config.yaml'
        _copy_arch_configs(config, exp_dir) # Copies AE and U-Net configs
        _create_experiment_readme(config, exp_config_path, exp_dir) # Creates README.md

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    return config, exp_dir, device

def load_core_models(config: dict, device: torch.device) -> tuple[UNet, nn.Module, AutoEncoder, LatentDiffusion]:
    """Loads U-Net, Scheduler, Autoencoder, and creates Diffusion Wrapper."""
    # Load U-Net
    print(f"Loading U-Net from {config['unet']['config_path']}", flush=True)
    latent_channels = config["latent_shape"][0]
    unet = UNet.from_config(
        config["unet"]["config_path"],
        latent_channels=latent_channels
    ).to(device)

    # Load Scheduler
    scheduler_config = config['scheduler']
    scheduler_class_name = scheduler_config['type']
    num_steps = scheduler_config['num_steps']
    
    print(f"Loading {scheduler_class_name} with {num_steps} steps", flush=True)

    # Use the dictionary map for a safe lookup
    scheduler_class = SCHEDULER_MAP.get(scheduler_class_name)
    
    if scheduler_class is None:
        raise ValueError(
            f"Scheduler type '{scheduler_class_name}' not found in SCHEDULER_MAP. "
            f"Available schedulers: {list(SCHEDULER_MAP.keys())}"
        )
        
    scheduler = scheduler_class(num_steps=num_steps).to(device)

    # Load Autoencoder
    print(f"Loading Autoencoder from {config['autoencoder']['config_path']}", flush=True)
    autoencoder = AutoEncoder.from_config(config["autoencoder"]["config_path"]).to(device)
    ae_ckpt_path = config["autoencoder"].get("checkpoint_path")
    if ae_ckpt_path:
        print(f"  Loading Autoencoder checkpoint: {ae_ckpt_path}", flush=True)
        # WARNING: Changed weights_only=False to True for security.
        # State dicts should not contain arbitrary pickled code.
        autoencoder.load_state_dict(
            torch.load(ae_ckpt_path, map_location=device, weights_only=True)
        )
    autoencoder.eval()

    # Create LatentDiffusion wrapper
    latent_shape = tuple(config["latent_shape"])
    diffusion_model = LatentDiffusion(
        unet=unet,
        scheduler=scheduler,
        autoencoder=autoencoder,
        latent_shape=latent_shape
    ).to(device)

    return unet, scheduler, autoencoder, diffusion_model

def load_data(config: dict, exp_dir: Path) -> tuple[DataLoader, DataLoader, UnifiedLayoutDataset]:
    """Loads dataset, performs split, creates dataloaders, and saves split info."""
    print("Loading unified dataset...", flush=True)
    
    room_manifest = config["dataset"]["room_manifest"]
    scene_manifest = config["dataset"]["scene_manifest"]
    pov_type = config["dataset"].get("pov_type")
    data_mode = config["dataset"].get("data_mode", "rooms_and_scenes")
    print(f"Loading dataset in mode: {data_mode}", flush=True)

    dataset = UnifiedLayoutDataset(
        room_manifest,
        scene_manifest,
        use_embeddings=True,
        pov_type=pov_type,
        data_mode=data_mode
    )

    # Perform Train/Val Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # TODO: Use torch.Generator for reproducible splits
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    # Get dataloader params from config
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    print(f"Using batch_size={batch_size}, num_workers={num_workers}", flush=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Save split info (using original dataset df and split indices)
    train_df = dataset.df.iloc[train_dataset.indices]
    val_df = dataset.df.iloc[val_dataset.indices]
    save_split_files(exp_dir, train_df, val_df)  # Assuming save_split_files exists

    return train_loader, val_loader, val_dataset

def create_mixer(config: dict, device: torch.device) -> nn.Module:
    """Creates the appropriate mixer based on config and args."""
    latent_hw = tuple(config["latent_shape"][-2:])

    # Get UNet conditioning channels (needed for mixer output size)
    with open(config["unet"]["config_path"], 'r') as f:
        unet_cfg = yaml.safe_load(f)
    cond_channels = unet_cfg["unet"]["cond_channels"]

    # Get embedding input dimensions from config
    mixer_config = config.get("mixer", {})
    # Get enabled flags
    use_pov = mixer_config.get("use_pov", True)
    use_graph = mixer_config.get("use_graph", True)

    pov_dim = mixer_config.get("pov_dim") if use_pov else None
    graph_dim = mixer_config.get("graph_dim") if use_graph else None
    print(f"Input dimensions - POV: {pov_dim} (Enabled: {use_pov}), Graph: {graph_dim} (Enabled: {use_graph})", flush=True)

    # Determine mixer type name
    mixer_type_name = mixer_config.get("type", "LinearConcatMixer")
    norm_type = mixer_config.get("norm_type", None)
    
    print(f"Creating {mixer_type_name} (out_channels={cond_channels}, target_size={latent_hw})...", flush=True)

    # Map type name string to class
    mixer_class_map = {
        "LinearConcatMixer": LinearConcatMixer,
        "NonLinearConcatMixer": NonLinearConcatMixer,
    }
    mixer_class = mixer_class_map.get(mixer_type_name)

    if mixer_class is None:
        print(f"ERROR: Unknown mixer type '{mixer_type_name}'. Defaulting to LinearConcatMixer.")
        mixer_class = LinearConcatMixer
        mixer_type_name = "LinearConcatMixer"

    # Prepare core arguments
    mixer_args = {
        "out_channels": cond_channels,
        "target_size": latent_hw,
        "pov_channels": pov_dim,
        "graph_channels": graph_dim,
        "norm_type": norm_type
    }

    # Add specific arguments conditionally
    if mixer_class == NonLinearConcatMixer:
        mlp_hidden_dim = mixer_config.get("mlp_hidden_dim", None)
        if mlp_hidden_dim is not None:
            mixer_args["hidden_dim_mlp"] = mlp_hidden_dim
            print(f"  Using MLP hidden dimension: {mlp_hidden_dim}", flush=True)
        else:
            print(f"  Using default MLP hidden dimension.", flush=True)

    # Instantiate
    try:
        mixer = mixer_class(**mixer_args).to(device)
        print(f"Mixer '{mixer_type_name}' instantiated successfully.", flush=True)
        print(f"  Mixer parameters are trainable.", flush=True)
        return mixer
    except TypeError as e:
        print(f"ERROR: Failed to instantiate mixer '{mixer_type_name}'. Args: {mixer_args}.")
        raise e

def create_optimizer_scheduler(config: dict, unet: nn.Module, mixer: nn.Module) -> tuple[Adam, CosineAnnealingLR]:
    """Creates Adam optimizer and CosineAnnealingLR scheduler."""
    params_to_train = list(unet.parameters()) + list(mixer.parameters())
    optimizer = Adam(params_to_train, lr=config["training"]["learning_rate"])
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])
    print(f"Optimizer training {len(params_to_train)} parameter groups (UNet + Mixer).", flush=True)
    return optimizer, scheduler_lr

def prepare_fixed_samples(config: dict, val_dataset, exp_dir: Path) -> list:
    """Selects and saves fixed validation samples for visualization."""
    num_fixed = min(config["training"].get("num_samples", 8), len(val_dataset))
    fixed_indices = random.sample(range(len(val_dataset)), num_fixed)
    fixed_samples = [val_dataset[i] for i in fixed_indices]
    fixed_indices_path = exp_dir / "fixed_indices.pt"
    torch.save(fixed_indices, fixed_indices_path)
    print(f"Saved {num_fixed} fixed sample indices to {fixed_indices_path}", flush=True)
    return fixed_samples

def resume_training(args: argparse.Namespace, exp_dir: Path, unet: nn.Module, mixer: nn.Module, optimizer: Adam, scheduler_lr: CosineAnnealingLR) -> tuple[int, float, dict]:
    """Handles loading checkpoint if resuming."""
    start_epoch = 0
    best_loss = float("inf")
    training_stats = init_training_stats()  # Initialize stats here

    latest_checkpoint = exp_dir / 'checkpoints' / 'latest.pt'
    if args.resume and latest_checkpoint.exists():
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        start_epoch, best_loss, loaded_stats = load_checkpoint(
            latest_checkpoint,
            models_dict={'unet': unet, 'mixer': mixer},
            optimizer=optimizer,
            scheduler_lr=scheduler_lr
        )
        training_stats.update(loaded_stats)  # Update stats dict
        start_epoch += 1  # Start from the next epoch
    else:
        if args.resume:
            print(f"Warning: --resume flag set but checkpoint not found at {latest_checkpoint}. Starting from scratch.")
        # If not resuming or checkpoint not found, stats remain initialized
    return start_epoch, best_loss, training_stats

def load_experiment_config(config_path: str | Path) -> dict:
    """Load experiment config from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_experiment_config(config: dict, exp_dir: str | Path):
    """Save experiment config to experiment directory"""
    exp_dir = Path(exp_dir)
    config_path = exp_dir / 'experiment_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved experiment config to {config_path}", flush=True)

def setup_experiment_dir(exp_dir: str | Path, resume: bool = False) -> Path:
    """
    Setup experiment directory structure.
    If resuming and dir exists, use existing.
    Otherwise, create new structure.
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
    
    # [CHANGED] Removed config copying, as it's now handled in setup_experiment

    print(f"Created experiment directory: {exp_dir}", flush=True)
    return exp_dir

def save_checkpoint(exp_dir: str | Path, epoch: int, state_dict: dict, training_stats: dict,
                    val_loss: float, best_loss: float, is_best: bool = False, save_periodic: bool = False):
    """
    Save checkpoints (latest, best, and optional periodic).
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

    # Save periodic lightweight AND SEPARATE checkpoints (model only)
    if save_periodic:
        # --- 1. Save the lightweight combined checkpoint ---
        periodic_checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            **{k: v for k, v in state_dict.items() if k in ['unet', 'mixer']}
        }
        periodic_path = ckpt_dir / f'epoch_{epoch+1}.pt'
        torch.save(periodic_checkpoint, periodic_path)
        print(f"Saved periodic combined checkpoint to {periodic_path}", flush=True)

        # --- 2. Save separate model-only files ---
        unet_path = ckpt_dir / f'unet_epoch_{epoch+1}.pt'
        mixer_path = ckpt_dir / f'mixer_epoch_{epoch+1}.pt'

        torch.save({'epoch': epoch, 'val_loss': val_loss, 'unet': state_dict['unet']}, unet_path)
        torch.save({'epoch': epoch, 'val_loss': val_loss, 'mixer': state_dict['mixer']}, mixer_path)
        print(f"Saved periodic separate models to {unet_path} and {mixer_path}", flush=True)

def load_checkpoint(checkpoint_path: str | Path, models_dict: dict[str, nn.Module],
                    optimizer: Adam | None = None, scheduler_lr: CosineAnnealingLR | None = None) -> tuple[int, float, dict]:
    """
    Load checkpoint and restore model states.
    """
    # WARNING: Changed weights_only=False to True for security.
    # State dicts should not contain arbitrary pickled code.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model states
    for name, model in models_dict.items():
        if name in checkpoint:
            model.load_state_dict(checkpoint[name])
        elif f'{name}_state_dict' in checkpoint:
            model.load_state_dict(checkpoint[f'{name}_state_dict'])

    # Load optimizer state
    if optimizer:
        if 'opt' in checkpoint:
            optimizer.load_state_dict(checkpoint['opt'])
        elif 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler_lr:
        if 'sched' in checkpoint and checkpoint['sched'] is not None:
            scheduler_lr.load_state_dict(checkpoint['sched'])
        elif 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler_lr.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float('inf'))
    training_stats = checkpoint.get('training_stats', {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'learning_rate': [], 'timestamps': []
    })

    print(f"Loaded checkpoint from epoch {epoch}, best_loss: {best_loss:.6f}", flush=True)
    return epoch, best_loss, training_stats

def init_training_stats() -> dict:
    """Initialize an empty training statistics dictionary."""
    return {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'learning_rate': [], 'timestamps': [],
        'train_corr_pov': [], 'train_corr_graph': [], # Train (RENAMED)
        'val_corr_pov': [], 'val_corr_graph': [], # Val
        'cond_std_pov': [], 'cond_std_graph': [],
        'dropout_ratio_pov': [], 'dropout_ratio_graph': [],
        'val_corr_pov_zero': [], 'val_corr_graph_zero': [],
        'val_corr_pov_shuffle': [], 'val_corr_graph_shuffle': [],
        'val_mse_pov_zero': [], 'val_mse_graph_zero': [],
        'val_mse_pov_shuffle': [], 'val_mse_graph_shuffle': []
    }

def update_training_stats(
    training_stats: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    train_corr_pov: float | None = None,
    train_corr_graph: float | None = None,
    val_corr_pov: float | None = None,
    val_corr_graph: float | None = None,
    cond_std_pov: float | None = None,
    cond_std_graph: float | None = None,
    dropout_ratio_pov: float | None = None,
    dropout_ratio_graph: float | None = None,
    val_corr_pov_zero: float | None = None,
    val_corr_graph_zero: float | None = None,
    val_corr_pov_shuffle: float | None = None,
    val_corr_graph_shuffle: float | None = None,
    val_mse_pov_zero: float | None = None,
    val_mse_graph_zero: float | None = None,
    val_mse_pov_shuffle: float | None = None,
    val_mse_graph_shuffle: float | None = None,):
    """Update training statistics with current epoch data and diagnostics."""
    training_stats['epochs'].append(epoch + 1)
    training_stats['train_loss'].append(train_loss)
    training_stats['val_loss'].append(val_loss)
    training_stats['learning_rate'].append(learning_rate)
    training_stats['timestamps'].append(datetime.now().isoformat())

    training_stats['train_corr_pov'].append(train_corr_pov)
    training_stats['train_corr_graph'].append(train_corr_graph)
    training_stats['val_corr_pov'].append(val_corr_pov)
    training_stats['val_corr_graph'].append(val_corr_graph)
    training_stats['cond_std_pov'].append(cond_std_pov)
    training_stats['cond_std_graph'].append(cond_std_graph)
    training_stats['dropout_ratio_pov'].append(dropout_ratio_pov)
    training_stats['dropout_ratio_graph'].append(dropout_ratio_graph)

    # --- Diagnostic metrics ---
    training_stats['val_corr_pov_zero'].append(val_corr_pov_zero)
    training_stats['val_corr_graph_zero'].append(val_corr_graph_zero)
    training_stats['val_corr_pov_shuffle'].append(val_corr_pov_shuffle)
    training_stats['val_corr_graph_shuffle'].append(val_corr_graph_shuffle)
    training_stats['val_mse_pov_zero'].append(val_mse_pov_zero)
    training_stats['val_mse_graph_zero'].append(val_mse_graph_zero)
    training_stats['val_mse_pov_shuffle'].append(val_mse_pov_shuffle)
    training_stats['val_mse_graph_shuffle'].append(val_mse_graph_shuffle)

    return training_stats

def save_training_stats(exp_dir: str | Path, training_stats: dict):
    """Save training statistics to JSON and generate plots"""
    exp_dir = Path(exp_dir)
    stats_path = exp_dir / 'logs' / 'training_stats.json'

    # Save JSON
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        # --- Main training curves (Loss, LR, Correlation) ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # Loss Plot
        if training_stats.get('train_loss'):
            axes[0].plot(training_stats['epochs'], training_stats['train_loss'], label='Train', marker='o', ms=3)
        if training_stats.get('val_loss'):
            axes[0].plot(training_stats['epochs'], training_stats['val_loss'], label='Val', marker='s', ms=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Train / Val Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning Rate Plot
        if training_stats.get('learning_rate'):
            axes[1].plot(training_stats['epochs'], training_stats['learning_rate'], color='orange', marker='o', ms=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)

        # Correlation Plot
        has_corr = any(v for k in ['train_corr_pov', 'train_corr_graph', 'val_corr_pov', 'val_corr_graph'] # (RENAMED)
                       for v in training_stats.get(k, []) if v is not None)
        if has_corr:
            if training_stats.get('train_corr_pov'): # (RENAMED)
                axes[2].plot(training_stats['epochs'], training_stats['train_corr_pov'], label='Train POV', marker='o', ms=3, linestyle='-', alpha=0.8) # (RENAMED)
            if training_stats.get('train_corr_graph'): # (RENAMED)
                axes[2].plot(training_stats['epochs'], training_stats['train_corr_graph'], label='Train Graph', marker='s', ms=3, linestyle='-', alpha=0.8) # (RENAMED)
        
            if training_stats.get('val_corr_pov'):
                axes[2].plot(training_stats['epochs'], training_stats['val_corr_pov'], label='Val POV', marker='o', ms=3, linestyle='--')
            if training_stats.get('val_corr_graph'):
                axes[2].plot(training_stats['epochs'], training_stats['val_corr_graph'], label='Val Graph', marker='s', ms=3, linestyle='--')

            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Correlation')
            axes[2].set_title('Conditioning Correlation')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = exp_dir / 'logs' / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved training plots to {plot_path}", flush=True)

        # --- Additional visual diagnostics ---

        # Condition Signal Strength
        has_std = any(v for k in ['cond_std_pov', 'cond_std_graph'] for v in training_stats.get(k, []) if v is not None)
        if has_std:
            fig_std, ax_std = plt.subplots(figsize=(6, 4))
            if training_stats.get('cond_std_pov'):
                ax_std.plot(training_stats['epochs'], training_stats['cond_std_pov'], label='POV σ', marker='o', ms=3)
            if training_stats.get('cond_std_graph'):
                ax_std.plot(training_stats['epochs'], training_stats['cond_std_graph'], label='Graph σ', marker='s', ms=3)
            ax_std.set_xlabel('Epoch')
            ax_std.set_ylabel('Std')
            ax_std.set_title('Condition Signal Strength')
            ax_std.legend()
            ax_std.grid(True, alpha=0.3)
            plt.tight_layout()
            sig_path = exp_dir / 'logs' / 'cond_signal_strength.png'
            plt.savefig(sig_path, dpi=150, bbox_inches='tight')
            plt.close(fig_std)
            print(f"Saved condition signal plot to {sig_path}", flush=True)

        # Dropout Ratio Plot
        has_dropout = any(v for k in ['dropout_ratio_pov', 'dropout_ratio_graph'] for v in training_stats.get(k, []) if v is not None)
        if has_dropout:
            fig_drop, ax_drop = plt.subplots(figsize=(6, 4))
            if training_stats.get('dropout_ratio_pov'):
                ax_drop.plot(training_stats['epochs'], training_stats['dropout_ratio_pov'], label='POV Dropout', marker='o', ms=3)
            if training_stats.get('dropout_ratio_graph'):
                ax_drop.plot(training_stats['epochs'], training_stats['dropout_ratio_graph'], label='Graph Dropout', marker='s', ms=3)
            ax_drop.set_xlabel('Epoch')
            ax_drop.set_ylabel('Dropout Ratio')
            ax_drop.set_title('Condition Dropout Frequency')
            ax_drop.legend()
            ax_drop.grid(True, alpha=0.3)
            plt.tight_layout()
            drop_path = exp_dir / 'logs' / 'cond_dropout.png'
            plt.savefig(drop_path, dpi=150, bbox_inches='tight')
            plt.close(fig_drop)
            print(f"Saved condition dropout plot to {drop_path}", flush=True)

        if 'val_corr_graph_zero' in training_stats:
            fig_d, ax_d = plt.subplots(figsize=(8,4))
            ax_d.plot(training_stats['epochs'], training_stats['val_corr_graph_zero'], label='Graph Zero', marker='o')
            ax_d.plot(training_stats['epochs'], training_stats['val_corr_graph_shuffle'], label='Graph Shuffle', marker='s')
            ax_d.plot(training_stats['epochs'], training_stats['val_corr_pov_zero'], label='POV Zero', marker='^')
            ax_d.plot(training_stats['epochs'], training_stats['val_corr_pov_shuffle'], label='POV Shuffle', marker='v')
            ax_d.set_xlabel('Epoch')
            ax_d.set_ylabel('Correlation')
            ax_d.set_title('Conditioning Diagnostic Correlations')
            ax_d.legend()
            ax_d.grid(True, alpha=0.3)
            plt.tight_layout()
            diag_path = exp_dir / 'logs' / 'diagnostic_correlations.png'
            plt.savefig(diag_path, dpi=150, bbox_inches='tight')
            plt.close(fig_d)


    except ImportError:
        print("Matplotlib not available, skipping plots", flush=True)
        
@torch.no_grad()
def compute_condition_correlations(proj_pov: torch.Tensor, proj_graph: torch.Tensor, noisy_latents: torch.Tensor) -> tuple[float | None, float | None]:
    """
    Compute cosine correlations between the noisy latent and each *already projected* conditioning source.
    
    This version is mixer-agnostic and "ablation-aware".
    If a projected tensor is all zeros (std dev is ~0), it means the condition was
    None (ablated). In this case, we return None for the metric, rather than
    computing and getting a NaN (which would be incorrectly logged as 0.0).
    """
    corr_pov_vals, corr_graph_vals = [], []
    with torch.no_grad():
        # Global average of the noisy latent tensor
        l = noisy_latents.mean(dim=[2, 3])  # Shape: [B, C_lat]

        # --- POV Correlation ---
        # Check if channels > 0 AND if the tensor is not all zeros (std > 1e-6)
        if proj_pov.shape[1] > 0 and proj_pov.std() > 1e-6:
            c_pov = proj_pov.mean(dim=[2, 3])  # Shape: [B, C_pov_out]
            
            min_ch_pov = min(c_pov.size(1), l.size(1))
            if min_ch_pov > 0:
                corr_p = torch.cosine_similarity(c_pov[:, :min_ch_pov], l[:, :min_ch_pov], dim=1)
                # We can safely use nan_to_num here, as any remaining NaNs 
                # would be from the latent, not from a zero-norm condition.
                corr_pov_vals.append(torch.nan_to_num(corr_p, nan=0.0).mean().item())

        # --- Graph Correlation ---
        # Check if channels > 0 AND if the tensor is not all zeros (std > 1e-6)
        if proj_graph.shape[1] > 0 and proj_graph.std() > 1e-6:
            c_graph = proj_graph.mean(dim=[2, 3])  # Shape: [B, C_graph_out]

            min_ch_graph = min(c_graph.size(1), l.size(1))
            if min_ch_graph > 0:
                corr_g = torch.cosine_similarity(c_graph[:, :min_ch_graph], l[:, :min_ch_graph], dim=1)
                corr_graph_vals.append(torch.nan_to_num(corr_g, nan=0.0).mean().item())

    # If the list is empty (because the condition was ablated), return None.
    # This will be logged as 'null' and create a gap in the plot.
    return (
        np.mean(corr_pov_vals) if corr_pov_vals else None,
        np.mean(corr_graph_vals) if corr_graph_vals else None,
    )


def train_epoch_unconditioned(unet: nn.Module, scheduler, dataloader: DataLoader,
                            optimizer: Adam, device: torch.device, epoch: int) -> float:
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


def _prepare_conditions(cond_pov: torch.Tensor | None, cond_graph: torch.Tensor | None,
                        cfg_train: dict) -> tuple[torch.Tensor | None, torch.Tensor | None, float, float]:
    """Helper to normalize, scale, and clip conditioning tensors."""
    norm_graph = cfg_train.get("normalize_graph", True)
    norm_pov = cfg_train.get("normalize_pov", False)
    scale_graph = cfg_train.get("cond_scale_graph", 5.0)
    scale_pov = cfg_train.get("cond_scale_pov", 1.0)
    clip_value = cfg_train.get("cond_clip_value", None)
    
    cond_std_pov = 0.0
    cond_std_graph = 0.0

    if cond_pov is not None:
        if norm_pov:
            mean, std = cond_pov.mean(), cond_pov.std()
            cond_pov = (cond_pov - mean) / (std + 1e-5)
        cond_pov = cond_pov * scale_pov
        if clip_value is not None:
            cond_pov = torch.clamp(cond_pov, -clip_value, clip_value)
        cond_std_pov = cond_pov.std().item()

    if cond_graph is not None:
        if norm_graph:
            mean, std = cond_graph.mean(), cond_graph.std()
            cond_graph = (cond_graph - mean) / (std + 1e-5)
        cond_graph = cond_graph * scale_graph
        if clip_value is not None:
            cond_graph = torch.clamp(cond_graph, -clip_value, clip_value)
        cond_std_graph = cond_graph.std().item()

    return cond_pov, cond_graph, cond_std_pov, cond_std_graph


def train_epoch_conditioned(
    unet: nn.Module,
    scheduler,
    mixer: nn.Module,
    dataloader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    epoch: int,
    config: dict,
    cfg_dropout_prob: float = 0.1,
    loss_fn: nn.Module = None) -> tuple[float, float, float, float, float, float, float]:

    """Train for one epoch with conditioning."""
    unet.train()
    total_loss = 0
    corr_pov_vals, corr_graph_vals = [], []
    cond_std_pov_vals, cond_std_graph_vals = [], []
    dropout_pov_events, dropout_graph_events, total_batches = 0, 0, 0

    # ---------------- Config ----------------
    cfg_train = config["training"]["cfg"]
    scale_mix = cfg_train.get("cond_scale_mix", 1.0)
    log_cond_stats = cfg_train.get("log_condition_stats", True)
    corr_every = cfg_train.get("compute_corr_every", 300)

    use_pov = config.get("mixer", {}).get("use_pov", True)
    use_graph = config.get("mixer", {}).get("use_graph", True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for i, batch in enumerate(pbar):
        total_batches += 1

        # --- Load and move tensors ---
        latents = batch["layout"].to(device)
        cond_pov_raw = batch.get("pov") if use_pov else None 
        cond_graph_raw = batch.get("graph") if use_graph else None
        if cond_pov_raw is not None:
            cond_pov_raw = cond_pov_raw.to(device)
        if cond_graph_raw is not None:
            cond_graph_raw = cond_graph_raw.to(device)
        
        B = latents.size(0)
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)

        # --- Classifier-free dropout ---
        cond_pov, cond_graph = cond_pov_raw, cond_graph_raw
        if torch.rand(1).item() < cfg_dropout_prob:
            if cond_pov is not None:
                cond_pov = torch.zeros_like(cond_pov)
                dropout_pov_events += 1
            if cond_graph is not None:
                cond_graph = torch.zeros_like(cond_graph)
                dropout_graph_events += 1

        # --- Normalize, scale, clip ---
        cond_pov, cond_graph, cond_std_pov, cond_std_graph = _prepare_conditions(
            cond_pov, cond_graph, cfg_train
        )
        cond_std_pov_vals.append(cond_std_pov)
        cond_std_graph_vals.append(cond_std_graph)

        # --- Fuse conditions ---
        cond = mixer([cond_pov, cond_graph]) * scale_mix
        cond_proj_pov, cond_proj_graph = mixer.get_projected_conditions(cond_pov_raw, cond_graph_raw)

        # --- Correlation diagnostics ---
        corr_pov, corr_graph = compute_condition_correlations(
            cond_proj_pov, cond_proj_graph, noisy_latents
        )
        corr_pov_vals.append(corr_pov)
        corr_graph_vals.append(corr_graph)

        # --- Forward + Backward ---
        noise_pred = unet(noisy_latents, t, cond=cond)
        alpha_bar_t = scheduler.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-5, max=0.999)
        x0_hat = (noisy_latents - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        x0_hat = torch.nan_to_num(x0_hat, nan=0.0, posinf=1.0, neginf=-1.0)


        if torch.isnan(noisy_latents).any():
            print(f"[epoch {epoch+1} | batch {i}] NaN in noisy_latents")
        if torch.isnan(cond).any():
            print(f"[epoch {epoch+1} | batch {i}] NaN in cond (mixer output)")
        if torch.isnan(noise_pred).any():
            print(f"[epoch {epoch+1} | batch {i}] NaN in noise_pred before loss")
        if torch.isnan(cond).any():
            print(f"[epoch {epoch+1} | batch {i}] NaN in noise target")


        if loss_fn is not None:
            loss, loss_logs = loss_fn(noise_pred, noise, x0_hat, cond)
        else:
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
    valid_corr_pov = [v for v in corr_pov_vals if v is not None]
    valid_corr_graph = [v for v in corr_graph_vals if v is not None]

    avg_loss = total_loss / len(dataloader)

    return (
        avg_loss,
        np.mean(valid_corr_pov) if valid_corr_pov else 0.0,
        np.mean(valid_corr_graph) if valid_corr_graph else 0.0,
        np.mean(cond_std_pov_vals) if cond_std_pov_vals else 0.0,
        np.mean(cond_std_graph_vals) if cond_std_graph_vals else 0.0,
        (dropout_pov_events / total_batches) if total_batches > 0 else 0.0,
        (dropout_graph_events / total_batches) if total_batches > 0 else 0.0,
    )


@torch.no_grad()
def validate_unconditioned(unet: nn.Module, scheduler, dataloader: DataLoader, device: torch.device) -> float:
    """Compute validation loss without conditioning"""
    unet.eval()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
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
    unet: nn.Module,
    scheduler,
    mixer: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict) -> tuple[float, float, float]:
    """Compute validation loss with conditioning."""
    unet.eval()
    total_loss = 0
    corr_pov_vals, corr_graph_vals = [], []

    # ---------------- Config ----------------
    cfg_train = config["training"]["cfg"]
    scale_mix = cfg_train.get("cond_scale_mix", 1.0)
    # Get enabled flags ONCE
    use_pov = config.get("mixer", {}).get("use_pov", True)
    use_graph = config.get("mixer", {}).get("use_graph", True)

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        latents = batch["layout"].to(device)
        cond_pov = batch.get("pov") if use_pov else None 
        cond_graph = batch.get("graph") if use_graph else None
        if cond_pov is not None:
            cond_pov = cond_pov.to(device)
        if cond_graph is not None:
            cond_graph = cond_graph.to(device)
        
        B = latents.size(0)
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(latents)
        noisy_latents, _ = scheduler.add_noise(latents, t, noise)

        # --- Normalize, scale, clip ---
        cond_pov, cond_graph, _, _ = _prepare_conditions(
            cond_pov, cond_graph, cfg_train
        )

        # --- Fuse conditions ---
        cond = mixer([cond_pov, cond_graph]) * scale_mix
        
        cond_proj_pov, cond_proj_graph = mixer.get_projected_conditions(cond_pov, cond_graph)
        # --- Correlation diagnostics ---
        corr_pov, corr_graph = compute_condition_correlations(
            cond_proj_pov, cond_proj_graph, noisy_latents
        )
        corr_pov_vals.append(corr_pov)
        corr_graph_vals.append(corr_graph)

        # --- Forward ---
        noise_pred = unet(noisy_latents, t, cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()
        pbar.set_postfix({'val_loss': loss.item()})

    unet.train()
    avg_loss = total_loss / len(dataloader)
    valid_corr_pov = [v for v in corr_pov_vals if v is not None]
    valid_corr_graph = [v for v in corr_graph_vals if v is not None]

    return (
        avg_loss, 
        np.mean(valid_corr_pov) if valid_corr_pov else 0.0, 
        np.mean(valid_corr_graph) if valid_corr_graph else 0.0
    )


def _compute_generation_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simple placeholder metric (MSE)."""
    return torch.mean((pred - target) ** 2).item()

@torch.no_grad()
def validate_generation_quality(diffusion_model: LatentDiffusion, mixer: nn.Module, dataloader: DataLoader,
                                device: torch.device, num_samples: int = 10) -> float:
    """Validate by generating from scratch and comparing to ground truth"""
    diffusion_model.eval()
    total_metric = 0
    
    samples_processed = 0
    for i, batch in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
            
        # Get conditions
        cond_pov = batch["pov"].to(device) if batch["pov"] is not None else None
        cond_graph = batch["graph"].to(device) if batch["graph"] is not None else None
        ground_truth = batch["layout"].to(device)
        
        # Fuse conditions
        conds = [c for c in [cond_pov, cond_graph] if c is not None]
        cond = mixer(conds)
        
        # Sample
        generated = diffusion_model.sample(cond=cond, batch_size=ground_truth.shape[0])
        
        # Compute generation metric
        metric = _compute_generation_mse(generated, ground_truth)
        total_metric += metric
        samples_processed += 1
    
    diffusion_model.train()
    return total_metric / max(1, samples_processed)

@torch.no_grad()
def diagnostic_condition_tests(
    unet: nn.Module,
    scheduler,
    mixer: nn.Module,
    diffusion_model: LatentDiffusion,
    dataloader: DataLoader,
    device: torch.device,
    config: dict) -> dict:
    """
    Run per-epoch diagnostics for conditioning integrity:
      1. Zero-conditioning test.
      2. Cross-pair (shuffled) conditioning test.
    Logs correlation and MSE for POV and graph conditions.
    """
    results = {
        'val_corr_pov_zero': 0.0, 'val_corr_graph_zero': 0.0,
        'val_corr_pov_shuffle': 0.0, 'val_corr_graph_shuffle': 0.0,
        'val_mse_pov_zero': 0.0, 'val_mse_graph_zero': 0.0,
        'val_mse_pov_shuffle': 0.0, 'val_mse_graph_shuffle': 0.0,
    }

    cfg_train = config['training']['cfg']
    scale_mix = cfg_train.get('cond_scale_mix', 1.0)
    use_pov = config.get('mixer', {}).get('use_pov', True)
    use_graph = config.get('mixer', {}).get('use_graph', True)

    corr_zero_pov, corr_zero_graph, corr_shuf_pov, corr_shuf_graph = [], [], [], []
    mse_zero_pov, mse_zero_graph, mse_shuf_pov, mse_shuf_graph = [], [], [], []

    for batch in tqdm(dataloader, desc="Diagnostics", leave=False):
        gt = batch["layout"].to(device)
        cond_pov = batch.get("pov").to(device) if use_pov and batch.get("pov") is not None else None
        cond_graph = batch.get("graph").to(device) if use_graph and batch.get("graph") is not None else None

        B = gt.size(0)
        t = torch.randint(0, scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(gt)
        noisy_latents, _ = scheduler.add_noise(gt, t, noise)

        # --- Zero-conditioning test ---
        pov_zero = torch.zeros_like(cond_pov) if cond_pov is not None else None
        graph_zero = torch.zeros_like(cond_graph) if cond_graph is not None else None
        cond_zero = mixer([pov_zero, graph_zero]) * scale_mix
        noise_pred_zero = unet(noisy_latents, t, cond=cond_zero)
        mse_zero = F.mse_loss(noise_pred_zero, noise).item()
        cond_proj_pov, cond_proj_graph = mixer.get_projected_conditions(pov_zero, graph_zero)
        corr_zp, corr_zg = compute_condition_correlations(cond_proj_pov, cond_proj_graph, noisy_latents)
        if corr_zp is not None: corr_zero_pov.append(corr_zp)
        if corr_zg is not None: corr_zero_graph.append(corr_zg)
        mse_zero_pov.append(mse_zero)
        mse_zero_graph.append(mse_zero)

        # --- Cross-pair shuffle test ---
        if cond_pov is not None:
            idx_shuf = torch.randperm(B, device=device)
            cond_pov_shuf = cond_pov[idx_shuf]
        else:
            cond_pov_shuf = None
        if cond_graph is not None:
            idx_shuf = torch.randperm(B, device=device)
            cond_graph_shuf = cond_graph[idx_shuf]
        else:
            cond_graph_shuf = None

        cond_shuf = mixer([cond_pov_shuf, cond_graph_shuf]) * scale_mix
        noise_pred_shuf = unet(noisy_latents, t, cond=cond_shuf)
        mse_shuf = F.mse_loss(noise_pred_shuf, noise).item()
        cond_proj_pov, cond_proj_graph = mixer.get_projected_conditions(cond_pov_shuf, cond_graph_shuf)
        corr_sp, corr_sg = compute_condition_correlations(cond_proj_pov, cond_proj_graph, noisy_latents)
        if corr_sp is not None: corr_shuf_pov.append(corr_sp)
        if corr_sg is not None: corr_shuf_graph.append(corr_sg)
        mse_shuf_pov.append(mse_shuf)
        mse_shuf_graph.append(mse_shuf)

    # Aggregate
    results['val_corr_pov_zero'] = float(np.mean(corr_zero_pov)) if corr_zero_pov else 0.0
    results['val_corr_graph_zero'] = float(np.mean(corr_zero_graph)) if corr_zero_graph else 0.0
    results['val_corr_pov_shuffle'] = float(np.mean(corr_shuf_pov)) if corr_shuf_pov else 0.0
    results['val_corr_graph_shuffle'] = float(np.mean(corr_shuf_graph)) if corr_shuf_graph else 0.0
    results['val_mse_pov_zero'] = float(np.mean(mse_zero_pov)) if mse_zero_pov else 0.0
    results['val_mse_graph_zero'] = float(np.mean(mse_zero_graph)) if mse_zero_graph else 0.0
    results['val_mse_pov_shuffle'] = float(np.mean(mse_shuf_pov)) if mse_shuf_pov else 0.0
    results['val_mse_graph_shuffle'] = float(np.mean(mse_shuf_graph)) if mse_shuf_graph else 0.0

    print(f"[Diagnostics] Zero corr POV={results['val_corr_pov_zero']:.3f}, Graph={results['val_corr_graph_zero']:.3f} | "
          f"Shuf corr POV={results['val_corr_pov_shuffle']:.3f}, Graph={results['val_corr_graph_shuffle']:.3f}", flush=True)

    return results

def save_split_files(exp_dir: str | Path, train_df, val_df):
    """Save train/val split information to CSV files"""
    exp_dir = Path(exp_dir)
    train_file = exp_dir / "trained_on.csv"
    val_file = exp_dir / "evaluated_on.csv"
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    print(f"Saved split files: {train_file}, {val_file}", flush=True)