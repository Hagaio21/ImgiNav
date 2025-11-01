"""
Training utility functions.
"""
import torch
import random
import numpy as np
import yaml
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from models.losses.base_loss import LOSS_REGISTRY


def set_deterministic(seed: int = 42, strict_determinism: bool = False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDNN deterministic settings (applies to most operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Strict deterministic algorithms (may warn for operations without deterministic impl)
    if strict_determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)


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
    dataset = ManifestDataset(**ds_cfg)
    return dataset


def build_loss(config):
    """Build loss function from config."""
    loss_cfg = config["training"]["loss"]
    loss_fn = LOSS_REGISTRY[loss_cfg["type"]].from_config(loss_cfg)
    return loss_fn


def build_optimizer(model, config):
    """Build optimizer from config using model's parameter_groups for trainable params."""
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    optimizer_type = config["training"].get("optimizer", "AdamW").lower()
    
    # Get parameter groups from model (respects frozen params and per-module LRs)
    param_groups = model.parameter_groups()
    
    # If no groups returned (all frozen or no per-module LRs), use trainable params with base LR
    if not param_groups:
        trainable_params = list(model.trainable_parameters())
        if not trainable_params:
            raise ValueError("No trainable parameters found in model!")
        param_groups = [{"params": trainable_params, "lr": lr}]
    else:
        # Set default LR for groups that don't have one specified
        for group in param_groups:
            if "lr" not in group:
                group["lr"] = lr
    
    # Add weight_decay to all groups
    for group in param_groups:
        group["weight_decay"] = weight_decay
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_type == "sgd":
        for group in param_groups:
            group["momentum"] = 0.9
        optimizer = torch.optim.SGD(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def get_device(config):
    """Get device from config or default."""
    device = config.get("training", {}).get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_scheduler(optimizer, config):
    """Build learning rate scheduler from config."""
    training_cfg = config.get("training", {})
    
    scheduler_type = training_cfg.get("scheduler", {}).get("type", None)
    if scheduler_type is None:
        return None
    
    epochs_target = config.get("experiment", {}).get("epochs_target", training_cfg.get("epochs", 100))
    
    if scheduler_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_target)
    elif scheduler_type.lower() == "linear":
        start_factor = training_cfg.get("scheduler", {}).get("start_factor", 1.0)
        end_factor = training_cfg.get("scheduler", {}).get("end_factor", 0.0)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=epochs_target)
    elif scheduler_type.lower() == "step":
        step_size = training_cfg.get("scheduler", {}).get("step_size", epochs_target // 3)
        gamma = training_cfg.get("scheduler", {}).get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        print(f"  Warning: Unknown scheduler type '{scheduler_type}', not using scheduler")
        return None
    
    return scheduler

