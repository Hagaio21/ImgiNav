"""
Training utility functions for building components and managing experiments.
"""
import os
import yaml
import torch
from pathlib import Path
from common.utils import safe_mkdir, write_json
from common.taxonomy import load_valid_colors


def build_optimizer(model, training_cfg):
    """
    Build optimizer from training config.
    
    Args:
        model: Model to optimize
        training_cfg: Training configuration dictionary
        
    Returns:
        Optimizer instance
    """
    opt_cfg = training_cfg.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    lr = opt_cfg.get("lr", training_cfg.get("lr", 1e-4))
    
    # Get trainable parameters (for diffusion models, only train backbone)
    params = model.parameters()
    if hasattr(model, "backbone") and hasattr(model, "autoencoder"):
        params = model.backbone.parameters()
    
    if opt_type == "adam":
        weight_decay = opt_cfg.get("weight_decay", 0.0)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "adamw":
        weight_decay = opt_cfg.get("weight_decay", 0.01)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        weight_decay = opt_cfg.get("weight_decay", 0.0)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def build_loss_function(loss_cfg):
    """
    Build loss function from config.
    
    Generic loss builder that instantiates loss classes based on config.
    
    Args:
        loss_cfg: Configuration dictionary with 'type' and loss-specific parameters
        
    Returns:
        Loss function instance
    """
    from models.losses.custom_loss import (
        StandardVAELoss, SegmentationVAELoss, DiffusionLoss, VGGPerceptualLoss
    )
    
    # Map loss type strings to loss classes
    loss_classes = {
        "standard": StandardVAELoss,
        "vae": StandardVAELoss,  # alias
        "segmentation": SegmentationVAELoss,
        "mse": DiffusionLoss,
        "diffusion": DiffusionLoss,
    }
    
    loss_type = loss_cfg.get("type", "standard").lower()
    
    if loss_type not in loss_classes:
        available = ", ".join(loss_classes.keys())
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: {available}")
    
    LossClass = loss_classes[loss_type]
    
    # Special handling for SegmentationVAELoss (needs taxonomy loading)
    if loss_type == "segmentation":
        taxonomy_path = loss_cfg.get("taxonomy_path")
        if not taxonomy_path:
            raise ValueError("[Loss] 'taxonomy_path' must be provided for segmentation loss")
        
        include_bg = loss_cfg.get("include_background", True)
        id_to_color, valid_ids = load_valid_colors(taxonomy_path, include_background=include_bg)
        
        print(f"[Loss] Using SegmentationVAELoss with {len(valid_ids)} class IDs")
        
        return LossClass(
            id_to_color=id_to_color,
            kl_weight=loss_cfg.get("kl_weight", 1e-6),
            lambda_seg=loss_cfg.get("lambda_seg", 1.0),
            lambda_mse=loss_cfg.get("lambda_mse", 1.0),
        )
    
    # Special handling for DiffusionLoss (may need VGG instance)
    elif loss_type in ("mse", "diffusion"):
        lambda_vgg = loss_cfg.get("lambda_vgg", 0.0)
        vgg_loss_fn = VGGPerceptualLoss() if lambda_vgg > 0 else None
        
        return LossClass(
            lambda_mse=loss_cfg.get("lambda_mse", 1.0),
            lambda_vgg=lambda_vgg,
            vgg_loss_fn=vgg_loss_fn,
        )
    
    # Generic handling for other losses - pass all config params except 'type'
    else:
        params = {k: v for k, v in loss_cfg.items() if k != "type"}
        print(f"[Loss] Using {LossClass.__name__} with params: {params}")
        return LossClass(**params)


def setup_experiment_directories(output_dir, ckpt_dir=None):
    """Setup output and checkpoint directories."""
    if ckpt_dir is None:
        ckpt_dir = os.path.join(output_dir, "checkpoints")
    safe_mkdir(Path(output_dir))
    safe_mkdir(Path(ckpt_dir))
    return output_dir, ckpt_dir


def save_experiment_config(config, output_dir, filename="experiment_config.yaml"):
    """Save experiment configuration to file."""
    config_path = os.path.join(output_dir, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    print(f"[Config] Saved experiment config to {config_path}")
    return config_path


def load_experiment_config(config_path):
    """Load experiment configuration from file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_embedding_shape(z, expected_dims=3):
    """Validate embedding tensor shape."""
    if z.dim() == 4 and z.shape[0] == 1:
        z = z.squeeze(0)
    
    if z.dim() != expected_dims:
        raise ValueError(f"Expected {expected_dims}D tensor, got {z.dim()}D with shape {z.shape}")
    
    return z


def setup_training_environment(seed=42, device=None):
    """Setup training environment (random seeds, device selection)."""
    from common.utils import set_seeds
    set_seeds(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def save_model_config(model, output_dir, filename="autoencoder_config.yaml"):
    """Save model configuration to file."""
    try:
        config_path = os.path.join(output_dir, filename)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model.to_config(), f)
        print(f"[Config] Saved: {config_path}", flush=True)
    except Exception as e:
        print(f"[Config] ERROR: Could not save config: {e}", flush=True)
