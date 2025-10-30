"""
Generic model builder that constructs any model type from configuration.
This module provides a unified interface for building models without requiring 
knowledge of specific model implementations in the training code.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from models.autoencoder import AutoEncoder
from models.diffusion import LatentDiffusion
from models.components.unet import DualUNet
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler


def build_model(model_cfg: Dict[str, Any], device: str = "cpu") -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:

    model_type = model_cfg.get("type", "autoencoder").lower()
    
    if model_type in ("autoencoder", "ae", "vae"):
        # AutoEncoder/VAE
        if "from_shape" in model_cfg or all(k in model_cfg for k in ["in_channels", "out_channels", "base_channels"]):
            # Build from shape parameters
            ae = AutoEncoder.from_shape(
                in_channels=model_cfg["in_channels"],
                out_channels=model_cfg["out_channels"],
                base_channels=model_cfg["base_channels"],
                latent_channels=model_cfg["latent_channels"],
                image_size=model_cfg["image_size"],
                latent_base=model_cfg["latent_base"],
                norm=model_cfg.get("norm"),
                act=model_cfg.get("act", "relu"),
                dropout=model_cfg.get("dropout", 0.0),
                num_classes=model_cfg.get("num_classes", None),
            )
            model_type_detected = model_cfg.get("type", "vae").lower()
            ae.deterministic = (model_type_detected == "ae")
        else:
            # Build from config dict/file
            ae = AutoEncoder.from_config(model_cfg.get("config", model_cfg))
        
        ae.encoder.print_summary()
        ae.decoder.print_summary()
        return ae.to(device), None
    
    elif model_type in ("diffusion", "latent_diffusion"):
        # Latent Diffusion
        if "autoencoder" not in model_cfg:
            raise ValueError("Diffusion model requires 'autoencoder' config")
        
        # Build autoencoder (frozen)
        if isinstance(model_cfg["autoencoder"], dict):
            # It's a config dict, build the autoencoder
            autoencoder = build_autoencoder(model_cfg["autoencoder"])
        else:
            # It's already a built autoencoder object
            autoencoder = model_cfg["autoencoder"]
        
        # Build diffusion components
        diff_cfg = model_cfg.get("diffusion", {})
        unet_cfg = diff_cfg.get("unet", {})
        scheduler_cfg = diff_cfg.get("scheduler", {})
        
        scheduler = build_scheduler(scheduler_cfg)
        
        # Handle unet config - can be dict with "config" key or direct config
        if isinstance(unet_cfg, dict) and "config" in unet_cfg:
            # Old format: unet.config is a file path
            unet = build_unet(unet_cfg.get("config"))
        elif isinstance(unet_cfg, dict) and "type" in unet_cfg:
            # New format: unet config is embedded
            unet = build_unet(unet_cfg)
        else:
            # Assume it's a config dict
            unet = build_unet(unet_cfg)
        
        # Determine latent shape
        latent_shape = (
            autoencoder.encoder.latent_channels,
            autoencoder.encoder.latent_base,
            autoencoder.encoder.latent_base,
        )
        
        # Build LatentDiffusion
        diffusion = LatentDiffusion(
            backbone=unet,
            scheduler=scheduler,
            autoencoder=autoencoder,
            latent_shape=latent_shape,
        ).to(device)
        
        return diffusion, autoencoder
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'autoencoder', 'vae', 'ae', or 'diffusion'.")


def build_autoencoder(ae_cfg: Dict[str, Any]) -> AutoEncoder:
    """
    Build autoencoder from config.
    
    Args:
        ae_cfg: Dict with keys:
            - "config": Path to config file or config dict
            - "checkpoint": Path to checkpoint file (optional)
    """
    ae_cfg_path = ae_cfg.get("config")
    ae_ckpt_path = ae_cfg.get("checkpoint")
    
    if ae_cfg_path is None:
        raise ValueError("Autoencoder config must be provided")
    
    # Build model from config (handles both file path and dict)
    if isinstance(ae_cfg_path, str):
        # It's a file path
        ae = AutoEncoder.from_config(ae_cfg_path)
    elif isinstance(ae_cfg_path, dict):
        # It's a config dict - pass it directly
        ae = AutoEncoder.from_config(ae_cfg_path)
    else:
        raise ValueError(f"Autoencoder config must be a string (file path) or dict, got {type(ae_cfg_path)}")
    
    # Load checkpoint if provided
    if ae_ckpt_path and Path(ae_ckpt_path).exists():
        state = torch.load(ae_ckpt_path, map_location="cpu")
        ae.load_state_dict(state.get("model", state), strict=False)
    elif ae_ckpt_path:
        print(f"[Warning] Autoencoder checkpoint not found: {ae_ckpt_path}, continuing without loading")
    
    ae.eval()
    return ae


def build_scheduler(sched_cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build noise scheduler from config."""
    sched_type = sched_cfg.get("type", "cosine").lower()
    num_steps = sched_cfg.get("num_steps", 1000)
    
    if sched_type == "cosine":
        return CosineScheduler(num_steps=num_steps)
    elif sched_type == "linear":
        return LinearScheduler(num_steps=num_steps)
    elif sched_type == "quadratic":
        return QuadraticScheduler(num_steps=num_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def build_unet(unet_cfg: Any) -> DualUNet:
    """
    Build UNet from config.
    
    Args:
        unet_cfg: Can be:
            - Path to config file (str)
            - Config dict
            - Dict with "config" key containing path
    """
    # Handle case where config is nested
    if isinstance(unet_cfg, dict) and "config" in unet_cfg and isinstance(unet_cfg["config"], str):
        # Load from file path
        config_path = unet_cfg["config"]
        with open(config_path, "r") as f:
            unet_cfg = yaml.safe_load(f)
    
    # DualUNet.from_config expects a dict, not a file path
    if isinstance(unet_cfg, str):
        # It's a file path
        with open(unet_cfg, "r") as f:
            unet_cfg = yaml.safe_load(f)
    
    # Extract model config if nested
    if "model" in unet_cfg:
        unet_cfg = unet_cfg["model"]
    
    return DualUNet.from_config(unet_cfg)

