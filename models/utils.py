"""
Utility functions for model components.

Consolidation notes:
- Added centralized `load_checkpoint_payload()` to consolidate gzip detection logic
- This was previously duplicated in base_component.py, diffusion.py, and autoencoder.py
"""

import torch
import torch.nn as nn
from pathlib import Path


def compute_num_groups(num_channels, requested_groups=8):
    """
    Compute valid number of groups for GroupNorm.
    
    GroupNorm requires that num_channels is divisible by num_groups.
    This function finds the largest valid divisor <= requested_groups.
    
    Args:
        num_channels: Number of channels in the tensor
        requested_groups: Desired number of groups (default: 8)
    
    Returns:
        Valid number of groups for GroupNorm
    """
    for g in range(min(requested_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def reparameterize(mu, logvar):
    """
    Reparameterization trick for VAE.
    
    This consolidates the duplicate reparameterization logic from VAE.sample()
    and VAEDecoder.forward().
    
    Args:
        mu: Mean tensor [B, C, H, W] or any shape
        logvar: Log variance tensor [B, C, H, W] or any shape (must match mu)
    
    Returns:
        Latent tensor z with same shape as mu
    """
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mu + std * epsilon


def load_checkpoint_payload(path, map_location="cpu"):
    """
    Load checkpoint payload with automatic gzip detection.
    
    This centralizes the repeated logic for loading checkpoints that may
    be gzip-compressed. Previously duplicated in:
    - base_component.py: BaseComponent.load_checkpoint()
    - diffusion.py: DiffusionModel._load_model_from_checkpoint()
    - autoencoder.py: Autoencoder.save_checkpoint() context
    
    Args:
        path: Path to checkpoint file
        map_location: Device to load tensors on (default: "cpu")
    
    Returns:
        Loaded checkpoint payload (dict with state_dict, config, etc.)
    """
    import gzip
    import pickle
    
    path = Path(path)
    
    # Try gzip first, fall back to torch.load
    try:
        with gzip.open(path, 'rb') as f:
            payload = pickle.load(f)
    except (gzip.BadGzipFile, OSError):
        payload = torch.load(path, map_location=map_location)
    
    return payload


def save_checkpoint_payload(payload, path, use_compression=False):
    """
    Save checkpoint payload with optional gzip compression.
    
    Args:
        payload: Checkpoint data dict
        path: Path to save checkpoint
        use_compression: If True, use gzip compression
    """
    import gzip
    import pickle
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if use_compression:
        with gzip.open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        torch.save(payload, path)


def load_checkpoint_weights(path, map_location="cpu"):
    """
    Helper function to load weights from a checkpoint file.
    Handles both direct state_dict files and checkpoint files with 'state_dict' key.
    
    Args:
        path: Path to checkpoint file
        map_location: Device to load on
    
    Returns:
        State dict (dict of parameter tensors)
    """
    payload = load_checkpoint_payload(path, map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload
