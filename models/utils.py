"""Utility functions for model components."""

import torch
import torch.nn as nn


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
    # Find the largest valid divisor <= requested_groups
    for g in range(min(requested_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # Fallback: single group


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
