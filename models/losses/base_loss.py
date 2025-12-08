import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from ..components.base_component import BaseComponent

LOSS_REGISTRY = {}

def register_loss(cls):
    """Decorator to register loss classes."""
    LOSS_REGISTRY[cls.__name__] = cls
    return cls

class LossComponent(BaseComponent):
    def _build(self):
        self.key = self._init_kwargs.get("key", None)
        # Support both "target" and "target_key" for backward compatibility
        self.target_key = self._init_kwargs.get("target_key") or self._init_kwargs.get("target", self.key)
        self.weight = self._init_kwargs.get("weight", 1.0)

    def forward(self, preds, targets):
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg):
        """Factory method used by CompositeLoss."""
        return cls(**cfg)



@register_loss
class MSELoss(LossComponent):
    def _build(self):
        super()._build()
        self.criterion = nn.MSELoss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(self.criterion.parameters(), torch.zeros(1)).device
            return torch.tensor(0.0, device=device), {}
        loss = self.criterion(preds[self.key], targets[self.target_key]) * self.weight
        return loss, {f"MSE_{self.key}": loss.detach()}


@register_loss
class SNRWeightedMSELoss(LossComponent):
    """
    SNR-weighted MSE loss for diffusion training.
    
    Applies time-step dependent weights based on Signal-to-Noise Ratio (SNR).
    SNR = alpha_bar / (1 - alpha_bar) for each timestep.
    
    Config:
        key: Key in preds for predicted noise (default: "pred_noise")
        target: Key in targets for target noise (default: "noise")
        weight: Loss weight multiplier (default: 1.0)
        snr_weight_mode: How to apply SNR weighting:
            - "snr": weight = SNR (linear)
            - "snr_squared": weight = SNR^2 (squared)
            - "inverse_snr": weight = 1 / (1 + SNR) (inverse)
            - "min_snr": weight = min(SNR, gamma) / SNR (min-SNR-gamma)
        min_snr_gamma: Gamma value for "min_snr" mode (default: 5.0)
        reduce_mean: If True, average over batch; if False, sum (default: True)
    """
    def _build(self):
        super()._build()
        self.snr_weight_mode = self._init_kwargs.get("snr_weight_mode", "snr").lower()
        self.min_snr_gamma = self._init_kwargs.get("min_snr_gamma", 5.0)
        self.reduce_mean = self._init_kwargs.get("reduce_mean", True)
        
        valid_modes = ["snr", "snr_squared", "inverse_snr", "min_snr"]
        if self.snr_weight_mode not in valid_modes:
            raise ValueError(f"snr_weight_mode must be one of {valid_modes}, got '{self.snr_weight_mode}'")

    def _compute_snr_weights(self, scheduler, timesteps):
        """
        Compute SNR weights for given timesteps.
        
        Args:
            scheduler: NoiseScheduler instance with alpha_bars
            timesteps: Tensor of timestep indices [B]
        
        Returns:
            weights: Tensor of SNR weights [B, 1, 1, 1] for broadcasting
        """
        device = timesteps.device
        alpha_bars = scheduler.alpha_bars.to(device)
        t = timesteps.long()
        
        # Get alpha_bar for each timestep: [B]
        alpha_bar_t = alpha_bars[t]
        
        # Compute SNR = alpha_bar / (1 - alpha_bar)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        snr = alpha_bar_t / (1 - alpha_bar_t + eps)
        
        # Apply weighting mode
        if self.snr_weight_mode == "snr":
            weights = snr
        elif self.snr_weight_mode == "snr_squared":
            weights = snr ** 2
        elif self.snr_weight_mode == "inverse_snr":
            weights = 1.0 / (1.0 + snr)
        elif self.snr_weight_mode == "min_snr":
            # min-SNR-gamma: weight = min(SNR, gamma) / SNR
            weights = torch.clamp(snr, max=self.min_snr_gamma) / (snr + eps)
        else:
            raise ValueError(f"Unknown snr_weight_mode: {self.snr_weight_mode}")
        
        # Reshape to [B, 1, 1, 1] for broadcasting over spatial dimensions
        while weights.dim() < 4:
            weights = weights.unsqueeze(-1)
        
        return weights

    def forward(self, preds, targets):
        # Check required keys
        if self.key not in preds or self.target_key not in targets:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        pred = preds[self.key]
        target = targets[self.target_key]
        
        # Check if scheduler and timesteps are available
        scheduler = preds.get("scheduler", None)
        timesteps = preds.get("timesteps", None)
        
        if scheduler is None or timesteps is None:
            # Fallback to standard MSE if SNR weighting not available
            device = pred.device
            mse = (pred - target) ** 2
            if self.reduce_mean:
                loss = mse.mean() * self.weight
            else:
                loss = mse.sum() * self.weight
            return loss, {f"SNRWeightedMSE_{self.key}": loss.detach(), f"MSE_{self.key}": loss.detach()}
        
        # Compute SNR weights
        snr_weights = self._compute_snr_weights(scheduler, timesteps)
        
        # Compute per-pixel MSE
        squared_diff = (pred - target) ** 2  # [B, C, H, W]
        
        # Apply SNR weights (broadcast over C, H, W dimensions)
        weighted_squared_diff = squared_diff * snr_weights
        
        # Reduce: mean or sum
        if self.reduce_mean:
            loss = weighted_squared_diff.mean() * self.weight
        else:
            loss = weighted_squared_diff.sum() * self.weight
        
        # Also compute unweighted MSE for comparison
        unweighted_loss = squared_diff.mean()
        
        return loss, {
            f"SNRWeightedMSE_{self.key}": loss.detach(),
            f"MSE_{self.key}": unweighted_loss.detach(),
        }


@register_loss
class KLDLoss(LossComponent):
    def forward(self, preds, targets=None):
        mu = preds.get("mu")
        logvar = preds.get("logvar")
        if mu is None or logvar is None:
            device = mu.device if mu is not None else torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld * self.weight
        return kld, {"KLD": kld.detach()}


@register_loss
class LatentStandardizationLoss(LossComponent):
    """
    Loss that encourages latents to be approximately N(0,1).
    Penalizes mean deviation from 0 and std deviation from 1.
    
    Uses a stronger penalty to ensure latents converge to N(0,1) distribution.
    Can use per-channel statistics to handle channel imbalances.
    """
    def _build(self):
        super()._build()
        # Optional: use L1 penalty for mean (more aggressive) or L2 (smoother)
        self.mean_penalty_type = self._init_kwargs.get("mean_penalty_type", "l2")  # "l1" or "l2"
        self.std_penalty_type = self._init_kwargs.get("std_penalty_type", "l2")  # "l1" or "l2"
        # Per-channel standardization (critical for handling channel imbalances)
        self.per_channel = self._init_kwargs.get("per_channel", True)  # Default to per-channel for better results
    
    def forward(self, preds, targets=None):
        # Extract latent from encoder output
        latent = preds.get(self.key)  # key should be "latent" or "mu"
        if latent is None:
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        if self.per_channel and latent.ndim == 4:  # [B, C, H, W]
            # Per-channel statistics (critical for handling channel imbalances)
            # Compute mean and std per channel across batch and spatial dimensions
            latent_mean_per_ch = latent.mean(dim=(0, 2, 3))  # [C]
            latent_std_per_ch = latent.std(dim=(0, 2, 3))  # [C]
            
            # Penalize mean ≠ 0 per channel
            if self.mean_penalty_type == "l1":
                mean_loss = torch.abs(latent_mean_per_ch).mean()  # L1 penalty (more aggressive)
            else:  # l2
                mean_loss = latent_mean_per_ch.pow(2).mean()  # L2 penalty on mean
            
            # Penalize std ≠ 1 per channel
            if self.std_penalty_type == "l1":
                std_loss = torch.abs(latent_std_per_ch - 1.0).mean()  # L1 penalty (more aggressive)
            else:  # l2
                std_loss = (latent_std_per_ch - 1.0).pow(2).mean()  # L2 penalty on std deviation from 1
            
            # Also track global stats for monitoring
            latent_flat = latent.reshape(latent.shape[0], -1)
            global_mean = latent_flat.mean()
            global_std = latent_flat.std()
        else:
            # Global statistics (original behavior)
            latent_flat = latent.reshape(latent.shape[0], -1)
            
            # Compute mean and std
            latent_mean = latent_flat.mean()
            latent_std = latent_flat.std()
            
            # Penalize mean ≠ 0
            if self.mean_penalty_type == "l1":
                mean_loss = torch.abs(latent_mean)  # L1 penalty (more aggressive)
            else:  # l2
                mean_loss = latent_mean.pow(2)  # L2 penalty on mean
            
            # Penalize std ≠ 1
            if self.std_penalty_type == "l1":
                std_loss = torch.abs(latent_std - 1.0)  # L1 penalty (more aggressive)
            else:  # l2
                std_loss = (latent_std - 1.0).pow(2)  # L2 penalty on std deviation from 1
            
            global_mean = latent_mean
            global_std = latent_std
        
        # Combined loss
        loss = (mean_loss + std_loss) * self.weight
        
        return loss, {
            f"LatentStd_Mean": mean_loss.detach(),
            f"LatentStd_Std": std_loss.detach(),
            f"LatentStd_MeanVal": global_mean.detach(),
            f"LatentStd_StdVal": global_std.detach(),
        }


# === Composite loss ==========================================================
@register_loss
class CompositeLoss(LossComponent):
    def _build(self):
        self.losses = nn.ModuleList()
        # Support both "losses" and "components" keys for backward compatibility
        sub_losses = self._init_kwargs.get("losses", None) or self._init_kwargs.get("components", [])
        for sub_cfg in sub_losses:
            loss_type = sub_cfg["type"]
            if loss_type not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss type: {loss_type}")
            self.losses.append(LOSS_REGISTRY[loss_type].from_config(sub_cfg))

    def forward(self, preds, targets):
        total = None
        logs = {}
        for loss_fn in self.losses:
            loss, sublog = loss_fn(preds, targets)
            if total is None:
                total = loss
            else:
                total = total + loss
            logs.update(sublog)
        # Ensure total is a tensor (if all losses returned zero, create a zero tensor)
        if total is None:
            # No losses, return zero tensor on appropriate device
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            total = torch.tensor(0.0, device=device, requires_grad=True)
        return total, logs


# Import reconstruction losses to ensure they are registered
# This must be at the end to avoid circular imports
try:
    from . import reconstruction_loss
except ImportError:
    pass  # reconstruction_loss may not exist in all installations
