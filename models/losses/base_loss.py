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


@register_loss
class ColorWeightedMSELoss(LossComponent):
    """
    MSE loss with color-based pixel weighting.
    Gives higher weight to pixels that are close to specific target colors.
    
    Config:
        key: Key in preds for predictions (default: "rgb")
        target: Key in targets for targets (default: "rgb")
        weight: Loss weight
        target_colors: List of target colors in RGB format [0-1] or [0-255]
                      Each entry: {"color": [r, g, b], "weight": float, "tolerance": float}
                      - color: RGB values (will be normalized to [0, 1] if > 1)
                      - weight: Weight to apply to pixels near this color
                      - tolerance: Distance threshold for matching (default: 0.1)
        color_space: "rgb" or "lab" for color distance computation (default: "rgb")
        fallback_weight: Weight for pixels not matching any target color (default: 1.0)
    """
    def _build(self):
        super()._build()
        
        # Get target colors from config
        target_colors_config = self._init_kwargs.get("target_colors", [])
        self.color_space = self._init_kwargs.get("color_space", "rgb")
        self.fallback_weight = self._init_kwargs.get("fallback_weight", 1.0)
        
        # Parse target colors
        self.target_colors = []
        for color_config in target_colors_config:
            color = color_config.get("color", [0, 0, 0])
            color_weight = color_config.get("weight", 1.0)
            tolerance = color_config.get("tolerance", 0.1)
            
            # Normalize color to [0, 1] if needed
            if isinstance(color, list):
                color = torch.tensor(color, dtype=torch.float32)
                if color.max() > 1.0:
                    color = color / 255.0
            else:
                color = torch.tensor(color, dtype=torch.float32)
                if color.max() > 1.0:
                    color = color / 255.0
            
            self.target_colors.append({
                "color": color,
                "weight": color_weight,
                "tolerance": tolerance
            })
        
        if len(self.target_colors) == 0:
            print("Warning: ColorWeightedMSELoss has no target colors specified. Using uniform weighting.")
    
    def _rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space."""
        # RGB to XYZ conversion
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        # Apply gamma correction
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        
        # RGB to XYZ matrix (D65 illuminant)
        matrix = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=rgb.device, dtype=rgb.dtype)
        
        xyz = torch.matmul(rgb_linear.permute(0, 2, 3, 1), matrix.t()).permute(0, 3, 1, 2)
        
        # Normalize by D65 white point
        xyz[:, 0] = xyz[:, 0] / 0.95047
        xyz[:, 2] = xyz[:, 2] / 1.08883
        
        # XYZ to LAB
        xyz = torch.clamp(xyz, 0.0, 1.0)
        mask = xyz > 0.008856
        fxyz = torch.where(mask, xyz ** (1.0/3.0), (7.787 * xyz + 16.0/116.0))
        
        L = 116.0 * fxyz[:, 1] - 16.0
        a = 500.0 * (fxyz[:, 0] - fxyz[:, 1])
        b = 200.0 * (fxyz[:, 1] - fxyz[:, 2])
        
        lab = torch.stack([L, a, b], dim=1)
        return lab
    
    def _compute_color_distance(self, rgb1, rgb2):
        """Compute color distance between two RGB tensors."""
        if self.color_space == "lab":
            lab1 = self._rgb_to_lab(rgb1)
            lab2 = self._rgb_to_lab(rgb2)
            # Delta E distance in LAB space
            diff = lab1 - lab2
            distance = torch.sqrt(torch.sum(diff ** 2, dim=1))
        else:  # rgb
            # L2 distance in RGB space
            diff = rgb1 - rgb2
            distance = torch.sqrt(torch.sum(diff ** 2, dim=1))
        
        return distance
    
    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            return torch.tensor(0.0, device=device), {f"ColorWeightedMSE_{self.key}": torch.tensor(0.0, device=device)}
        
        pred = preds[self.key]  # [B, 3, H, W] in [-1, 1] or [0, 1]
        target = targets[self.target_key]  # [B, 3, H, W] in [-1, 1] or [0, 1]
        
        # Normalize to [0, 1] if in [-1, 1] range
        if pred.min() < 0:
            pred_normalized = (pred + 1.0) / 2.0
        else:
            pred_normalized = pred
        
        if target.min() < 0:
            target_normalized = (target + 1.0) / 2.0
        else:
            target_normalized = target
        
        device = pred.device
        B, C, H, W = pred.shape
        
        # Initialize weight map with fallback weight
        weight_map = torch.ones(B, H, W, device=device, dtype=torch.float32) * self.fallback_weight
        
        # For each target color, compute distance and apply weight
        for color_config in self.target_colors:
            target_color = color_config["color"].to(device)  # [3]
            color_weight = color_config["weight"]
            tolerance = color_config["tolerance"]
            
            # Expand target color to match spatial dimensions: [B, 3, H, W]
            target_color_expanded = target_color.view(1, 3, 1, 1).expand(B, 3, H, W)
            
            # Compute distance from each pixel to target color (using target image)
            # This way we weight based on what the pixel should be, not what it is
            distance = self._compute_color_distance(target_normalized, target_color_expanded)  # [B, H, W]
            
            # Apply weight based on distance (closer = higher weight)
            # Use exponential decay: weight = color_weight * exp(-distance / tolerance)
            # Or use step function: weight = color_weight if distance < tolerance
            mask = distance < tolerance
            weight_map[mask] = torch.maximum(weight_map[mask], torch.tensor(color_weight, device=device))
        
        # Expand weight map to match pred shape: [B, 1, H, W]
        weight_map = weight_map.unsqueeze(1)
        
        # Compute per-pixel MSE
        squared_diff = (pred - target) ** 2  # [B, 3, H, W]
        
        # Apply color-based weights
        weighted_squared_diff = squared_diff * weight_map
        
        # Average over all dimensions
        loss = weighted_squared_diff.mean() * self.weight
        
        # Also compute unweighted MSE for comparison
        unweighted_loss = squared_diff.mean()
        
        return loss, {
            f"ColorWeightedMSE_{self.key}": loss.detach(),
            f"MSE_{self.key}": unweighted_loss.detach(),
        }


@register_loss
class LatentStructuralLossAE(LossComponent):
    """
    Latent-space structural loss for autoencoder training.
    
    Computes Sobel or Laplacian gradients on both predicted and target latents,
    then minimizes their difference. This encourages the autoencoder to learn
    spatially coherent latent representations where boundaries and transitions
    align with real layout geometry.
    
    Unlike LatentStructuralLoss (for diffusion), this version works directly
    on latents without requiring a diffusion scheduler or SNR weighting.
    
    Config:
        key: Key in preds for predicted latent (default: "latent")
        target: Key in targets for target latent (default: "latent")
        weight: Loss weight (default: 1.0)
        gradient_type: "sobel" or "laplacian" (default: "sobel")
        reduction: "mean" or "sum" for gradient magnitude reduction (default: "mean")
    """
    
    def _build(self):
        super()._build()
        # Set defaults if not specified
        if self.key is None:
            self.key = "latent"
        if self.target_key is None:
            self.target_key = "latent"
        
        # Gradient computation type
        self.gradient_type = self._init_kwargs.get("gradient_type", "sobel").lower()
        if self.gradient_type not in ["sobel", "laplacian"]:
            raise ValueError(f"gradient_type must be 'sobel' or 'laplacian', got '{self.gradient_type}'")
        
        # Reduction method for gradient magnitude
        self.reduction = self._init_kwargs.get("reduction", "mean")
        
        # Build Sobel kernels if needed
        if self.gradient_type == "sobel":
            # Sobel kernels for x and y gradients
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
        
        # Laplacian kernel
        elif self.gradient_type == "laplacian":
            laplacian = torch.tensor([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("laplacian", laplacian)
    
    def _compute_gradients(self, latents):
        """
        Compute gradients on latents using Sobel or Laplacian operator.
        
        Args:
            latents: Tensor [B, C, H, W]
        
        Returns:
            Gradient magnitude tensor [B, C, H, W] (for Sobel) or [B, C, H, W] (for Laplacian)
        """
        B, C, H, W = latents.shape
        device = latents.device
        
        if self.gradient_type == "sobel":
            # Apply Sobel filters to each channel
            # Expand kernels to match number of channels and ensure they're on the same device
            sobel_x = self.sobel_x.to(device).expand(C, 1, 3, 3)
            sobel_y = self.sobel_y.to(device).expand(C, 1, 3, 3)
            
            # Compute gradients for each channel
            grad_x = F.conv2d(latents, sobel_x, padding=1, groups=C)
            grad_y = F.conv2d(latents, sobel_y, padding=1, groups=C)
            
            # Compute gradient magnitude
            gradients = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
        
        else:  # laplacian
            # Apply Laplacian filter to each channel
            laplacian_kernel = self.laplacian.to(device).expand(C, 1, 3, 3)
            gradients = F.conv2d(latents, laplacian_kernel, padding=1, groups=C)
            # Take absolute value for Laplacian (second derivative can be negative)
            gradients = torch.abs(gradients)
        
        return gradients
    
    def forward(self, preds, targets):
        # Check required keys
        if self.key not in preds or self.target_key not in targets:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        pred_latents = preds[self.key]
        target = targets[self.target_key]
        
        # Handle case where target is an image (for autoencoder training)
        # We need to downsample and convert to grayscale or use per-channel gradients
        if target.shape[1] == 3 and pred_latents.shape[1] != 3:
            # Target is RGB image, pred is latent - downsample target to match latent resolution
            target_h, target_w = target.shape[2], target.shape[3]
            latent_h, latent_w = pred_latents.shape[2], pred_latents.shape[3]
            
            if target_h != latent_h or target_w != latent_w:
                # Downsample target image to match latent resolution
                target = F.interpolate(target, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            
            # Convert RGB to grayscale for gradient computation (or use per-channel)
            # For simplicity, convert to grayscale: 0.299*R + 0.587*G + 0.114*B
            target_gray = (0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3])
            # Expand to match latent channels (average across channels)
            if pred_latents.shape[1] > 1:
                target_latents = target_gray.expand(-1, pred_latents.shape[1], -1, -1)
            else:
                target_latents = target_gray
        else:
            # Target is already a latent or matches shape
            target_latents = target
        
        # Ensure shapes match
        if target_latents.shape != pred_latents.shape:
            # If still mismatched, interpolate
            target_latents = F.interpolate(
                target_latents, 
                size=(pred_latents.shape[2], pred_latents.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            # Handle channel mismatch by averaging or repeating
            if target_latents.shape[1] != pred_latents.shape[1]:
                if target_latents.shape[1] == 1:
                    target_latents = target_latents.expand(-1, pred_latents.shape[1], -1, -1)
                else:
                    # Average channels
                    target_latents = target_latents.mean(dim=1, keepdim=True).expand(-1, pred_latents.shape[1], -1, -1)
        
        # Compute gradients on both predicted and target latents
        pred_gradients = self._compute_gradients(pred_latents)
        target_gradients = self._compute_gradients(target_latents)
        
        # Compute structural loss (L1 difference between gradient magnitudes)
        # This preserves boundaries and transitions in latent space
        gradient_diff = torch.abs(pred_gradients - target_gradients)
        
        if self.reduction == "mean":
            loss = gradient_diff.mean() * self.weight
        else:  # sum
            loss = gradient_diff.sum() / (pred_gradients.numel()) * self.weight
        
        return loss, {
            f"latent_structural_ae_{self.gradient_type}": loss.detach(),
            f"latent_structural_ae_{self.gradient_type}_pred_mean": pred_gradients.mean().detach(),
            f"latent_structural_ae_{self.gradient_type}_target_mean": target_gradients.mean().detach(),
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
