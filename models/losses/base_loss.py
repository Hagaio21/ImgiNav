import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..components.base_component import BaseComponent

LOSS_REGISTRY = {}

def register_loss(cls):
    """Decorator to register loss classes."""
    LOSS_REGISTRY[cls.__name__] = cls
    return cls

class LossComponent(BaseComponent):
    def _build(self):
        self.key = self._init_kwargs.get("key", None)
        self.target_key = self._init_kwargs.get("target", self.key)
        self.weight = self._init_kwargs.get("weight", 1.0)

    def forward(self, preds, targets):
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg):
        """Factory method used by CompositeLoss."""
        return cls(**cfg)



@register_loss
class L1Loss(LossComponent):
    def _build(self):
        super()._build()
        self.criterion = nn.L1Loss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(self.criterion.parameters(), torch.zeros(1)).device
            return torch.tensor(0.0, device=device), {}
        loss = self.criterion(preds[self.key], targets[self.target_key]) * self.weight
        return loss, {f"L1_{self.key}": loss.detach()}


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
class CrossEntropyLoss(LossComponent):
    def _build(self):
        super()._build()
        self.ignore_index = self._init_kwargs.get("ignore_index", -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, preds, targets):
        pred = preds[self.key]
        tgt = targets[self.target_key]

        # Handle string labels FIRST (before checking tensor attributes)
        # Convert sample_type string ("room"/"scene") to class index for binary classification
        if isinstance(tgt, str) or (isinstance(tgt, list) and len(tgt) > 0 and isinstance(tgt[0], str)):
            from .loss_utils import sample_type_to_class_index
            tgt = sample_type_to_class_index(tgt, ignore_index=self.ignore_index)
            if isinstance(tgt, torch.Tensor):
                tgt = tgt.to(pred.device)
        # Convert RGB layout to segmentation mask (4D with 3 channels = RGB image)
        elif isinstance(tgt, torch.Tensor) and tgt.ndim == 4 and tgt.shape[1] == 3:
            from .loss_utils import create_seg_mask
            tgt = create_seg_mask(tgt, ignore_index=self.ignore_index).to(tgt.device)
        elif isinstance(tgt, torch.Tensor) and tgt.ndim == 4 and tgt.shape[1] == 1:
            tgt = tgt.squeeze(1)
        # Convert numeric labels (legacy room_id support) - defaults to room (0) if not found
        elif isinstance(tgt, torch.Tensor) and (tgt.ndim == 0 or tgt.ndim == 1):
            # For numeric values, assume 0 or "0000" means scene (1), others are room (0)
            tgt = (tgt == 0).long().to(pred.device)  # 0 -> 1 (scene), others -> 0 (room)
        elif not isinstance(tgt, torch.Tensor):
            # Handle non-tensor numeric values
            tgt = torch.tensor(1 if (isinstance(tgt, (int, float)) and (tgt == 0 or str(tgt) == "0000")) else 0, 
                             dtype=torch.long, device=pred.device)
        
        if tgt.dtype != torch.long:
            tgt = tgt.long()

        loss = self.criterion(pred, tgt) * self.weight
        logs = {f"CE_{self.key}": loss.detach()}
        return loss, logs


@register_loss
class PerceptualLoss(LossComponent):
    def _build(self):
        super()._build()
        from torchvision.models import vgg16
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self._device_set = False

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(self.vgg.parameters()).device if self._device_set else preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            return torch.tensor(0.0, device=device), {}
        
        # Ensure VGG is on the same device as inputs
        pred_tensor = preds[self.key]
        target_tensor = targets[self.target_key]
        device = pred_tensor.device
        
        if not self._device_set or next(self.vgg.parameters()).device != device:
            self.vgg = self.vgg.to(device)
            self._device_set = True
        
        f_pred = self.vgg(pred_tensor)
        f_tgt = self.vgg(target_tensor)
        loss = self.criterion(f_pred, f_tgt) * self.weight
        return loss, {f"Perceptual_{self.key}": loss.detach()}


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
        for sub_cfg in self._init_kwargs.get("losses", []):
            loss_type = sub_cfg["type"]
            if loss_type not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss type: {loss_type}")
            self.losses.append(LOSS_REGISTRY[loss_type].from_config(sub_cfg))

    def forward(self, preds, targets):
        total, logs = 0.0, {}
        for loss_fn in self.losses:
            loss, sublog = loss_fn(preds, targets)
            total += loss
            logs.update(sublog)
        return total, logs
