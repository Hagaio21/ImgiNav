"""
Diffusion-specific loss components.
These losses require model components (scheduler, discriminator, decoder) as context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import LossComponent, register_loss


@register_loss
class SNRWeightedNoiseLoss(LossComponent):
    """
    SNR-weighted noise prediction loss for diffusion training.
    
    Computes: loss = mean((pred_noise - noise)^2 * w) where w = snr / (1 + snr)
    and snr = alpha_bar / (1 - alpha_bar)
    
    Config:
        key: Key in preds for predicted noise (default: "pred_noise")
        target: Key in targets for target noise (default: "noise")
        weight: Loss weight (default: 1.0)
    
    Requires scheduler and timesteps to be passed in preds:
        preds["scheduler"]: Diffusion scheduler (for alpha_bars)
        preds["timesteps"]: Timestep tensor [B]
    """
    
    def _build(self):
        super()._build()
        # Set defaults if not specified
        if self.key is None:
            self.key = "pred_noise"
        if self.target_key is None:
            self.target_key = "noise"
    
    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        pred_noise = preds[self.key]
        noise_target = targets[self.target_key]
        scheduler = preds.get("scheduler")
        timesteps = preds.get("timesteps")
        
        if scheduler is None or timesteps is None:
            # Fallback to unweighted MSE if scheduler/timesteps not provided
            loss = ((pred_noise - noise_target).pow(2)).mean() * self.weight
            return loss, {f"noise_loss": loss.detach()}
        
        # Compute SNR-weighted loss
        device_obj = pred_noise.device
        alpha_bars = scheduler.alpha_bars.to(device_obj)
        alpha_bar = alpha_bars[timesteps].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        snr = alpha_bar / (1 - alpha_bar + 1e-8)  # Signal-to-noise ratio
        w = snr / (1 + snr)  # SNR weighting
        
        loss = ((pred_noise - noise_target).pow(2) * w).mean() * self.weight
        return loss, {f"noise_loss": loss.detach()}


@register_loss
class DiscriminatorLoss(LossComponent):
    """
    Discriminator adversarial loss for diffusion training.
    
    Computes: loss = -log(viability_score.mean()) * weight
    where viability_score comes from discriminator(latents)
    
    Config:
        key: Key in preds for latent tensor (default: "latent")
        weight: Loss weight (default: 1.0)
        target: Not used (discriminator doesn't need targets)
    
    Requires discriminator to be passed in preds:
        preds["discriminator"]: LatentDiscriminator model
    """
    
    def _build(self):
        super()._build()
        # Set defaults if not specified
        if self.key is None:
            self.key = "latent"
        # Discriminator loss doesn't use targets
    
    def forward(self, preds, targets):
        if self.key not in preds:
            device = preds.get(self.key, torch.zeros(1))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        latents = preds[self.key]
        discriminator = preds.get("discriminator")
        
        if discriminator is None:
            # Return zero loss if discriminator not provided
            device = latents.device
            return torch.tensor(0.0, device=device), {}
        
        # Get viability scores from discriminator
        viability_scores = discriminator(latents)  # [B, 1] in [0, 1]
        
        # Adversarial loss: maximize score (push toward 1.0 = viable)
        loss = -torch.log(viability_scores.mean() + 1e-8) * self.weight
        
        return loss, {
            "discriminator_loss": loss.detach(),
            "viability_score": viability_scores.mean().detach()
        }


@register_loss
class LatentStructuralLoss(LossComponent):
    """
    Latent-space structural loss for diffusion training.
    
    Computes Sobel or Laplacian gradients on both predicted and ground-truth latents,
    then minimizes their difference. This encourages the UNet to learn spatially coherent
    latent representations where boundaries and transitions align with real layout geometry.
    
    Config:
        key: Key in preds for predicted noise (default: "pred_noise")
        target: Key in targets for ground-truth latent (default: "latent")
        weight: Loss weight (default: 1.0)
        gradient_type: "sobel" or "laplacian" (default: "sobel")
        reduction: "mean" or "sum" for gradient magnitude reduction (default: "mean")
    
    Requires scheduler, timesteps, and noisy_latent to be passed in preds:
        preds["scheduler"]: Diffusion scheduler (for alpha_bars)
        preds["timesteps"]: Timestep tensor [B]
        preds["noisy_latent"]: Noisy latents [B, C, H, W]
    """
    
    def _build(self):
        super()._build()
        # Set defaults if not specified
        if self.key is None:
            self.key = "pred_noise"
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
        
        if self.gradient_type == "sobel":
            # Apply Sobel filters to each channel
            # Expand kernels to match number of channels
            sobel_x = self.sobel_x.expand(C, 1, 3, 3)
            sobel_y = self.sobel_y.expand(C, 1, 3, 3)
            
            # Compute gradients for each channel
            grad_x = F.conv2d(latents, sobel_x, padding=1, groups=C)
            grad_y = F.conv2d(latents, sobel_y, padding=1, groups=C)
            
            # Compute gradient magnitude
            gradients = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
        
        else:  # laplacian
            # Apply Laplacian filter to each channel
            laplacian_kernel = self.laplacian.expand(C, 1, 3, 3)
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
        
        pred_noise = preds[self.key]
        gt_latents = targets[self.target_key]
        scheduler = preds.get("scheduler")
        timesteps = preds.get("timesteps")
        noisy_latents = preds.get("noisy_latent")
        
        # Check if we have all required components
        if scheduler is None or timesteps is None or noisy_latents is None:
            device = pred_noise.device if isinstance(pred_noise, torch.Tensor) else torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        # Compute predicted clean latents from noisy latents and predicted noise
        device_obj = pred_noise.device
        alpha_bars = scheduler.alpha_bars.to(device_obj)
        alpha_bar = alpha_bars[timesteps].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Predict x0 from noisy latents: x0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
        pred_latents = (noisy_latents - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp(min=1e-8)
        # Clamp to avoid numerical issues
        pred_latents = torch.clamp(pred_latents, -10.0, 10.0)
        
        # Compute gradients on both predicted and ground-truth latents
        pred_gradients = self._compute_gradients(pred_latents)
        gt_gradients = self._compute_gradients(gt_latents)
        
        # Compute loss as L1 difference between gradient magnitudes
        # This preserves boundaries and transitions in latent space
        if self.reduction == "mean":
            loss = F.l1_loss(pred_gradients, gt_gradients) * self.weight
        else:  # sum
            loss = torch.abs(pred_gradients - gt_gradients).sum() / (pred_gradients.numel()) * self.weight
        
        return loss, {
            "latent_structural_loss": loss.detach(),
            f"latent_structural_{self.gradient_type}_pred_mean": pred_gradients.mean().detach(),
            f"latent_structural_{self.gradient_type}_gt_mean": gt_gradients.mean().detach(),
        }

