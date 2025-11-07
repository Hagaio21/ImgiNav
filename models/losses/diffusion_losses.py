"""
Diffusion-specific loss components.
These losses require model components (scheduler, discriminator, decoder) as context.
"""

import torch
import torch.nn as nn
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

