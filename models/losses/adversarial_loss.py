"""
Adversarial loss with gradient reconnection for diffusion training.

This loss implements the gradient reconnection approach:
1. Full T-step sampling is done with no_grad, producing detached x_fake
2. Discriminator is trained on real vs these fully sampled fakes
3. To apply discriminator loss to diffusion model, reconnect gradients:
   - Feed x_fake back into model at t=0 to get x0_pred
   - Compute discriminator loss on x0_pred (not x_fake)
   - Gradients flow from discriminator → x0_pred → model, without passing through sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import LossComponent, register_loss


@register_loss
class DiscriminatorLossWithReconnection(LossComponent):
    """
    Discriminator adversarial loss with gradient reconnection for diffusion training.
    
    Uses the gradient reconnection approach:
    1. Full T-step sampling is done with no_grad, producing detached x_fake
    2. Discriminator is trained on real vs these fully sampled fakes
    3. To apply discriminator loss to diffusion model, reconnect gradients:
       - Feed x_fake back into model at t=0 to get x0_pred
       - Compute discriminator loss on x0_pred (not x_fake)
       - Gradients flow from discriminator → x0_pred → model, without passing through sampling
    
    Config:
        key: Key in preds for x0_pred (default: "x0_pred", set automatically)
        weight: Loss weight (default: 1.0)
        consistency_weight: Weight for consistency term ensuring x0_pred ≈ x_fake (default: 0.1)
        target: Not used (discriminator doesn't need targets)
    
    Requires:
        preds["discriminator"]: LatentDiscriminator model
        preds["x_fake"]: Fully sampled fake latents [B, C, H, W] (detached, from no_grad sampling)
        preds["model"]: DiffusionModel (for gradient reconnection)
        preds["cond"]: Optional conditioning (for gradient reconnection)
    """
    
    def _build(self):
        super()._build()
        # Key will be "x0_pred" which is set by compute_loss_with_reconnection
        if self.key is None:
            self.key = "x0_pred"
        # Consistency weight for distribution matching (x0_pred ≈ x_fake)
        self.consistency_weight = self._init_kwargs.get("consistency_weight", 0.1)
        print(f"DiscriminatorLossWithReconnection initialized with key='{self.key}', weight={self.weight}, consistency_weight={self.consistency_weight}")
    
    def forward(self, preds, targets):
        discriminator = preds.get("discriminator")
        if discriminator is None:
            if not hasattr(self, '_warned_no_discriminator'):
                print(f"WARNING: DiscriminatorLossWithReconnection - discriminator not found in preds. Available keys: {list(preds.keys())}")
                self._warned_no_discriminator = True
            device = preds.get("x_fake", torch.zeros(1))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        # Check if we have fully sampled fakes (x_fake) for gradient reconnection
        x_fake = preds.get("x_fake")
        model = preds.get("model")
        
        if x_fake is None or model is None:
            if not hasattr(self, '_warned_no_x_fake'):
                print(f"WARNING: DiscriminatorLossWithReconnection - x_fake or model not found in preds. Available keys: {list(preds.keys())}")
                self._warned_no_x_fake = True
            device = preds.get("x_fake", torch.zeros(1))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        # Gradient reconnection approach: feed x_fake back into model at t=0
        # This creates a valid gradient path without backpropagating through sampling
        cond = preds.get("cond", None)
        device_obj = x_fake.device
        
        # Create t=0 timesteps for all samples
        batch_size = x_fake.shape[0]
        t_zero = torch.zeros(batch_size, dtype=torch.long, device=device_obj)
        
        # Reconnect gradients: single forward pass at t=0
        # This gives us x0_pred which has gradients connected to the model
        model_outputs = model(x_fake, t_zero, cond=cond, noise=None)
        x0_pred = model_outputs.get("pred_latent")
        
        if x0_pred is None:
            # Fallback: if model doesn't return pred_latent, compute it manually
            # At t=0, the model should predict the input (or very close to it)
            # But we still need to go through the model for gradients
            pred_noise = model_outputs.get("pred_noise")
            if pred_noise is not None:
                # At t=0, alpha_bar ≈ 1.0, so x0_pred ≈ x_fake - 0 * pred_noise ≈ x_fake
                # But we want gradients, so use the model's prediction
                scheduler = model.scheduler
                alpha_bars = scheduler.alpha_bars.to(device_obj)
                alpha_bar = alpha_bars[t_zero].view(-1, 1, 1, 1)
                noisy_latent = model_outputs.get("noisy_latent", x_fake)
                x0_pred = (noisy_latent - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp(min=1e-8)
                x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
            else:
                # Last resort: use the input (but this won't have gradients through model)
                x0_pred = x_fake
        
        # Ensure discriminator is in eval mode
        if discriminator.training:
            discriminator.eval()
        
        # DISTRIBUTION CONSISTENCY: Ensure x0_pred ≈ x_fake at t=0
        # This minimizes distribution mismatch between discriminator training (x_fake) 
        # and diffusion training (x0_pred)
        # At t=0, the model should predict x_fake almost exactly (with minimal noise)
        consistency_loss = F.mse_loss(x0_pred, x_fake.detach())
        
        # Compute discriminator loss on x0_pred (which has gradients)
        # Note: We use x0_pred (not x_fake) to maintain gradient flow, but the consistency
        # term ensures they're close, minimizing distribution mismatch
        viability_scores = discriminator(x0_pred)  # [B, 1] in [0, 1]
        
        # Adversarial loss: encourage model to generate latents that fool discriminator
        # High when fake (score→0), low when real (score→1)
        adversarial_loss = -torch.log(viability_scores.mean() + 1e-8) * self.weight
        
        # Combined loss: adversarial + consistency
        loss = adversarial_loss + self.consistency_weight * consistency_loss
        
        # Debug logging (once)
        if not hasattr(self, '_logged_reconnection_stats'):
            score_mean = viability_scores.mean().item()
            score_min = viability_scores.min().item()
            score_max = viability_scores.max().item()
            score_std = viability_scores.std().item()
            x0_fake_diff = (x0_pred - x_fake.detach()).abs().mean().item()
            print(f"DiscriminatorLossWithReconnection DEBUG - Scores: mean={score_mean:.6f}, min={score_min:.6f}, max={score_max:.6f}, std={score_std:.6f}")
            print(f"DiscriminatorLossWithReconnection DEBUG - x_fake shape: {x_fake.shape}, range: [{x_fake.min().item():.3f}, {x_fake.max().item():.3f}]")
            print(f"DiscriminatorLossWithReconnection DEBUG - x0_pred shape: {x0_pred.shape}, range: [{x0_pred.min().item():.3f}, {x0_pred.max().item():.3f}]")
            print(f"DiscriminatorLossWithReconnection DEBUG - Distribution consistency: |x0_pred - x_fake|_mean = {x0_fake_diff:.6f}")
            print(f"DiscriminatorLossWithReconnection DEBUG - Consistency loss: {consistency_loss.item():.6f}, Adversarial loss: {adversarial_loss.item():.6f}")
            print(f"DiscriminatorLossWithReconnection DEBUG - Gradient reconnection successful (x0_pred requires_grad: {x0_pred.requires_grad})")
            self._logged_reconnection_stats = True
        
        return loss, {
            "discriminator_loss": adversarial_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "total_discriminator_loss": loss.detach(),
            "viability_score": viability_scores.mean().detach(),
            "x0_pred_min": x0_pred.min().detach().item(),
            "x0_pred_max": x0_pred.max().detach().item(),
            "x0_fake_diff": (x0_pred - x_fake.detach()).abs().mean().detach().item(),
        }

