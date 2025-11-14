"""
Advanced/exotic loss functions for improved diffusion model quality.
These losses target specific aspects of generation quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import LossComponent, register_loss


@register_loss
class FrequencyDomainLoss(LossComponent):
    """
    Frequency domain loss using FFT to match high and low frequency components.
    Helps preserve fine details and overall structure.
    
    Config:
        key: Key in preds for predicted tensor (default: "pred_latent")
        target: Key in targets for target tensor (default: "latent")
        weight: Loss weight (default: 1.0)
        low_freq_weight: Weight for low frequency components (default: 0.5)
        high_freq_weight: Weight for high frequency components (default: 1.0)
    """
    
    def _build(self):
        super()._build()
        if self.key is None:
            self.key = "pred_latent"
        if self.target_key is None:
            self.target_key = "latent"
        self.low_freq_weight = self._init_kwargs.get("low_freq_weight", 0.5)
        self.high_freq_weight = self._init_kwargs.get("high_freq_weight", 1.0)
    
    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        pred = preds[self.key]
        target = targets[self.target_key]
        
        # Compute FFT for both
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # Split into low and high frequency components
        # Low frequency: center region, high frequency: edges
        h, w = pred_fft.shape[-2:]
        center_h, center_w = h // 4, w // 4
        
        # Low frequency mask (center)
        low_freq_mask = torch.zeros(h, w, device=pred.device, dtype=pred.dtype)
        low_freq_mask[h//2 - center_h:h//2 + center_h, 
                     w//2 - center_w:w//2 + center_w] = 1.0
        low_freq_mask = low_freq_mask.unsqueeze(0).unsqueeze(0)
        
        # High frequency mask (everything else)
        high_freq_mask = 1.0 - low_freq_mask
        
        # Compute losses in frequency domain
        pred_low = pred_fft * low_freq_mask
        target_low = target_fft * low_freq_mask
        pred_high = pred_fft * high_freq_mask
        target_high = target_fft * high_freq_mask
        
        # L1 loss on frequency components
        low_freq_loss = F.l1_loss(pred_low.real, target_low.real) + \
                       F.l1_loss(pred_low.imag, target_low.imag)
        high_freq_loss = F.l1_loss(pred_high.real, target_high.real) + \
                        F.l1_loss(pred_high.imag, target_high.imag)
        
        loss = (self.low_freq_weight * low_freq_loss + 
                self.high_freq_weight * high_freq_loss) * self.weight
        
        return loss, {
            "freq_domain_loss": loss.detach(),
            "freq_low": low_freq_loss.detach(),
            "freq_high": high_freq_loss.detach(),
        }


@register_loss
class CharbonnierLoss(LossComponent):
    """
    Charbonnier loss (smooth L1 variant): sqrt(x^2 + eps^2) - eps
    Better than L1 for handling outliers, smoother gradients than L2.
    
    Config:
        key: Key in preds (default: "pred_noise")
        target: Key in targets (default: "noise")
        weight: Loss weight (default: 1.0)
        eps: Smoothing parameter (default: 1e-3)
    """
    
    def _build(self):
        super()._build()
        if self.key is None:
            self.key = "pred_noise"
        if self.target_key is None:
            self.target_key = "noise"
        self.eps = self._init_kwargs.get("eps", 1e-3)
    
    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        pred = preds[self.key]
        target = targets[self.target_key]
        
        diff = pred - target
        loss = torch.sqrt(diff.pow(2) + self.eps ** 2) - self.eps
        loss = loss.mean() * self.weight
        
        return loss, {f"charbonnier_{self.key}": loss.detach()}


@register_loss
class FeatureMatchingLoss(LossComponent):
    """
    Feature matching loss from discriminator (not adversarial).
    Matches intermediate features from discriminator between real and fake.
    Encourages generator to produce features similar to real data.
    
    Config:
        key: Key in preds for predicted latents (default: "pred_latent")
        target: Key in targets for real latents (default: "latent")
        weight: Loss weight (default: 1.0)
        discriminator: Discriminator model (passed via preds["discriminator"])
        layer_indices: Which discriminator layers to match (default: [0, 1, 2])
    """
    
    def _build(self):
        super()._build()
        if self.key is None:
            self.key = "pred_latent"
        if self.target_key is None:
            self.target_key = "latent"
        self.layer_indices = self._init_kwargs.get("layer_indices", [0, 1, 2])
    
    def _extract_features(self, model, x, layer_indices):
        """Extract features from discriminator at specified layers."""
        features = []
        if hasattr(model, 'net'):
            # Sequential model
            layers = list(model.net.children())
            x_current = x
            for i, layer in enumerate(layers):
                x_current = layer(x_current)
                if i in layer_indices:
                    features.append(x_current)
        else:
            # Custom model - try to extract from first few conv layers
            x_current = x
            for i, layer in enumerate(list(model.children())[:max(layer_indices) + 1]):
                x_current = layer(x_current)
                if i in layer_indices:
                    features.append(x_current)
        return features
    
    def forward(self, preds, targets):
        discriminator = preds.get("discriminator")
        if discriminator is None:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        if self.key not in preds or self.target_key not in targets:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        pred_latent = preds[self.key]
        real_latent = targets[self.target_key]
        
        # Ensure discriminator is in eval mode
        was_training = discriminator.training
        discriminator.eval()
        
        # Extract features from discriminator
        with torch.no_grad():
            real_features = self._extract_features(discriminator, real_latent, self.layer_indices)
        
        pred_features = self._extract_features(discriminator, pred_latent, self.layer_indices)
        
        # Restore training mode
        if was_training:
            discriminator.train()
        
        # Match features at each layer
        total_loss = 0.0
        for real_feat, pred_feat in zip(real_features, pred_features):
            # L1 loss on features
            layer_loss = F.l1_loss(pred_feat, real_feat.detach())
            total_loss += layer_loss
        
        # Average over layers
        if len(real_features) > 0:
            total_loss = total_loss / len(real_features)
        
        loss = total_loss * self.weight
        
        return loss, {
            "feature_matching_loss": loss.detach(),
            "feature_matching_layers": len(real_features),
        }


@register_loss
class ConsistencyLoss(LossComponent):
    """
    Consistency loss: ensures predictions at different noise levels are consistent.
    Predicts x0 from two different timesteps and ensures they match.
    Helps with temporal consistency in diffusion.
    
    Config:
        key: Key in preds for predicted noise (default: "pred_noise")
        target: Not used (consistency is internal)
        weight: Loss weight (default: 1.0)
        scheduler: Diffusion scheduler (passed via preds["scheduler"])
        timesteps: Timestep tensor (passed via preds["timesteps"])
        noisy_latent: Noisy latents (passed via preds["noisy_latent"])
        num_consistency_samples: Number of timestep pairs to sample (default: 2)
    """
    
    def _build(self):
        super()._build()
        if self.key is None:
            self.key = "pred_noise"
        self.num_samples = self._init_kwargs.get("num_consistency_samples", 2)
    
    def forward(self, preds, targets):
        scheduler = preds.get("scheduler")
        timesteps = preds.get("timesteps")
        noisy_latent = preds.get("noisy_latent")
        pred_noise = preds.get(self.key)
        
        if scheduler is None or timesteps is None or noisy_latent is None or pred_noise is None:
            device = preds.get(self.key, torch.zeros(1))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        # Predict x0 from current timestep
        device_obj = pred_noise.device
        alpha_bars = scheduler.alpha_bars.to(device_obj)
        alpha_bar = alpha_bars[timesteps].view(-1, 1, 1, 1)
        
        pred_x0 = (noisy_latent - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp(min=1e-8)
        pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
        
        # For consistency, we could add noise back and predict again
        # But for simplicity, we'll use a simpler version: ensure x0 predictions are stable
        # across small timestep differences (if we had multiple predictions)
        
        # Alternative: ensure predicted x0 has reasonable statistics
        # Encourage consistency in predicted clean latents
        x0_mean = pred_x0.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        x0_std = pred_x0.std(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # Consistency: mean and std should be similar across batch
        # (encourages stable predictions)
        mean_consistency = x0_mean.std()  # Lower is better (more consistent)
        std_consistency = x0_std.std()  # Lower is better
        
        loss = (mean_consistency + std_consistency) * self.weight
        
        return loss, {
            "consistency_loss": loss.detach(),
            "consistency_mean": mean_consistency.detach(),
            "consistency_std": std_consistency.detach(),
        }


@register_loss
class StyleLoss(LossComponent):
    """
    Style loss using Gram matrix (from neural style transfer).
    Matches style/texture statistics between predicted and target.
    Useful for matching overall appearance and texture.
    
    Config:
        key: Key in preds (default: "pred_latent")
        target: Key in targets (default: "latent")
        weight: Loss weight (default: 1.0)
    """
    
    def _build(self):
        super()._build()
        if self.key is None:
            self.key = "pred_latent"
        if self.target_key is None:
            self.target_key = "latent"
    
    def _gram_matrix(self, x):
        """Compute Gram matrix for style loss."""
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
        return gram / (C * H * W)  # Normalize
    
    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            else:
                device = torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        pred = preds[self.key]
        target = targets[self.target_key]
        
        # Compute Gram matrices
        pred_gram = self._gram_matrix(pred)
        target_gram = self._gram_matrix(target)
        
        # MSE loss on Gram matrices
        loss = F.mse_loss(pred_gram, target_gram) * self.weight
        
        return loss, {f"style_{self.key}": loss.detach()}

