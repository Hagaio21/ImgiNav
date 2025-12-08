"""
Encoder models for autoencoders and VAEs.

Consolidation notes:
- Extracted `_extract_tensor_from_input()` helper to reduce code duplication
- Moved channel dropout logic to base class `_apply_channel_dropout()`
- VAEEncoder now properly inherits from Encoder, only overriding necessary parts
"""

import torch
import torch.nn as nn

from .components.base_component import BaseComponent
from .utils import compute_num_groups


def _extract_tensor_from_input(x):
    """
    Extract tensor from various input types (dict, tensor).
    
    This consolidates the repeated dict unpacking logic that was duplicated
    across Encoder.forward(), VAEEncoder.forward(), and Autoencoder.forward().
    
    Args:
        x: Input which can be:
           - torch.Tensor directly
           - dict with keys like "rgb", "input", "x", "data"
    
    Returns:
        torch.Tensor extracted from the input
    """
    if isinstance(x, torch.Tensor):
        return x
    
    if not isinstance(x, dict):
        raise TypeError(f"Expected Tensor or dict, got {type(x)}")
    
    # Priority order for input keys
    priority_keys = ["rgb", "input", "x", "data"]
    
    for key in priority_keys:
        if key in x:
            return x[key]
    
    # Single entry dict - use that value
    if len(x) == 1:
        return next(iter(x.values()))
    
    # Fallback: find first tensor value
    for v in x.values():
        if isinstance(v, torch.Tensor):
            return v
    
    raise ValueError(f"Could not extract tensor from dict with keys: {list(x.keys())}")


class Encoder(BaseComponent):
    """
    Deterministic encoder - outputs latent directly.
    
    Architecture per downsampling level:
        Conv 3x3 → Norm → Act → Conv 4x4 stride 2 → Norm → Act
    
    Channel progression (base_ch, down_steps from config):
        Level i: channels = base_ch * (2 ** i)
        Example with base_ch=64, down_steps=3:
            Level 0: 3 → 64 → 64      (256→128)
            Level 1: 64 → 128 → 128   (128→64)
            Level 2: 128 → 256 → 256  (64→32)
            Refinement: 256 → 256
            Latent proj: 256 → latent_ch
    
    Config:
        in_channels: Input channels (default: 3)
        base_channels: Base channel count (default: 64)
        downsampling_steps: Number of downsample levels (default: 3)
        latent_channels: Output latent channels (default: 4)
        norm_groups: Groups for GroupNorm (default: 8)
        activation: Activation function (default: SiLU)
        channel_dropout: Channel dropout rate (default: 0.0, disabled)
    """
    
    def _build(self):
        in_ch = self._init_kwargs.get("in_channels", 3)
        base_ch = self._init_kwargs.get("base_channels", 64)
        down_steps = self._init_kwargs.get("downsampling_steps", 3)
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        act = getattr(nn, self._init_kwargs.get("activation", "SiLU"))()
        self.channel_dropout_rate = self._init_kwargs.get("channel_dropout", 0.0)

        layers = []
        
        for i in range(down_steps):
            out_ch = base_ch * (2 ** i)
            valid_groups = compute_num_groups(out_ch, norm_groups)
            
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(valid_groups, out_ch),
                act,
                nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(valid_groups, out_ch),
                act,
            ]
            in_ch = out_ch
        
        # Final feature channels from last level
        final_ch = base_ch * (2 ** (down_steps - 1))
        valid_groups = compute_num_groups(final_ch, norm_groups)
        
        # Refinement layer at highest channel count
        layers += [
            nn.Conv2d(final_ch, final_ch, 3, padding=1),
            nn.GroupNorm(valid_groups, final_ch),
            act,
        ]
        
        self.feature_extractor = nn.Sequential(*layers)
        self._feature_channels = final_ch
        
        # Latent projection (overridden in VAEEncoder)
        self.latent_proj = nn.Conv2d(final_ch, latent_ch, 1)

    def _apply_channel_dropout(self, features):
        """
        Apply channel dropout during training if enabled.
        
        Extracted from forward() to avoid duplication between Encoder and VAEEncoder.
        
        Args:
            features: Feature tensor [B, C, H, W]
        
        Returns:
            Features with channel dropout applied (if training and rate > 0)
        """
        if self.channel_dropout_rate > 0.0 and self.training:
            channels = features.shape[1]
            device = features.device
            
            # Create random mask for channels: [channels]
            keep_prob = 1.0 - self.channel_dropout_rate
            channel_mask = torch.bernoulli(torch.ones(channels, device=device) * keep_prob)
            
            # Reshape to [1, channels, 1, 1] for broadcasting
            channel_mask = channel_mask.view(1, channels, 1, 1)
            features = features * channel_mask
        
        return features

    def _extract_features(self, x):
        """
        Extract features from input, handling dict inputs.
        
        Shared between Encoder and VAEEncoder to avoid duplication.
        
        Args:
            x: Input tensor [B, C, H, W] or dict
        
        Returns:
            Feature tensor after feature_extractor and channel dropout
        """
        x = _extract_tensor_from_input(x)
        features = self.feature_extractor(x)
        features = self._apply_channel_dropout(features)
        return features

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] or dict
        
        Returns:
            Dict: {"latent": z, "latent_features": features}
        """
        features = self._extract_features(x)
        z = self.latent_proj(features)
        return {"latent": z, "latent_features": features}
    
    def get_input_shape(self, batch_size=1):
        in_ch = self._init_kwargs.get("in_channels", 3)
        return (batch_size, in_ch, None, None)
    
    def get_output_shape(self, batch_size=1):
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        return {
            "latent": (batch_size, latent_ch, None, None),
            "latent_features": (batch_size, self._feature_channels, None, None)
        }


class VAEEncoder(Encoder):
    """
    Variational encoder - outputs mu and logvar.
    
    Inherits from Encoder and only overrides the projection heads
    and forward method output.
    """
    
    def _build(self):
        super()._build()
        
        # Remove deterministic projection
        if hasattr(self, 'latent_proj'):
            delattr(self, 'latent_proj')
        
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        self.mu_head = nn.Conv2d(self._feature_channels, latent_ch, 1)
        self.logvar_head = nn.Conv2d(self._feature_channels, latent_ch, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] or dict
        
        Returns:
            Dict: {"mu": mu, "logvar": logvar, "latent_features": features}
        """
        # Use parent's _extract_features which handles dict unpacking and channel dropout
        features = self._extract_features(x)
        
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        return {"mu": mu, "logvar": logvar, "latent_features": features}
    
    def get_output_shape(self, batch_size=1):
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        return {
            "mu": (batch_size, latent_ch, None, None),
            "logvar": (batch_size, latent_ch, None, None),
            "latent_features": (batch_size, self._feature_channels, None, None)
        }
