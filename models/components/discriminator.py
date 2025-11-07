"""
Latent space discriminator for distinguishing viable vs non-viable layout latents.
"""

import torch
import torch.nn as nn
from .base_component import BaseComponent


class LatentDiscriminator(BaseComponent):
    """
    Discriminator that operates in latent space.
    Distinguishes between viable (real) and non-viable (fake) layout latents.
    Outputs continuous score [0, 1] where 1 = viable, 0 = non-viable.
    """
    
    def _build(self):
        latent_channels = self._init_kwargs.get("latent_channels", 16)
        base_channels = self._init_kwargs.get("base_channels", 64)
        num_layers = self._init_kwargs.get("num_layers", 4)
        use_batch_norm = self._init_kwargs.get("use_batch_norm", True)
        
        layers = []
        in_channels = latent_channels
        
        # Downsample layers
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        
        # Final classification layer
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()  # Output continuous value [0, 1]
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, latents):
        """
        Args:
            latents: [B, C, H, W] latent tensors
        
        Returns:
            scores: [B, 1] probability scores (1 = viable, 0 = non-viable)
        """
        return self.net(latents)
    
    def to_config(self):
        cfg = super().to_config()
        cfg["latent_channels"] = self._init_kwargs.get("latent_channels", 16)
        cfg["base_channels"] = self._init_kwargs.get("base_channels", 64)
        cfg["num_layers"] = self._init_kwargs.get("num_layers", 4)
        cfg["use_batch_norm"] = self._init_kwargs.get("use_batch_norm", True)
        return cfg
    
    @classmethod
    def from_config(cls, cfg):
        """Create discriminator from config dict."""
        return cls(**cfg)

