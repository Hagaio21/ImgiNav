import torch
import torch.nn as nn

from .components.base_component import BaseComponent

class Encoder(BaseComponent):
    """Deterministic encoder - outputs latent directly."""
    
    def _build(self):
        act = getattr(nn, self._init_kwargs.get("activation", "SiLU"))()
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        in_ch = self._init_kwargs.get("in_channels", 3)
        out_ch = self._init_kwargs.get("base_channels", 64)
        down_steps = self._init_kwargs.get("downsampling_steps", 4)
        latent_ch = self._init_kwargs.get("latent_channels", 4)

        # Build feature extractor
        layers = []
        for _ in range(down_steps):
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(norm_groups, out_ch),
                act,
                nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(norm_groups, out_ch),
                act,
            ]
            in_ch = out_ch
            out_ch *= 2
        
        # Final feature extraction layer (before latent projection)
        layers += [
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(norm_groups, in_ch),
            act,
        ]
        self.feature_extractor = nn.Sequential(*layers)
        # Store feature extractor output channels for subclasses
        self._feature_channels = in_ch
        
        # Regular deterministic encoder: project features to latent
        self.latent_proj = nn.Conv2d(in_ch, latent_ch, 1)

    def forward(self, x):
        """
        Forward pass. Returns deterministic latent.
        
        Returns:
            Dictionary: {"latent": z, "latent_features": features}
        """
        # Extract features
        features = self.feature_extractor(x)
        # Project features to latent
        z = self.latent_proj(features)
        return {"latent": z, "latent_features": features}
    
    def get_input_shape(self, batch_size=1):
        """Get expected input shape."""
        in_ch = self._init_kwargs.get("in_channels", 3)
        # Assume input is RGB image (512x512 by default, but shape is flexible)
        return (batch_size, in_ch, None, None)  # H, W are flexible
    
    def get_output_shape(self, batch_size=1):
        """Get expected output shape."""
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        down_steps = self._init_kwargs.get("downsampling_steps", 4)
        # Latent is downsampled by 2^down_steps
        # If input is 512x512, output is 512/(2^down_steps) = 32x32
        spatial_res = None  # Depends on input, but typically 32x32 for 512x512 input
        return {
            "latent": (batch_size, latent_ch, spatial_res, spatial_res),
            "latent_features": (batch_size, self._feature_channels, spatial_res, spatial_res)
        }


class VAEEncoder(Encoder):
    """Variational encoder - outputs mu and logvar for VAE."""
    
    def _build(self):
        # Call parent to build feature extractor
        super()._build()
        
        # Remove the deterministic latent projection
        if hasattr(self, 'latent_proj'):
            delattr(self, 'latent_proj')
        
        # Get latent_channels from config
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        # Use feature channels from parent
        in_ch = self._feature_channels
        
        # VAE mode: output mu and logvar from features
        self.mu_head = nn.Conv2d(in_ch, latent_ch, 1)
        self.logvar_head = nn.Conv2d(in_ch, latent_ch, 1)
    
    def forward(self, x):
        """
        Forward pass. Returns mu and logvar for VAE.
        
        Returns:
            Dictionary: {"mu": mu, "logvar": logvar, "latent_features": features}
        """
        # Extract features (from parent)
        features = self.feature_extractor(x)
        
        # VAE mode: project features to mu and logvar
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        return {"mu": mu, "logvar": logvar, "latent_features": features}
    
    def get_output_shape(self, batch_size=1):
        """Get expected output shape."""
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        down_steps = self._init_kwargs.get("downsampling_steps", 4)
        spatial_res = None  # Depends on input, but typically 32x32 for 512x512 input
        return {
            "mu": (batch_size, latent_ch, spatial_res, spatial_res),
            "logvar": (batch_size, latent_ch, spatial_res, spatial_res),
            "latent_features": (batch_size, self._feature_channels, spatial_res, spatial_res)
        }
