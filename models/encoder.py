import torch
import torch.nn as nn

from .components.base_component import BaseComponent
from .utils import compute_num_groups


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
            # Channel count for this level: doubles each level
            out_ch = base_ch * (2 ** i)
            valid_groups = compute_num_groups(out_ch, norm_groups)
            
            layers += [
                # First conv: change channels (in_ch → out_ch)
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(valid_groups, out_ch),
                act,
                # Second conv: downsample, keep channels (out_ch → out_ch)
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
        
        # Latent projection
        self.latent_proj = nn.Conv2d(final_ch, latent_ch, 1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] or DataFlow
        
        Returns:
            DataFlow: {"latent": z, "latent_features": features}
        """
        if isinstance(x, dict) and not isinstance(x, torch.Tensor):
            if "rgb" in x:
                x = x["rgb"]
            elif "input" in x:
                x = x["input"]
            elif "x" in x:
                x = x["x"]
            elif len(x) == 1:
                x = next(iter(x.values()))
            else:
                for key in ["rgb", "input", "x", "data"]:
                    if key in x:
                        x = x[key]
                        break
                else:
                    x = next(v for v in x.values() if isinstance(v, torch.Tensor))
        
        features = self.feature_extractor(x)
        
        # Apply channel dropout if enabled (only during training)
        if self.channel_dropout_rate > 0.0 and self.training:
            batch_size, channels, height, width = features.shape
            # Create random mask for channels: [channels]
            channel_mask = torch.bernoulli(torch.ones(channels, device=features.device) * (1.0 - self.channel_dropout_rate))
            # Reshape to [1, channels, 1, 1] for broadcasting
            channel_mask = channel_mask.view(1, channels, 1, 1)
            features = features * channel_mask
        
        z = self.latent_proj(features)
        return self._to_dataflow({"latent": z, "latent_features": features})
    
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
    """Variational encoder - outputs mu and logvar."""
    
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
            x: Input tensor [B, C, H, W] or DataFlow
        
        Returns:
            DataFlow: {"mu": mu, "logvar": logvar, "latent_features": features}
        """
        if isinstance(x, dict) and not isinstance(x, torch.Tensor):
            if "rgb" in x:
                x = x["rgb"]
            elif "input" in x:
                x = x["input"]
            elif "x" in x:
                x = x["x"]
            elif len(x) == 1:
                x = next(iter(x.values()))
            else:
                for key in ["rgb", "input", "x", "data"]:
                    if key in x:
                        x = x[key]
                        break
                else:
                    x = next(v for v in x.values() if isinstance(v, torch.Tensor))
        
        features = self.feature_extractor(x)
        
        # Apply channel dropout if enabled (only during training)
        if self.channel_dropout_rate > 0.0 and self.training:
            batch_size, channels, height, width = features.shape
            # Create random mask for channels: [channels]
            channel_mask = torch.bernoulli(torch.ones(channels, device=features.device) * (1.0 - self.channel_dropout_rate))
            # Reshape to [1, channels, 1, 1] for broadcasting
            channel_mask = channel_mask.view(1, channels, 1, 1)
            features = features * channel_mask
        
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        return self._to_dataflow({"mu": mu, "logvar": logvar, "latent_features": features})
    
    def get_output_shape(self, batch_size=1):
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        return {
            "mu": (batch_size, latent_ch, None, None),
            "logvar": (batch_size, latent_ch, None, None),
            "latent_features": (batch_size, self._feature_channels, None, None)
        }