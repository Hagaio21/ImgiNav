import torch
import torch.nn as nn

from .components.base_component import BaseComponent
from .components.blocks import DownBlock, MidBlock
from .utils import compute_num_groups


class Encoder(BaseComponent):
    """
    Deterministic encoder with residual blocks.
    
    Uses shared block classes with time_dim=None (no time conditioning).
    
    Required config:
        in_channels: Input channels
        latent_channels: Latent space channels
        base_channels: Base channel count
        channel_multipliers: Channel multipliers per level
        num_res_blocks: Residual blocks per level
        norm_groups: GroupNorm groups
    
    Optional config:
        dropout: Dropout rate (default: 0.0)
    """
    
    def _build(self):
        in_ch = self._init_kwargs["in_channels"]
        latent_ch = self._init_kwargs["latent_channels"]
        base_ch = self._init_kwargs["base_channels"]
        ch_mults = self._init_kwargs["channel_multipliers"]
        num_res = self._init_kwargs["num_res_blocks"]
        norm_groups = self._init_kwargs["norm_groups"]
        dropout = self._init_kwargs.get("dropout", 0.0)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # Downsampling blocks (time_dim=None for VAE)
        self.down_blocks = nn.ModuleList()
        channels = [base_ch] + [base_ch * m for m in ch_mults]
        
        for i in range(len(ch_mults)):
            self.down_blocks.append(
                DownBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    time_dim=None,  # No time conditioning for VAE
                    num_res_blocks=num_res,
                    norm_groups=norm_groups,
                    dropout=dropout
                )
            )
        
        # Middle block
        mid_ch = channels[-1]
        self.mid_block = MidBlock(
            channels=mid_ch,
            time_dim=None,  # No time conditioning for VAE
            norm_groups=norm_groups,
            dropout=dropout
        )
        
        # Output
        self.norm_out = nn.GroupNorm(compute_num_groups(mid_ch, norm_groups), mid_ch)
        self.act_out = nn.SiLU()
        
        # Store feature channels
        self._feature_channels = mid_ch
        
        # Latent projection
        self.latent_proj = nn.Conv2d(mid_ch, latent_ch, 1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] or DataFlow
        
        Returns:
            DataFlow: {"latent": z, "latent_features": features}
        """
        # Handle DataFlow input
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
        
        # Initial conv
        x = self.conv_in(x)
        
        # Downsampling (no skip connections needed for VAE encoder)
        for block in self.down_blocks:
            x = block(x, t_emb=None, return_skip=False)
        
        # Middle
        x = self.mid_block(x)
        
        # Output
        features = self.act_out(self.norm_out(x))
        z = self.latent_proj(features)
        
        return self._to_dataflow({"latent": z, "latent_features": features})
    
    def get_input_shape(self, batch_size=1):
        in_ch = self._init_kwargs["in_channels"]
        return (batch_size, in_ch, None, None)
    
    def get_output_shape(self, batch_size=1):
        latent_ch = self._init_kwargs["latent_channels"]
        return {
            "latent": (batch_size, latent_ch, None, None),
            "latent_features": (batch_size, self._feature_channels, None, None)
        }


class VAEEncoder(Encoder):
    """
    Variational encoder - outputs mu and logvar.
    
    Same required config as Encoder.
    """
    
    def _build(self):
        super()._build()
        
        # Remove deterministic projection
        if hasattr(self, 'latent_proj'):
            delattr(self, 'latent_proj')
        
        # VAE heads
        latent_ch = self._init_kwargs["latent_channels"]
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
        # Handle DataFlow input
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
        
        # Initial conv
        x = self.conv_in(x)
        
        # Downsampling
        for block in self.down_blocks:
            x = block(x, t_emb=None, return_skip=False)
        
        # Middle
        x = self.mid_block(x)
        
        # Output
        features = self.act_out(self.norm_out(x))
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        return self._to_dataflow({
            "mu": mu,
            "logvar": logvar,
            "latent_features": features
        })
    
    def get_output_shape(self, batch_size=1):
        latent_ch = self._init_kwargs["latent_channels"]
        return {
            "mu": (batch_size, latent_ch, None, None),
            "logvar": (batch_size, latent_ch, None, None),
            "latent_features": (batch_size, self._feature_channels, None, None)
        }