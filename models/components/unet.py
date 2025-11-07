import torch
import torch.nn as nn
from .base_component import BaseComponent
from .blocks import TimeEmbedding, DownBlock, UpBlock, ResidualBlock


class Unet(BaseComponent):
    """Simplified UNet without conditioning support (removed DualUNet conditioning logic)."""
    
    def _build(self):
        in_ch = self._init_kwargs.get("in_channels", 3)
        out_ch = self._init_kwargs.get("out_channels", 3)
        base_ch = self._init_kwargs.get("base_channels", 64)
        depth = self._init_kwargs.get("depth", 4)
        num_res_blocks = self._init_kwargs.get("num_res_blocks", 1)
        time_dim = self._init_kwargs.get("time_dim", 128)
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        dropout = self._init_kwargs.get("dropout", 0.0)

        self.time_mlp = TimeEmbedding(time_dim)

        self.downs = nn.ModuleList()
        prev_ch = in_ch
        feats = []

        for i in range(depth):
            ch = base_ch * (2 ** i)
            self.downs.append(DownBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout))
            prev_ch = ch
            feats.append(ch)

        self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_dim, norm_groups, dropout)

        self.ups = nn.ModuleList()
        for ch in reversed(feats):
            self.ups.append(UpBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout))
            prev_ch = ch

        self.final = nn.Conv2d(prev_ch, out_ch, 1)

    def forward(self, x_t, t, cond=None):
        """Forward pass. cond parameter is ignored (kept for API compatibility)."""
        t_emb = self.time_mlp(t.float())
        skips = []

        for down in self.downs:
            x_t, skip = down(x_t, t_emb)
            skips.append(skip)

        x_t = self.bottleneck(x_t, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)

        return self.final(x_t)
    
    def freeze_blocks(self, block_names):
        """Freeze specific blocks by name."""
        if isinstance(block_names, str):
            block_names = [block_names]
        
        for name in block_names:
            if name == "downs":
                for block in self.downs:
                    for p in block.parameters():
                        p.requires_grad = False
            elif name == "ups":
                for block in self.ups:
                    for p in block.parameters():
                        p.requires_grad = False
            elif name == "bottleneck":
                for p in self.bottleneck.parameters():
                    p.requires_grad = False
            elif name == "time_mlp":
                for p in self.time_mlp.parameters():
                    p.requires_grad = False
            elif name == "final":
                for p in self.final.parameters():
                    p.requires_grad = False
            else:
                raise ValueError(f"Unknown block name: {name}")
    
    def freeze_downblocks(self):
        """Freeze all downsampling blocks (for ControlNet attachment)."""
        self.freeze_blocks(["downs"])
    
    def freeze_upblocks(self):
        """Freeze all upsampling blocks."""
        self.freeze_blocks(["ups"])
    
    def get_skip_connections(self, x_t, t):
        """
        Forward pass that returns skip connections for ControlNet attachment.
        
        Returns:
            tuple: (output, skips) where skips is a list of skip connection tensors
        """
        t_emb = self.time_mlp(t.float())
        skips = []
        
        for down in self.downs:
            x_t, skip = down(x_t, t_emb)
            skips.append(skip)
        
        x_t = self.bottleneck(x_t, t_emb)
        
        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)
        
        return self.final(x_t), skips

    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "in_channels": self._init_kwargs.get("in_channels", 3),
            "out_channels": self._init_kwargs.get("out_channels", 3),
            "base_channels": self._init_kwargs.get("base_channels", 64),
            "depth": self._init_kwargs.get("depth", 4),
            "num_res_blocks": self._init_kwargs.get("num_res_blocks", 1),
            "time_dim": self._init_kwargs.get("time_dim", 128),
            "norm_groups": self._init_kwargs.get("norm_groups", 8),
            "dropout": self._init_kwargs.get("dropout", 0.0),
        })
        return cfg


# Backward compatibility: Keep DualUNet as an alias for Unet
# This allows old checkpoints to load without breaking
class DualUNet(Unet):
    """
    Deprecated: DualUNet is now an alias for Unet.
    
    This class is kept for backward compatibility with old checkpoints.
    Use Unet instead for new code. The migration script can convert old checkpoints.
    """
    def _build(self):
        # Remove conditioning-related kwargs before building
        kwargs = self._init_kwargs.copy()
        kwargs.pop("cond_channels", None)
        kwargs.pop("fusion_mode", None)
        kwargs.pop("cond_mult", None)
        self._init_kwargs = kwargs
        super()._build()
