import torch
import torch.nn as nn
from .base_component import BaseComponent
from .blocks import (
    TimeEmbedding, DownBlock, UpBlock, ResidualBlock,
    DownBlockWithAttention, UpBlockWithAttention, ResidualBlockWithAttention
)


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


class UnetWithAttention(BaseComponent):
    """
    UNet with self-attention support.
    
    Similar to Unet but uses ResidualBlockWithAttention, DownBlockWithAttention,
    and UpBlockWithAttention to enable attention in residual blocks.
    
    Config:
        use_attention: If True, enable attention in all residual blocks (default: True)
        attention_heads: Number of attention heads per block (default: None, auto)
        attention_at: List of where to apply attention: ["bottleneck", "downs", "ups"] (default: all)
    """
    
    def _build(self):
        in_ch = self._init_kwargs.get("in_channels", 3)
        out_ch = self._init_kwargs.get("out_channels", 3)
        base_ch = self._init_kwargs.get("base_channels", 64)
        depth = self._init_kwargs.get("depth", 4)
        num_res_blocks = self._init_kwargs.get("num_res_blocks", 1)
        time_dim = self._init_kwargs.get("time_dim", 128)
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        dropout = self._init_kwargs.get("dropout", 0.0)
        
        # Attention configuration
        use_attention = self._init_kwargs.get("use_attention", True)
        attention_heads = self._init_kwargs.get("attention_heads", None)
        attention_at = self._init_kwargs.get("attention_at", ["bottleneck", "downs", "ups"])
        
        if not isinstance(attention_at, list):
            attention_at = [attention_at] if attention_at else []
        
        # Conditioning configuration (optional, backward compatible)
        cond_dim = self._init_kwargs.get("cond_dim", 0)
        num_cond_classes = self._init_kwargs.get("num_cond_classes", 0)
        
        self.time_mlp = TimeEmbedding(time_dim)
        
        # Build condition embedding if specified
        if cond_dim > 0 and num_cond_classes > 0:
            from .blocks import ConditionEmbedding
            self.cond_embedding = ConditionEmbedding(num_cond_classes, cond_dim)
        else:
            self.cond_embedding = None
            cond_dim = 0  # Ensure cond_dim is 0 if not using conditioning

        self.downs = nn.ModuleList()
        prev_ch = in_ch
        feats = []

        # Build downsampling blocks
        use_attn_downs = use_attention and "downs" in attention_at
        for i in range(depth):
            ch = base_ch * (2 ** i)
            if use_attn_downs:
                self.downs.append(DownBlockWithAttention(
                    prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout,
                    use_attention=True, attention_heads=attention_heads, cond_dim=cond_dim
                ))
            else:
                self.downs.append(DownBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout, cond_dim))
            prev_ch = ch
            feats.append(ch)

        # Bottleneck with optional attention
        use_attn_bottleneck = use_attention and "bottleneck" in attention_at
        if use_attn_bottleneck:
            self.bottleneck = ResidualBlockWithAttention(
                prev_ch, prev_ch, time_dim, norm_groups, dropout,
                use_attention=True, attention_heads=attention_heads, cond_dim=cond_dim
            )
        else:
            self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_dim, norm_groups, dropout, cond_dim)

        # Build upsampling blocks
        self.ups = nn.ModuleList()
        use_attn_ups = use_attention and "ups" in attention_at
        for ch in reversed(feats):
            if use_attn_ups:
                self.ups.append(UpBlockWithAttention(
                    prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout,
                    use_attention=True, attention_heads=attention_heads, cond_dim=cond_dim
                ))
            else:
                self.ups.append(UpBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout, cond_dim))
            prev_ch = ch

        self.final = nn.Conv2d(prev_ch, out_ch, 1)

    def forward(self, x_t, t, cond=None):
        """
        Forward pass with optional conditioning.
        
        Args:
            x_t: Noisy latents [B, C, H, W]
            t: Timesteps [B]
            cond: Optional condition IDs [B] where 0=ROOM, 1=SCENE. If None, no conditioning is used.
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        t_emb = self.time_mlp(t.float())
        
        # Process conditioning if available
        cond_emb = None
        if self.cond_embedding is not None and cond is not None:
            cond_emb = self.cond_embedding(cond)  # [B, cond_dim]
        
        skips = []

        for down in self.downs:
            x_t, skip = down(x_t, t_emb, cond_emb)
            skips.append(skip)

        x_t = self.bottleneck(x_t, t_emb, cond_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb, cond_emb)

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
    
    def get_skip_connections(self, x_t, t, cond=None):
        """
        Forward pass that returns skip connections for ControlNet attachment.
        
        Args:
            x_t: Noisy latents [B, C, H, W]
            t: Timesteps [B]
            cond: Optional condition IDs [B] where 0=ROOM, 1=SCENE. If None, no conditioning is used.
        
        Returns:
            tuple: (output, skips) where skips is a list of skip connection tensors
        """
        t_emb = self.time_mlp(t.float())
        
        # Process conditioning if available
        cond_emb = None
        if self.cond_embedding is not None and cond is not None:
            cond_emb = self.cond_embedding(cond)  # [B, cond_dim]
        
        skips = []
        
        for down in self.downs:
            x_t, skip = down(x_t, t_emb, cond_emb)
            skips.append(skip)
        
        x_t = self.bottleneck(x_t, t_emb, cond_emb)
        
        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb, cond_emb)
        
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
            "use_attention": self._init_kwargs.get("use_attention", True),
            "attention_heads": self._init_kwargs.get("attention_heads", None),
            "attention_at": self._init_kwargs.get("attention_at", ["bottleneck", "downs", "ups"]),
            "cond_dim": self._init_kwargs.get("cond_dim", 0),
            "num_cond_classes": self._init_kwargs.get("num_cond_classes", 0),
        })
        return cfg
