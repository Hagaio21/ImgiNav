import torch
import torch.nn as nn
from .base_component import BaseComponent
from .blocks import (
    TimeEmbedding, DownBlock, UpBlock, ResidualBlock,
    DownBlockWithAttention, UpBlockWithAttention, ResidualBlockWithAttention
)


class UnetWithAttention(BaseComponent):
    """
    Unified UNet architecture with optional self-attention support.
    
    This class replaces both the old Unet and UnetWithAttention classes.
    When use_attention=False, it behaves like the old Unet (using standard blocks).
    When use_attention=True, it uses attention-enabled blocks.
    
    Config:
        use_attention: If True, enable attention in residual blocks (default: True).
                      When False, uses standard ResidualBlock/DownBlock/UpBlock.
        attention_heads: Number of attention heads per block (default: None, auto)
        attention_at: List of where to apply attention: ["bottleneck", "downs", "ups"] (default: all)
        enable_cross_attention: Enable cross-attention for conditioning signals (default: False)
        conditioning_channels: Number of channels in conditioning signal (required if enable_cross_attention=True)
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
        enable_cross_attention = self._init_kwargs.get("enable_cross_attention", False)
        conditioning_channels = self._init_kwargs.get("conditioning_channels", None)
        
        if not isinstance(attention_at, list):
            attention_at = [attention_at] if attention_at else []
        
        self.time_mlp = TimeEmbedding(time_dim)

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
                    use_attention=True, attention_heads=attention_heads,
                    enable_cross_attention=enable_cross_attention,
                    conditioning_channels=conditioning_channels
                ))
            else:
                self.downs.append(DownBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout))
            prev_ch = ch
            feats.append(ch)

        # Bottleneck with optional attention
        use_attn_bottleneck = use_attention and "bottleneck" in attention_at
        if use_attn_bottleneck:
            self.bottleneck = ResidualBlockWithAttention(
                prev_ch, prev_ch, time_dim, norm_groups, dropout,
                use_attention=True, attention_heads=attention_heads,
                enable_cross_attention=enable_cross_attention,
                conditioning_channels=conditioning_channels
            )
        else:
            self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_dim, norm_groups, dropout)

        # Build upsampling blocks
        self.ups = nn.ModuleList()
        use_attn_ups = use_attention and "ups" in attention_at
        for ch in reversed(feats):
            if use_attn_ups:
                self.ups.append(UpBlockWithAttention(
                    prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout,
                    use_attention=True, attention_heads=attention_heads,
                    enable_cross_attention=enable_cross_attention,
                    conditioning_channels=conditioning_channels
                ))
            else:
                self.ups.append(UpBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout))
            prev_ch = ch

        self.final = nn.Conv2d(prev_ch, out_ch, 1)

    def forward(self, x_t, t, cond=None, conditioning_signal=None):
        """
        Forward pass with conditioning signals for cross-attention.
        
        Args:
            x_t: Noisy latents [B, C, H, W]
            t: Timesteps [B]
            cond: Deprecated - kept for API compatibility, ignored
            conditioning_signal: Optional conditioning signal tensor [B, C_cond, H_cond, W_cond] for cross-attention
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        t_emb = self.time_mlp(t.float())
        
        skips = []

        for down in self.downs:
            if isinstance(down, DownBlockWithAttention):
                x_t, skip = down(x_t, t_emb, conditioning_signal=conditioning_signal)
            else:
                x_t, skip = down(x_t, t_emb)
            skips.append(skip)

        # Pass conditioning_signal to bottleneck if it's a ResidualBlockWithAttention
        if isinstance(self.bottleneck, ResidualBlockWithAttention):
            x_t = self.bottleneck(x_t, t_emb, conditioning_signal=conditioning_signal)
        else:
            x_t = self.bottleneck(x_t, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            if isinstance(up, UpBlockWithAttention):
                x_t = up(x_t, skip, t_emb, conditioning_signal=conditioning_signal)
            else:
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
            "use_attention": self._init_kwargs.get("use_attention", True),
            "attention_heads": self._init_kwargs.get("attention_heads", None),
            "attention_at": self._init_kwargs.get("attention_at", ["bottleneck", "downs", "ups"]),
            "enable_cross_attention": self._init_kwargs.get("enable_cross_attention", False),
        })
        return cfg
