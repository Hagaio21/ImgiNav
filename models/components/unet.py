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
    
    # Default configuration values (defined once, used in both _build and to_config)
    DEFAULT_CONFIG = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 64,
        "depth": 4,
        "num_res_blocks": 1,
        "time_dim": 128,
        "norm_groups": 8,
        "dropout": 0.0,
        "use_attention": True,
        "attention_heads": None,
        "attention_at": ["bottleneck", "downs", "ups"],
        "enable_cross_attention": False,
        "window_size": None,  # If None, use full attention. If int, use windowed attention with this window size
    }

    def _build(self):
        in_ch = self._init_kwargs.get("in_channels", self.DEFAULT_CONFIG["in_channels"])
        out_ch = self._init_kwargs.get("out_channels", self.DEFAULT_CONFIG["out_channels"])
        base_ch = self._init_kwargs.get("base_channels", self.DEFAULT_CONFIG["base_channels"])
        depth = self._init_kwargs.get("depth", self.DEFAULT_CONFIG["depth"])
        num_res_blocks = self._init_kwargs.get("num_res_blocks", self.DEFAULT_CONFIG["num_res_blocks"])
        time_dim = self._init_kwargs.get("time_dim", self.DEFAULT_CONFIG["time_dim"])
        norm_groups = self._init_kwargs.get("norm_groups", self.DEFAULT_CONFIG["norm_groups"])
        dropout = self._init_kwargs.get("dropout", self.DEFAULT_CONFIG["dropout"])
        
        # Attention configuration
        use_attention = self._init_kwargs.get("use_attention", self.DEFAULT_CONFIG["use_attention"])
        attention_heads = self._init_kwargs.get("attention_heads", self.DEFAULT_CONFIG["attention_heads"])
        attention_at = self._init_kwargs.get("attention_at", self.DEFAULT_CONFIG["attention_at"])
        enable_cross_attention = self._init_kwargs.get("enable_cross_attention", self.DEFAULT_CONFIG["enable_cross_attention"])
        conditioning_channels = self._init_kwargs.get("conditioning_channels", None)
        window_size = self._init_kwargs.get("window_size", self.DEFAULT_CONFIG["window_size"])
        
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
                # Alternate between shifted and non-shifted windows for cross-window communication
                # Use shifted windows on even layers, non-shifted on odd layers
                layer_window_size = window_size if window_size is not None else None
                # Note: shift is handled internally by SelfAttentionBlock based on shift_size
                self.downs.append(DownBlockWithAttention(
                    prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout,
                    use_attention=True, attention_heads=attention_heads,
                    enable_cross_attention=enable_cross_attention,
                    conditioning_channels=conditioning_channels,
                    window_size=layer_window_size
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
                conditioning_channels=conditioning_channels,
                window_size=window_size
            )
        else:
            self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_dim, norm_groups, dropout)

        # Build upsampling blocks
        self.ups = nn.ModuleList()
        use_attn_ups = use_attention and "ups" in attention_at
        for ch in reversed(feats):
            if use_attn_ups:
                layer_window_size = window_size if window_size is not None else None
                self.ups.append(UpBlockWithAttention(
                    prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout,
                    use_attention=True, attention_heads=attention_heads,
                    enable_cross_attention=enable_cross_attention,
                    conditioning_channels=conditioning_channels,
                    window_size=layer_window_size
                ))
            else:
                self.ups.append(UpBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout))
            prev_ch = ch

        self.final = nn.Conv2d(prev_ch, out_ch, 1)

    def forward(self, x_t, t, conditioning_signal=None):
        """
        Forward pass with conditioning signals for cross-attention.
        
        Args:
            x_t: Noisy latents [B, C, H, W]
            t: Timesteps [B]
            conditioning_signal: Optional conditioning signal tensor [B, C_cond, H_cond, W_cond] for cross-attention
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        t_emb = self.time_mlp(t.float())
        
        skips = []

        for down in self.downs:
            x_t, skip = down(x_t, t_emb, conditioning_signal=conditioning_signal)
            skips.append(skip)

        x_t = self.bottleneck(x_t, t_emb, conditioning_signal=conditioning_signal)

        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb, conditioning_signal=conditioning_signal)

        return self.final(x_t)
    
    def get_input_shape(self, batch_size=1):
        """Get expected input shape."""
        in_ch = self._init_kwargs.get("in_channels", self.DEFAULT_CONFIG["in_channels"])
        # Latent shape depends on encoder/decoder, but typically matches latent_channels
        # For 512x512 images with 4 downsampling steps, latent is 32x32
        return (batch_size, in_ch, None, None)  # H, W are flexible
    
    def get_output_shape(self, batch_size=1):
        """Get expected output shape (same as input for UNet)."""
        out_ch = self._init_kwargs.get("out_channels", self.DEFAULT_CONFIG["out_channels"])
        # UNet outputs same spatial resolution as input
        return (batch_size, out_ch, None, None)  # H, W match input

    def to_config(self):
        return super().to_config()
