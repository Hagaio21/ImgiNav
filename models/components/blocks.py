import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(1, dim * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t = self.act(self.fc1(t))
        return self.fc2(t)


from ..utils import compute_num_groups


def _compute_num_heads(channels, target_heads_per_32=1):
    """
    Compute valid number of attention heads that divides channels evenly.
    
    Args:
        channels: Number of channels
        target_heads_per_32: Target number of heads per 32 channels (default: 1, i.e., channels // 32)
    
    Returns:
        Valid num_heads that divides channels evenly
    """
    # Target: channels // 32 heads (or similar ratio)
    target = max(1, channels // (32 // target_heads_per_32))
    
    # Find the largest divisor of channels that is <= target
    for num_heads in range(min(target, channels), 0, -1):
        if channels % num_heads == 0:
            return num_heads
    return 1  # Fallback: single head


class ResidualBlock(nn.Module):
    """
    Residual block with optional time embedding.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        time_dim: Time embedding dimension (None to disable time conditioning)
        norm_groups: Number of groups for GroupNorm
        dropout: Dropout rate
    """
    def __init__(self, in_ch, out_ch, time_dim=None, norm_groups=8, dropout=0.0):
        super().__init__()
        norm_groups_in = compute_num_groups(in_ch, norm_groups)
        norm_groups_out = compute_num_groups(out_ch, norm_groups)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Dropout after first conv
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # Time embedding (optional)
        self.use_time_emb = time_dim is not None
        if self.use_time_emb:
            self.time_emb = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(norm_groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Dropout after second conv
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _compute_features(self, x, t_emb=None):
        """Compute features up to conv2 output (before skip connection)."""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.dropout1(h)

        # Add time embedding if enabled and provided
        if self.use_time_emb and t_emb is not None:
            t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        h = self.dropout2(h)
        return h

    def forward(self, x, t_emb=None, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            t_emb: Optional time embedding [B, time_dim]
            **kwargs: Additional kwargs (ignored, for API consistency)
        
        Returns:
            Output tensor [B, out_ch, H, W]
        """
        h = self._compute_features(x, t_emb)
        return h + self.skip(x)


class DownBlock(nn.Module):
    """
    Downsampling block with optional time embedding.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        time_dim: Time embedding dimension (None to disable time conditioning)
        num_res_blocks: Number of residual blocks
        norm_groups: Number of groups for GroupNorm
        dropout: Dropout rate
    """
    def __init__(self, in_ch, out_ch, time_dim=None, num_res_blocks=1, norm_groups=8, dropout=0.0):
        super().__init__()
        self.use_time_emb = time_dim is not None
        
        self.res_blocks = nn.ModuleList([
            self._create_res_block(
                in_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout
            )
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def _create_res_block(self, in_ch, out_ch, time_dim, norm_groups, dropout):
        """Factory method to create residual blocks. Override in subclasses to use different block types."""
        return ResidualBlock(in_ch, out_ch, time_dim, norm_groups, dropout)

    def forward(self, x, t_emb=None, return_skip=True, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            t_emb: Optional time embedding [B, time_dim]
            return_skip: If True, return (output, skip). If False, return output only.
            **kwargs: Additional kwargs passed to res blocks
        
        Returns:
            If return_skip: (downsampled output, skip connection)
            Else: downsampled output only
        """
        for res in self.res_blocks:
            x = res(x, t_emb, **kwargs)
        skip = x
        x = self.downsample(x)
        
        if return_skip:
            return x, skip
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with optional time embedding and skip connections.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        time_dim: Time embedding dimension (None to disable time conditioning)
        num_res_blocks: Number of residual blocks
        norm_groups: Number of groups for GroupNorm
        dropout: Dropout rate
        use_skip_connection: If True, expects skip connection input (for UNet). If False, no skip (for VAE decoder).
    """
    def __init__(self, in_ch, out_ch, time_dim=None, num_res_blocks=1, norm_groups=8, dropout=0.0, use_skip_connection=True):
        super().__init__()
        self.use_time_emb = time_dim is not None
        self.use_skip_connection = use_skip_connection

        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)

        # First res block input channels depends on skip connection
        first_in_ch = out_ch + out_ch if use_skip_connection else out_ch
        
        self.res_blocks = nn.ModuleList([
            self._create_res_block(
                first_in_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout
            )
            for i in range(num_res_blocks)
        ])

    def _create_res_block(self, in_ch, out_ch, time_dim, norm_groups, dropout):
        """Factory method to create residual blocks. Override in subclasses to use different block types."""
        return ResidualBlock(in_ch, out_ch, time_dim, norm_groups, dropout)

    def forward(self, x, skip=None, t_emb=None, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            skip: Optional skip connection tensor [B, C, H*2, W*2]
            t_emb: Optional time embedding [B, time_dim]
            **kwargs: Additional kwargs passed to res blocks
        
        Returns:
            Output tensor [B, out_ch, H*2, W*2]
        """
        x = self.upsample(x)
        
        if self.use_skip_connection:
            if skip is None:
                raise ValueError("UpBlock with use_skip_connection=True requires skip tensor")
            x = torch.cat([x, skip], dim=1)
        
        for res in self.res_blocks:
            x = res(x, t_emb, **kwargs)
        return x


class MidBlock(nn.Module):
    """
    Middle block for processing at lowest resolution.
    
    Args:
        channels: Number of channels
        time_dim: Time embedding dimension (None to disable time conditioning)
        norm_groups: Number of groups for GroupNorm
        dropout: Dropout rate
        num_blocks: Number of residual blocks (default: 2)
    """
    def __init__(self, channels, time_dim=None, norm_groups=8, dropout=0.0, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_dim, norm_groups, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x, t_emb=None, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            t_emb: Optional time embedding [B, time_dim]
            **kwargs: Additional kwargs passed to res blocks
        
        Returns:
            Output tensor [B, C, H, W]
        """
        for block in self.blocks:
            x = block(x, t_emb, **kwargs)
        return x

class SelfAttentionBlock(nn.Module):

    def __init__(self, channels, num_heads=None, norm_groups=8, enable_cross_attention=False, conditioning_channels=None, window_size=None):
        super().__init__()
        self.channels = channels
        self.norm_groups = compute_num_groups(channels, norm_groups)
        self.enable_cross_attention = enable_cross_attention
        self.window_size = window_size  # If None, use full attention
        
        if num_heads is None:
            num_heads = _compute_num_heads(channels, target_heads_per_32=1)
        else:
            if channels % num_heads != 0:
                num_heads = _compute_num_heads(channels, target_heads_per_32=1)
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(self.norm_groups, channels)
        self.act = nn.SiLU()
        self.q_proj = nn.Conv2d(channels, channels, 1)
        
        if enable_cross_attention:
            self.k_proj = nn.Conv2d(channels, channels, 1)
            self.v_proj = nn.Conv2d(channels, channels, 1)
            # Initialize ctrl_proj if conditioning_channels is provided and different from channels
            if conditioning_channels is not None and conditioning_channels != channels:
                self.ctrl_proj = nn.Conv2d(conditioning_channels, channels, 1)
            else:
                self.ctrl_proj = None
        else:
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
        
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # For windowed attention: use shifted windows for cross-window communication
        # Shift by half window size to enable communication between adjacent windows
        self.shift_size = window_size // 2 if window_size is not None and window_size > 1 else 0
        
    def window_partition(self, x, window_size):
        """
        Partition input into non-overlapping windows.
        
        Args:
            x: Input tensor [B, C, H, W]
            window_size: Window size (int, assumes square windows)
        
        Returns:
            Windows [B * num_windows, C, window_size, window_size]
            (Hp, Wp): Padded height and width
            num_windows: (num_windows_h, num_windows_w)
        """
        B, C, H, W = x.shape
        
        # Pad if necessary
        pad_l = pad_t = 0
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            Hp, Wp = H + pad_b, W + pad_r
        else:
            Hp, Wp = H, W
        
        num_windows_h = Hp // window_size
        num_windows_w = Wp // window_size
        num_windows = num_windows_h * num_windows_w
        
        # Reshape to windows: [B, C, Hp, Wp] -> [B * num_windows, C, window_size, window_size]
        x = x.view(B, C, num_windows_h, window_size, num_windows_w, window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * num_windows, C, window_size, window_size)
        
        return x, (Hp, Wp), (num_windows_h, num_windows_w)
    
    def window_reverse(self, windows, window_size, Hp, Wp, H, W):
        """
        Reverse window partition.
        
        Args:
            windows: Windows [B * num_windows, C, window_size, window_size]
            window_size: Window size
            Hp, Wp: Padded height and width
            H, W: Original height and width
        
        Returns:
            Reconstructed tensor [B, C, H, W]
        """
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        num_windows_h = Hp // window_size
        num_windows_w = Wp // window_size
        
        # Reshape back: [B * num_windows, C, window_size, window_size] -> [B, C, Hp, Wp]
        x = windows.view(B, num_windows_h, num_windows_w, -1, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, -1, Hp, Wp)
        
        # Crop padding
        if Hp > H or Wp > W:
            x = x[:, :, :H, :W]
        
        return x
    
    def forward(self, x, **kwargs):
        """
        Args:
            x: Input tensor [B, C, H, W]
            **kwargs: May include conditioning_signal [B, C_cond, H_cond, W_cond]
                     If provided and cross-attention is enabled, uses it for K, V
        
        Returns:
            Output tensor [B, C, H, W]
        """
        conditioning_signal = kwargs.get("conditioning_signal", None)
        B, C, H, W = x.shape
        
        # Window shifting for cross-window communication (alternate between shifted and non-shifted)
        if self.window_size is not None and self.shift_size > 0:
            # Cyclic shift
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
        
        h = self.act(self.norm(shifted_x))
        q = self.q_proj(h)
        
        if self.enable_cross_attention:
            if conditioning_signal is not None:
                # Cross-attention: use conditioning signal for K, V
                cond_signal = conditioning_signal
                
                # Ensure dtype matches input x (for mixed precision training)
                cond_signal = cond_signal.to(dtype=x.dtype)
                
                if cond_signal.shape[2:] != (H, W):
                    cond_signal = F.interpolate(
                        cond_signal, size=(H, W), mode='bilinear', align_corners=False
                    )
                
                # Apply same shift to conditioning signal if windowed
                if self.window_size is not None and self.shift_size > 0:
                    cond_signal = torch.roll(cond_signal, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
                
                if cond_signal.shape[1] != C:
                    if self.ctrl_proj is None:
                        raise RuntimeError(
                            f"Conditioning signal has {cond_signal.shape[1]} channels but attention block expects {C} channels. "
                            f"ctrl_proj was not initialized. Set conditioning_channels={cond_signal.shape[1]} when creating the attention block."
                        )
                    cond_signal = self.ctrl_proj(cond_signal)
                
                k = self.k_proj(cond_signal)
                v = self.v_proj(cond_signal)
                del cond_signal
            else:
                # Self-attention fallback: use input for K, V when conditioning_signal is None (CFG dropout)
                k = self.k_proj(h)
                v = self.v_proj(h)
        else:
            # Standard self-attention: use qkv projection
            qkv = self.qkv(h)
            q, k, v = qkv.chunk(3, dim=1)
        
        if C % self.num_heads != 0:
            raise ValueError(
                f"Channels ({C}) must be divisible by num_heads ({self.num_heads}). "
                f"This should have been caught at initialization. "
                f"Model may have been modified incorrectly."
            )
        head_dim = C // self.num_heads
        
        # Windowed attention or full attention
        if self.window_size is not None:
            # Windowed attention
            q_windows, (Hp, Wp), (num_windows_h, num_windows_w) = self.window_partition(q, self.window_size)
            k_windows, _, _ = self.window_partition(k, self.window_size)
            v_windows, _, _ = self.window_partition(v, self.window_size)
            
            B_windows = q_windows.shape[0]
            window_size_sq = self.window_size * self.window_size
            
            # Reshape for attention: [B_windows, C, window_size, window_size] -> [B_windows, num_heads, head_dim, window_size_sq]
            q_windows = q_windows.view(B_windows, self.num_heads, head_dim, window_size_sq)
            k_windows = k_windows.view(B_windows, self.num_heads, head_dim, window_size_sq)
            v_windows = v_windows.view(B_windows, self.num_heads, head_dim, window_size_sq)
            
            q_windows = q_windows.transpose(-2, -1)  # [B_windows, num_heads, window_size_sq, head_dim]
            k_windows = k_windows.transpose(-2, -1)
            v_windows = v_windows.transpose(-2, -1)
            
            scale = (head_dim ** -0.5)
            attn = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out_windows = torch.matmul(attn, v_windows)  # [B_windows, num_heads, window_size_sq, head_dim]
            
            # Reshape back: [B_windows, num_heads, window_size_sq, head_dim] -> [B_windows, C, window_size, window_size]
            out_windows = out_windows.transpose(-2, -1).contiguous()
            out_windows = out_windows.view(B_windows, C, self.window_size, self.window_size)
            
            # Reverse window partition
            out = self.window_reverse(out_windows, self.window_size, Hp, Wp, H, W)
            
            # Reverse cyclic shift
            if self.shift_size > 0:
                out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            # Full attention (original implementation with chunking for memory)
            q = q.view(B, self.num_heads, head_dim, H * W)
            k = k.view(B, self.num_heads, head_dim, H * W)
            v = v.view(B, self.num_heads, head_dim, H * W)
            
            q = q.transpose(-2, -1)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            
            scale = (head_dim ** -0.5)
            seq_len = H * W
            
            if seq_len <= 64:
                chunk_size = seq_len
            elif seq_len <= 256:
                chunk_size = 64
            else:
                chunk_size = 32
            
            if seq_len > chunk_size:
                out_chunks = []
                k_t = k.transpose(-2, -1)
                for i in range(0, seq_len, chunk_size):
                    end_idx = min(i + chunk_size, seq_len)
                    q_chunk = q[:, :, i:end_idx, :]
                    attn_chunk = torch.matmul(q_chunk, k_t) * scale
                    attn_chunk = F.softmax(attn_chunk, dim=-1)
                    out_chunk = torch.matmul(attn_chunk, v)
                    out_chunks.append(out_chunk)
                    del attn_chunk, q_chunk, out_chunk
                
                out = torch.cat(out_chunks, dim=2)
                del out_chunks, k_t
            else:
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = F.softmax(attn, dim=-1)
                out = torch.matmul(attn, v)
            
            out = out.transpose(-2, -1).contiguous()
            out = out.view(B, C, H, W)
        
        out = self.proj(out)
        return x + out

class ResidualBlockWithAttention(ResidualBlock):
    """
    Residual block with optional self-attention.
    Extends ResidualBlock by adding attention after the second conv.
    """
    def __init__(self, in_ch, out_ch, time_dim=None, norm_groups=8, dropout=0.0, use_attention=False, attention_heads=None, enable_cross_attention=False, conditioning_channels=None, window_size=None):
        super().__init__(in_ch, out_ch, time_dim, norm_groups, dropout)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttentionBlock(
                out_ch, num_heads=attention_heads, norm_groups=norm_groups,
                enable_cross_attention=enable_cross_attention,
                conditioning_channels=conditioning_channels,
                window_size=window_size
            )
        else:
            self.attention = None

    def forward(self, x, t_emb=None, **kwargs):
        h = self._compute_features(x, t_emb)
        
        if self.use_attention:
            h = self.attention(h, **kwargs)

        return h + self.skip(x)

class DownBlockWithAttention(DownBlock):
    """DownBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim=None, num_res_blocks=1, norm_groups=8, dropout=0.0, 
                 use_attention=False, attention_heads=None, enable_cross_attention=False, conditioning_channels=None, window_size=None):
        # Store attention params for _create_res_block
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.enable_cross_attention = enable_cross_attention
        self.conditioning_channels = conditioning_channels
        self.window_size = window_size
        super().__init__(in_ch, out_ch, time_dim, num_res_blocks, norm_groups, dropout)

    def _create_res_block(self, in_ch, out_ch, time_dim, norm_groups, dropout):
        """Override to use ResidualBlockWithAttention instead of ResidualBlock."""
        return ResidualBlockWithAttention(
            in_ch, out_ch, time_dim, norm_groups, dropout,
            use_attention=self.use_attention, attention_heads=self.attention_heads,
            enable_cross_attention=self.enable_cross_attention,
            conditioning_channels=self.conditioning_channels,
            window_size=self.window_size
        )


class UpBlockWithAttention(UpBlock):
    """UpBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim=None, num_res_blocks=1, norm_groups=8, dropout=0.0,
                 use_attention=False, attention_heads=None, enable_cross_attention=False, conditioning_channels=None,
                 use_skip_connection=True, window_size=None):
        # Store attention params for _create_res_block
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.enable_cross_attention = enable_cross_attention
        self.conditioning_channels = conditioning_channels
        self.window_size = window_size
        super().__init__(in_ch, out_ch, time_dim, num_res_blocks, norm_groups, dropout, use_skip_connection)

    def _create_res_block(self, in_ch, out_ch, time_dim, norm_groups, dropout):
        """Override to use ResidualBlockWithAttention instead of ResidualBlock."""
        return ResidualBlockWithAttention(
            in_ch, out_ch, time_dim, norm_groups, dropout,
            use_attention=self.use_attention, attention_heads=self.attention_heads,
            enable_cross_attention=self.enable_cross_attention,
            conditioning_channels=self.conditioning_channels,
            window_size=self.window_size
        )