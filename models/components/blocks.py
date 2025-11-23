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


class ConditionEmbedding(nn.Module):
    """
    Simple embedding for discrete conditions (room/scene IDs).
    Similar to TimeEmbedding but for categorical labels.
    """
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)
    
    def forward(self, cond_ids):
        """
        Args:
            cond_ids: [B] tensor of class indices (0=ROOM, 1=SCENE)
        
        Returns:
            [B, dim] tensor of condition embeddings
        """
        return self.embedding(cond_ids)


def _compute_num_groups(num_channels, requested_groups=8):
    """Compute valid number of groups for GroupNorm."""
    # Find the largest valid divisor <= requested_groups
    for g in range(min(requested_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # Fallback: single group


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
    def __init__(self, in_ch, out_ch, time_dim, norm_groups=8, dropout=0.0):
        super().__init__()
        norm_groups_in = _compute_num_groups(in_ch, norm_groups)
        norm_groups_out = _compute_num_groups(out_ch, norm_groups)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Dropout after first conv
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.time_emb = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(norm_groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Dropout after second conv
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.dropout1(h)

        t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t  # Add time embedding

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        h = self.dropout2(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout)
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb):
        for res in self.res_blocks:
            x = res(x, t_emb)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_ch + out_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout)
            for i in range(num_res_blocks)
        ])

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, t_emb)
        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for UNet with optional cross-attention support for conditioning signals.
    Applies self-attention to capture long-range spatial dependencies.
    Can optionally use cross-attention with conditioning signals as keys/values.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: channels // 32, min 1)
        norm_groups: Number of groups for GroupNorm (default: 8)
        enable_cross_attention: If True, enables cross-attention with conditioning signals (default: False)
    """
    def __init__(self, channels, num_heads=None, norm_groups=8, enable_cross_attention=False, conditioning_channels=None):
        super().__init__()
        self.channels = channels
        self.norm_groups = _compute_num_groups(channels, norm_groups)
        self.enable_cross_attention = enable_cross_attention
        
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
        
    def forward(self, x, conditioning_signal=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            conditioning_signal: Optional conditioning signal tensor [B, C_cond, H_cond, W_cond]
                             If provided and cross-attention is enabled, uses it for K, V
        
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        h = self.act(self.norm(x))
        q = self.q_proj(h)
        
        if self.enable_cross_attention and conditioning_signal is not None:
            cond_signal = conditioning_signal
            
            if cond_signal.shape[2:] != (H, W):
                cond_signal = F.interpolate(
                    cond_signal, size=(H, W), mode='bilinear', align_corners=False
                )
            
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
            # Self-attention: use qkv projection
            qkv = self.qkv(h)
            q, k, v = qkv.chunk(3, dim=1)
        
        if C % self.num_heads != 0:
            raise ValueError(
                f"Channels ({C}) must be divisible by num_heads ({self.num_heads}). "
                f"This should have been caught at initialization. "
                f"Model may have been modified incorrectly."
            )
        head_dim = C // self.num_heads
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


class ResidualBlockWithAttention(nn.Module):
    """
    Residual block with optional self-attention.
    Similar to ResidualBlock but can include attention after the second conv.
    """
    def __init__(self, in_ch, out_ch, time_dim, norm_groups=8, dropout=0.0, use_attention=False, attention_heads=None, enable_cross_attention=False, conditioning_channels=None):
        super().__init__()
        norm_groups_in = _compute_num_groups(in_ch, norm_groups)
        norm_groups_out = _compute_num_groups(out_ch, norm_groups)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Dropout after first conv
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.time_emb = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(norm_groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttentionBlock(
                out_ch, num_heads=attention_heads, norm_groups=norm_groups,
                enable_cross_attention=enable_cross_attention,
                conditioning_channels=conditioning_channels
            )
        else:
            self.attention = None

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, conditioning_signal=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.dropout1(h)

        t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t  # Add time embedding

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        h = self.dropout2(h)
        
        if self.use_attention:
            h = self.attention(h, conditioning_signal=conditioning_signal)

        return h + self.skip(x)


class DownBlockWithAttention(nn.Module):
    """DownBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0, 
                 use_attention=False, attention_heads=None, enable_cross_attention=False, conditioning_channels=None):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlockWithAttention(
                in_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout,
                use_attention=use_attention, attention_heads=attention_heads,
                enable_cross_attention=enable_cross_attention,
                conditioning_channels=conditioning_channels
            )
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb, conditioning_signal=None):
        for res in self.res_blocks:
            x = res(x, t_emb, conditioning_signal=conditioning_signal)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlockWithAttention(nn.Module):
    """UpBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0,
                 use_attention=False, attention_heads=None, enable_cross_attention=False, conditioning_channels=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.res_blocks = nn.ModuleList([
            ResidualBlockWithAttention(
                out_ch + out_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout,
                use_attention=use_attention, attention_heads=attention_heads,
                enable_cross_attention=enable_cross_attention,
                conditioning_channels=conditioning_channels
            )
            for i in range(num_res_blocks)
        ])

    def forward(self, x, skip, t_emb, conditioning_signal=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, t_emb, conditioning_signal=conditioning_signal)
        return x