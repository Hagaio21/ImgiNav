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
    def __init__(self, in_ch, out_ch, time_dim, norm_groups=8, dropout=0.0, cond_dim=0):
        super().__init__()
        norm_groups_in = _compute_num_groups(in_ch, norm_groups)
        norm_groups_out = _compute_num_groups(out_ch, norm_groups)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Dropout after first conv
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.time_emb = nn.Linear(time_dim, out_ch)
        
        # Optional conditioning embedding
        if cond_dim > 0:
            self.cond_emb = nn.Linear(cond_dim, out_ch)
        else:
            self.cond_emb = None

        self.norm2 = nn.GroupNorm(norm_groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Dropout after second conv
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond_emb=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.dropout1(h)

        t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        if self.cond_emb is not None and cond_emb is not None:
            c = self.cond_emb(cond_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t + c  # Add both time and condition embeddings
        else:
            h = h + t  # Only time embedding (backward compatible)

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        h = self.dropout2(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0, cond_dim=0):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout, cond_dim)
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb, cond_emb=None):
        for res in self.res_blocks:
            x = res(x, t_emb, cond_emb)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0, cond_dim=0):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_ch + out_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout, cond_dim)
            for i in range(num_res_blocks)
        ])

    def forward(self, x, skip, t_emb, cond_emb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, t_emb, cond_emb)
        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for UNet.
    Applies self-attention to capture long-range spatial dependencies.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: channels // 32, min 1)
        norm_groups: Number of groups for GroupNorm (default: 8)
    """
    def __init__(self, channels, num_heads=None, norm_groups=8):
        super().__init__()
        self.channels = channels
        self.norm_groups = _compute_num_groups(channels, norm_groups)
        
        # Ensure num_heads divides channels evenly
        if num_heads is None:
            num_heads = _compute_num_heads(channels, target_heads_per_32=1)
        else:
            # If explicitly provided, ensure it divides channels
            if channels % num_heads != 0:
                # Find the closest valid num_heads
                num_heads = _compute_num_heads(channels, target_heads_per_32=1)
        self.num_heads = num_heads
        
        # GroupNorm + SiLU + QKV projection
        self.norm = nn.GroupNorm(self.norm_groups, channels)
        self.act = nn.SiLU()
        
        # Multi-head attention: Q, K, V projections
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Normalize and activate
        h = self.act(self.norm(x))
        
        # Compute Q, K, V
        qkv = self.qkv(h)  # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B, C, H, W]
        
        # Reshape for multi-head attention: [B, C, H, W] -> [B, num_heads, C//num_heads, H*W]
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        k = k.view(B, self.num_heads, head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        v = v.view(B, self.num_heads, head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        
        # Transpose for attention computation: [B, num_heads, head_dim, H*W] -> [B, num_heads, H*W, head_dim]
        q = q.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        k = k.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        v = v.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        
        # Scaled dot-product attention
        # Attention scores: [B, num_heads, H*W, H*W]
        scale = (head_dim ** -0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values: [B, num_heads, H*W, head_dim]
        out = torch.matmul(attn, v)
        
        # Reshape back: [B, num_heads, H*W, head_dim] -> [B, C, H, W]
        out = out.transpose(-2, -1).contiguous()  # [B, num_heads, head_dim, H*W]
        out = out.view(B, C, H, W)
        
        # Project and residual connection
        out = self.proj(out)
        return x + out  # Residual connection


class ResidualBlockWithAttention(nn.Module):
    """
    Residual block with optional self-attention.
    Similar to ResidualBlock but can include attention after the second conv.
    """
    def __init__(self, in_ch, out_ch, time_dim, norm_groups=8, dropout=0.0, use_attention=False, attention_heads=None, cond_dim=0):
        super().__init__()
        norm_groups_in = _compute_num_groups(in_ch, norm_groups)
        norm_groups_out = _compute_num_groups(out_ch, norm_groups)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Dropout after first conv
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.time_emb = nn.Linear(time_dim, out_ch)
        
        # Optional conditioning embedding
        if cond_dim > 0:
            self.cond_emb = nn.Linear(cond_dim, out_ch)
        else:
            self.cond_emb = None

        self.norm2 = nn.GroupNorm(norm_groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Dropout after second conv
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        
        # Optional self-attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttentionBlock(out_ch, num_heads=attention_heads, norm_groups=norm_groups)
        else:
            self.attention = None

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond_emb=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.dropout1(h)

        t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        if self.cond_emb is not None and cond_emb is not None:
            c = self.cond_emb(cond_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t + c  # Add both time and condition embeddings
        else:
            h = h + t  # Only time embedding (backward compatible)

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        h = self.dropout2(h)
        
        # Apply attention if enabled
        if self.use_attention:
            h = self.attention(h)

        return h + self.skip(x)


class DownBlockWithAttention(nn.Module):
    """DownBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0, 
                 use_attention=False, attention_heads=None, cond_dim=0):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlockWithAttention(
                in_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout,
                use_attention=use_attention, attention_heads=attention_heads, cond_dim=cond_dim
            )
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb, cond_emb=None):
        for res in self.res_blocks:
            x = res(x, t_emb, cond_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlockWithAttention(nn.Module):
    """UpBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0,
                 use_attention=False, attention_heads=None, cond_dim=0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.res_blocks = nn.ModuleList([
            ResidualBlockWithAttention(
                out_ch + out_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout,
                use_attention=use_attention, attention_heads=attention_heads, cond_dim=cond_dim
            )
            for i in range(num_res_blocks)
        ])

    def forward(self, x, skip, t_emb, cond_emb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, t_emb, cond_emb)
        return x