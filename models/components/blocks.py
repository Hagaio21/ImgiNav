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
    Self-attention block for UNet with optional cross-attention support for ControlNet signals.
    Applies self-attention to capture long-range spatial dependencies.
    Can optionally use cross-attention with ControlNet signals as keys/values.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: channels // 32, min 1)
        norm_groups: Number of groups for GroupNorm (default: 8)
        enable_cross_attention: If True, enables cross-attention with ControlNet signals (default: False)
    """
    def __init__(self, channels, num_heads=None, norm_groups=8, enable_cross_attention=False):
        super().__init__()
        self.channels = channels
        self.norm_groups = _compute_num_groups(channels, norm_groups)
        self.enable_cross_attention = enable_cross_attention
        
        # Ensure num_heads divides channels evenly
        if num_heads is None:
            num_heads = _compute_num_heads(channels, target_heads_per_32=1)
        else:
            # If explicitly provided, ensure it divides channels
            if channels % num_heads != 0:
                # Find the closest valid num_heads
                num_heads = _compute_num_heads(channels, target_heads_per_32=1)
        self.num_heads = num_heads
        
        # GroupNorm + SiLU + Q projection (always needed)
        self.norm = nn.GroupNorm(self.norm_groups, channels)
        self.act = nn.SiLU()
        
        # Query projection (always needed)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        
        if enable_cross_attention:
            # For cross-attention: separate K, V projections for ControlNet signals
            # ControlNet signals may have different channel dimensions, so we use adaptive projections
            # We'll project controlnet signals to match channels if needed
            self.k_proj = nn.Conv2d(channels, channels, 1)  # Will be applied to control signals
            self.v_proj = nn.Conv2d(channels, channels, 1)  # Will be applied to control signals
            # Note: If controlnet signals have different channels, we'll create projection on first use
            # This is stored as a module attribute but initialized lazily
            self._ctrl_proj = None
            self._ctrl_proj_channels = None
        else:
            # For self-attention: QKV projection
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
        
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, controlnet_signal=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            controlnet_signal: Optional ControlNet signal tensor [B, C_ctrl, H_ctrl, W_ctrl]
                             If provided and cross-attention is enabled, uses it for K, V
        
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Normalize and activate
        h = self.act(self.norm(x))
        
        # Compute Query from input
        q = self.q_proj(h)  # [B, C, H, W]
        
        if self.enable_cross_attention and controlnet_signal is not None:
            # Cross-attention: Q from input, K and V from ControlNet signal
            # Ensure controlnet_signal has the same spatial dimensions (or can be interpolated)
            if controlnet_signal.shape[2:] != (H, W):
                # Interpolate controlnet signal to match spatial dimensions
                controlnet_signal = F.interpolate(
                    controlnet_signal, size=(H, W), mode='bilinear', align_corners=False
                )
            
            # Ensure channel match - if different, use 1x1 conv to project
            if controlnet_signal.shape[1] != C:
                # Project controlnet signal to match channels
                ctrl_in_channels = controlnet_signal.shape[1]
                if self._ctrl_proj is None or self._ctrl_proj_channels != ctrl_in_channels:
                    # Create or recreate projection with the correct input channels
                    self._ctrl_proj = nn.Conv2d(ctrl_in_channels, C, 1).to(controlnet_signal.device)
                    self._ctrl_proj_channels = ctrl_in_channels
                    # Register as a submodule so it's saved/loaded properly
                    # Use a unique name to avoid conflicts
                    self.add_module('ctrl_proj', self._ctrl_proj)
                controlnet_signal = self._ctrl_proj(controlnet_signal)
            
            # Compute K, V from controlnet signal
            k = self.k_proj(controlnet_signal)  # [B, C, H, W]
            v = self.v_proj(controlnet_signal)  # [B, C, H, W]
        else:
            # Self-attention: Q, K, V all from input
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
        
        # Use chunked attention for large sequences to reduce memory
        # Chunk size: process in chunks to avoid large intermediate tensors
        seq_len = H * W
        # Reduce chunk size for better memory efficiency with cross-attention
        # Smaller chunks = less memory per attention matrix
        chunk_size = 256  # Reduced from 512 to 256 for better memory efficiency
        
        if seq_len > chunk_size:
            # Chunked attention for memory efficiency
            out_chunks = []
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                q_chunk = q[:, :, i:end_idx, :]  # [B, num_heads, chunk_len, head_dim]
                
                # Compute attention scores for this chunk
                attn_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale  # [B, num_heads, chunk_len, seq_len]
                attn_chunk = F.softmax(attn_chunk, dim=-1)
                
                # Apply to values
                out_chunk = torch.matmul(attn_chunk, v)  # [B, num_heads, chunk_len, head_dim]
                out_chunks.append(out_chunk)
            
            out = torch.cat(out_chunks, dim=2)  # [B, num_heads, seq_len, head_dim]
        else:
            # Standard attention for small sequences
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
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
            # Check if cross-attention should be enabled (can be set via config)
            enable_cross_attention = getattr(self, '_enable_cross_attention', False)
            self.attention = SelfAttentionBlock(
                out_ch, num_heads=attention_heads, norm_groups=norm_groups,
                enable_cross_attention=enable_cross_attention
            )
        else:
            self.attention = None

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond_emb=None, controlnet_signal=None):
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
            h = self.attention(h, controlnet_signal=controlnet_signal)

        return h + self.skip(x)


class DownBlockWithAttention(nn.Module):
    """DownBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0, 
                 use_attention=False, attention_heads=None, cond_dim=0, enable_cross_attention=False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlockWithAttention(
                in_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout,
                use_attention=use_attention, attention_heads=attention_heads, cond_dim=cond_dim
            )
            for i in range(num_res_blocks)
        ])
        # Set cross-attention flag on attention blocks
        if enable_cross_attention:
            for res_block in self.res_blocks:
                if hasattr(res_block, 'attention') and res_block.attention is not None:
                    res_block.attention.enable_cross_attention = True
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb, cond_emb=None, controlnet_signal=None):
        for res in self.res_blocks:
            x = res(x, t_emb, cond_emb, controlnet_signal=controlnet_signal)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlockWithAttention(nn.Module):
    """UpBlock that uses ResidualBlockWithAttention instead of ResidualBlock."""
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm_groups=8, dropout=0.0,
                 use_attention=False, attention_heads=None, cond_dim=0, enable_cross_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.res_blocks = nn.ModuleList([
            ResidualBlockWithAttention(
                out_ch + out_ch if i == 0 else out_ch, out_ch, time_dim, norm_groups, dropout,
                use_attention=use_attention, attention_heads=attention_heads, cond_dim=cond_dim
            )
            for i in range(num_res_blocks)
        ])
        # Set cross-attention flag on attention blocks
        if enable_cross_attention:
            for res_block in self.res_blocks:
                if hasattr(res_block, 'attention') and res_block.attention is not None:
                    res_block.attention.enable_cross_attention = True

    def forward(self, x, skip, t_emb, cond_emb=None, controlnet_signal=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, t_emb, cond_emb, controlnet_signal=controlnet_signal)
        return x