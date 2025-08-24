import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """A block with two convolutional layers, batch norm, and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """A downsampling block using MaxPool followed by DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """An upsampling block that handles concatenation with a skip connection."""
    def __init__(self, in_channels_up, in_channels_skip, out_channels):
        super().__init__()
        # The up-sampling layer only takes the channels from the previous feature map
        self.up = nn.ConvTranspose2d(in_channels_up, in_channels_up // 2, kernel_size=2, stride=2)
        # The subsequent convolution takes the concatenated channels
        self.conv = DoubleConv((in_channels_up // 2) + in_channels_skip, out_channels)

    def forward(self, x1, x2):
        # x1 is the feature map from the previous layer, x2 is the skip connection
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CrossAttention(nn.Module):
    """A generic cross-attention module."""
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query, context):
        B, T, C = query.shape
        B, S, D = context.shape
        q = self.to_q(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)