"""Model construction utilities and helper functions."""
import torch.nn as nn
from .maps import NORM_MAP, ACT_MAP


def build_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
               norm=None, act=None, dropout=0.0, transpose=False):
    """Build a convolutional layer with optional norm, activation, and dropout."""
    if transpose:
        output_padding = 0 if stride <= 1 else stride - 1
        conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
    else:
        conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    layers = [conv]

    if norm:
        norm_cls = NORM_MAP.get(norm)
        if norm_cls:
            layers.append(norm_cls(out_channels))

    if act:
        act_cls = ACT_MAP.get(act)
        if act_cls:
            layers.append(act_cls(inplace=False))

    if dropout and dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers)


def build_norm_layer(norm_type, num_channels, target_size=None):
    """Build a normalization layer using NORM_MAP."""
    norm_cls = NORM_MAP.get(norm_type)
    if norm_cls is None:
        raise ValueError(f"Unsupported norm_type: {norm_type}. Supported: {list(NORM_MAP.keys())}")
    
    if norm_type == "layer":
        if target_size is None:
            raise ValueError("target_size required for layer norm")
        H, W = target_size
        return norm_cls([num_channels, H, W], elementwise_affine=True)
    elif norm_type in (None, "group", "default"):
        return norm_cls(1, num_channels, affine=True)
    elif norm_type in ("batch", "instance"):
        return norm_cls(num_channels, affine=True)
    else:
        return norm_cls()
