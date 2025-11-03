import torch
import torch.nn as nn


class LatentNormalizer(nn.Module):
    """
    Normalizes latents to approximately zero mean and unit variance.
    Uses learnable shift and scale parameters that are trained end-to-end.
    
    Normalization: z_norm = (z - shift) / (scale + eps)
    Denormalization: z = z_norm * scale + shift
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        # Learnable parameters: shift (mean) and scale (std)
        # Initialized to zero shift and unit scale
        self.register_parameter('shift', nn.Parameter(torch.zeros(1, num_channels, 1, 1)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, num_channels, 1, 1)))
    
    def forward(self, x, reverse=False):
        """
        Args:
            x: Input tensor [B, C, H, W]
            reverse: If True, denormalize (multiply scale and add shift)
                    If False, normalize (subtract shift and divide by scale)
        """
        # Clamp log_scale for numerical stability
        scale = torch.exp(self.log_scale.clamp(min=-10, max=10))
        
        if reverse:
            # Denormalize: x = (x * scale) + shift
            return x * scale + self.shift
        else:
            # Normalize: x = (x - shift) / (scale + eps)
            return (x - self.shift) / (scale + self.eps)

