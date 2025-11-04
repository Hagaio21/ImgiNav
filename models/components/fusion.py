"""
Fusion components for combining control features with skip connections.

Different fusion strategies:
- AddFusion: Simple element-wise addition
- LearnedFusion: Simple FILM-style learned fusion (gamma * skip + beta)
- FILMFusion: Explicit FILM with configurable MLP/conv and optional normalization/residual
- ConcatFusion: Channel concatenation (requires architecture changes)
- ScaledAddFusion: Scaled addition with learned per-channel scale
- GatedFusion: Learned gating between skip and control features
"""

import torch
import torch.nn as nn
from .base_component import BaseComponent


class BaseFusion(BaseComponent):
    """Base class for fusion operations."""
    
    def forward(self, skip, ctrl_feat):
        """
        Fuse control features with skip connections.
        
        Args:
            skip: Skip connection from UNet [B, C, H, W]
            ctrl_feat: Control feature from adapter [B, C, H, W]
        
        Returns:
            Fused feature [B, C, H, W] (or [B, 2*C, H, W] for concat)
        """
        raise NotImplementedError


class AddFusion(BaseFusion):
    """Simple element-wise addition fusion."""
    
    def _build(self):
        # No parameters needed - just addition
        pass
    
    def forward(self, skip, ctrl_feat):
        return skip + ctrl_feat


class LearnedFusion(BaseFusion):
    """FILM-style learned fusion: gamma * skip + beta."""
    
    def _build(self):
        channels = self._init_kwargs.get("channels")
        if channels is None:
            raise ValueError("LearnedFusion requires 'channels' parameter")
        
        # Small convs to produce gamma and beta from control features
        self.gamma = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Conv2d(channels, channels, 1)
        
        # Initialize to identity (gamma=1, beta=0) for backward compatibility
        # This makes it start like AddFusion
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)
    
    def forward(self, skip, ctrl_feat):
        gamma = self.gamma(ctrl_feat)
        beta = self.beta(ctrl_feat)
        return gamma * skip + beta


class FILMFusion(BaseFusion):
    """
    Explicit FILM (Feature-wise Linear Modulation) fusion.
    
    FILM generates affine transformation parameters (gamma, beta) from conditioning
    and applies: output = gamma * skip + beta
    
    This is more flexible than simple addition and can learn to modulate features
    per-channel and per-spatial-location.
    """
    
    def _build(self):
        channels = self._init_kwargs.get("channels")
        if channels is None:
            raise ValueError("FILMFusion requires 'channels' parameter")
        
        # Method to generate gamma/beta: 'conv' or 'mlp'
        method = self._init_kwargs.get("method", "conv")
        use_residual = self._init_kwargs.get("use_residual", False)
        use_normalization = self._init_kwargs.get("use_normalization", False)
        
        self.use_residual = use_residual
        self.use_normalization = use_normalization
        
        if method == "conv":
            # Simple 1x1 conv to generate gamma and beta
            self.gamma_net = nn.Conv2d(channels, channels, 1)
            self.beta_net = nn.Conv2d(channels, channels, 1)
        elif method == "mlp":
            # MLP with normalization for richer transformation
            hidden_dim = channels * 2
            self.gamma_net = nn.Sequential(
                nn.Conv2d(channels, hidden_dim, 1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, channels, 1)
            )
            self.beta_net = nn.Sequential(
                nn.Conv2d(channels, hidden_dim, 1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, channels, 1)
            )
        else:
            raise ValueError(f"Unknown FILM method: {method}. Choose 'conv' or 'mlp'")
        
        # Initialize gamma to ~1 and beta to ~0 for identity initialization
        # This makes it start like AddFusion
        if method == "conv":
            nn.init.ones_(self.gamma_net.weight)
            nn.init.zeros_(self.beta_net.weight)
            if self.gamma_net.bias is not None:
                nn.init.zeros_(self.gamma_net.bias)
            if self.beta_net.bias is not None:
                nn.init.zeros_(self.beta_net.bias)
        else:
            # For MLP, initialize last layer to produce identity
            # Initialize last conv layer biases to produce ~1 for gamma, ~0 for beta
            nn.init.zeros_(self.gamma_net[-1].weight)
            nn.init.ones_(self.gamma_net[-1].bias)
            nn.init.zeros_(self.beta_net[-1].weight)
            nn.init.zeros_(self.beta_net[-1].bias)
        
        # Optional normalization
        if use_normalization:
            self.norm = nn.GroupNorm(8, channels)
        else:
            self.norm = nn.Identity()
    
    def forward(self, skip, ctrl_feat):
        # Generate gamma and beta from control features
        gamma = self.gamma_net(ctrl_feat)
        beta = self.beta_net(ctrl_feat)
        
        # Apply FILM: gamma * skip + beta
        output = gamma * skip + beta
        
        # Optional normalization
        output = self.norm(output)
        
        # Optional residual connection (skip + modulated)
        if self.use_residual:
            output = output + skip
        
        return output


class ConcatFusion(BaseFusion):
    """Channel concatenation fusion (doubles channels)."""
    
    def _build(self):
        channels = self._init_kwargs.get("channels")
        if channels is None:
            raise ValueError("ConcatFusion requires 'channels' parameter")
        
        # Optional: learnable projection back to original channels
        # This would require modifying UpBlock architecture
        self.project = self._init_kwargs.get("project", False)
        if self.project:
            self.proj_conv = nn.Conv2d(channels * 2, channels, 1)
        else:
            self.proj_conv = None
    
    def forward(self, skip, ctrl_feat):
        # Concatenate along channel dimension
        fused = torch.cat([skip, ctrl_feat], dim=1)  # [B, 2*C, H, W]
        
        # Optionally project back to original channels
        if self.project and self.proj_conv is not None:
            fused = self.proj_conv(fused)
        
        return fused


class ScaledAddFusion(BaseFusion):
    """Scaled addition with learned scale factor."""
    
    def _build(self):
        channels = self._init_kwargs.get("channels")
        if channels is None:
            raise ValueError("ScaledAddFusion requires 'channels' parameter")
        
        # Learnable scale factor per channel
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
    
    def forward(self, skip, ctrl_feat):
        return skip + self.scale * ctrl_feat


class GatedFusion(BaseFusion):
    """Gated fusion: learnable gating between skip and control features."""
    
    def _build(self):
        channels = self._init_kwargs.get("channels")
        if channels is None:
            raise ValueError("GatedFusion requires 'channels' parameter")
        
        # Gate that learns how much to use control features
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()  # Gate between 0 and 1
        )
    
    def forward(self, skip, ctrl_feat):
        # Compute gate from both features
        gate_input = torch.cat([skip, ctrl_feat], dim=1)
        gate_value = self.gate(gate_input)
        
        # Gated combination
        return gate_value * skip + (1 - gate_value) * ctrl_feat


# Registry for fusion types
FUSION_REGISTRY = {
    "add": AddFusion,
    "learned": LearnedFusion,
    "film": FILMFusion,  # Explicit FILM implementation
    "concat": ConcatFusion,
    "scaled_add": ScaledAddFusion,
    "gated": GatedFusion,
}

