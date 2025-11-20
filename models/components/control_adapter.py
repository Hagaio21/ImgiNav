"""
Control adapter implementations for converting conditions to control features.

Different adapter architectures can be used:
- SimpleAdapter: Basic linear/conv projections
- MLPAdapter: Multi-layer with non-linearities
- DeepAdapter: Deeper networks for more complex conditioning
"""

import torch
import torch.nn as nn
from .base_component import BaseComponent


class BaseControlAdapter(BaseComponent):
    """Base class for control adapters."""
    
    def forward(self, text_emb, pov_emb):
        """
        Convert conditioning inputs to multi-scale control features.
        
        Args:
            text_emb: Text/graph embeddings [B, text_dim]
            pov_emb: POV embeddings [B, pov_dim, H, W] or [B, pov_dim] (spatial or non-spatial)
        
        Returns:
            List of control features, one per UNet level [feat_level0, feat_level1, ...]
            Each feature should be [B, channels, H, W] matching skip connection dimensions
        """
        raise NotImplementedError


class SimpleAdapter(BaseControlAdapter):
    """Simple adapter: basic linear/conv projections."""
    
    def _build(self):
        text_dim = self._init_kwargs.get("text_dim", 768)
        pov_dim = self._init_kwargs.get("pov_dim", 256)
        base_channels = self._init_kwargs.get("base_channels", 64)
        depth = self._init_kwargs.get("depth", 4)
        pov_is_spatial = self._init_kwargs.get("pov_is_spatial", True)
        
        self.pov_is_spatial = pov_is_spatial
        
        # Text projection: Linear layers with non-linearity
        self.text_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, base_channels * (2 ** i)),
                nn.SiLU(),
                nn.Linear(base_channels * (2 ** i), base_channels * (2 ** i))
            )
            for i in range(depth)
        ])
        
        # POV projection: depends on whether it's spatial or not
        if pov_is_spatial:
            # Spatial: use Conv2d
            self.pov_proj = nn.ModuleList([
                nn.Conv2d(pov_dim, base_channels * (2 ** i), 1)
                for i in range(depth)
            ])
        else:
            # Non-spatial: use Linear layers (like text)
            self.pov_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(pov_dim, base_channels * (2 ** i)),
                    nn.SiLU(),
                    nn.Linear(base_channels * (2 ** i), base_channels * (2 ** i))
                )
                for i in range(depth)
            ])
    
    def forward(self, text_emb, pov_emb):
        feats = []
        for tp, pp in zip(self.text_proj, self.pov_proj):
            t = tp(text_emb).unsqueeze(-1).unsqueeze(-1)  # [B, ch, 1, 1]
            
            if self.pov_is_spatial:
                p = pp(pov_emb)  # [B, ch, H, W]
            else:
                p = pp(pov_emb).unsqueeze(-1).unsqueeze(-1)  # [B, ch, 1, 1]
            
            # Combine text and POV features
            # No normalization - let scaled_add fusion learn the appropriate scale
            feats.append(t + p)
        return feats


class MLPAdapter(BaseControlAdapter):
    """MLP adapter: non-linear projections with normalization."""
    
    def _build(self):
        text_dim = self._init_kwargs.get("text_dim", 768)
        pov_dim = self._init_kwargs.get("pov_dim", 256)
        base_channels = self._init_kwargs.get("base_channels", 64)
        depth = self._init_kwargs.get("depth", 4)
        pov_is_spatial = self._init_kwargs.get("pov_is_spatial", True)
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        use_final_activation = self._init_kwargs.get("use_final_activation", True)
        
        self.pov_is_spatial = pov_is_spatial
        
        # Text projection: deeper MLP with normalization
        self.text_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, base_channels * (2 ** i)),
                nn.SiLU(),
                nn.Linear(base_channels * (2 ** i), base_channels * (2 ** i)),
                nn.SiLU() if use_final_activation else nn.Identity()
            )
            for i in range(depth)
        ])
        
        # POV projection: non-linear with normalization
        if pov_is_spatial:
            self.pov_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(pov_dim, base_channels * (2 ** i), 1),
                    nn.GroupNorm(norm_groups, base_channels * (2 ** i)),
                    nn.SiLU()
                )
                for i in range(depth)
            ])
        else:
            self.pov_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(pov_dim, base_channels * (2 ** i)),
                    nn.SiLU(),
                    nn.Linear(base_channels * (2 ** i), base_channels * (2 ** i)),
                    nn.SiLU() if use_final_activation else nn.Identity()
                )
                for i in range(depth)
            ])
        
        # Optional: non-linear combination after adding text + pov
        self.combine_activation = nn.SiLU() if use_final_activation else nn.Identity()
    
    def forward(self, text_emb, pov_emb):
        feats = []
        for tp, pp in zip(self.text_proj, self.pov_proj):
            t = tp(text_emb).unsqueeze(-1).unsqueeze(-1)
            
            if self.pov_is_spatial:
                p = pp(pov_emb)
            else:
                p = pp(pov_emb).unsqueeze(-1).unsqueeze(-1)
            
            combined = t + p
            # No normalization - let scaled_add fusion learn the appropriate scale
            feats.append(self.combine_activation(combined))
        return feats


class DeepAdapter(BaseControlAdapter):
    """Deep adapter: multiple layers for more complex conditioning."""
    
    def _build(self):
        text_dim = self._init_kwargs.get("text_dim", 768)
        pov_dim = self._init_kwargs.get("pov_dim", 256)
        base_channels = self._init_kwargs.get("base_channels", 64)
        depth = self._init_kwargs.get("depth", 4)
        pov_is_spatial = self._init_kwargs.get("pov_is_spatial", True)
        num_layers = self._init_kwargs.get("num_layers", 3)  # Depth of MLP per level
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        
        self.pov_is_spatial = pov_is_spatial
        
        # Text projection: deep MLP
        self.text_proj = nn.ModuleList([])
        for i in range(depth):
            layers = []
            ch = base_channels * (2 ** i)
            layers.append(nn.Linear(text_dim, ch))
            layers.append(nn.SiLU())
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(ch, ch))
                layers.append(nn.SiLU())
            
            self.text_proj.append(nn.Sequential(*layers))
        
        # POV projection: deep with normalization
        self.pov_proj = nn.ModuleList([])
        for i in range(depth):
            ch = base_channels * (2 ** i)
            if pov_is_spatial:
                layers = [nn.Conv2d(pov_dim, ch, 1)]
                layers.append(nn.GroupNorm(norm_groups, ch))
                layers.append(nn.SiLU())
                
                for _ in range(num_layers - 1):
                    layers.append(nn.Conv2d(ch, ch, 1))
                    layers.append(nn.GroupNorm(norm_groups, ch))
                    layers.append(nn.SiLU())
            else:
                layers = [nn.Linear(pov_dim, ch), nn.SiLU()]
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(ch, ch))
                    layers.append(nn.SiLU())
            
            self.pov_proj.append(nn.Sequential(*layers))
    
    def forward(self, text_emb, pov_emb):
        feats = []
        for tp, pp in zip(self.text_proj, self.pov_proj):
            t = tp(text_emb).unsqueeze(-1).unsqueeze(-1)
            
            if self.pov_is_spatial:
                p = pp(pov_emb)
            else:
                p = pp(pov_emb).unsqueeze(-1).unsqueeze(-1)
            
            combined = t + p
            # No normalization - let scaled_add fusion learn the appropriate scale
            feats.append(combined)
        return feats
