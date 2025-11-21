# models/components/embedding_projection.py
import torch
import torch.nn as nn
from .base_component import BaseComponent


class EmbeddingToSpatial(BaseComponent):
    """
    Projects 1D embeddings to spatial feature maps for cross-attention.
    
    Combines text and POV embeddings into a single spatial feature map
    that can be used in cross-attention layers.
    
    Config:
        text_dim: Dimension of text/graph embeddings (default: 384)
        pov_dim: Dimension of POV embeddings (default: 512)
        output_channels: Number of output channels for spatial features (default: matches UNet base_channels)
        spatial_size: Target spatial size (H, W) - will be interpolated to match UNet resolution (default: (64, 64))
        combine_method: How to combine text and POV embeddings: "add", "concat", "concat_proj" (default: "concat_proj")
    """
    
    def _build(self):
        text_dim = self._init_kwargs.get("text_dim", 384)
        pov_dim = self._init_kwargs.get("pov_dim", 512)
        output_channels = self._init_kwargs.get("output_channels", 96)  # Default to match UNet base_channels
        spatial_size = self._init_kwargs.get("spatial_size", (64, 64))
        combine_method = self._init_kwargs.get("combine_method", "concat_proj")
        
        self.text_dim = text_dim
        self.pov_dim = pov_dim
        self.output_channels = output_channels
        self.spatial_size = spatial_size
        self.combine_method = combine_method
        
        if combine_method == "add":
            # Add embeddings (requires same dimension)
            if text_dim != pov_dim:
                # Project to common dimension
                common_dim = max(text_dim, pov_dim)
                self.text_proj = nn.Linear(text_dim, common_dim)
                self.pov_proj = nn.Linear(pov_dim, common_dim)
                combined_dim = common_dim
            else:
                self.text_proj = nn.Identity()
                self.pov_proj = nn.Identity()
                combined_dim = text_dim
        elif combine_method == "concat":
            # Concatenate embeddings
            combined_dim = text_dim + pov_dim
            self.text_proj = nn.Identity()
            self.pov_proj = nn.Identity()
        elif combine_method == "concat_proj":
            # Concatenate then project (most flexible)
            combined_dim = text_dim + pov_dim
            self.text_proj = nn.Identity()
            self.pov_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")
        
        # Project combined embeddings to spatial features
        # Output: [B, output_channels, H, W]
        spatial_elements = output_channels * spatial_size[0] * spatial_size[1]
        self.spatial_proj = nn.Sequential(
            nn.Linear(combined_dim, spatial_elements * 2),  # Intermediate layer
            nn.SiLU(),
            nn.Linear(spatial_elements * 2, spatial_elements)
        )
        
    def forward(self, text_emb, pov_emb):
        """
        Convert embeddings to spatial feature map.
        
        Args:
            text_emb: Text/graph embeddings [B, text_dim]
            pov_emb: POV embeddings [B, pov_dim]
        
        Returns:
            Spatial feature map [B, output_channels, H, W]
        """
        B = text_emb.shape[0]
        
        # Project embeddings if needed
        text_feat = self.text_proj(text_emb)  # [B, text_dim or common_dim]
        pov_feat = self.pov_proj(pov_emb)  # [B, pov_dim or common_dim]
        
        # Combine embeddings
        if self.combine_method == "add":
            combined = text_feat + pov_feat  # [B, common_dim]
        elif self.combine_method in ["concat", "concat_proj"]:
            combined = torch.cat([text_feat, pov_feat], dim=1)  # [B, text_dim + pov_dim]
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")
        
        # Project to spatial features
        spatial_flat = self.spatial_proj(combined)  # [B, output_channels * H * W]
        spatial = spatial_flat.view(B, self.output_channels, self.spatial_size[0], self.spatial_size[1])
        
        return spatial
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "text_dim": self.text_dim,
            "pov_dim": self.pov_dim,
            "output_channels": self.output_channels,
            "spatial_size": self.spatial_size,
            "combine_method": self.combine_method,
        })
        return cfg

