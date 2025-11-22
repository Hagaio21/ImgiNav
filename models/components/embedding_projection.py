# models/components/embedding_projection.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .base_component import BaseComponent

class CLIPEmbeddingToSpatial(BaseComponent):
    """
    Projects 1D embeddings to spatial feature maps using CLIP projections.
    
    First projects embeddings through CLIP joint space (from trained VAE),
    then converts to spatial features for cross-attention.
    
    This ensures spatial features (K, V) are in the same semantic space
    as VAE latents (Q), improving cross-attention alignment.
    
    Config:
        clip_projections: CLIPProjections instance or path to VAE checkpoint with CLIP projections
        output_channels: Number of output channels for spatial features (default: matches UNet base_channels)
        spatial_size: Target spatial size (H, W) - will be interpolated to match UNet resolution (default: (64, 64))
        combine_method: How to combine text and POV in CLIP space: "add", "average" (default: "average")
    """
    
    def _build(self):
        output_channels = self._init_kwargs.get("output_channels", 96)
        spatial_size = self._init_kwargs.get("spatial_size", (64, 64))
        combine_method = self._init_kwargs.get("combine_method", "average")
        clip_projections = self._init_kwargs.get("clip_projections", None)
        
        self.output_channels = output_channels
        self.spatial_size = spatial_size
        self.combine_method = combine_method
        
        # Load or use provided CLIP projections
        if clip_projections is None:
            raise ValueError("CLIPEmbeddingToSpatial requires clip_projections (CLIPProjections instance or VAE checkpoint path)")
        
        if isinstance(clip_projections, str) or isinstance(clip_projections, Path):
            # Load from VAE checkpoint
            from models.autoencoder import Autoencoder
            checkpoint_path = Path(clip_projections)
            autoencoder = Autoencoder.load_checkpoint(checkpoint_path, map_location="cpu")
            if not hasattr(autoencoder, 'clip_projections') or autoencoder.clip_projections is None:
                raise ValueError(f"VAE checkpoint {checkpoint_path} does not have CLIP projections")
            self.clip_projections = autoencoder.clip_projections
        else:
            # Use provided CLIPProjections instance
            self.clip_projections = clip_projections
        
        # Project from joint space (256-dim) to spatial features
        # Input: joint space embeddings [B, 256]
        # Output: spatial features [B, output_channels, H, W]
        spatial_elements = output_channels * spatial_size[0] * spatial_size[1]
        self.spatial_proj = nn.Sequential(
            nn.Linear(256, spatial_elements * 2),  # 256 is CLIP projection_dim
            nn.LayerNorm(spatial_elements * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(spatial_elements * 2, spatial_elements)
        )
    
    def forward(self, text_emb, pov_emb):
        """
        Convert embeddings to spatial feature map via CLIP joint space.
        
        Args:
            text_emb: Text/graph embeddings [B, text_dim]
            pov_emb: POV embeddings [B, pov_dim]
        
        Returns:
            Spatial feature map [B, output_channels, H, W]
        """
        B = text_emb.shape[0]
        
        # Flatten embeddings if needed
        if text_emb.dim() > 2:
            text_emb = text_emb.flatten(start_dim=1)
        if pov_emb.dim() > 2:
            pov_emb = pov_emb.flatten(start_dim=1)
        
        # Project text and POV through CLIP projections to joint space
        # This aligns embeddings with VAE features (which are also in this space)
        text_proj = self.clip_projections.text_proj(text_emb)  # [B, 256]
        pov_proj = self.clip_projections.pov_proj(pov_emb)  # [B, 256]
        
        # Normalize
        text_proj = F.normalize(text_proj, p=2, dim=1)
        pov_proj = F.normalize(pov_proj, p=2, dim=1)
        
        # Combine in joint space
        if self.combine_method == "add":
            combined_emb = text_proj + pov_proj
        elif self.combine_method == "average":
            combined_emb = (text_proj + pov_proj) / 2.0
        else:
            combined_emb = (text_proj + pov_proj) / 2.0
        
        combined_emb = F.normalize(combined_emb, p=2, dim=1)  # [B, 256] in joint space
        
        # Project from joint space to spatial features
        spatial_flat = self.spatial_proj(combined_emb)  # [B, output_channels * H * W]
        spatial = spatial_flat.view(B, self.output_channels, self.spatial_size[0], self.spatial_size[1])
        
        return spatial
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "output_channels": self.output_channels,
            "spatial_size": self.spatial_size,
            "combine_method": self.combine_method,
        })
        # Note: clip_projections is not serialized (loaded from checkpoint)
        return cfg


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

