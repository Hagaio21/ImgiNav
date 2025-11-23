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
        
        # Efficient projection from joint space (256-dim) to spatial features
        # Use a simple learned projection: 256 -> output_channels
        # This is trainable and learns to convert CLIP embeddings to spatial features
        # Small enough to not collapse, but expressive enough to learn useful mappings
        self.spatial_proj = nn.Linear(256, output_channels)
        
        # Initialize to preserve information (Xavier/Glorot initialization)
        nn.init.xavier_uniform_(self.spatial_proj.weight)
        if self.spatial_proj.bias is not None:
            nn.init.zeros_(self.spatial_proj.bias)
    
    def forward(self, text_emb, pov_emb):
        """
        Convert embeddings to spatial feature map via CLIP joint space.
        
        Args:
            text_emb: Text/graph embeddings [B, text_dim] or None
            pov_emb: POV embeddings [B, pov_dim] or None
        
        Returns:
            Spatial feature map [B, output_channels, H, W]
        """
        if text_emb is None and pov_emb is None:
            raise ValueError("At least one of text_emb or pov_emb must be provided")
        
        if text_emb is not None:
            B = text_emb.shape[0]
            if text_emb.dim() > 2:
                text_emb = text_emb.flatten(start_dim=1)
            # Ensure dtype matches CLIP projection parameters (for mixed precision training)
            text_proj_dtype = next(self.clip_projections.text_proj.parameters()).dtype
            text_emb = text_emb.to(dtype=text_proj_dtype)
            text_proj = self.clip_projections.text_proj(text_emb)
            text_proj = F.normalize(text_proj, p=2, dim=1)
        else:
            B = pov_emb.shape[0]
            text_proj = None
        
        if pov_emb is not None:
            if pov_emb.dim() > 2:
                pov_emb = pov_emb.flatten(start_dim=1)
            # Ensure dtype matches CLIP projection parameters (for mixed precision training)
            pov_proj_dtype = next(self.clip_projections.pov_proj.parameters()).dtype
            pov_emb = pov_emb.to(dtype=pov_proj_dtype)
            pov_proj = self.clip_projections.pov_proj(pov_emb)
            pov_proj = F.normalize(pov_proj, p=2, dim=1)
        else:
            pov_proj = None
        
        if text_proj is not None and pov_proj is not None:
            if self.combine_method == "add":
                combined_emb = text_proj + pov_proj
            elif self.combine_method == "average":
                combined_emb = (text_proj + pov_proj) / 2.0
            else:
                combined_emb = (text_proj + pov_proj) / 2.0
        elif text_proj is not None:
            combined_emb = text_proj
        else:
            combined_emb = pov_proj
        
        combined_emb = F.normalize(combined_emb, p=2, dim=1)  # [B, 256] in joint space
        
        # Project from joint space to output_channels
        spatial_flat = self.spatial_proj(combined_emb)  # [B, output_channels]
        
        # Reshape to spatial: [B, output_channels] -> [B, output_channels, 1, 1] -> [B, output_channels, H, W]
        # This broadcasts the same features across spatial dimensions (global conditioning)
        spatial = spatial_flat.unsqueeze(-1).unsqueeze(-1)  # [B, output_channels, 1, 1]
        spatial = F.interpolate(spatial, size=self.spatial_size, mode='bilinear', align_corners=False)  # [B, output_channels, H, W]
        
        return spatial
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "output_channels": self.output_channels,
            "spatial_size": self.spatial_size,
            "combine_method": self.combine_method,
        })
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
        
        # Safety check: prevent unreasonably large spatial projections
        MAX_SPATIAL_ELEMENTS = 10_000_000  # 10M elements max
        if spatial_elements > MAX_SPATIAL_ELEMENTS:
            raise ValueError(
                f"spatial_elements is too large: {spatial_elements} "
                f"(output_channels={output_channels}, spatial_size={spatial_size}). "
                f"This would create a Linear layer with {spatial_elements * 2} input features, "
                f"requiring ~{spatial_elements * 2 * spatial_elements * 4 / 1e9:.1f}GB of memory. "
                f"Check that spatial_size is set to latent dimensions (e.g., (64, 64)), not image dimensions."
            )
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(combined_dim, spatial_elements * 2),  # Intermediate layer
            nn.SiLU(),
            nn.Linear(spatial_elements * 2, spatial_elements)
        )
        
    def forward(self, text_emb, pov_emb):
        """
        Convert embeddings to spatial feature map.
        
        Args:
            text_emb: Text/graph embeddings [B, text_dim] or None
            pov_emb: POV embeddings [B, pov_dim] or None
        
        Returns:
            Spatial feature map [B, output_channels, H, W]
        """
        if text_emb is None and pov_emb is None:
            raise ValueError("At least one of text_emb or pov_emb must be provided")
        
        if text_emb is not None:
            B = text_emb.shape[0]
            text_feat = self.text_proj(text_emb)
        else:
            B = pov_emb.shape[0]
            text_feat = None
        
        if pov_emb is not None:
            pov_feat = self.pov_proj(pov_emb)
        else:
            pov_feat = None
        
        if text_feat is not None and pov_feat is not None:
            if self.combine_method == "add":
                combined = text_feat + pov_feat
            elif self.combine_method in ["concat", "concat_proj"]:
                combined = torch.cat([text_feat, pov_feat], dim=1)
            else:
                raise ValueError(f"Unknown combine_method: {self.combine_method}")
        elif text_feat is not None:
            if self.combine_method in ["concat", "concat_proj"]:
                zero_pov = torch.zeros(B, self.pov_dim, device=text_feat.device, dtype=text_feat.dtype)
                pov_feat = self.pov_proj(zero_pov)
                combined = torch.cat([text_feat, pov_feat], dim=1)
            else:
                combined = text_feat
        else:
            if self.combine_method in ["concat", "concat_proj"]:
                zero_text = torch.zeros(B, self.text_dim, device=pov_feat.device, dtype=pov_feat.dtype)
                text_feat = self.text_proj(zero_text)
                combined = torch.cat([text_feat, pov_feat], dim=1)
            else:
                combined = pov_feat
        
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

