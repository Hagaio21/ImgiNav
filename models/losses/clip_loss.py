"""
CLIP-style contrastive loss to align VAE latents with text/POV embeddings.

Creates a joint embedding space where:
- VAE encoder features (projected) are close to matching text/POV embeddings
- VAE encoder features are far from non-matching embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import LossComponent, register_loss


class CLIPProjections(nn.Module):
    """
    Standalone CLIP projection layers that can be attached to a model.
    These project VAE features, text embeddings, and POV embeddings to a joint space.
    
    Supports both global and spatial alignment modes:
    - Global mode (default): Pools spatial features to global, aligns global-to-global
    - Spatial mode: Preserves spatial structure, projects global conditions to spatial dimensions
    """
    def __init__(self, projection_dim=256, text_dim=384, pov_dim=512, latent_dim=None, 
                 spatial_alignment=False):
        super().__init__()
        self.projection_dim = projection_dim
        self.text_dim = text_dim
        self.pov_dim = pov_dim
        self._latent_dim = latent_dim
        self.spatial_alignment = spatial_alignment
        
        # Text embedding -> joint space
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # POV embedding -> joint space
        self.pov_proj = nn.Sequential(
            nn.Linear(pov_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # VAE latent features -> joint space (will be initialized dynamically)
        self.latent_proj = None
        if latent_dim is not None:
            self._init_latent_proj(latent_dim)
        
        # Spatial projection for global conditions (only used in spatial_alignment mode)
        # Projects global embeddings to spatial feature maps
        self.spatial_text_proj = None
        self.spatial_pov_proj = None
        if spatial_alignment:
            # These will be initialized dynamically based on spatial dimensions
            self._spatial_h = None
            self._spatial_w = None
    
    def _init_latent_proj(self, latent_dim, device=None):
        """Initialize latent projection."""
        if self.latent_proj is None or self._latent_dim != latent_dim:
            self._latent_dim = latent_dim
            proj = nn.Sequential(
                nn.Linear(latent_dim, self.projection_dim * 2),
                nn.LayerNorm(self.projection_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.projection_dim * 2, self.projection_dim),
                nn.LayerNorm(self.projection_dim)
            )
            if device is not None:
                proj = proj.to(device)
            self.latent_proj = proj
    
    def _init_spatial_projections(self, h, w, device=None):
        """Initialize spatial projections for global conditions."""
        if self._spatial_h == h and self._spatial_w == w and self.spatial_text_proj is not None:
            return  # Already initialized
        
        self._spatial_h = h
        self._spatial_w = w
        spatial_elements = self.projection_dim * h * w
        
        # Project global text embedding to spatial [B, projection_dim, H, W]
        self.spatial_text_proj = nn.Sequential(
            nn.Linear(self.text_dim, spatial_elements * 2),
            nn.LayerNorm(spatial_elements * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(spatial_elements * 2, spatial_elements)
        )
        
        # Project global POV embedding to spatial [B, projection_dim, H, W]
        self.spatial_pov_proj = nn.Sequential(
            nn.Linear(self.pov_dim, spatial_elements * 2),
            nn.LayerNorm(spatial_elements * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(spatial_elements * 2, spatial_elements)
        )
        
        if device is not None:
            self.spatial_text_proj = self.spatial_text_proj.to(device)
            self.spatial_pov_proj = self.spatial_pov_proj.to(device)
    
    def forward(self, latent_features, text_emb, pov_emb, combine_method="average"):
        """
        Project all embeddings to joint space.
        
        Args:
            latent_features: VAE features [B, C, H, W] or [B, D]
            text_emb: Text embeddings [B, text_dim]
            pov_emb: POV embeddings [B, pov_dim] or None (for scenes)
            combine_method: How to combine text and POV ("add", "concat", "average")
        
        Returns:
            Global mode: (latent_proj, combined_emb) where both are [B, projection_dim]
            Spatial mode: (latent_proj, combined_emb) where both are [B, projection_dim, H, W]
        """
        if self.spatial_alignment and latent_features.dim() == 4:
            # Spatial alignment mode: preserve spatial structure
            B, C, H, W = latent_features.shape
            
            # Initialize spatial projections if needed
            self._init_spatial_projections(H, W, latent_features.device)
            
            # Project VAE features spatially: [B, C, H, W] -> [B, projection_dim, H, W]
            # Use 1x1 conv to project channels
            if self.latent_proj is None or self._latent_dim != C:
                self._latent_dim = C
                # Use Conv2d for spatial projection instead of Linear
                self.latent_proj = nn.Sequential(
                    nn.Conv2d(C, self.projection_dim * 2, 1),
                    nn.GroupNorm(8, self.projection_dim * 2),
                    nn.GELU(),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(self.projection_dim * 2, self.projection_dim, 1),
                    nn.GroupNorm(8, self.projection_dim)
                ).to(latent_features.device)
            
            latent_proj = self.latent_proj(latent_features)  # [B, projection_dim, H, W]
            latent_proj = F.normalize(latent_proj, p=2, dim=1)  # Normalize per spatial location
            
            # Project global conditions to spatial dimensions
            text_emb_flat = text_emb.flatten(start_dim=1) if text_emb.dim() > 2 else text_emb
            text_spatial_flat = self.spatial_text_proj(text_emb_flat)  # [B, projection_dim * H * W]
            text_spatial = text_spatial_flat.view(B, self.projection_dim, H, W)
            text_spatial = F.normalize(text_spatial, p=2, dim=1)
            
            if pov_emb is None:
                combined_emb = text_spatial
            else:
                pov_emb_flat = pov_emb.flatten(start_dim=1) if pov_emb.dim() > 2 else pov_emb
                pov_spatial_flat = self.spatial_pov_proj(pov_emb_flat)  # [B, projection_dim * H * W]
                pov_spatial = pov_spatial_flat.view(B, self.projection_dim, H, W)
                pov_spatial = F.normalize(pov_spatial, p=2, dim=1)
                
                # Combine text and POV spatially
                if combine_method == "add":
                    combined_emb = text_spatial + pov_spatial
                elif combine_method == "average":
                    combined_emb = (text_spatial + pov_spatial) / 2.0
                else:
                    combined_emb = (text_spatial + pov_spatial) / 2.0
                
                combined_emb = F.normalize(combined_emb, p=2, dim=1)
            
            return latent_proj, combined_emb
        
        # Global alignment mode (original behavior)
        # Flatten spatial dimensions if needed
        if latent_features.dim() > 2:
            if latent_features.dim() == 4:
                latent_features = F.adaptive_avg_pool2d(latent_features, 1).squeeze(-1).squeeze(-1)
            else:
                latent_features = latent_features.flatten(start_dim=1)
        
        # Initialize projection if needed (or re-initialize if dimension changed)
        latent_dim = latent_features.shape[1]
        if self.latent_proj is None or self._latent_dim != latent_dim:
            self._init_latent_proj(latent_dim, latent_features.device)
        
        # Flatten embeddings if needed and ensure consistent dtype
        if text_emb.dim() > 2:
            text_emb = text_emb.flatten(start_dim=1)
        if pov_emb is not None and pov_emb.dim() > 2:
            pov_emb = pov_emb.flatten(start_dim=1)
        
        # Ensure embeddings match projection dtype (projections are float32)
        text_emb = text_emb.to(dtype=next(self.text_proj.parameters()).dtype)
        if pov_emb is not None:
            pov_emb = pov_emb.to(dtype=next(self.pov_proj.parameters()).dtype)
        latent_features = latent_features.to(dtype=next(self.latent_proj.parameters()).dtype)
        
        # Project to joint space
        latent_proj = self.latent_proj(latent_features)
        text_proj = self.text_proj(text_emb)
        
        # Normalize
        latent_proj = F.normalize(latent_proj, p=2, dim=1)
        text_proj = F.normalize(text_proj, p=2, dim=1)
        
        # Handle missing pov_emb (scenes don't have POV embeddings)
        if pov_emb is None:
            # For scenes, use only text_emb as the combined embedding
            combined_emb = text_proj
        else:
            # Flatten POV embedding if needed
            if pov_emb.dim() > 2:
                pov_emb = pov_emb.flatten(start_dim=1)
            
            # Project POV to joint space
            pov_proj = self.pov_proj(pov_emb)
            pov_proj = F.normalize(pov_proj, p=2, dim=1)
            
            # Combine text and POV
            if combine_method == "add":
                combined_emb = text_proj + pov_proj
            elif combine_method == "average":
                combined_emb = (text_proj + pov_proj) / 2.0
            else:
                combined_emb = (text_proj + pov_proj) / 2.0
        
        combined_emb = F.normalize(combined_emb, p=2, dim=1)
        
        return latent_proj, combined_emb


@register_loss
class CLIPLoss(LossComponent):
    """
    CLIP-style contrastive loss to align VAE latents with text/POV embeddings.
    
    Creates a joint embedding space where:
    - VAE encoder features (projected) are close to matching text/POV embeddings
    - VAE encoder features are far from non-matching embeddings
    
    Config:
        key: Key in preds for VAE latent features (default: "latent_features")
        text_key: Key in targets for text embeddings (default: "text_emb")
        pov_key: Key in targets for POV embeddings (default: "pov_emb")
        temperature: Temperature for contrastive loss (default: 0.07)
        weight: Loss weight (default: 0.1)
        projection_dim: Dimension for joint embedding space (default: 256)
        latent_dim: Dimension of latent features (will be inferred if not provided)
        text_dim: Dimension of text embeddings (default: 384)
        pov_dim: Dimension of POV embeddings (default: 512)
        combine_method: How to combine text and POV embeddings: "add", "concat", "average" (default: "average")
        spatial_alignment: If True, preserves spatial structure and aligns per-pixel (default: False)
                          When True, projects global conditions to spatial dimensions for alignment
    """
    
    def _build(self):
        super()._build()
        self.key = self._init_kwargs.get("key", "latent_features")
        self.text_key = self._init_kwargs.get("text_key", "text_emb")
        self.pov_key = self._init_kwargs.get("pov_key", "pov_emb")
        self.temperature = self._init_kwargs.get("temperature", 0.07)
        self.projection_dim = self._init_kwargs.get("projection_dim", 256)
        self.combine_method = self._init_kwargs.get("combine_method", "average")
        
        # Embedding dimensions
        text_dim = self._init_kwargs.get("text_dim", 384)
        pov_dim = self._init_kwargs.get("pov_dim", 512)
        latent_dim = self._init_kwargs.get("latent_dim", None)
        spatial_alignment = self._init_kwargs.get("spatial_alignment", False)
        
        # Check if projections are provided from model (attached to Autoencoder)
        # If not, create our own
        self.use_model_projections = self._init_kwargs.get("use_model_projections", False)
        self.spatial_alignment = spatial_alignment
        
        if not self.use_model_projections:
            # Create our own projection layers
            self.projections = CLIPProjections(
                projection_dim=self.projection_dim,
                text_dim=text_dim,
                pov_dim=pov_dim,
                latent_dim=latent_dim,
                spatial_alignment=spatial_alignment
            )
        else:
            # Will use projections from model (set via set_projections method)
            self.projections = None
    
    def set_projections(self, projections):
        """Set projection layers from model (when attached to Autoencoder)."""
        self.projections = projections
        self.use_model_projections = True
    
    def forward(self, preds, targets):
        """
        Compute CLIP contrastive loss.
        
        Args:
            preds: Dict with VAE latent features [B, C, H, W] or [B, D]
            targets: Dict with text_emb [B, text_dim] and pov_emb [B, pov_dim]
        
        Returns:
            (loss, logs_dict)
        """
        # Get VAE latent features
        latent_features = preds.get(self.key)
        if latent_features is None:
            # Return zero loss - but this should not happen in normal training
            # If it does, we can't connect to computation graph, so return a simple zero
            device = next(self.text_proj.parameters()).device if hasattr(self, 'text_proj') else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        # Verify latent_features has gradients (critical for gradient flow)
        if not latent_features.requires_grad:
            # If latent_features doesn't require grad, we can't compute gradients
            # This might happen if the encoder is frozen or if features are detached
            # Try to create a connected loss using projection parameters
            if self.projections is not None:
                # Use a projection parameter to create a connected zero loss
                # This ensures gradients can flow to projection parameters
                proj_param = next(self.projections.parameters())
                if proj_param.requires_grad:
                    # Create zero loss connected to projection parameter
                    zero_loss = (proj_param * 0.0).sum() * 0.0
                    return zero_loss, {}
            # If no projections or they don't require grad, return simple zero
            # This will cause an error downstream, which is better than silent failure
            return torch.tensor(0.0, device=latent_features.device, requires_grad=True), {}
        
        # Get text and POV embeddings
        text_emb = targets.get(self.text_key)
        pov_emb = targets.get(self.pov_key)
        
        # Handle missing text_emb (should not happen, but handle gracefully)
        if text_emb is None:
            # Return zero loss connected to computation graph via latent_features
            return (latent_features * 0.0).sum() * 0.0, {}
        
        # Use projections (either from model or our own)
        if self.projections is None:
            # Return zero loss connected to computation graph via latent_features
            return (latent_features * 0.0).sum() * 0.0, {}
        
        # Note: text_emb and pov_emb are pre-computed and may be detached
        # This is fine - the projections will still compute gradients for their parameters
        # The key is that latent_features has gradients, which will flow through latent_proj
        
        # Project to joint embedding space
        # If pov_emb is None (scenes), the projections will handle it by using only text_emb
        latent_proj, combined_emb = self.projections(
            latent_features, text_emb, pov_emb, combine_method=self.combine_method
        )
        
        # Verify latent_proj has gradients (it should, since it comes from latent_features)
        # This is critical - if latent_proj doesn't have gradients, the loss won't flow back
        if not latent_proj.requires_grad:
            # This should not happen if latent_features has gradients
            # Check if projections are trainable
            proj_has_grad = any(p.requires_grad for p in self.projections.parameters())
            raise RuntimeError(
                f"latent_proj does not require gradients! "
                f"latent_features.requires_grad={latent_features.requires_grad}, "
                f"projections.trainable={proj_has_grad}"
            )
        
        # Verify projections are actually being used (check if they're part of the computation graph)
        # This ensures the projections are the same instance as model.clip_projections
        proj_param = next(self.projections.parameters())
        if not proj_param.requires_grad:
            raise RuntimeError("CLIP projection parameters do not require gradients! Check that projections are included in optimizer.")
        
        # Handle spatial vs global alignment
        if self.spatial_alignment and latent_proj.dim() == 4:
            # Spatial alignment: compute per-pixel alignment loss
            # latent_proj: [B, projection_dim, H, W]
            # combined_emb: [B, projection_dim, H, W]
            B, C, H, W = latent_proj.shape
            
            # Flatten spatial dimensions: [B, projection_dim, H*W]
            latent_flat = latent_proj.view(B, C, H * W)  # [B, C, H*W]
            combined_flat = combined_emb.view(B, C, H * W)  # [B, C, H*W]
            
            # Transpose for matrix multiplication: [B, H*W, C]
            latent_flat = latent_flat.transpose(1, 2)  # [B, H*W, C]
            combined_flat = combined_flat.transpose(1, 2)  # [B, H*W, C]
            
            # Compute similarity matrix per spatial location
            # For each spatial location, compute similarity across batch
            # latent_flat: [B, H*W, C], combined_flat: [B, H*W, C]
            # We want: for each spatial location, compute [B, B] similarity matrix
            # Then average over spatial locations
            
            # Reshape to [B*H*W, C] for batch-wise similarity computation
            latent_all = latent_flat.reshape(B * H * W, C)  # [B*H*W, C]
            combined_all = combined_flat.reshape(B * H * W, C)  # [B*H*W, C]
            
            # Compute similarity: [B*H*W, B*H*W]
            # But we want per-spatial-location: for each of H*W locations, compute [B, B] similarity
            # So we need to group by spatial location
            
            # Alternative: compute per-pixel MSE/alignment and average
            # This is simpler and still provides spatial alignment signal
            per_pixel_loss = F.mse_loss(latent_flat, combined_flat, reduction='none')  # [B, H*W, C]
            per_pixel_loss = per_pixel_loss.mean(dim=2)  # [B, H*W] - average over channels
            loss = per_pixel_loss.mean()  # Average over batch and spatial dimensions
            
            # Also compute contrastive loss on global pooled features for stability
            # Pool spatial features to global
            latent_global = F.adaptive_avg_pool2d(latent_proj, 1).squeeze(-1).squeeze(-1)  # [B, C]
            combined_global = F.adaptive_avg_pool2d(combined_emb, 1).squeeze(-1).squeeze(-1)  # [B, C]
            
            # Normalize
            latent_global = F.normalize(latent_global, p=2, dim=1)
            combined_global = F.normalize(combined_global, p=2, dim=1)
            
            # Contrastive loss on global features
            logits = latent_global @ combined_global.T / self.temperature  # [B, B]
            labels = torch.arange(B, device=logits.device, dtype=torch.long)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            contrastive_loss = (loss_i2t + loss_t2i) / 2.0
            
            # Combine spatial alignment loss with contrastive loss
            # Weight spatial loss more heavily since it's the main objective
            loss = 0.7 * loss + 0.3 * contrastive_loss
            
            loss_i2t = loss_t2i = loss  # For logging
        else:
            # Global alignment (original behavior)
            # Compute similarity matrix
            # latent_proj @ combined_emb.T -> [B, B]
            # Note: combined_emb may not have gradients (from detached text_emb/pov_emb),
            # but gradients will still flow through latent_proj
            B = latent_proj.shape[0]
            logits = latent_proj @ combined_emb.T / self.temperature  # [B, B]
            
            # Verify logits has gradients (it should, since latent_proj has gradients)
            if not logits.requires_grad:
                raise RuntimeError(
                    f"logits does not require gradients! "
                    f"latent_proj.requires_grad={latent_proj.requires_grad}, "
                    f"combined_emb.requires_grad={combined_emb.requires_grad}"
                )
            
            # Labels: diagonal elements are positive pairs
            labels = torch.arange(B, device=logits.device, dtype=torch.long)
            
            # Symmetric loss: image-to-text and text-to-image
            # Cross-entropy will compute gradients through logits, which flows through latent_proj
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2.0
        
        # Final verification: loss must have gradients
        if not loss.requires_grad:
            raise RuntimeError(
                f"CLIP loss does not require gradients! "
                f"loss_i2t.requires_grad={loss_i2t.requires_grad}, "
                f"loss_t2i.requires_grad={loss_t2i.requires_grad}"
            )
        
        return loss * self.weight, {
            f"clip_loss": loss.detach(),
            f"clip_loss_i2t": loss_i2t.detach(),
            f"clip_loss_t2i": loss_t2i.detach(),
        }

