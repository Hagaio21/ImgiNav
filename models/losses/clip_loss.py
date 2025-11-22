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
    """
    def __init__(self, projection_dim=256, text_dim=384, pov_dim=512, latent_dim=None):
        super().__init__()
        self.projection_dim = projection_dim
        self.text_dim = text_dim
        self.pov_dim = pov_dim
        self._latent_dim = latent_dim
        
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
    
    def forward(self, latent_features, text_emb, pov_emb, combine_method="average"):
        """
        Project all embeddings to joint space.
        
        Args:
            latent_features: VAE features [B, C, H, W] or [B, D]
            text_emb: Text embeddings [B, text_dim]
            pov_emb: POV embeddings [B, pov_dim] or None (for scenes)
            combine_method: How to combine text and POV ("add", "concat", "average")
        
        Returns:
            (latent_proj, combined_emb) where both are [B, projection_dim]
        """
        # Flatten spatial dimensions if needed
        if latent_features.dim() > 2:
            if latent_features.dim() == 4:
                latent_features = F.adaptive_avg_pool2d(latent_features, 1).squeeze(-1).squeeze(-1)
            else:
                latent_features = latent_features.flatten(start_dim=1)
        
        # Initialize projection if needed
        latent_dim = latent_features.shape[1]
        if self.latent_proj is None:
            self._init_latent_proj(latent_dim, latent_features.device)
        
        # Flatten embeddings if needed
        if text_emb.dim() > 2:
            text_emb = text_emb.flatten(start_dim=1)
        
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
        
        # Check if projections are provided from model (attached to Autoencoder)
        # If not, create our own
        self.use_model_projections = self._init_kwargs.get("use_model_projections", False)
        
        if not self.use_model_projections:
            # Create our own projection layers
            self.projections = CLIPProjections(
                projection_dim=self.projection_dim,
                text_dim=text_dim,
                pov_dim=pov_dim,
                latent_dim=latent_dim
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
            # Return zero loss connected to a parameter from projections to ensure gradient flow
            if self.projections is not None:
                # Use a projection parameter to create a connected zero loss
                proj_param = next(self.projections.parameters())
                return (proj_param * 0.0).sum() * 0.0, {}
            else:
                # No projections available, return simple zero
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
        
        # Compute similarity matrix
        # latent_proj @ combined_emb.T -> [B, B]
        B = latent_proj.shape[0]
        logits = latent_proj @ combined_emb.T / self.temperature  # [B, B]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=logits.device, dtype=torch.long)
        
        # Symmetric loss: image-to-text and text-to-image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss * self.weight, {
            f"clip_loss": loss.detach(),
            f"clip_loss_i2t": loss_i2t.detach(),
            f"clip_loss_t2i": loss_t2i.detach(),
        }

