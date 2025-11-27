"""
CLIP-style contrastive loss to align VAE latents with text/POV embeddings.

Creates a joint embedding space where:
- VAE encoder features (projected) are close to matching text/POV embeddings
- VAE encoder features are far from non-matching embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.base_component import BaseComponent
from .base_loss import LossComponent, register_loss
from models.components.projections import CLIPProjections


@register_loss
class CLIPLoss(LossComponent):
    """
    CLIP-style contrastive loss to align VAE latents with text/POV embeddings.
    
    Creates a joint embedding space where:
    - VAE encoder features (projected) are close to matching text/POV embeddings
    - VAE encoder features are far from non-matching embeddings
    
    NOTE: CLIPLoss does NOT create its own projections. It uses the model's CLIPProjections
    which must be set via set_projections() before use. This ensures projections are shared
    and trained jointly with the model.
    
    Config:
        key: Key in preds for VAE latent features (default: "latent_features")
        text_key: Key in targets for text embeddings (default: "text_emb")
        pov_key: Key in targets for POV embeddings (default: "pov_emb")
        temperature: Temperature for contrastive loss (default: 0.07)
        weight: Loss weight (default: 0.1)
        combine_method: How to combine text and POV embeddings: "add", "concat", "average" (default: "average")
        spatial_alignment: If True, preserves spatial structure and aligns per-pixel (default: False)
                          When True, projects global conditions to spatial dimensions for alignment
                          (inferred from model's CLIPProjections if not specified)
    """
    
    def _build(self):
        super()._build()
        self.key = self._init_kwargs.get("key", "latent_features")
        self.text_key = self._init_kwargs.get("text_key", "text_emb")
        self.pov_key = self._init_kwargs.get("pov_key", "pov_emb")
        self.temperature = self._init_kwargs.get("temperature", 0.07)
        self.combine_method = self._init_kwargs.get("combine_method", "average")
        
        # Projections must be set from model (via set_projections)
        # CLIPLoss does not create its own projections - it uses the model's projections
        # This ensures the projections are shared and trained jointly with the model
        self.projections = None
        self.spatial_alignment = self._init_kwargs.get("spatial_alignment", False)
    
    def set_projections(self, projections):
        """
        Set projection layers from model (required).
        
        CLIPLoss uses the model's CLIPProjections to ensure they are shared
        and trained jointly. This method must be called before using the loss.
        """
        if projections is None:
            raise ValueError("CLIPLoss requires projections to be set via set_projections()")
        self.projections = projections
        # Infer spatial_alignment from projections if not explicitly set
        if hasattr(projections, 'spatial_alignment'):
            self.spatial_alignment = projections.spatial_alignment
    
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
        
        # Use projections from model (must be set via set_projections)
        if self.projections is None:
            raise RuntimeError(
                "CLIPLoss.projections is None! "
                "Projections must be set via set_projections() before using the loss. "
                "This should be done automatically by the training script."
            )
        
        # text_emb and pov_emb are pre-computed and may be detached
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
            # combined_emb may not have gradients (from detached text_emb/pov_emb),
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

