

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.base_component import BaseComponent
from .base_loss import LossComponent, register_loss
from models.components.projections import CLIPProjections


@register_loss
class CLIPLoss(LossComponent):

    
    def _build(self):
        super()._build()
        self.key = self._init_kwargs.get("key", "latent_features")
        self.text_key = self._init_kwargs.get("text_key", "text_emb")
        self.pov_key = self._init_kwargs.get("pov_key", "pov_emb")
        self.temperature = self._init_kwargs.get("temperature", 0.07)
        self.combine_method = self._init_kwargs.get("combine_method", "average")

        self.projections = None
        self.spatial_alignment = self._init_kwargs.get("spatial_alignment", False)
    
    def set_projections(self, projections):

        if projections is None:
            raise ValueError("CLIPLoss requires projections to be set via set_projections()")
        self.projections = projections
        # Infer spatial_alignment from projections if not explicitly set
        if hasattr(projections, 'spatial_alignment'):
            self.spatial_alignment = projections.spatial_alignment
    
    def forward(self, preds, targets):

        # Get VAE latent features
        latent_features = preds.get(self.key)
        if latent_features is None:

            device = next(self.text_proj.parameters()).device if hasattr(self, 'text_proj') else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        # Verify latent_features has gradients (critical for gradient flow)
        if not latent_features.requires_grad:

            if self.projections is not None:

                proj_param = next(self.projections.parameters())
                if proj_param.requires_grad:
                    # Create zero loss connected to projection parameter
                    zero_loss = (proj_param * 0.0).sum() * 0.0
                    return zero_loss, {}

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

        latent_proj, combined_emb = self.projections(
            latent_features, text_emb, pov_emb, combine_method=self.combine_method
        )

        if not latent_proj.requires_grad:

            proj_has_grad = any(p.requires_grad for p in self.projections.parameters())
            raise RuntimeError(
                f"latent_proj does not require gradients! "
                f"latent_features.requires_grad={latent_features.requires_grad}, "
                f"projections.trainable={proj_has_grad}"
            )
        

        proj_param = next(self.projections.parameters())
        if not proj_param.requires_grad:
            raise RuntimeError("CLIP projection parameters do not require gradients! Check that projections are included in optimizer.")
        
        # Handle spatial vs global alignment
        if self.spatial_alignment and latent_proj.dim() == 4:

            B, C, H, W = latent_proj.shape
            
            latent_flat = latent_proj.view(B, C, H * W)  # [B, C, H*W]
            combined_flat = combined_emb.view(B, C, H * W)  # [B, C, H*W]
            
            # Transpose for matrix multiplication: [B, H*W, C]
            latent_flat = latent_flat.transpose(1, 2)  # [B, H*W, C]
            combined_flat = combined_flat.transpose(1, 2)  # [B, H*W, C]

            per_pixel_loss = F.mse_loss(latent_flat, combined_flat, reduction='none')  # [B, H*W, C]
            per_pixel_loss = per_pixel_loss.mean(dim=2)  # [B, H*W] - average over channels
            loss = per_pixel_loss.mean()  # Average over batch and spatial dimensions
            

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
            

            loss = 0.7 * loss + 0.3 * contrastive_loss
            
            loss_i2t = loss_t2i = loss  # For logging
        else:

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

