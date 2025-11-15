"""
Semantic loss component for Stage 2 diffusion training.
Combines segmentation and perceptual losses on decoded images.
"""

import torch
import torch.nn as nn
from .base_loss import LossComponent, register_loss


@register_loss
class SemanticLoss(LossComponent):
    """
    Semantic loss that combines segmentation and perceptual losses.
    Used in Stage 2 diffusion training to ensure decoded layouts are viable.
    
    Config:
        decoded_rgb_key: Key in preds for decoded RGB (default: "decoded_rgb")
        decoded_seg_key: Key in preds for decoded segmentation (default: "decoded_segmentation")
        target_rgb_key: Key in targets for target RGB (default: "rgb")
        target_seg_key: Key in targets for target segmentation (default: "segmentation")
        segmentation_loss: Config for segmentation loss (optional)
            - type: Loss type (e.g., "CrossEntropyLoss")
            - key: Key in preds for segmentation (default: "segmentation")
            - target: Key in targets for target segmentation (default: "segmentation")
            - weight: Loss weight
        perceptual_loss: Config for perceptual loss (optional)
            - type: Loss type (e.g., "PerceptualLoss")
            - key: Key in preds for RGB (default: "rgb")
            - target: Key in targets for target RGB (default: "rgb")
            - weight: Loss weight
    
    This loss acts as a key mapper: it reads from decoded_* keys in preds and maps
    them to standard keys for sub-losses, maintaining consistency with the dict interface.
    """
    def _build(self):
        super()._build()
        
        # Get key mappings (configurable)
        self.decoded_rgb_key = self._init_kwargs.get("decoded_rgb_key", "decoded_rgb")
        self.decoded_seg_key = self._init_kwargs.get("decoded_seg_key", "decoded_segmentation")
        self.target_rgb_key = self._init_kwargs.get("target_rgb_key", "rgb")
        self.target_seg_key = self._init_kwargs.get("target_seg_key", "segmentation")
        
        # Build segmentation loss
        seg_cfg = self._init_kwargs.get("segmentation_loss", {})
        if seg_cfg:
            from .base_loss import LOSS_REGISTRY
            seg_type = seg_cfg.get("type", "CrossEntropyLoss")
            if seg_type not in LOSS_REGISTRY:
                raise ValueError(f"Unknown segmentation loss type: {seg_type}")
            # Create a copy to avoid modifying original config
            seg_cfg_copy = seg_cfg.copy()
            # Ensure key and target are set correctly
            seg_cfg_copy.setdefault("key", "segmentation")
            seg_cfg_copy.setdefault("target", "segmentation")
            self.seg_loss = LOSS_REGISTRY[seg_type].from_config(seg_cfg_copy)
        else:
            self.seg_loss = None
        
        # Build perceptual loss
        perc_cfg = self._init_kwargs.get("perceptual_loss", {})
        if perc_cfg:
            from .base_loss import LOSS_REGISTRY
            perc_type = perc_cfg.get("type", "PerceptualLoss")
            if perc_type not in LOSS_REGISTRY:
                raise ValueError(f"Unknown perceptual loss type: {perc_type}")
            # Create a copy to avoid modifying original config
            perc_cfg_copy = perc_cfg.copy()
            # Ensure key and target are set correctly
            perc_cfg_copy.setdefault("key", "rgb")
            perc_cfg_copy.setdefault("target", "rgb")
            self.perc_loss = LOSS_REGISTRY[perc_type].from_config(perc_cfg_copy)
        else:
            self.perc_loss = None
        
        if self.seg_loss is None and self.perc_loss is None:
            raise ValueError("SemanticLoss requires at least one of segmentation_loss or perceptual_loss")
    
    def forward(self, preds, targets):
        """
        Compute semantic losses on decoded outputs.
        
        Args:
            preds: Dictionary with:
                - "decoded_rgb": Decoded RGB images [B, 3, H, W]
                - "decoded_segmentation": Decoded segmentation logits [B, C, H, W]
            targets: Dictionary with:
                - "rgb": Target RGB images [B, 3, H, W]
                - "segmentation": Target segmentation masks [B, H, W] or [B, 3, H, W] (RGB)
        
        Returns:
            (total_loss, logs_dict)
        """
        total_loss = 0.0
        logs = {}
        
        # Segmentation loss
        if self.seg_loss is not None:
            if self.decoded_seg_key in preds and self.target_seg_key in targets:
                # Map decoded_segmentation -> segmentation for the loss
                # This maintains consistency: sub-loss uses standard key names
                seg_preds = {"segmentation": preds[self.decoded_seg_key]}
                seg_targets = {"segmentation": targets[self.target_seg_key]}
                seg_loss_val, seg_logs = self.seg_loss(seg_preds, seg_targets)
                total_loss += seg_loss_val
                logs.update({f"seg_{k}": v for k, v in seg_logs.items()})
        
        # Perceptual loss
        if self.perc_loss is not None:
            if self.decoded_rgb_key in preds and self.target_rgb_key in targets:
                # Normalize RGB to [0, 1] for perceptual loss (VGG expects [0, 1])
                # Decoder outputs tanh ([-1, 1]), so normalize directly without checking
                decoded_rgb = (preds[self.decoded_rgb_key] + 1.0) / 2.0
                target_rgb = targets[self.target_rgb_key]
                # Target RGB may be [-1, 1] or [0, 1], normalize if needed
                if target_rgb.min() < 0:
                    target_rgb = (target_rgb + 1.0) / 2.0
                
                # Map decoded_rgb -> rgb for the loss
                # This maintains consistency: sub-loss uses standard key names
                perc_preds = {"rgb": decoded_rgb}
                perc_targets = {"rgb": target_rgb}
                perc_loss_val, perc_logs = self.perc_loss(perc_preds, perc_targets)
                total_loss += perc_loss_val
                logs.update({f"perc_{k}": v for k, v in perc_logs.items()})
        
        return total_loss, logs

