import torch
import torch.nn as nn
from . import register_conditioning
from .base_conditioning import BaseConditioningModule
from ..blocks import CrossAttention

@register_conditioning("ImageCrossAttention")
class ImageCrossAttentionConditioning(BaseConditioningModule):
    def __init__(self, config: dict, vision_encoder: nn.Module): # <-- RECEIVES encoder
        super().__init__(config)
        self.vision_encoder = vision_encoder
        
        # Freeze the encoder
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        feature_map_channels = self.info['feature_map_channels']
        context_dim = self.vision_encoder.info['architecture'][-1]
        num_heads = self.info.get('num_heads', 8)
        
        self.attention = CrossAttention(
            query_dim=feature_map_channels,
            context_dim=context_dim,
            num_heads=num_heads
        )

    def forward(self, x, time_emb, condition_image):
        device = x.device
        self.vision_encoder.to(device)
        B, C, H, W = x.shape

        with torch.no_grad():
            context = self.vision_encoder(condition_image.to(device))

        context = context.view(B, context.shape[1], -1).transpose(1, 2)
        query = x.view(B, C, -1).transpose(1, 2)
        attn_output = self.attention(query, context)
        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        return x + attn_output