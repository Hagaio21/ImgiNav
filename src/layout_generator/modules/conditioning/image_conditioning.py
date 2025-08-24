import torch
import torch.nn as nn
from . import register_conditioning
from .base_conditioning import BaseConditioningModule
from ..blocks import CrossAttention
from ...utils.factories import create_model

@register_conditioning("ImageCrossAttention")
class ImageCrossAttentionConditioning(BaseConditioningModule):
    """
    Injects a conditioning image using cross-attention.

    This module is specifically designed to handle an image as the condition.
    It uses a pre-trained vision encoder (your 'Encoder' model) to get an
    embedding, which then serves as the context for the cross-attention
    mechanism.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        feature_map_channels = self.info['feature_map_channels']
        num_heads = self.info.get('num_heads', 8)

        # 1. Build the vision encoder using its dedicated config file
        encoder_config_path = self.info['vision_encoder_config_path']
        self.vision_encoder = create_model(encoder_config_path, device='cpu')

        # 2. Load the pre-trained weights for the vision encoder
        checkpoint_path = self.info.get('vision_encoder_checkpoint')
        if checkpoint_path:
            print(f"[ImageConditioning] Loading vision encoder checkpoint from: {checkpoint_path}")
            self.vision_encoder.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        # 3. Freeze the encoder's weights
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 4. Get the context dimension from the encoder's own config
        context_dim = self.vision_encoder.info['architecture'][-1]
        
        self.attention = CrossAttention(
            query_dim=feature_map_channels,
            context_dim=context_dim,
            num_heads=num_heads
        )

    def forward(self, x, time_emb, condition_image):
        device = x.device
        self.vision_encoder.to(device)
        B, C, H, W = x.shape

        # Get the context embedding from the frozen vision encoder
        with torch.no_grad():
            context = self.vision_encoder(condition_image.to(device))

        # Reshape for attention: [B, D, H', W'] -> [B, H'*W', D]
        context = context.view(B, context.shape[1], -1).transpose(1, 2)
        # Reshape query: [B, C, H, W] -> [B, H*W, C]
        query = x.view(B, C, -1).transpose(1, 2)
        
        attn_output = self.attention(query, context)
        
        # Reshape back and add as a residual connection
        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        return x + attn_output