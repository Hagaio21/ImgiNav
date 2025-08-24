import torch
import torch.nn as nn
from . import register_conditioning
from .base_conditioning import BaseConditioningModule
from ..blocks import CrossAttention
from ...utils.factories import create_model

@register_conditioning("TextCrossAttention")
class TextCrossAttentionConditioning(BaseConditioningModule):
    """
    Injects text conditioning using a frozen, pre-trained text encoder.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        feature_map_channels = self.info['feature_map_channels']
        num_heads = self.info.get('num_heads', 8)

        # 1. Build the text encoder (e.g., BertEncoder) using its config file
        encoder_config_path = self.info['text_encoder_config_path']
        self.text_encoder = create_model(encoder_config_path, device='cpu')

        # 2. Freeze the text encoder's weights
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # 3. Get the context dimension from the encoder's own config
        context_dim = self.text_encoder.info['architecture'][-1]
        
        self.attention = CrossAttention(
            query_dim=feature_map_channels,
            context_dim=context_dim,
            num_heads=num_heads
        )

    def forward(self, x, time_emb, raw_text_strings):
        device = x.device
        self.text_encoder.to(device)
        B, C, H, W = x.shape

        # Get the context embedding from the frozen text encoder
        with torch.no_grad():
            context = self.text_encoder(raw_text_strings) # -> [B, SeqLen, D_ctx]

        # Reshape query for attention: [B, C, H, W] -> [B, H*W, C]
        query = x.view(B, C, -1).transpose(1, 2)
        
        attn_output = self.attention(query, context)
        
        # Reshape back and add as a residual connection
        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        return x + attn_output