import torch
import torch.nn as nn
from . import register_conditioning
from .base_conditioning import BaseConditioningModule
from ..blocks import CrossAttention
from ...utils.factories import create_model # Use our factory to build the encoder dependency

@register_conditioning("CrossAttention")
class CrossAttentionConditioning(BaseConditioningModule):
    """
    A generic cross-attention conditioning module.

    This module is modality-agnostic. It takes a raw conditioning input
    (like an image or text), passes it through a specified 'encoder' model
    to get a context embedding, and then injects that context into the
    UNet's feature map using cross-attention.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        feature_map_channels = self.info['feature_map_channels']
        num_heads = self.info.get('num_heads', 8)

        # --- Dynamically Build the Encoder Dependency ---
        encoder_config_path = self.info['encoder_config_path']
        self.encoder = create_model(encoder_config_path, device='cpu') # Build on CPU

        # Load checkpoint if provided
        checkpoint_path = self.info.get('encoder_checkpoint')
        if checkpoint_path:
            print(f"[Conditioning] Loading encoder checkpoint: {checkpoint_path}")
            self.encoder.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        # Freeze the encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Determine context dimension from the loaded encoder's config
        # This assumes the last channel in the architecture is the context dim
        context_dim = self.encoder.info['architecture'][-1]

        self.attention = CrossAttention(
            query_dim=feature_map_channels,
            context_dim=context_dim,
            num_heads=num_heads
        )

    def forward(self, x, time_emb, raw_condition):
        device = x.device
        self.encoder.to(device)

        B, C, H, W = x.shape

        with torch.no_grad():
            context = self.encoder(raw_condition.to(device)) # -> [B, D_ctx, H', W'] or [B, SeqLen, D_ctx]

        # Reshape for attention based on context dimension
        if context.ndim == 4: # Image-like context
            context = context.view(B, context.shape[1], -1).transpose(1, 2)
        elif context.ndim == 2: # Simple vector context
            context = context.unsqueeze(1)
        # if context is [B, SeqLen, D_ctx], it's ready

        query = x.view(B, C, -1).transpose(1, 2)

        attn_output = self.attention(query, context)

        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        return x + attn_output