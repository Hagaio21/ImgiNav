import torch.nn as nn
from . import register_conditioning
from .base_conditioning import BaseConditioningModule

@register_conditioning("VectorAdditive")
class VectorAdditiveConditioning(BaseConditioningModule):
    """
    A smart conditioning strategy that pre-builds projection layers for all
    feature map sizes it will encounter in the UNet.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.cond_dim = self.info['cond_dim']
        self.time_dim = self.info['time_dim']
        
        # Initial projection for the raw condition vector
        self.cond_projection = nn.Linear(self.cond_dim, self.time_dim)
        
        # --- THIS IS THE FIX ---
        # Pre-build a projection layer for each unique channel size in the UNet architecture
        self.time_projections = nn.ModuleDict()
        unet_channel_sizes = self.info['unet_channel_sizes']
        for size in set(unet_channel_sizes):
            self.time_projections[str(size)] = nn.Linear(self.time_dim, size)

    def forward(self, x, time_emb, condition):
        # 1. Create the base combined embedding
        cond_emb = self.cond_projection(condition)
        combined_emb = time_emb + cond_emb

        # 2. Project the embedding to match the feature map's channel dimension
        out_channels = x.shape[1]
        projected_emb = self.time_projections[str(out_channels)](combined_emb)

        # 3. Add the correctly-sized embedding to the feature map
        return x + projected_emb[:, :, None, None]