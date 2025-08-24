import torch.nn as nn
from . import register_conditioning
from .base_conditioning import BaseConditioningModule

@register_conditioning("VectorAdditive")
class VectorAdditiveConditioning(BaseConditioningModule):
    """
    A simple conditioning strategy that projects a condition vector
    and adds it to the time embedding, then injects it into the feature map.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        cond_dim = self.info['cond_dim']
        time_dim = self.info['time_dim']
        self.projection = nn.Linear(cond_dim, time_dim)

    def forward(self, x, time_emb, condition):
        cond_emb = self.projection(condition)
        combined_emb = time_emb + cond_emb
        
        # Add the combined embedding to every spatial location of the feature map
        return x + combined_emb[:, :, None, None]