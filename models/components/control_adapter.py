# models/components/control_adapter.py
import torch
import torch.nn as nn
from .base_component import BaseComponent


class ControlAdapter(BaseComponent):
    """Learned adapter: (text_emb, pov_emb) → multi-scale control features."""
    def _build(self):
        text_dim = self._init_kwargs.get("text_dim", 768)
        pov_dim = self._init_kwargs.get("pov_dim", 256)
        base_channels = self._init_kwargs.get("base_channels", 64)
        depth = self._init_kwargs.get("depth", 4)

        self.text_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, base_channels * (2 ** i)),
                nn.SiLU(),
                nn.Linear(base_channels * (2 ** i), base_channels * (2 ** i))
            )
            for i in range(depth)
        ])
        self.pov_proj = nn.ModuleList([
            nn.Conv2d(pov_dim, base_channels * (2 ** i), 1)
            for i in range(depth)
        ])

    def forward(self, text_emb, pov_emb):
        feats = []
        for tp, pp in zip(self.text_proj, self.pov_proj):
            t = tp(text_emb).unsqueeze(-1).unsqueeze(-1)
            p = pp(pov_emb)
            feats.append(t + p)
        return feats
