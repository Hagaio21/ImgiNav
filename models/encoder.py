import torch.nn as nn

from .components.base_component import BaseComponent

class Encoder(BaseComponent):
    def _build(self):
        act = getattr(nn, self._init_kwargs.get("activation", "SiLU"))()
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        in_ch = self._init_kwargs.get("in_channels", 3)
        out_ch = self._init_kwargs.get("base_channels", 64)
        down_steps = self._init_kwargs.get("downsampling_steps", 4)
        latent_ch = self._init_kwargs.get("latent_channels", 4)

        layers = []
        for _ in range(down_steps):
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(norm_groups, out_ch),
                act,
                nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(norm_groups, out_ch),
                act,
            ]
            in_ch = out_ch
            out_ch *= 2

        layers += [
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(norm_groups, in_ch),
            act,
            nn.Conv2d(in_ch, latent_ch, 1),
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
