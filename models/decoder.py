import torch
import torch.nn as nn
from .components.base_component import BaseComponent
from .components.heads import HEAD_REGISTRY


class Decoder(BaseComponent):
    def _build(self):
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        base_ch = self._init_kwargs.get("base_channels", 64)
        up_steps = self._init_kwargs.get("upsampling_steps", 4)
        activation = getattr(nn, self._init_kwargs.get("activation", "SiLU"))()
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        head_cfgs = self._init_kwargs.get("heads", [])

        layers = []
        in_ch = latent_ch
        out_ch = base_ch * (2 ** (up_steps - 1))

        layers += [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(norm_groups, out_ch),
            activation,
        ]

        for _ in range(up_steps):
            layers += [
                nn.ConvTranspose2d(out_ch, out_ch // 2, 4, stride=2, padding=1),
                nn.GroupNorm(norm_groups, out_ch // 2),
                activation,
            ]
            out_ch //= 2

        self.shared_decoder = nn.Sequential(*layers)
        self.shared_out_channels = out_ch

        self.heads = nn.ModuleDict()
        for cfg in head_cfgs:
            head_type = cfg.get("type", "DecoderHead")
            head_name = cfg.get("name", head_type.lower())
            cls = HEAD_REGISTRY.get(head_type)
            if cls is None:
                raise ValueError(f"Unknown head type '{head_type}'")
            cfg["in_channels"] = cfg.get("in_channels", out_ch)
            self.heads[head_name] = cls.from_config(cfg)

    def forward(self, z):
        feats = self.shared_decoder(z)
        return {name: head(feats) for name, head in self.heads.items()}

    def to_config(self):
        cfg = super().to_config()
        cfg["heads"] = [head.to_config() for head in self.heads.values()]
        return cfg
