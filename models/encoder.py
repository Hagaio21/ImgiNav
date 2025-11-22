import torch
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
        self.variational = self._init_kwargs.get("variational", False)

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

        if self.variational:
            # VAE mode: output mu and logvar
            layers += [
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GroupNorm(norm_groups, in_ch),
                act,
            ]
            self.feature_extractor = nn.Sequential(*layers)
            self.mu_head = nn.Conv2d(in_ch, latent_ch, 1)
            self.logvar_head = nn.Conv2d(in_ch, latent_ch, 1)
        else:
            # Regular deterministic encoder
            layers += [
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GroupNorm(norm_groups, in_ch),
                act,
                nn.Conv2d(in_ch, latent_ch, 1),
            ]
            self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass. Always returns a dictionary.
        
        Returns:
            - Regular mode: {"latent": z}
            - VAE mode: {"mu": mu, "logvar": logvar, "latent_features": features}
        """
        if self.variational:
            features = self.feature_extractor(x)
            mu = self.mu_head(features)
            logvar = self.logvar_head(features)
            return {"mu": mu, "logvar": logvar, "latent_features": features}
        else:
            z = self.encoder(x)
            return {"latent": z}
