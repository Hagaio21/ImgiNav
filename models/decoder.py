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

    def forward(self, z_or_dict):
        """
        Forward pass. Accepts dict from encoder.
        
        Args:
            z_or_dict: Dictionary containing:
                - "latent": tensor z (regular mode)
                - "mu" and "logvar": tensors (VAE mode - will reparameterize)
        
        Returns:
            Dictionary with outputs from all heads
        """
        if isinstance(z_or_dict, dict):
            if "latent" in z_or_dict:
                # Regular mode: use provided latent
                z = z_or_dict["latent"]
            elif "mu" in z_or_dict and "logvar" in z_or_dict:
                # VAE mode: reparameterization trick
                mu = z_or_dict["mu"]
                logvar = z_or_dict["logvar"]
                std = torch.exp(0.5 * logvar)
                epsilon = torch.randn_like(std)
                z = mu + std * epsilon
            else:
                raise ValueError(f"Dict must contain 'latent' or 'mu'/'logvar'. Got keys: {list(z_or_dict.keys())}")
        else:
            raise TypeError(f"Decoder forward expects dict, got {type(z_or_dict)}")
        
        feats = self.shared_decoder(z)
        outputs = {name: head(feats) for name, head in self.heads.items()}
        
        # Convert RGB from [-1, 1] (tanh) to [0, 255] for proper image values
        if "rgb" in outputs:
            rgb = outputs["rgb"]  # RGB in [-1, 1] range from tanh
            # Convert to [0, 255] range
            rgb = (rgb + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            rgb = rgb * 255.0  # [0, 1] -> [0, 255]
            rgb = torch.clamp(rgb, 0.0, 255.0)  # Ensure valid range
            outputs["rgb"] = rgb
        
        return outputs

    def to_config(self):
        cfg = super().to_config()
        cfg["heads"] = [head.to_config() for head in self.heads.values()]
        return cfg
