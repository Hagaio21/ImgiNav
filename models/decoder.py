"""
Decoder models for autoencoders and VAEs.

Consolidation notes:
- No major changes needed; decoder was already clean
- Minor simplification of error messages
"""

import torch
import torch.nn as nn
from .components.base_component import BaseComponent
from .utils import compute_num_groups, reparameterize


class Decoder(BaseComponent):
    """
    Deterministic decoder - symmetric with Encoder.
    
    Architecture mirrors encoder in reverse. Each level:
        ConvT 4x4 stride 2 → Norm → Act → Conv 3x3 → Norm → Act
    
    Config:
        latent_channels: Input latent channels (default: 4)
        base_channels: Base channel count (default: 64)
        upsampling_steps: Number of upsample levels (default: 3)
        norm_groups: Groups for GroupNorm (default: 8)
        activation: Activation function (default: SiLU)
        heads: List of head configurations
    """
    
    def _build(self):
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        base_ch = self._init_kwargs.get("base_channels", 64)
        up_steps = self._init_kwargs.get("upsampling_steps", 3)
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        act = getattr(nn, self._init_kwargs.get("activation", "SiLU"))()
        head_cfgs = self._init_kwargs.get("heads", [])

        layers = []
        
        # Start with highest channel count
        highest_ch = base_ch * (2 ** (up_steps - 1))
        valid_groups = compute_num_groups(highest_ch, norm_groups)
        
        # Initial projection from latent to highest channel count
        layers += [
            nn.Conv2d(latent_ch, highest_ch, 3, padding=1),
            nn.GroupNorm(valid_groups, highest_ch),
            act,
        ]
        
        # Refinement layer (mirrors encoder's final refinement)
        layers += [
            nn.Conv2d(highest_ch, highest_ch, 3, padding=1),
            nn.GroupNorm(valid_groups, highest_ch),
            act,
        ]

        # Upsampling levels
        in_ch = highest_ch
        for i in range(up_steps):
            encoder_mirror_level = up_steps - 2 - i
            out_ch = base_ch * (2 ** encoder_mirror_level) if encoder_mirror_level >= 0 else base_ch
            
            valid_groups_in = compute_num_groups(in_ch, norm_groups)
            valid_groups_out = compute_num_groups(out_ch, norm_groups)
            
            layers += [
                nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1),
                nn.GroupNorm(valid_groups_in, in_ch),
                act,
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(valid_groups_out, out_ch),
                act,
            ]
            in_ch = out_ch

        self.shared_decoder = nn.Sequential(*layers)
        self.shared_out_channels = in_ch

        # Build heads
        self.heads = nn.ModuleDict()
        for cfg in head_cfgs:
            head_type = cfg.get("type", "DecoderHead")
            head_name = cfg.get("name", head_type.lower())
            cfg["in_channels"] = cfg.get("in_channels", in_ch)
            self.heads[head_name] = self.create_component_from_config(cfg, default_type=head_type)

    def _extract_latent(self, z_or_dict):
        """Extract latent tensor from input dict."""
        if not isinstance(z_or_dict, dict):
            raise TypeError(f"Decoder expects dict, got {type(z_or_dict)}")
        
        if "latent" not in z_or_dict:
            raise ValueError(f"Decoder expects dict with 'latent' key. Got: {list(z_or_dict.keys())}")
        
        return z_or_dict["latent"]

    def forward(self, z_or_dict):
        """
        Forward pass.
        
        Args:
            z_or_dict: Dict containing "latent": tensor z
        
        Returns:
            Dict with outputs from all heads
        """
        z = self._extract_latent(z_or_dict)
        feats = self.shared_decoder(z)
        outputs = {name: head(feats) for name, head in self.heads.items()}
        return outputs
    
    def get_input_shape(self, batch_size=1):
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        return {"latent": (batch_size, latent_ch, None, None)}
    
    def get_output_shape(self, batch_size=1):
        outputs = {}
        for name, head in self.heads.items():
            if hasattr(head, 'get_output_shape'):
                outputs[name] = head.get_output_shape(batch_size)
            else:
                out_ch = getattr(head, 'out_channels', 3)
                outputs[name] = (batch_size, out_ch, None, None)
        return outputs

    def to_config(self):
        cfg = super().to_config()
        cfg["heads"] = [head.to_config() for head in self.heads.values()]
        return cfg


class VAEDecoder(Decoder):
    """
    Variational decoder - handles mu/logvar with reparameterization.
    
    Can accept either:
    - {"latent": z} - use z directly
    - {"mu": mu, "logvar": logvar} - reparameterize to get z
    """
    
    def _extract_latent(self, z_or_dict):
        """Extract or sample latent from input dict."""
        if not isinstance(z_or_dict, dict):
            raise TypeError(f"VAEDecoder expects dict, got {type(z_or_dict)}")
        
        if "latent" in z_or_dict:
            return z_or_dict["latent"]
        
        if "mu" in z_or_dict and "logvar" in z_or_dict:
            return reparameterize(z_or_dict["mu"], z_or_dict["logvar"])
        
        raise ValueError(f"VAEDecoder expects 'latent' or 'mu'/'logvar'. Got: {list(z_or_dict.keys())}")
    
    def forward(self, z_or_dict):
        """
        Forward pass.
        
        Args:
            z_or_dict: Dict containing "latent" or "mu"/"logvar"
        
        Returns:
            Dict with outputs from all heads
        """
        z = self._extract_latent(z_or_dict)
        feats = self.shared_decoder(z)
        outputs = {name: head(feats) for name, head in self.heads.items()}
        return outputs
