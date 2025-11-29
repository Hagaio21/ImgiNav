import torch
import torch.nn as nn
from .components.base_component import BaseComponent
from .utils import compute_num_groups, reparameterize


class Decoder(BaseComponent):
    """
    Deterministic decoder - symmetric with Encoder.
    
    Architecture per upsampling level:
        ConvT 4x4 stride 2 → Norm → Act → Conv 3x3 → Norm → Act
    
    This matches encoder's 2 convs per level, making it symmetric.
    
    Config:
        latent_channels: Input latent channels (default: 4)
        base_channels: Base channel count (default: 64)
        upsampling_steps: Number of upsample levels (default: 3)
        norm_groups: Groups for GroupNorm (default: 8)
        activation: Activation function (default: SiLU)
        heads: List of head configurations
    
    For 32x32 latent with upsampling_steps=3:
        32 → 64 → 128 → 256 (output: 256x256)
    """
    
    def _build(self):
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        base_ch = self._init_kwargs.get("base_channels", 64)
        up_steps = self._init_kwargs.get("upsampling_steps", 3)
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        act = getattr(nn, self._init_kwargs.get("activation", "SiLU"))()
        head_cfgs = self._init_kwargs.get("heads", [])

        layers = []
        
        # Initial projection from latent
        # Start with highest channel count (matching encoder's final channels)
        out_ch = base_ch * (2 ** (up_steps - 1))
        valid_groups = compute_num_groups(out_ch, norm_groups)
        layers += [
            nn.Conv2d(latent_ch, out_ch, 3, padding=1),
            nn.GroupNorm(valid_groups, out_ch),
            act,
        ]

        # Upsampling levels - symmetric with encoder
        for i in range(up_steps):
            out_ch_next = out_ch // 2 if i < up_steps - 1 else base_ch
            valid_groups = compute_num_groups(out_ch_next, norm_groups)
            layers += [
                # Upsample
                nn.ConvTranspose2d(out_ch, out_ch_next, 4, stride=2, padding=1),
                nn.GroupNorm(valid_groups, out_ch_next),
                act,
                # Refine (added for symmetry with encoder)
                nn.Conv2d(out_ch_next, out_ch_next, 3, padding=1),
                nn.GroupNorm(valid_groups, out_ch_next),
                act,
            ]
            out_ch = out_ch_next

        self.shared_decoder = nn.Sequential(*layers)
        self.shared_out_channels = out_ch

        # Build heads
        self.heads = nn.ModuleDict()
        for cfg in head_cfgs:
            head_type = cfg.get("type", "DecoderHead")
            head_name = cfg.get("name", head_type.lower())
            cfg["in_channels"] = cfg.get("in_channels", out_ch)
            self.heads[head_name] = self.create_component_from_config(cfg, default_type=head_type)

    def forward(self, z_or_dict):
        """
        Forward pass.
        
        Args:
            z_or_dict: DataFlow or dict containing "latent": tensor z
        
        Returns:
            DataFlow with outputs from all heads
        """
        if isinstance(z_or_dict, dict):
            if "latent" not in z_or_dict:
                raise ValueError(f"Decoder expects dict with 'latent' key. Got: {list(z_or_dict.keys())}")
            z = z_or_dict["latent"]
        else:
            raise TypeError(f"Decoder expects dict, got {type(z_or_dict)}")
        
        feats = self.shared_decoder(z)
        outputs = {name: head(feats) for name, head in self.heads.items()}
        return self._to_dataflow(outputs)
    
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
    """Variational decoder - handles mu/logvar with reparameterization."""
    
    def forward(self, z_or_dict):
        """
        Forward pass.
        
        Args:
            z_or_dict: Dict containing "latent" or "mu"/"logvar"
        
        Returns:
            DataFlow with outputs from all heads
        """
        if isinstance(z_or_dict, dict):
            if "latent" in z_or_dict:
                z = z_or_dict["latent"]
            elif "mu" in z_or_dict and "logvar" in z_or_dict:
                z = reparameterize(z_or_dict["mu"], z_or_dict["logvar"])
            else:
                raise ValueError(f"VAEDecoder expects 'latent' or 'mu'/'logvar'. Got: {list(z_or_dict.keys())}")
        else:
            raise TypeError(f"VAEDecoder expects dict, got {type(z_or_dict)}")
        
        feats = self.shared_decoder(z)
        outputs = {name: head(feats) for name, head in self.heads.items()}
        return self._to_dataflow(outputs)