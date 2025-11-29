import torch
import torch.nn as nn
from .components.base_component import BaseComponent
from .utils import compute_num_groups, reparameterize


class Decoder(BaseComponent):
    """
    Deterministic decoder - symmetric with Encoder.
    
    Architecture mirrors encoder in reverse. Each level:
        ConvT 4x4 stride 2 → Norm → Act → Conv 3x3 → Norm → Act
    
    This mirrors encoder's per-level structure (in reverse):
        Encoder: Conv 3x3 (change channels) → Conv 4x4 stride 2 (downsample, keep channels)
        Decoder: ConvT 4x4 stride 2 (upsample, keep channels) → Conv 3x3 (change channels)
    
    Channel progression (base_ch, up_steps from config) - exact mirror of encoder:
        Level i (decoder) mirrors Level (up_steps - 1 - i) (encoder)
        
        Example with base_ch=64, up_steps=3:
            Initial proj: latent_ch → base_ch * 2^(up_steps-1)  [4 → 256]
            Refinement: same channels                          [256 → 256]
            Level 0: 256 → 256 (upsample), 256 → 128 (change)  [mirrors encoder level 2]
            Level 1: 128 → 128 (upsample), 128 → 64 (change)   [mirrors encoder level 1]
            Level 2: 64 → 64 (upsample), 64 → 64 (change)      [mirrors encoder level 0]
            Head: 64 → out_channels
    
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
        
        # Start with highest channel count (matching encoder's final channels before latent proj)
        # This is base_ch * 2^(up_steps-1)
        highest_ch = base_ch * (2 ** (up_steps - 1))
        valid_groups = compute_num_groups(highest_ch, norm_groups)
        
        # Initial projection from latent to highest channel count
        layers += [
            nn.Conv2d(latent_ch, highest_ch, 3, padding=1),
            nn.GroupNorm(valid_groups, highest_ch),
            act,
        ]
        
        # Refinement layer (mirrors encoder's final refinement layer)
        layers += [
            nn.Conv2d(highest_ch, highest_ch, 3, padding=1),
            nn.GroupNorm(valid_groups, highest_ch),
            act,
        ]

        # Upsampling levels - mirror encoder levels in reverse order
        in_ch = highest_ch
        for i in range(up_steps):
            # Mirror encoder level (up_steps - 1 - i)
            # Encoder level j has channels = base_ch * 2^j after the first conv
            # We're transitioning from encoder level (up_steps-1-i) to level (up_steps-2-i)
            # Encoder level (up_steps-1-i) has channels = base_ch * 2^(up_steps-1-i)
            # Encoder level (up_steps-2-i) has channels = base_ch * 2^(up_steps-2-i)
            # So we go from base_ch * 2^(up_steps-1-i) to base_ch * 2^(up_steps-2-i)
            
            encoder_mirror_level = up_steps - 2 - i  # The encoder level we're transitioning TO
            if encoder_mirror_level >= 0:
                out_ch = base_ch * (2 ** encoder_mirror_level)
            else:
                out_ch = base_ch
            
            valid_groups_in = compute_num_groups(in_ch, norm_groups)
            valid_groups_out = compute_num_groups(out_ch, norm_groups)
            
            layers += [
                # First: upsample and keep channels (mirrors encoder's second conv in reverse)
                nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1),
                nn.GroupNorm(valid_groups_in, in_ch),
                act,
                # Second: change channels (mirrors encoder's first conv in reverse)
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(valid_groups_out, out_ch),
                act,
            ]
            in_ch = out_ch

        self.shared_decoder = nn.Sequential(*layers)
        self.shared_out_channels = in_ch  # Should be base_ch

        # Build heads
        self.heads = nn.ModuleDict()
        for cfg in head_cfgs:
            head_type = cfg.get("type", "DecoderHead")
            head_name = cfg.get("name", head_type.lower())
            cfg["in_channels"] = cfg.get("in_channels", in_ch)
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