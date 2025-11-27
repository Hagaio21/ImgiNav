import torch
import torch.nn as nn
from .components.base_component import BaseComponent
from .utils import compute_num_groups, reparameterize


class Decoder(BaseComponent):
    """Deterministic decoder - expects latent tensor."""
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

        # Compute valid num_groups for initial layer
        valid_groups = compute_num_groups(out_ch, norm_groups)
        layers += [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(valid_groups, out_ch),
            activation,
        ]

        for _ in range(up_steps):
            out_ch_next = out_ch // 2
            # Compute valid num_groups for this layer
            valid_groups = compute_num_groups(out_ch_next, norm_groups)
            layers += [
                nn.ConvTranspose2d(out_ch, out_ch_next, 4, stride=2, padding=1),
                nn.GroupNorm(valid_groups, out_ch_next),
                activation,
            ]
            out_ch = out_ch_next

        self.shared_decoder = nn.Sequential(*layers)
        self.shared_out_channels = out_ch

        self.heads = nn.ModuleDict()
        for cfg in head_cfgs:
            head_type = cfg.get("type", "DecoderHead")
            head_name = cfg.get("name", head_type.lower())
            cfg["in_channels"] = cfg.get("in_channels", out_ch)
            # Use unified component registry
            self.heads[head_name] = self.create_component_from_config(cfg, default_type=head_type)

    def forward(self, z_or_dict):
        """
        Forward pass. Accepts dict/DataFlow with latent tensor.
        
        Args:
            z_or_dict: DataFlow or dict containing "latent": tensor z
        
        Returns:
            DataFlow with outputs from all heads
        """
        # Handle DataFlow or dict input
        if isinstance(z_or_dict, dict):
            if "latent" not in z_or_dict:
                raise ValueError(f"Decoder expects dict/DataFlow with 'latent' key. Got keys: {list(z_or_dict.keys())}")
            z = z_or_dict["latent"]
        else:
            raise TypeError(f"Decoder forward expects dict/DataFlow, got {type(z_or_dict)}")
        
        feats = self.shared_decoder(z)
        outputs = {name: head(feats) for name, head in self.heads.items()}
        
        # Keep RGB in [-1, 1] range (tanh output) for training compatibility
        # Conversion to [0, 255] should be done when saving images, not here
        return self._to_dataflow(outputs)
    
    def get_input_shape(self, batch_size=1):
        """Get expected input shape."""
        latent_ch = self._init_kwargs.get("latent_channels", 4)
        up_steps = self._init_kwargs.get("upsampling_steps", 4)
        # Latent is typically 32x32 for 512x512 output (downsampled by 2^4)
        spatial_res = None  # Depends on encoder, but typically 32x32
        return {"latent": (batch_size, latent_ch, spatial_res, spatial_res)}
    
    def get_output_shape(self, batch_size=1):
        """Get expected output shape."""
        up_steps = self._init_kwargs.get("upsampling_steps", 4)
        # Output is upsampled by 2^up_steps
        # If latent is 32x32, output is 32*(2^4) = 512x512
        spatial_res = None  # Depends on input, but typically 512x512
        outputs = {}
        for name, head in self.heads.items():
            # Each head outputs its own shape
            if hasattr(head, 'get_output_shape'):
                outputs[name] = head.get_output_shape(batch_size)
            else:
                # Default: assume head outputs same spatial resolution
                out_ch = getattr(head, 'out_channels', 3)
                outputs[name] = (batch_size, out_ch, spatial_res, spatial_res)
        return outputs

    def to_config(self):
        cfg = super().to_config()
        cfg["heads"] = [head.to_config() for head in self.heads.values()]
        return cfg


class VAEDecoder(Decoder):
    """Variational decoder - handles mu/logvar and performs reparameterization."""
    
    def forward(self, z_or_dict):
        """
        Forward pass. Accepts dict with mu/logvar or latent.
        
        Args:
            z_or_dict: Dictionary containing:
                - "latent": tensor z (already sampled)
                - "mu" and "logvar": tensors (will reparameterize)
        
        Returns:
            Dictionary with outputs from all heads
        """
        if isinstance(z_or_dict, dict):
            if "latent" in z_or_dict:
                # Already sampled: use provided latent
                z = z_or_dict["latent"]
            elif "mu" in z_or_dict and "logvar" in z_or_dict:
                # VAE mode: reparameterization trick
                mu = z_or_dict["mu"]
                logvar = z_or_dict["logvar"]
                z = reparameterize(mu, logvar)
            else:
                raise ValueError(
                    f"VAEDecoder expects dict with 'latent' or 'mu'/'logvar' keys. "
                    f"Got keys: {list(z_or_dict.keys())}"
                )
        else:
            raise TypeError(f"VAEDecoder forward expects dict, got {type(z_or_dict)}")
        
        feats = self.shared_decoder(z)
        outputs = {name: head(feats) for name, head in self.heads.items()}
        
        # Keep RGB in [-1, 1] range (tanh output) for training compatibility
        # Conversion to [0, 255] should be done when saving images, not here
        return self._to_dataflow(outputs)
