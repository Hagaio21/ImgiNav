import torch
import torch.nn as nn
from .components.base_component import BaseComponent
from .components.blocks import UpBlock, MidBlock
from .utils import compute_num_groups, reparameterize


class Decoder(BaseComponent):
    """
    Deterministic decoder with residual blocks.
    
    Uses shared block classes with time_dim=None (no time conditioning).
    
    Required config:
        latent_channels: Latent space channels
        base_channels: Base channel count
        channel_multipliers: Channel multipliers per level (reversed from encoder)
        num_res_blocks: Residual blocks per level
        norm_groups: GroupNorm groups
        heads: List of head configurations
    
    Optional config:
        dropout: Dropout rate (default: 0.0)
    """
    
    def _build(self):
        latent_ch = self._init_kwargs["latent_channels"]
        base_ch = self._init_kwargs["base_channels"]
        ch_mults = self._init_kwargs["channel_multipliers"]
        num_res = self._init_kwargs["num_res_blocks"]
        norm_groups = self._init_kwargs["norm_groups"]
        dropout = self._init_kwargs.get("dropout", 0.0)
        head_cfgs = self._init_kwargs.get("heads", [])
        
        # Channel progression
        channels = [base_ch * m for m in ch_mults] + [base_ch]
        
        # Input projection
        self.conv_in = nn.Conv2d(latent_ch, channels[0], 3, padding=1)
        
        # Middle block
        self.mid_block = MidBlock(
            channels=channels[0],
            time_dim=None,  # No time conditioning for VAE
            norm_groups=norm_groups,
            dropout=dropout
        )
        
        # Upsampling blocks (time_dim=None, no skip connections for VAE)
        self.up_blocks = nn.ModuleList()
        for i in range(len(ch_mults)):
            self.up_blocks.append(
                UpBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    time_dim=None,  # No time conditioning for VAE
                    num_res_blocks=num_res,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    use_skip_connection=False  # No skip connections for VAE decoder
                )
            )
        
        # Output
        out_ch = channels[-1]
        self.norm_out = nn.GroupNorm(compute_num_groups(out_ch, norm_groups), out_ch)
        self.act_out = nn.SiLU()
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
            z_or_dict: DataFlow or dict with "latent" key
        
        Returns:
            DataFlow with head outputs
        """
        if isinstance(z_or_dict, dict):
            if "latent" not in z_or_dict:
                raise ValueError(f"Decoder expects 'latent' key. Got: {list(z_or_dict.keys())}")
            z = z_or_dict["latent"]
        else:
            raise TypeError(f"Decoder expects dict, got {type(z_or_dict)}")
        
        # Input projection
        x = self.conv_in(z)
        
        # Middle
        x = self.mid_block(x)
        
        # Upsampling (no skip connections for VAE)
        for block in self.up_blocks:
            x = block(x, skip=None, t_emb=None)
        
        # Output
        x = self.act_out(self.norm_out(x))
        
        # Heads
        outputs = {name: head(x) for name, head in self.heads.items()}
        
        return self._to_dataflow(outputs)
    
    def get_input_shape(self, batch_size=1):
        latent_ch = self._init_kwargs["latent_channels"]
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
    
    Same required config as Decoder.
    """
    
    def forward(self, z_or_dict):
        """
        Forward pass with reparameterization support.
        
        Args:
            z_or_dict: Dict with "latent" OR "mu"/"logvar" keys
        
        Returns:
            DataFlow with head outputs
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
        
        # Input projection
        x = self.conv_in(z)
        
        # Middle
        x = self.mid_block(x)
        
        # Upsampling
        for block in self.up_blocks:
            x = block(x, skip=None, t_emb=None)
        
        # Output
        x = self.act_out(self.norm_out(x))
        
        # Heads
        outputs = {name: head(x) for name, head in self.heads.items()}
        
        return self._to_dataflow(outputs)