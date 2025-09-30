import torch
import torch.nn as nn


# --- helpers ---
def make_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
               norm=None, act=None, dropout=0.0, transpose=False):
    """Create a single conv/deconv layer with optional norm, activation, dropout"""
    
    if transpose:
        # Calculate output_padding for transposed conv
        output_padding = 0
        if stride > 1:
            output_padding = stride - 1
        
        conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
    else:
        conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding
        )

    layers = [conv]

    if norm == "batch":
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == "instance":
        layers.append(nn.InstanceNorm2d(out_channels))

    # Use non-inplace operations to avoid gradient issues
    if act == "relu":
        layers.append(nn.ReLU(inplace=False))  # Changed to inplace=False
    elif act == "leakyrelu":
        layers.append(nn.LeakyReLU(0.2, inplace=False))  # Changed to inplace=False
    elif act == "tanh":
        layers.append(nn.Tanh())
    elif act == "sigmoid":
        layers.append(nn.Sigmoid())

    if dropout and dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers)


# --- Encoder ---
class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 layers_cfg: list[dict],
                 latent_dim: int,
                 image_size: int,
                 global_norm=None,
                 global_act=None,
                 global_dropout: float = 0.0):
        super().__init__()

        layers = []
        prev = in_channels
        factor = 1
        
        # Track dimensions for debugging
        current_size = image_size
        
        for i, cfg in enumerate(layers_cfg):
            out_ch = cfg["out_channels"]
            k = cfg.get("kernel_size", 3)
            s = cfg.get("stride", 1)
            p = cfg.get("padding", 1)
            norm = cfg.get("norm")
            act = cfg.get("act")
            drop = cfg.get("dropout", 0.0)
            
            layers.append(make_layer(prev, out_ch, k, s, p, norm, act, drop))
            prev = out_ch
            factor *= s
            current_size = current_size // s
            print(f"Encoder layer {i}: in={prev}, out={out_ch}, size {current_size}x{current_size}")
        
        self.conv = nn.Sequential(*layers)
        
        # Calculate flattened size
        reduced_size = image_size // factor
        flatten_dim = prev * reduced_size * reduced_size
        
        self.fc = nn.Linear(flatten_dim, latent_dim)
        
        # Store for reference
        self.reduced_size = reduced_size
        self.final_channels = prev
        
        print(f"Encoder output: {prev} channels, {reduced_size}x{reduced_size} -> {latent_dim} latent dim")

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return z


# --- Decoder ---
class ConvDecoder(nn.Module):
    def __init__(self,
                 out_channels: int,
                 latent_dim: int,
                 encoder_final_channels: int,
                 encoder_reduced_size: int,
                 encoder_layers_cfg: list[dict],
                 image_size: int):
        super().__init__()
        
        # Start from encoder's final state
        self.reduced_size = encoder_reduced_size
        self.start_channels = encoder_final_channels
        
        # Project from latent to spatial
        flatten_dim = self.start_channels * self.reduced_size * self.reduced_size
        self.fc = nn.Linear(latent_dim, flatten_dim)
        
        # Build decoder layers by reversing encoder
        layers = []
        prev = self.start_channels
        current_size = self.reduced_size
        
        # Process encoder layers in reverse order
        reversed_configs = list(reversed(encoder_layers_cfg))
        
        # Track the previous output channel count
        prev_ch = self.start_channels
        
        for i in range(len(reversed_configs)):
            # Input channels is the previous layer's output
            in_ch = prev_ch
            
            # Determine output channels
            if i < len(reversed_configs) - 1:
                # Mirror the encoder: go from 128 -> 64 -> 32
                out_ch = reversed_configs[i+1]["out_channels"]
            else:
                # Last layer: use enough channels to avoid issues
                out_ch = 32  # Keep reasonable channel count before final conv
            
            # Get other parameters from the corresponding encoder layer
            cfg = reversed_configs[i]
            k = cfg.get("kernel_size", 3)
            s = cfg.get("stride", 1)
            p = cfg.get("padding", 1)
            norm = cfg.get("norm")
            act = cfg.get("act")
            drop = cfg.get("dropout", 0.0)
            
            layers.append(make_layer(in_ch, out_ch, k, s, p, norm, act, drop, transpose=True))
            current_size = current_size * s
            print(f"Decoder layer {i}: in={in_ch}, out={out_ch}, size {current_size}x{current_size}")
            
            # Update prev_ch for next iteration
            prev_ch = out_ch
        
        prev = prev_ch  # For final conv layer
        
        self.deconv = nn.Sequential(*layers)
        
        # Final conv to get exact output channels
        self.final = nn.Conv2d(prev, out_channels, kernel_size=3, padding=1)
        print(f"Decoder final conv: {prev} -> {out_channels} channels")

    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.start_channels, self.reduced_size, self.reduced_size)
        
        # Upsample
        x = self.deconv(x)
        
        # Final conv and activation
        x = self.final(x)
        x = torch.sigmoid(x)
        
        return x


# --- AutoEncoder ---
class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    @classmethod
    def from_config(cls, cfg: dict | str):
        import yaml
        if isinstance(cfg, str):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

        enc_cfg = cfg["encoder"]
        dec_cfg = cfg["decoder"]

        # Build encoder
        encoder = ConvEncoder(
            in_channels=enc_cfg["in_channels"],
            layers_cfg=enc_cfg["layers"],
            latent_dim=enc_cfg["latent_dim"],
            image_size=enc_cfg["image_size"],
            global_norm=enc_cfg.get("global_norm"),
            global_act=enc_cfg.get("global_act"),
            global_dropout=enc_cfg.get("global_dropout", 0.0),
        )

        # Build decoder using encoder's configuration
        decoder = ConvDecoder(
            out_channels=dec_cfg["out_channels"],
            latent_dim=dec_cfg["latent_dim"],
            encoder_final_channels=encoder.final_channels,
            encoder_reduced_size=encoder.reduced_size,
            encoder_layers_cfg=enc_cfg["layers"],
            image_size=dec_cfg["image_size"]
        )

        return cls(encoder, decoder)