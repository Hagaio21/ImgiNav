import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional


# --- helpers ---
def make_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
               norm=None, act=None, dropout=0.0, transpose=False):
    """Create a single conv/deconv layer with optional norm, activation, dropout"""
    if transpose:
        output_padding = 0 if stride <= 1 else stride - 1
        conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
    else:
        conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    layers = [conv]

    if norm == "batch":
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == "instance":
        layers.append(nn.InstanceNorm2d(out_channels))

    if act == "relu":
        layers.append(nn.ReLU(inplace=False))
    elif act == "leakyrelu":
        layers.append(nn.LeakyReLU(0.2, inplace=False))
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
                 layers_cfg: List[Dict[str, Any]],
                 latent_dim: int,
                 image_size: int,
                 latent_channels: int,
                 latent_base: int,
                 global_norm: Optional[str] = None,
                 global_act: Optional[str] = None,
                 global_dropout: float = 0.0):
        super().__init__()

        layers = []
        prev = in_channels
        current_size = image_size

        for i, cfg in enumerate(layers_cfg):
            in_ch = prev
            out_ch = cfg["out_channels"]
            k = cfg.get("kernel_size", 3)
            s = cfg.get("stride", 1)
            p = cfg.get("padding", 1)
            norm = cfg.get("norm", global_norm)
            act = cfg.get("act", global_act)
            drop = cfg.get("dropout", global_dropout)

            layers.append(make_layer(prev, out_ch, k, s, p, norm, act, drop))
            prev = out_ch
            current_size //= s
            print(f"[Encoder] Layer {i}: in={in_ch}, out={out_ch}, size={current_size}x{current_size}")

        self.conv = nn.Sequential(*layers)
        
        # Final layer maps conv output to target latent cube shape
        # Input: prev channels at current_size x current_size
        # Output: latent_channels at latent_base x latent_base
        self.to_latent = nn.Sequential(
            nn.Conv2d(prev, latent_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d((latent_base, latent_base))
        )
        
        self.latent_channels = latent_channels
        self.latent_base = latent_base
        self.conv_output_channels = prev
        self.conv_output_size = current_size
        
        print(f"[Encoder] Conv output: {prev}x{current_size}x{current_size}")
        print(f"[Encoder] Latent output: {latent_channels}x{latent_base}x{latent_base}")

    def forward(self, x):
        x = self.conv(x)
        z = self.to_latent(x)
        z = z / (z.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-8) # normalization added
        return z


# --- Decoder ---
class ConvDecoder(nn.Module):
    def __init__(self,
                 out_channels: int,
                 latent_dim: int,
                 encoder_layers_cfg: List[Dict[str, Any]],
                 image_size: int,
                 latent_channels: int,
                 latent_base: int,
                 global_norm: Optional[str] = None,
                 global_act: Optional[str] = None,
                 global_dropout: float = 0.0):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_base = latent_base
        
        # Calculate what the encoder's final conv output size would be
        start_size = image_size
        for cfg in encoder_layers_cfg:
            start_size //= cfg.get("stride", 1)
        
        start_channels = encoder_layers_cfg[-1]["out_channels"]
        
        print(f"[Decoder] Latent input: {latent_channels}x{latent_base}x{latent_base}")
        print(f"[Decoder] Decoder start: {start_channels}x{start_size}x{start_size}")
        
        # Map from latent cube to decoder starting point
        self.from_latent = nn.Sequential(
            nn.Upsample(size=(start_size, start_size), mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels, start_channels, kernel_size=1)
        )
        
        # Build deconv layers (mirror of encoder)
        layers = []
        prev_ch = start_channels
        current_size = start_size
        reversed_configs = list(reversed(encoder_layers_cfg))

        for i, cfg in enumerate(reversed_configs):
            in_ch = prev_ch
            out_ch = cfg["out_channels"]
            k = cfg.get("kernel_size", 3)
            s = cfg.get("stride", 1)
            p = cfg.get("padding", 1)
            norm = cfg.get("norm", global_norm)
            act = cfg.get("act", global_act)
            drop = cfg.get("dropout", global_dropout)

            layers.append(make_layer(in_ch, out_ch, k, s, p, norm, act, drop, transpose=True))
            current_size *= s
            print(f"[Decoder] Layer {i}: in={in_ch}, out={out_ch}, size={current_size}x{current_size}")
            prev_ch = out_ch

        self.deconv = nn.Sequential(*layers)
        self.final = nn.Conv2d(prev_ch, out_channels, kernel_size=3, padding=1)
        print(f"[Decoder] Final output: {out_channels}x{current_size}x{current_size}")

    def forward(self, z):
        x = self.from_latent(z)
        x = self.deconv(x)
        x = self.final(x)
        return torch.sigmoid(x)


# --- AutoEncoder wrapper ---
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

        encoder = ConvEncoder(
            in_channels=enc_cfg["in_channels"],
            layers_cfg=enc_cfg["layers"],
            latent_dim=enc_cfg["latent_dim"],
            image_size=enc_cfg["image_size"],
            latent_channels=enc_cfg["latent_channels"],
            latent_base=enc_cfg["latent_base"],
            global_norm=enc_cfg.get("global_norm"),
            global_act=enc_cfg.get("global_act"),
            global_dropout=enc_cfg.get("global_dropout", 0.0),
        )

        decoder = ConvDecoder(
            out_channels=dec_cfg["out_channels"],
            latent_dim=dec_cfg["latent_dim"],
            encoder_layers_cfg=enc_cfg["layers"],
            image_size=dec_cfg["image_size"],
            latent_channels=dec_cfg["latent_channels"],
            latent_base=dec_cfg["latent_base"],
            global_norm=dec_cfg.get("global_norm"),
            global_act=dec_cfg.get("global_act"),
            global_dropout=dec_cfg.get("global_dropout", 0.0),
        )

        return cls(encoder, decoder)