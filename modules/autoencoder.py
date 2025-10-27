import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import yaml


# --- helpers ---
def make_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
               norm=None, act=None, dropout=0.0, transpose=False):
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

        for cfg in layers_cfg:
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

        self.conv = nn.Sequential(*layers)

        self.to_latent_mu = nn.Sequential(
            nn.Conv2d(prev, latent_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d((latent_base, latent_base))
        )
        self.to_latent_logvar = nn.Sequential(
            nn.Conv2d(prev, latent_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d((latent_base, latent_base))
        )

        self.latent_channels = latent_channels
        self.latent_base = latent_base
        self.conv_output_channels = prev
        self.conv_output_size = current_size
        self.image_size = image_size

    def forward(self, x):
        x = self.conv(x)
        mu = self.to_latent_mu(x)
        logvar = self.to_latent_logvar(x)
        return mu, logvar

    def print_summary(self):
        print(f"[Encoder] Conv output: {self.conv_output_channels}x{self.conv_output_size}x{self.conv_output_size}")
        print(f"[Encoder] Latent output: {self.latent_channels}x{self.latent_base}x{self.latent_base}")


# --- Decoder (supports optional segmentation head) ---
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
                 use_sigmoid: bool = False,
                 global_dropout: float = 0.0,
                 num_classes: Optional[int] = None):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_base = latent_base
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid

        start_size = image_size
        for cfg in encoder_layers_cfg:
            start_size //= cfg.get("stride", 1)
        start_channels = encoder_layers_cfg[-1]["out_channels"]

        self.from_latent = nn.Sequential(
            nn.Upsample(size=(start_size, start_size), mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels, start_channels, kernel_size=1)
        )

        layers = []
        prev_ch = start_channels
        current_size = start_size
        reversed_configs = list(reversed(encoder_layers_cfg))

        for cfg in reversed_configs:
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
            prev_ch = out_ch

        self.deconv = nn.Sequential(*layers)
        self.final_rgb = nn.Conv2d(prev_ch, out_channels, kernel_size=3, padding=1)

        # optional segmentation head
        self.seg_head = None
        if num_classes is not None and num_classes > 0:
            self.seg_head = nn.Conv2d(prev_ch, num_classes, kernel_size=1)

        self.output_size = current_size
        self.output_channels = out_channels

    def forward(self, z):
        x = self.from_latent(z)
        x = self.deconv(x)
        rgb_out = torch.sigmoid(self.final_rgb(x)) if self.use_sigmoid else self.final_rgb(x)
        seg_logits = self.seg_head(x) if self.seg_head is not None else None
        return rgb_out, seg_logits

    def print_summary(self):
        print(f"[Decoder] Final RGB output: {self.output_channels}x{self.output_size}x{self.output_size}")
        if self.seg_head is not None:
            print(f"[Decoder] Segmentation head: {self.seg_head.out_channels} classes")


# --- AutoEncoder wrapper (VAE + optional segmentation) ---
class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, deterministic: bool = True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, deterministic=True):
        mu, logvar = self.encoder(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        rgb_out, seg_logits = self.decoder(z)
        return {"recon": rgb_out, "seg_logits": seg_logits, "mu": mu, "logvar": logvar}



    @torch.no_grad()
    def decode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        rgb_out, seg_logits = self.decoder(z)
        return rgb_out, seg_logits

    @torch.no_grad()
    def sample(self, z):
        self.eval()
        rgb_out, seg_logits = self.decoder(z)
        return rgb_out, seg_logits

    @torch.no_grad()
    def encode(self, x):
        self.eval()
        mu, logvar = self.encoder(x)
        return mu, logvar

    @torch.no_grad()
    def encode_latent(self, x, deterministic: bool = True):
        self.eval()
        mu, logvar = self.encoder(x)
        return mu if deterministic else self.reparameterize(mu, logvar)

    def to_config(self):
        """Return YAML-compatible config that fully reproduces architecture."""
        enc = self.encoder
        dec = self.decoder

        def infer_norm_act(seq):
            norm = None
            act = None
            for m in seq.modules():
                if isinstance(m, nn.BatchNorm2d):
                    norm = "batch"
                elif isinstance(m, nn.InstanceNorm2d):
                    norm = "instance"
                elif isinstance(m, nn.ReLU):
                    act = "relu"
                elif isinstance(m, nn.LeakyReLU):
                    act = "leakyrelu"
                elif isinstance(m, nn.Tanh):
                    act = "tanh"
                elif isinstance(m, nn.Sigmoid):
                    act = "sigmoid"
            return norm, act

        enc_norm, enc_act = infer_norm_act(enc.conv)
        dec_norm, dec_act = infer_norm_act(dec.deconv)

        layers_cfg = []
        for layer in enc.conv:
            if isinstance(layer, nn.Sequential) and isinstance(layer[0], nn.Conv2d):
                conv = layer[0]
                layers_cfg.append({
                    "out_channels": conv.out_channels,
                    "kernel_size": conv.kernel_size[0],
                    "stride": conv.stride[0],
                    "padding": conv.padding[0],
                })

        cfg = {
            "encoder": {
                "in_channels": enc.conv[0][0].in_channels if hasattr(enc.conv[0][0], "in_channels") else None,
                "layers": layers_cfg,
                "image_size": enc.image_size,
                "latent_channels": enc.latent_channels,
                "latent_base": enc.latent_base,
                "global_norm": enc_norm,
                "global_act": enc_act,
                "global_dropout": 0.0,
            },
            "decoder": {
                "out_channels": dec.output_channels,
                "image_size": dec.output_size,
                "latent_channels": dec.latent_channels,
                "latent_base": dec.latent_base,
                "global_norm": dec_norm,
                "global_act": dec_act,
                "global_dropout": 0.0,
                "use_sigmoid": getattr(dec, "use_sigmoid", False),
                "num_classes": getattr(dec, "num_classes", None),
            },
        }
        return cfg

    @classmethod
    def from_config(cls, cfg: dict | str):
        if isinstance(cfg, str):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

        if "encoder" not in cfg and "decoder" not in cfg:
            return cls.from_shape(**cfg)

        enc_cfg = cfg["encoder"]
        dec_cfg = cfg["decoder"]

        encoder = ConvEncoder(
            in_channels=enc_cfg["in_channels"],
            layers_cfg=enc_cfg["layers"],
            latent_dim=enc_cfg.get("latent_dim", 0),
            image_size=enc_cfg["image_size"],
            latent_channels=enc_cfg["latent_channels"],
            latent_base=enc_cfg["latent_base"],
            global_norm=enc_cfg.get("global_norm"),
            global_act=enc_cfg.get("global_act"),
            global_dropout=enc_cfg.get("global_dropout", 0.0),
        )

        decoder = ConvDecoder(
            out_channels=dec_cfg["out_channels"],
            latent_dim=dec_cfg.get("latent_dim", 0),
            encoder_layers_cfg=enc_cfg["layers"],
            image_size=dec_cfg["image_size"],
            latent_channels=dec_cfg["latent_channels"],
            latent_base=dec_cfg["latent_base"],
            global_norm=dec_cfg.get("global_norm"),
            global_act=dec_cfg.get("global_act"),
            global_dropout=dec_cfg.get("global_dropout", 0.0),
            use_sigmoid=dec_cfg.get("use_sigmoid", False),
            num_classes=dec_cfg.get("num_classes", None),
        )

        # read type if present (default to vae)
        model_type = cfg.get("type", "vae").lower()
        deterministic = model_type == "ae"

        return cls(encoder, decoder, deterministic=deterministic)

    @classmethod
    def from_shape(cls,
                   in_channels: int,
                   out_channels: int,
                   base_channels: int,
                   latent_channels: int,
                   image_size: int,
                   latent_base: int,
                   norm: Optional[str] = None,
                   act: Optional[str] = "relu",
                   dropout: float = 0.0,
                   kernel_size: int = 3,
                   stride: int = 2,
                   padding: int = 1,
                   num_classes: Optional[int] = None):
        ratio = image_size // latent_base
        depth = int(torch.log2(torch.tensor(ratio)).item())
        if 2 ** depth != ratio:
            raise ValueError("image_size / latent_base must be a power of 2")

        layers_cfg = []
        ch = base_channels
        for i in range(depth):
            layers_cfg.append({
                "out_channels": ch,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "norm": norm,
                "act": act,
                "dropout": dropout,
            })
            ch *= 2

        encoder = ConvEncoder(
            in_channels=in_channels,
            layers_cfg=layers_cfg,
            latent_dim=0,
            image_size=image_size,
            latent_channels=latent_channels,
            latent_base=latent_base,
            global_norm=norm,
            global_act=act,
            global_dropout=dropout,
        )

        decoder = ConvDecoder(
            out_channels=out_channels,
            latent_dim=0,
            encoder_layers_cfg=layers_cfg,
            image_size=image_size,
            latent_channels=latent_channels,
            latent_base=latent_base,
            global_norm=norm,
            global_act=act,
            global_dropout=dropout,
            use_sigmoid=True,
            num_classes=num_classes,
        )

        return cls(encoder, decoder)
