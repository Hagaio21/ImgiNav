import math
import torch
import torch.nn as nn
from typing import Optional, List

"""
config file format:

# The U-Net is defined by the autoencoder latent shape [C, H, W].
# Example: latent = [8, 32, 32] → C=8, H=W=32

unet:
  in_channels: 8        <- set equal to latent_channels (C)
  cond_channels: 4     <- condition channels to concat (0 if none)
  out_channels: 8       <- must equal latent_channels (predict noise)
  base_channels: 64     <- internal width, scales capacity (not tied to latent)
  depth: 4              <- how many downsamples. must satisfy H/2^depth >= 1.
                           (for H=32, max depth=5)
  num_res_blocks: 2     <- residual blocks per stage
  time_dim: 128         <- timestep embedding size
  norm: batch           <- normalization
  act: relu             <- activation

Relation:
- Latent [C,H,W] fixes in_channels, out_channels, and spatial H,W.
- Config only changes internal computation, not I/O resolution.
"""

# --- sinusoidal time embedding ---
def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings like in DDPM/Transformer.
    timesteps: [B] int64
    Returns: [B, dim]
    """
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half_dim, device=device).float() / half_dim
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, t: torch.Tensor):
        emb = timestep_embedding(t, self.fc1.in_features)
        return self.fc2(self.act(self.fc1(emb)))


# --- residual block with time conditioning ---
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, norm=None, act=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(out_ch) if norm == "batch" else nn.Identity()
            self.norm2 = nn.BatchNorm2d(out_ch) if norm == "batch" else nn.Identity()
        elif norm == "group":
            num_groups = min(32, out_ch // 4) if out_ch >= 8 else 1
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.SiLU()  # default

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm=None, act=None):
        super().__init__()
        blocks = [ResidualBlock(in_ch, out_ch, time_dim, norm, act)]
        for _ in range(num_res_blocks - 1):
            blocks.append(ResidualBlock(out_ch, out_ch, time_dim, norm, act))
        self.res = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        h = self.res[0](x, t_emb)
        for b in self.res[1:]:
            h = b(h, t_emb)
        return self.pool(h), h


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1, norm=None, act=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        blocks = [ResidualBlock(out_ch * 2, out_ch, time_dim, norm, act)]
        for _ in range(num_res_blocks - 1):
            blocks.append(ResidualBlock(out_ch, out_ch, time_dim, norm, act))
        self.res = nn.ModuleList(blocks)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for b in self.res:
            x = b(x, t_emb)
        return x


# --- U-Net denoiser ---
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond_channels: int = 0,
                 base_channels: int = 64,
                 depth: int = 4,
                 num_res_blocks: int = 1,
                 time_dim: int = 128,
                 norm: Optional[str] = None,
                 act: Optional[str] = None):
        super().__init__()

        self.time_mlp = TimeEmbedding(time_dim)

        total_in = in_channels + cond_channels

        # learned fusion for latent + condition
        if cond_channels > 0:
            self.fusion = nn.Sequential(
                            nn.Conv2d(total_in, base_channels, 1),
                            nn.SiLU(),
                            nn.Conv2d(base_channels, in_channels, 1)
                         )
        else:
            self.fusion = nn.Identity()

        # encoder
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        feats = []
        for i in range(depth):
            ch = base_channels * (2 ** i)
            self.downs.append(DownBlock(prev_ch, ch, time_dim, num_res_blocks, norm, act))
            feats.append(ch)
            prev_ch = ch

        # bottleneck
        self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_dim, norm, act)

        # decoder
        self.ups = nn.ModuleList()
        for ch in reversed(feats):
            self.ups.append(UpBlock(prev_ch, ch, time_dim, num_res_blocks, norm, act))
            prev_ch = ch

        # final conv
        self.final = nn.Conv2d(prev_ch, out_channels, 1)

    def forward(self, x_t, t, cond=None):
        if cond is not None:
            x_t = torch.cat([x_t, cond], dim=1)
        x_t = self.fusion(x_t)

        t_emb = self.time_mlp(t)  # [B, time_dim]

        skips = []
        for down in self.downs:
            x_t, skip = down(x_t, t_emb)
            skips.append(skip)

        x_t = self.bottleneck(x_t, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)

        return self.final(x_t)

    def print_summary(self):
        print("\n=== UNet Summary ===")
        print(f"in_channels     : {self.downs[0].res[0].conv1.in_channels}")
        print(f"out_channels    : {self.final.out_channels}")
        print(f"base_channels   : {self.downs[0].res[0].conv1.out_channels}")
        print(f"depth           : {len(self.downs)}")
        print(f"time_dim        : {self.time_mlp.fc1.in_features}")
        print(f"num_res_blocks  : {len(self.downs[0].res)}")
        print(f"norm            : {type(self.downs[0].res[0].norm1).__name__}")
        print(f"act             : {type(self.downs[0].res[0].act).__name__}")
        print("encoder channels:",
              [b.res[0].conv1.out_channels for b in self.downs])
        print("decoder channels:",
              [b.res[0].conv1.out_channels for b in self.ups])
        print("====================\n")

    def to_config(self):
        """Return YAML-compatible config matching from_config() schema."""
        cfg = {
            "unet": {
                "in_channels": self.downs[0].res[0].conv1.in_channels,
                "cond_channels": 0
                    if isinstance(self.fusion, torch.nn.Identity)
                    else self.fusion[0].in_channels - self.downs[0].res[0].conv1.in_channels,
                "out_channels": self.final.out_channels,
                "base_channels": self.downs[0].res[0].conv1.out_channels,
                "depth": len(self.downs),
                "num_res_blocks": len(self.downs[0].res),
                "time_dim": self.time_mlp.fc1.in_features,
                "norm": "batch" if isinstance(self.downs[0].res[0].norm1, torch.nn.BatchNorm2d)
                        else "group" if isinstance(self.downs[0].res[0].norm1, torch.nn.GroupNorm)
                        else None,
                "act": "relu" if isinstance(self.downs[0].res[0].act, torch.nn.ReLU)
                        else "leakyrelu" if isinstance(self.downs[0].res[0].act, torch.nn.LeakyReLU)
                        else "silu",
            }
        }
        return cfg

    @classmethod
    def from_shape(cls,
                   in_channels: int,
                   out_channels: int,
                   base_channels: int,
                   depth: int,
                   num_res_blocks: int = 2,
                   time_dim: int = 256,
                   cond_channels: int = 0,
                   norm: str = "batch",
                   act: str = "relu"):
        """
        Construct a UNet directly from shape parameters.
        Mirrors 'from_config' but does not require a YAML config.
        """
        cfg = {
            "unet": {
                "in_channels": in_channels,
                "cond_channels": cond_channels,
                "out_channels": out_channels,
                "base_channels": base_channels,
                "depth": depth,
                "num_res_blocks": num_res_blocks,
                "time_dim": time_dim,
                "norm": norm,
                "act": act,
            }
        }
        return cls.from_config(cfg)

    @classmethod
    def from_config(cls, cfg: dict | str, latent_channels: Optional[int] = None, latent_base: Optional[int] = None):
        """
        Build U-Net from config file or dict.
        Handles both flat and nested configs ({'unet': {...}}).
        """
        import yaml, math

        # Load YAML if path
        if isinstance(cfg, str):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

        # unwrap if nested
        if "unet" in cfg:
            cfg = cfg["unet"]

        # Override input/output channels if given
        if latent_channels is not None:
            cfg["in_channels"] = latent_channels
            cfg["out_channels"] = latent_channels

        # Validate depth vs. latent base
        if latent_base is not None:
            max_depth = int(math.log2(latent_base))
            if cfg.get("depth", 0) > max_depth:
                raise ValueError(
                    f"Depth {cfg['depth']} too large for latent_base {latent_base}. "
                    f"Max allowed = {max_depth}"
                )

        return cls(**cfg)

