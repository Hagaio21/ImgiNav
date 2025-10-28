import torch
import torch.nn as nn
from blocks import TimeEmbedding, DownBlock, UpBlock, ResidualBlock

class ConditionFusion(nn.Module):
    """Implements F(x, cond) with selectable mode."""
    def __init__(self, mode: str, x_channels: int, cond_channels: int):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.op = nn.Conv2d(x_channels + cond_channels, x_channels, 1)
        elif mode == "add":
            self.op = nn.Identity()
        elif mode == "film":
            self.gamma = nn.Conv2d(cond_channels, x_channels, 1)
            self.beta = nn.Conv2d(cond_channels, x_channels, 1)
        elif mode in ("none", None):
            self.op = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, x, cond):
        if cond is None or self.mode in ("none", None):
            return x
        if self.mode == "concat":
            return self.op(torch.cat([x, cond], dim=1))
        elif self.mode == "add":
            return x + cond
        elif self.mode == "film":
            return self.gamma(cond) * x + self.beta(cond)
        return x

class DualUNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond_channels: int = 0,
                 base_channels: int = 64,
                 depth: int = 4,
                 num_res_blocks: int = 1,
                 time_dim: int = 128,
                 fusion_mode: str = "none",
                 cond_mult: float = 1.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.base_channels = base_channels
        self.depth = depth
        self.num_res_blocks = num_res_blocks
        self.time_dim = time_dim
        self.cond_mult = cond_mult
        self.time_mlp = TimeEmbedding(time_dim)
        self.use_cond = cond_channels > 0
        self.fusion_mode = fusion_mode

        # encoder
        self.downs = nn.ModuleList()
        self.cond_downs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        prev_ch = in_channels
        cond_prev_ch = int(cond_channels * cond_mult) if cond_channels > 0 else 0
        feats = []

        for i in range(depth):
            ch = base_channels * (2 ** i)
            self.downs.append(DownBlock(prev_ch, ch, time_dim, num_res_blocks))
            if self.use_cond:
                cond_ch = int(base_channels * cond_mult * (2 ** i))
                self.cond_downs.append(DownBlock(cond_prev_ch, cond_ch, time_dim, num_res_blocks))
                self.fusions.append(ConditionFusion(fusion_mode, ch, cond_ch))
                cond_prev_ch = cond_ch
            prev_ch = ch
            feats.append(ch)

        # bottleneck
        self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_dim)
        self.bottleneck_fusion = ConditionFusion(fusion_mode, prev_ch, cond_prev_ch if self.use_cond else 0)

        # decoder
        self.ups = nn.ModuleList()
        for ch in reversed(feats):
            self.ups.append(UpBlock(prev_ch, ch, time_dim, num_res_blocks))
            prev_ch = ch

        self.final = nn.Conv2d(prev_ch, out_channels, 1)

    def forward(self, x_t, t, cond=None):
        t_emb = self.time_mlp(t)
        skips = []

        # Down path
        if self.use_cond and cond is not None:
            cond_feats = cond
        else:
            cond_feats = None

        for i, down in enumerate(self.downs):
            x_t, skip = down(x_t, t_emb)
            skips.append(skip)
            if self.use_cond and cond is not None:
                cond_feats, _ = self.cond_downs[i](cond_feats, t_emb)
                x_t = self.fusions[i](x_t, cond_feats)

        # Bottleneck
        x_t = self.bottleneck(x_t, t_emb)
        if self.use_cond and cond is not None:
            x_t = self.bottleneck_fusion(x_t, cond_feats)

        # Up path
        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)

        return self.final(x_t)

    def to_config(self):
        """Return a dictionary describing this DualUNet architecture."""
        return {
            "model": {
                "type": "dual_unet",
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "cond_channels": self.cond_channels,
                "base_channels": self.base_channels,
                "depth": self.depth,
                "num_res_blocks": self.num_res_blocks,
                "time_dim": self.time_dim,
                "fusion_mode": self.fusion_mode,
                "cond_mult": self.cond_mult,
                "act": getattr(self, "act", None),
                "norm": getattr(self, "norm", None),
            }
        }

    @classmethod
    def from_config(cls, cfg: dict):
        """Instantiate a DualUNet from a saved configuration dictionary."""
        model_cfg = cfg.get("model", cfg)

        return cls(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            cond_channels=model_cfg.get("cond_channels", 0),
            base_channels=model_cfg["base_channels"],
            depth=model_cfg["depth"],
            num_res_blocks=model_cfg["num_res_blocks"],
            time_dim=model_cfg["time_dim"],
            fusion_mode=model_cfg.get("fusion_mode", "none"),
            cond_mult=model_cfg.get("cond_mult", 1.0),
        )


    def print_summary(self):
        print("\n=== DualUNet Summary ===")
        print(f"base_channels   : {self.downs[0].res_blocks[0].conv1.out_channels}")
        print(f"depth           : {len(self.downs)}")
        print(f"time_dim        : {self.time_mlp.fc1.in_features}")
        print(f"fusion_mode     : {self.fusion_mode}")
        print(f"use_cond        : {self.use_cond}")
        print("========================\n")
