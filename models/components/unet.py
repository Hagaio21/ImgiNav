import torch
import torch.nn as nn
from .base_component import BaseComponent
from .blocks import TimeEmbedding, DownBlock, UpBlock, ResidualBlock

class ConditionFusion(BaseComponent):
    def _build(self):
        mode = self._init_kwargs.get("mode", "none")
        x_ch = self._init_kwargs.get("x_channels")
        cond_ch = self._init_kwargs.get("cond_channels", 0)

        self.mode = mode
        if mode == "concat":
            self.op = nn.Conv2d(x_ch + cond_ch, x_ch, 1)
        elif mode == "add":
            self.op = nn.Identity()
        elif mode == "film":
            self.gamma = nn.Conv2d(cond_ch, x_ch, 1)
            self.beta = nn.Conv2d(cond_ch, x_ch, 1)
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

class DualUNet(BaseComponent):
    def _build(self):
        in_ch = self._init_kwargs.get("in_channels", 3)
        out_ch = self._init_kwargs.get("out_channels", 3)
        cond_ch = self._init_kwargs.get("cond_channels", 0)
        base_ch = self._init_kwargs.get("base_channels", 64)
        depth = self._init_kwargs.get("depth", 4)
        num_res_blocks = self._init_kwargs.get("num_res_blocks", 1)
        time_dim = self._init_kwargs.get("time_dim", 128)
        fusion_mode = self._init_kwargs.get("fusion_mode", "none")
        cond_mult = self._init_kwargs.get("cond_mult", 1.0)
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        dropout = self._init_kwargs.get("dropout", 0.0)

        self.time_mlp = TimeEmbedding(time_dim)
        self.use_cond = cond_ch > 0

        self.downs = nn.ModuleList()
        self.cond_downs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        prev_ch = in_ch
        cond_prev_ch = int(cond_ch * cond_mult) if cond_ch > 0 else 0
        feats = []

        for i in range(depth):
            ch = base_ch * (2 ** i)
            self.downs.append(DownBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout))
            if self.use_cond:
                cond_c = int(base_ch * cond_mult * (2 ** i))
                self.cond_downs.append(DownBlock(cond_prev_ch, cond_c, time_dim, num_res_blocks, norm_groups, dropout))
                fusion_cfg = {
                    "mode": fusion_mode,
                    "x_channels": ch,
                    "cond_channels": cond_c,
                }
                self.fusions.append(ConditionFusion(**fusion_cfg))
                cond_prev_ch = cond_c
            prev_ch = ch
            feats.append(ch)

        self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_dim, norm_groups, dropout)
        self.bottleneck_fusion = ConditionFusion(
            mode=fusion_mode,
            x_channels=prev_ch,
            cond_channels=cond_prev_ch if self.use_cond else 0,
        )

        self.ups = nn.ModuleList()
        for ch in reversed(feats):
            self.ups.append(UpBlock(prev_ch, ch, time_dim, num_res_blocks, norm_groups, dropout))
            prev_ch = ch

        self.final = nn.Conv2d(prev_ch, out_ch, 1)

    def forward(self, x_t, t, cond=None):
        t_emb = self.time_mlp(t.float())
        skips = []
        cond_feats = cond if self.use_cond and cond is not None else None

        for i, down in enumerate(self.downs):
            x_t, skip = down(x_t, t_emb)
            skips.append(skip)
            if self.use_cond and cond is not None:
                cond_feats, _ = self.cond_downs[i](cond_feats, t_emb)
                x_t = self.fusions[i](x_t, cond_feats)

        x_t = self.bottleneck(x_t, t_emb)
        if self.use_cond and cond is not None:
            x_t = self.bottleneck_fusion(x_t, cond_feats)

        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)

        return self.final(x_t)
    
    def freeze_blocks(self, block_names):

        if isinstance(block_names, str):
            block_names = [block_names]
        
        for name in block_names:
            if name == "downs":
                for block in self.downs:
                    for p in block.parameters():
                        p.requires_grad = False
            elif name == "ups":
                for block in self.ups:
                    for p in block.parameters():
                        p.requires_grad = False
            elif name == "bottleneck":
                for p in self.bottleneck.parameters():
                    p.requires_grad = False
            elif name == "time_mlp":
                for p in self.time_mlp.parameters():
                    p.requires_grad = False
            elif name == "final":
                for p in self.final.parameters():
                    p.requires_grad = False
            else:
                raise ValueError(f"Unknown block name: {name}")
    
    def freeze_downblocks(self):
        """Freeze all downsampling blocks (for ControlNet attachment)."""
        self.freeze_blocks(["downs"])
    
    def freeze_upblocks(self):
        """Freeze all upsampling blocks."""
        self.freeze_blocks(["ups"])
    
    def get_skip_connections(self, x_t, t):
        """
        Forward pass that returns skip connections for ControlNet attachment.
        
        Returns:
            tuple: (output, skips) where skips is a list of skip connection tensors
        """
        t_emb = self.time_mlp(t.float())
        skips = []
        cond_feats = None  # ControlNet doesn't use cond_downs
        
        for i, down in enumerate(self.downs):
            x_t, skip = down(x_t, t_emb)
            skips.append(skip)
        
        x_t = self.bottleneck(x_t, t_emb)
        
        for up, skip in zip(self.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)
        
        return self.final(x_t), skips

    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "in_channels": self._init_kwargs.get("in_channels", 3),
            "out_channels": self._init_kwargs.get("out_channels", 3),
            "cond_channels": self._init_kwargs.get("cond_channels", 0),
            "base_channels": self._init_kwargs.get("base_channels", 64),
            "depth": self._init_kwargs.get("depth", 4),
            "num_res_blocks": self._init_kwargs.get("num_res_blocks", 1),
            "time_dim": self._init_kwargs.get("time_dim", 128),
            "fusion_mode": self._init_kwargs.get("fusion_mode", "none"),
            "cond_mult": self._init_kwargs.get("cond_mult", 1.0),
            "norm_groups": self._init_kwargs.get("norm_groups", 8),
            "dropout": self._init_kwargs.get("dropout", 0.0),
        })
        return cfg
