# models/components/controlnet.py
import torch
from .base_component import BaseComponent
from .unet import Unet


class ControlNet(BaseComponent):
    """Frozen UNet with injected control features."""
    def _build(self):
        base_unet_cfg = self._init_kwargs.get("base_unet")
        adapter_cfg = self._init_kwargs.get("adapter")

        if base_unet_cfg is None or adapter_cfg is None:
            raise ValueError("ControlNet requires both 'base_unet' and 'adapter' configs")

        self.base_unet = Unet.from_config(base_unet_cfg)
        self.base_unet.freeze_downblocks()

        from .control_adapter import ControlAdapter
        self.adapter = ControlAdapter.from_config(adapter_cfg)

        self.fuse_mode = self._init_kwargs.get("fuse_mode", "add")

    def forward(self, x_t, t, text_emb, pov_emb):
        ctrl_feats = self.adapter(text_emb, pov_emb)
        t_emb = self.base_unet.time_mlp(t.float())
        skips = []

        for i, down in enumerate(self.base_unet.downs):
            x_t, skip = down(x_t, t_emb)
            if i < len(ctrl_feats):
                skip = skip + ctrl_feats[i] if self.fuse_mode == "add" else torch.cat([skip, ctrl_feats[i]], dim=1)
            skips.append(skip)

        x_t = self.base_unet.bottleneck(x_t, t_emb)
        for up, skip in zip(self.base_unet.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)
        return self.base_unet.final(x_t)
