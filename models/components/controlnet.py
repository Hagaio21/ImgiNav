# models/components/controlnet.py
import torch
import torch.nn as nn
from .base_component import BaseComponent
from .unet import Unet
from .fusion import FUSION_REGISTRY


class ControlNet(BaseComponent):
    """Frozen UNet with injected control features."""
    def _build(self):
        base_unet_cfg = self._init_kwargs.get("base_unet")
        adapter_cfg = self._init_kwargs.get("adapter")
        fusion_cfg = self._init_kwargs.get("fusion", {})
        
        if base_unet_cfg is None or adapter_cfg is None:
            raise ValueError("ControlNet requires both 'base_unet' and 'adapter' configs")
        
        self.base_unet = Unet.from_config(base_unet_cfg)
        self.base_unet.freeze_downblocks()
        
        # Build adapter (supports multiple types)
        adapter_type = adapter_cfg.get("type", "SimpleAdapter")
        if adapter_type == "ControlAdapter":
            adapter_type = "SimpleAdapter"  # Backward compatibility
        
        from .control_adapter import BaseControlAdapter, SimpleAdapter, MLPAdapter, DeepAdapter
        
        ADAPTER_REGISTRY = {
            "SimpleAdapter": SimpleAdapter,
            "MLPAdapter": MLPAdapter,
            "DeepAdapter": DeepAdapter,
        }
        
        if adapter_type not in ADAPTER_REGISTRY:
            raise ValueError(f"Unknown adapter type: {adapter_type}. Choose from {list(ADAPTER_REGISTRY.keys())}")
        
        adapter_cls = ADAPTER_REGISTRY[adapter_type]
        adapter_config = {k: v for k, v in adapter_cfg.items() if k != "type"}
        self.adapter = adapter_cls.from_config(adapter_config)
        
        # Build fusion layers (one per UNet level)
        # Get channel counts from UNet config
        base_channels = base_unet_cfg.get("base_channels", 64)
        depth = base_unet_cfg.get("depth", 4)
        
        # Backward compatibility: support old fuse_mode parameter
        if "fuse_mode" in self._init_kwargs and "type" not in fusion_cfg:
            fuse_mode = self._init_kwargs["fuse_mode"]
            fusion_type = fuse_mode  # Use old fuse_mode as fusion type
        else:
            fusion_type = fusion_cfg.get("type", "add")
        
        if fusion_type not in FUSION_REGISTRY:
            raise ValueError(f"Unknown fusion type: {fusion_type}. Choose from {list(FUSION_REGISTRY.keys())}")
        
        fusion_cls = FUSION_REGISTRY[fusion_type]
        
        # Create fusion layer for each UNet level
        fusion_config = {k: v for k, v in fusion_cfg.items() if k != "type"}
        self.fusion_layers = nn.ModuleList([
            fusion_cls(channels=base_channels * (2 ** i), **fusion_config)
            for i in range(depth)
        ])

    def forward(self, x_t, t, text_emb, pov_emb):
        ctrl_feats = self.adapter(text_emb, pov_emb)
        t_emb = self.base_unet.time_mlp(t.float())
        skips = []
        
        for i, down in enumerate(self.base_unet.downs):
            x_t, skip = down(x_t, t_emb)
            if i < len(ctrl_feats) and i < len(self.fusion_layers):
                # Use fusion layer to combine skip and control features
                skip = self.fusion_layers[i](skip, ctrl_feats[i])
            skips.append(skip)
        
        x_t = self.base_unet.bottleneck(x_t, t_emb)
        for up, skip in zip(self.base_unet.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)
        return self.base_unet.final(x_t)
