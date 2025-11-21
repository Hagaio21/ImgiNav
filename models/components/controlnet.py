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
        
        # Verify init_scale is working for ScaledAddFusion
        if fusion_type == "scaled_add" and "init_scale" in fusion_config:
            expected_init_scale = fusion_config["init_scale"]
            print(f"[ControlNet Init] Verifying init_scale={expected_init_scale} for ScaledAddFusion layers:")
            for i, fusion_layer in enumerate(self.fusion_layers):
                # Try multiple ways to access the scale parameter
                actual_init_scale = None
                if hasattr(fusion_layer, 'scale'):
                    actual_init_scale = fusion_layer.scale.mean().item()
                elif 'scale' in fusion_layer.state_dict():
                    actual_init_scale = fusion_layer.state_dict()['scale'].mean().item()
                else:
                    # Try to get it from named_parameters
                    for name, param in fusion_layer.named_parameters():
                        if name == 'scale':
                            actual_init_scale = param.mean().item()
                            break
                
                if actual_init_scale is not None:
                    if abs(actual_init_scale - expected_init_scale) > 1e-6:
                        print(f"  WARNING: Level {i} scale={actual_init_scale:.6f}, expected={expected_init_scale:.6f}")
                    else:
                        print(f"  ✓ Level {i} scale={actual_init_scale:.6f} (correct)")
                else:
                    print(f"  ERROR: Level {i} fusion layer does not have 'scale' attribute! Type: {type(fusion_layer)}")
                    # Print all attributes for debugging
                    print(f"    Available attributes: {[attr for attr in dir(fusion_layer) if not attr.startswith('_')]}")
                    print(f"    State dict keys: {list(fusion_layer.state_dict().keys())}")

    def forward(self, x_t, t, text_emb, pov_emb):
        ctrl_feats = self.adapter(text_emb, pov_emb)
        t_emb = self.base_unet.time_mlp(t.float())
        skips = []
        
        # Debug: log control feature magnitudes (only in training, periodically)
        debug_control_features = getattr(self, '_debug_control_features', False)
        if debug_control_features and self.training and torch.rand(1).item() < 0.01:  # Log 1% of batches
            for i, ctrl_feat in enumerate(ctrl_feats):
                ctrl_mag = ctrl_feat.abs().mean().item()
                print(f"[ControlNet Debug] Level {i}: ctrl_feat_magnitude={ctrl_mag:.6f}")
        
        for i, down in enumerate(self.base_unet.downs):
            x_t, skip = down(x_t, t_emb)
            if i < len(ctrl_feats) and i < len(self.fusion_layers):
                # Expand control features to match skip connection spatial dimensions
                # Control features are [B, ch, 1, 1] (from global context), need to expand to [B, ch, H, W]
                ctrl_feat = ctrl_feats[i]
                if ctrl_feat.shape[2] == 1 and ctrl_feat.shape[3] == 1:
                    # Expand 1x1 features to match skip spatial dimensions using bilinear interpolation
                    skip_h, skip_w = skip.shape[2], skip.shape[3]
                    if skip_h > 1 or skip_w > 1:
                        ctrl_feat = torch.nn.functional.interpolate(
                            ctrl_feat, size=(skip_h, skip_w), mode='bilinear', align_corners=False
                        )
                
                # Debug: log skip vs control feature magnitudes
                if debug_control_features and self.training and torch.rand(1).item() < 0.01:
                    skip_mag_val = skip.abs().mean().item()
                    ctrl_mag_val = ctrl_feat.abs().mean().item()
                    ratio = ctrl_mag_val / (skip_mag_val + 1e-8)
                    # Get fusion scale if it's ScaledAddFusion
                    fusion_scale_str = "N/A"
                    effective_ctrl_mag = ctrl_mag_val
                    fusion_layer = self.fusion_layers[i]
                    try:
                        # Check if it's ScaledAddFusion by class name or by checking for scale attribute
                        if hasattr(fusion_layer, 'scale'):
                            fusion_scale = fusion_layer.scale.mean().item()
                            fusion_scale_str = f"{fusion_scale:.6f}"
                            effective_ctrl_mag = ctrl_mag_val * fusion_scale
                        elif 'ScaledAddFusion' in str(type(fusion_layer)):
                            # Try to access scale via getattr or state_dict
                            if 'scale' in fusion_layer.state_dict():
                                fusion_scale = fusion_layer.state_dict()['scale'].mean().item()
                                fusion_scale_str = f"{fusion_scale:.6f}"
                                effective_ctrl_mag = ctrl_mag_val * fusion_scale
                    except Exception as e:
                        # Silently fail - fusion_scale_str will remain "N/A"
                        pass
                    print(f"[ControlNet Debug] Level {i}: skip_mag={skip_mag_val:.6f}, ctrl_mag={ctrl_mag_val:.6f}, ratio={ratio:.6f}, fusion_scale={fusion_scale_str}, effective_ctrl_mag={effective_ctrl_mag:.6f}, skip_shape={skip.shape}, ctrl_shape={ctrl_feat.shape}")
                
                # Use fusion layer to combine skip and control features
                # The ScaledAddFusion layer has a learnable scale parameter (initialized to 0.0 for Zero Convolution)
                # that will learn the appropriate magnitude naturally without hard normalization
                skip = self.fusion_layers[i](skip, ctrl_feat)
            skips.append(skip)
        
        x_t = self.base_unet.bottleneck(x_t, t_emb)
        for up, skip in zip(self.base_unet.ups, reversed(skips)):
            x_t = up(x_t, skip, t_emb)
        return self.base_unet.final(x_t)
