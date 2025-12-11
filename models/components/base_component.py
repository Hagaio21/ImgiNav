"""
Base component class for all model modules.

Consolidation notes:
- Uses centralized checkpoint loading from models.utils
- Removed inline gzip/pickle logic (now in utils.load_checkpoint_payload)
- Simplified some methods
- Removed DataFlow in favor of plain dict communication
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path


class BaseComponent(nn.Module):
    """
    Configurable base for all modules.
    Handles config I/O, checkpointing, dependency loading, freezing, and param grouping.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._init_kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._build()

    def _build(self):
        """Override in subclasses to build model components."""
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # -----------------------
    # Config I/O
    # -----------------------
    def to_config(self):
        cfg = {"type": self.__class__.__name__}
        cfg.update({
            k: v for k, v in self._init_kwargs.items()
            if isinstance(v, (int, float, str, bool, list, dict, type(None)))
        })
        return cfg
    
    def _update_config_with_defaults(self, cfg, *keys, **defaults):
        """Helper method to update config with explicit field values."""
        for key in keys:
            if key in defaults:
                cfg[key] = self._init_kwargs.get(key, defaults[key])
            else:
                cfg[key] = self._init_kwargs.get(key)
        
        for key, default_value in defaults.items():
            if key not in keys:
                cfg[key] = self._init_kwargs.get(key, default_value)
        
        return cfg

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.get("model", cfg)
        cfg = {k: v for k, v in cfg.items() if k != "type"}
        return cls(**cfg)

    def save_config(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_config(), f)

    @classmethod
    def load_config(cls, path):
        with open(path, "r") as f:
            return cls.from_config(yaml.safe_load(f))

    # -----------------------
    # Checkpoint handling
    # -----------------------
    def save_checkpoint(self, path, include_config=True, use_compression=False, **extra_state):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            include_config: Whether to include model config
            use_compression: If True, save with gzip compression
            **extra_state: Additional state to save
        """
        from ..utils import save_checkpoint_payload
        
        payload = {"state_dict": self.state_dict()}
        if include_config:
            payload["config"] = self.to_config()
        payload.update(extra_state)
        
        save_checkpoint_payload(payload, path, use_compression=use_compression)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu"):
        """Load model checkpoint."""
        from ..utils import load_checkpoint_payload
        
        payload = load_checkpoint_payload(path, map_location)
        config = payload.get("config")
        model = cls.from_config(config) if config else cls()
        model.load_state_dict(payload["state_dict"], strict=False)
        return model

    # -----------------------
    # Dependency loading
    # -----------------------
    def load_subcomponent(self, key, cls_map):
        """Load a subcomponent from config using a registry."""
        from ..utils import load_checkpoint_weights
        
        sub_cfg = self._init_kwargs.get(key)
        if sub_cfg is None:
            raise ValueError(f"Missing subcomponent config for '{key}'")

        sub_type = sub_cfg.get("type")
        if sub_type not in cls_map:
            raise ValueError(f"Unknown subcomponent type '{sub_type}' for '{key}'. Available: {list(cls_map.keys())}")

        sub_cls = cls_map[sub_type]
        instance = sub_cls.from_config(sub_cfg)

        ckpt_path = sub_cfg.get("checkpoint")
        if ckpt_path:
            weights = load_checkpoint_weights(ckpt_path, map_location="cpu")
            instance.load_state_dict(weights, strict=False)

        if sub_cfg.get("frozen", False):
            instance.freeze()

        return instance
    
    def load_subcomponent_from_registry(self, key, registry=None, default_type=None):
        """Load a subcomponent from config using unified component registry."""
        from .registry import create_component
        
        sub_cfg = self._init_kwargs.get(key)
        if sub_cfg is None:
            return None
        
        if not isinstance(sub_cfg, dict):
            return sub_cfg  # Already an instance
        
        if "type" not in sub_cfg:
            if default_type:
                sub_cfg = {**sub_cfg, "type": default_type}
            else:
                raise ValueError(f"Subcomponent config for '{key}' must have 'type' field or provide default_type")
        
        return create_component(sub_cfg, registry=registry)
    
    def create_component(self, key, default_type=None):
        """Simplified method to create a component from config."""
        return self.load_subcomponent_from_registry(key, registry=None, default_type=default_type)
    
    def create_component_from_config(self, cfg, default_type=None):
        """Create a component directly from a config dict."""
        from .registry import create_component
        
        if not isinstance(cfg, dict):
            return cfg
        
        if "type" not in cfg and default_type:
            cfg = {**cfg, "type": default_type}
        
        return create_component(cfg)

    # -----------------------
    # Freezing and training control
    # -----------------------
    def freeze(self, modules=None):
        """Freeze all or specific modules."""
        if modules is None:
            for p in self.parameters():
                p.requires_grad = False
            return
        if isinstance(modules, str):
            modules = [modules]
        for name, module in self.named_children():
            if name in modules:
                for p in module.parameters():
                    p.requires_grad = False

    def set_trainable(self, modules=None):
        """Make all or specific modules trainable."""
        if modules is None:
            for p in self.parameters():
                p.requires_grad = True
            return
        if isinstance(modules, str):
            modules = [modules]
        for name, module in self.named_children():
            if name in modules:
                for p in module.parameters():
                    p.requires_grad = True

    # -----------------------
    # Parameter utilities
    # -----------------------
    def trainable_parameters(self):
        """Iterate over trainable parameters."""
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_groups(self):
        """
        Automatically create optimizer parameter groups.
        Reads keys like '<submodule>_lr' from self._init_kwargs.
        """
        groups = []
        child_params = set()
        
        for name, module in self.named_children():
            trainable = [p for p in module.parameters() if p.requires_grad]
            if not trainable:
                continue
            child_params.update(trainable)
            lr_key = f"{name}_lr"
            lr_value = self._init_kwargs.get(lr_key)
            group = {"params": trainable}
            if lr_value is not None:
                group["lr"] = lr_value
            groups.append(group)

        top_params = [p for p in self.parameters() if p.requires_grad and p not in child_params]
        if top_params:
            base_lr = self._init_kwargs.get("lr")
            group = {"params": top_params}
            if base_lr is not None:
                group["lr"] = base_lr
            groups.append(group)

        return groups
    
    # -----------------------
    # Shape information
    # -----------------------
    def get_input_shape(self, batch_size=1):
        """Get expected input shape for this component."""
        return None
    
    def get_output_shape(self, batch_size=1):
        """Get expected output shape for this component."""
        return None
    
    def get_shape_info(self, batch_size=1):
        """Get comprehensive shape information."""
        return {
            "input": self.get_input_shape(batch_size),
            "output": self.get_output_shape(batch_size)
        }
    
    def infer_shapes_from_forward(self, *sample_inputs, batch_size=1):
        """Infer input/output shapes by running a forward pass."""
        self.eval()
        with torch.no_grad():
            try:
                input_shapes = {}
                if len(sample_inputs) == 1 and isinstance(sample_inputs[0], dict):
                    input_shapes = {
                        k: tuple(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) 
                        for k, v in sample_inputs[0].items()
                    }
                else:
                    input_shapes = {
                        f"arg_{i}": tuple(arg.shape) if isinstance(arg, torch.Tensor) else str(type(arg))
                        for i, arg in enumerate(sample_inputs)
                    }
                
                output = self.forward(*sample_inputs)
                
                if isinstance(output, dict):
                    output_shapes = {
                        k: tuple(v.shape) if isinstance(v, torch.Tensor) else str(type(v))
                        for k, v in output.items()
                    }
                elif isinstance(output, torch.Tensor):
                    output_shapes = tuple(output.shape)
                else:
                    output_shapes = str(type(output))
                
                return {"input": input_shapes, "output": output_shapes}
            except Exception as e:
                return {
                    "input": input_shapes if 'input_shapes' in locals() else None,
                    "output": None,
                    "error": str(e)
                }
