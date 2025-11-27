import torch
import torch.nn as nn
import yaml
from pathlib import Path


def load_checkpoint_weights(path, map_location="cpu"):
    """
    Helper function to load weights from a checkpoint file.
    Handles both direct state_dict files and checkpoint files with 'state_dict' key.
    
    Args:
        path: Path to checkpoint file
        map_location: Device to load on
    
    Returns:
        State dict (dict of parameter tensors)
    """
    weights = torch.load(path, map_location=map_location)
    if "state_dict" in weights:
        return weights["state_dict"]
    return weights


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
        """
        Helper method to update config with explicit field values from _init_kwargs.
        Useful for subclasses that want to be explicit about which fields to include.
        
        Args:
            cfg: Config dict to update
            *keys: Keys to include from _init_kwargs (with their current values)
            **defaults: Default values for keys not in _init_kwargs
        
        Returns:
            Updated config dict
        
        Example:
            def to_config(self):
                cfg = super().to_config()
                return self._update_config_with_defaults(
                    cfg,
                    "in_channels", "out_channels", "base_channels",
                    in_channels=3, out_channels=3, base_channels=64
                )
        """
        for key in keys:
            if key in defaults:
                cfg[key] = self._init_kwargs.get(key, defaults[key])
            else:
                cfg[key] = self._init_kwargs.get(key)
        
        # Also add any defaults that weren't in keys
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
    def save_checkpoint(self, path, include_config=True, **extra_state):
        """
        Save model checkpoint. Override in subclasses for extended checkpointing.
        
        Args:
            path: Path to save checkpoint
            include_config: Whether to include model config
            **extra_state: Additional state to save (e.g., optimizer, step, epoch, etc.)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": self.state_dict()}
        if include_config:
            payload["config"] = self.to_config()
        payload.update(extra_state)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu"):
        """Load model checkpoint. Override in subclasses for extended checkpointing."""
        payload = torch.load(path, map_location=map_location)
        config = payload.get("config")
        model = cls.from_config(config) if config else cls()
        # Use strict=False for backward compatibility
        model.load_state_dict(payload["state_dict"], strict=False)
        return model

    # -----------------------
    # Dependency loading
    # -----------------------
    def load_subcomponent(self, key, cls_map):
        """
        Load a subcomponent from config using a registry.
        
        Args:
            key: Key in _init_kwargs for the subcomponent config
            cls_map: Registry dict mapping type names to classes
        
        Returns:
            Instance of the registered class
        """
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
        """
        Load a subcomponent from config using a registry with automatic type detection.
        
        Args:
            key: Key in _init_kwargs for the subcomponent config
            registry: Optional specific registry to use (default: uses unified COMPONENT_REGISTRY)
            default_type: Default type if not specified in config
        
        Returns:
            Instance of the registered class
        """
        # Lazy import to avoid circular dependency
        from .registry import create_component
        
        sub_cfg = self._init_kwargs.get(key)
        if sub_cfg is None:
            return None
        
        if not isinstance(sub_cfg, dict):
            # Already an instance, return as-is
            return sub_cfg
        
        # Ensure type is set
        if "type" not in sub_cfg:
            if default_type:
                sub_cfg = {**sub_cfg, "type": default_type}
            else:
                raise ValueError(f"Subcomponent config for '{key}' must have 'type' field or provide default_type")
        
        # Use unified create_component function
        return create_component(sub_cfg, registry=registry)
    
    def create_component(self, key, default_type=None):
        """
        Simplified method to create a component from config.
        
        This is a convenience wrapper around load_subcomponent_from_registry
        that uses the unified component registry.
        
        Args:
            key: Key in _init_kwargs for the component config
            default_type: Default type if not specified in config
        
        Returns:
            Instance of the registered class, or None if config not found
        
        Examples:
            >>> # In _build method:
            >>> self.clip_projections = self.create_component("clip_projection", default_type="CLIPProjections")
            >>> self.scheduler = self.create_component("scheduler", default_type="CosineScheduler")
        """
        return self.load_subcomponent_from_registry(key, registry=None, default_type=default_type)
    
    def create_component_from_config(self, cfg, default_type=None):
        """
        Create a component directly from a config dict (not from _init_kwargs).
        
        Useful for creating multiple components of the same type (e.g., multiple heads).
        
        Args:
            cfg: Config dict with "type" field
            default_type: Default type if not specified in config
        
        Returns:
            Instance of the registered class
        
        Examples:
            >>> # In _build method, creating multiple heads:
            >>> for head_cfg in head_configs:
            >>>     head = self.create_component_from_config(head_cfg, default_type="DecoderHead")
        """
        # Lazy import to avoid circular dependency
        from .registry import create_component
        
        if not isinstance(cfg, dict):
            return cfg  # Already an instance
        
        if "type" not in cfg and default_type:
            cfg = {**cfg, "type": default_type}
        
        return create_component(cfg)

    # -----------------------
    # Freezing and training control
    # -----------------------
    def freeze(self, modules=None):
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
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_groups(self):
        """
        Automatically create optimizer parameter groups.

        - Reads keys like '<submodule>_lr' from self._init_kwargs
        - Returns a list of {params, lr} dicts for optimizer creation.
        """
        groups = []
        
        # Track parameters that belong to child modules
        child_params = set()
        
        # First, collect parameters from child modules
        for name, module in self.named_children():
            trainable = [p for p in module.parameters() if p.requires_grad]
            if not trainable:
                continue
            # Track these parameters
            child_params.update(trainable)
            lr_key = f"{name}_lr"
            lr_value = self._init_kwargs.get(lr_key, None)
            group = {"params": trainable}
            if lr_value is not None:
                group["lr"] = lr_value
            groups.append(group)

        # Collect top-level parameters that are NOT in child modules
        top_params = [p for p in self.parameters() if p.requires_grad and p not in child_params]
        if top_params:
            base_lr = self._init_kwargs.get("lr", None)
            group = {"params": top_params}
            if base_lr is not None:
                group["lr"] = base_lr
            groups.append(group)

        return groups
    
    # -----------------------
    # Shape information
    # -----------------------
    def get_input_shape(self, batch_size=1):
        """
        Get expected input shape for this component.
        
        Args:
            batch_size: Batch size for shape inference (default: 1)
        
        Returns:
            Input shape tuple or dict of input shapes, or None if not determinable
        """
        # Override in subclasses to provide shape information
        return None
    
    def get_output_shape(self, batch_size=1):
        """
        Get expected output shape for this component.
        
        Args:
            batch_size: Batch size for shape inference (default: 1)
        
        Returns:
            Output shape tuple or dict of output shapes, or None if not determinable
        """
        # Override in subclasses to provide shape information
        return None
    
    def get_shape_info(self, batch_size=1):
        """
        Get comprehensive shape information for this component.
        
        Args:
            batch_size: Batch size for shape inference (default: 1)
        
        Returns:
            Dict with 'input' and 'output' keys, each containing shape information
        """
        return {
            "input": self.get_input_shape(batch_size),
            "output": self.get_output_shape(batch_size)
        }
    
    def infer_shapes_from_forward(self, *sample_inputs, batch_size=1):
        """
        Infer input/output shapes by running a forward pass with sample inputs.
        
        Args:
            *sample_inputs: Sample input tensors or dicts
            batch_size: Batch size (if inputs need to be created)
        
        Returns:
            Dict with 'input' and 'output' keys containing inferred shapes
        """
        self.eval()
        with torch.no_grad():
            try:
                # Record input shapes
                input_shapes = {}
                if len(sample_inputs) == 1 and isinstance(sample_inputs[0], dict):
                    input_shapes = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) 
                                   for k, v in sample_inputs[0].items()}
                else:
                    input_shapes = {f"arg_{i}": tuple(arg.shape) if isinstance(arg, torch.Tensor) else str(type(arg))
                                   for i, arg in enumerate(sample_inputs)}
                
                # Run forward pass
                output = self.forward(*sample_inputs)
                
                # Record output shapes
                if isinstance(output, dict):
                    output_shapes = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else str(type(v))
                                     for k, v in output.items()}
                elif isinstance(output, torch.Tensor):
                    output_shapes = tuple(output.shape)
                else:
                    output_shapes = str(type(output))
                
                return {
                    "input": input_shapes,
                    "output": output_shapes
                }
            except Exception as e:
                return {
                    "input": input_shapes if 'input_shapes' in locals() else None,
                    "output": None,
                    "error": str(e)
                }