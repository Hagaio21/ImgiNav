import torch
import torch.nn as nn
import yaml


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
    def save_checkpoint(self, path, include_config=True):
        """Save model checkpoint. Override in subclasses for extended checkpointing."""
        payload = {"state_dict": self.state_dict()}
        if include_config:
            payload["config"] = self.to_config()
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu"):
        """Load model checkpoint. Override in subclasses for extended checkpointing."""
        payload = torch.load(path, map_location=map_location)
        config = payload.get("config")
        model = cls.from_config(config) if config else cls()
        model.load_state_dict(payload["state_dict"])
        return model

    # -----------------------
    # Dependency loading
    # -----------------------
    def load_subcomponent(self, key, cls_map):
        sub_cfg = self._init_kwargs.get(key)
        if sub_cfg is None:
            raise ValueError(f"Missing subcomponent config for '{key}'")

        sub_type = sub_cfg.get("type")
        if sub_type not in cls_map:
            raise ValueError(f"Unknown subcomponent type '{sub_type}' for '{key}'")

        sub_cls = cls_map[sub_type]
        instance = sub_cls.from_config(sub_cfg)

        ckpt_path = sub_cfg.get("checkpoint")
        if ckpt_path:
            weights = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in weights:
                weights = weights["state_dict"]
            instance.load_state_dict(weights)

        if sub_cfg.get("frozen", False):
            instance.freeze()

        return instance

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
