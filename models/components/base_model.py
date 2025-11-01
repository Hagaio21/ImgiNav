"""
BaseModel - Subclass of BaseComponent for trainable full models.
Provides extended checkpointing capabilities for training state.
"""
import torch
from pathlib import Path
from .base_component import BaseComponent


class BaseModel(BaseComponent):
    """
    Base class for full trainable models (e.g., Autoencoder, DiffusionModel).
    
    Extends BaseComponent with model-specific functionality.
    All checkpoint methods from BaseComponent are inherited and work as-is.
    Subclasses can override for extended checkpointing (optimizer, epoch, etc.).
    """
    
    def save_checkpoint(self, path, include_config=True, **extra_state):
        """
        Save model checkpoint with optional extra training state.
        
        Args:
            path: Path to save checkpoint
            include_config: Whether to include model config
            **extra_state: Additional state to save (e.g., optimizer, epoch, etc.)
        """
        path = Path(path)
        payload = {"state_dict": self.state_dict()}
        if include_config:
            payload["config"] = self.to_config()
        payload.update(extra_state)
        torch.save(payload, path)
    
    @classmethod
    def load_checkpoint(cls, path, map_location="cpu", return_extra=False):
        """
        Load model checkpoint, optionally returning extra state.
        
        Args:
            path: Path to checkpoint
            map_location: Device to load on
            return_extra: If True, return tuple (model, extra_state_dict)
            
        Returns:
            If return_extra=False: just the model (backward compatible)
            If return_extra=True: (model, extra_state_dict) tuple where 
                extra_state_dict contains any additional state (optimizer, epoch, etc.)
        """
        path = Path(path)
        payload = torch.load(path, map_location=map_location)
        
        config = payload.get("config")
        model = cls.from_config(config) if config else cls()
        model.load_state_dict(payload["state_dict"])
        
        if return_extra:
            # Return model and any extra state (optimizer, epoch, etc.)
            extra_state = {k: v for k, v in payload.items() 
                          if k not in ["state_dict", "config"]}
            return model, extra_state
        
        return model

