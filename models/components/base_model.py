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
    def load_checkpoint(cls, path, map_location="cpu", return_extra=False, config=None, strict=True):
        """
        Load model checkpoint, optionally returning extra state.
        
        Args:
            path: Path to checkpoint
            map_location: Device to load on
            return_extra: If True, return tuple (model, extra_state_dict)
            config: Optional config dict to use instead of saved config (useful when resuming training)
            strict: If True, require exact match. If False, filter out mismatched keys (default: True)
            
        Returns:
            If return_extra=False: just the model (backward compatible)
            If return_extra=True: (model, extra_state_dict) tuple where 
                extra_state_dict contains any additional state (optimizer, epoch, etc.)
        """
        path = Path(path)
        payload = torch.load(path, map_location=map_location)
        
        # Use provided config if available, otherwise use saved config
        model_config = config if config is not None else payload.get("config")
        model = cls.from_config(model_config) if model_config else cls()
        
        # Load state dict
        state_dict = payload["state_dict"]
        if strict:
            model.load_state_dict(state_dict)
        else:
            # Filter state dict to only include keys that match in shape
            # Also skip CLIP projection keys if they don't match (they're not needed for encoding)
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            clip_proj_keys = []
            
            for key, value in state_dict.items():
                # Skip CLIP projection keys if they don't match (optional for encoding)
                is_clip_proj = key.startswith("clip_projections.")
                
                if key in model_state_dict:
                    # Check if shapes match
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        if is_clip_proj:
                            clip_proj_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
                        else:
                            skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
                else:
                    # Key doesn't exist in model
                    if is_clip_proj:
                        clip_proj_keys.append(f"{key} (not in model)")
                    else:
                        skipped_keys.append(f"{key} (not in model)")
            
            if clip_proj_keys:
                # CLIP projection mismatches are expected when loading for encoding (not needed)
                import warnings
                warnings.warn(
                    f"Skipping {len(clip_proj_keys)} CLIP projection keys (not needed for encoding). "
                    f"This is normal when loading spatial CLIP VAE checkpoints for embedding."
                )
            
            if skipped_keys:
                import warnings
                warnings.warn(
                    f"Skipping {len(skipped_keys)} keys when loading checkpoint (strict=False). "
                    f"First few: {skipped_keys[:5]}"
                )
            
            # Load filtered state dict
            model.load_state_dict(filtered_state_dict, strict=False)
        
        if return_extra:
            # Return model and any extra state (optimizer, epoch, etc.)
            extra_state = {k: v for k, v in payload.items() 
                          if k not in ["state_dict", "config"]}
            return model, extra_state
        
        return model

