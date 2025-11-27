"""
Unified component registry system.

This module provides a general component registry that automatically discovers
and creates components from all available registries (projections, losses, 
schedulers, heads, etc.).

Usage:
    from models.components.registry import create_component
    
    # Automatically finds component from any registry
    component = create_component({"type": "TextProjection", "text_dim": 384})
    
    # Or use in BaseComponent
    self.clip_projections = self.create_component("clip_projection")
"""

from typing import Dict, Type, Optional, Any
import warnings

# Import all registries
from .projections import PROJECTION_REGISTRY
from .scheduler import SCHEDULER_REGISTRY
from .heads import HEAD_REGISTRY
from ..losses.base_loss import LOSS_REGISTRY

# Import model components to register
from ..encoder import Encoder
from ..decoder import Decoder
from .unet import UnetWithAttention

# Master component registry - combines all registries
COMPONENT_REGISTRY: Dict[str, Type] = {}

# Track which registry each component belongs to (for better error messages)
_COMPONENT_SOURCES: Dict[str, str] = {}


def _register_components_from_registry(registry, source_name):
    """Helper to register components from a registry with source tracking."""
    global COMPONENT_REGISTRY, _COMPONENT_SOURCES
    for name, cls in registry.items():
        if name in COMPONENT_REGISTRY:
            warnings.warn(
                f"Component '{name}' already registered. "
                f"Overwriting {_COMPONENT_SOURCES.get(name, 'unknown')} with {source_name}."
            )
        COMPONENT_REGISTRY[name] = cls
        _COMPONENT_SOURCES[name] = source_name

def _register_all_components():
    """Register all components from all registries into the master registry."""
    global COMPONENT_REGISTRY, _COMPONENT_SOURCES
    
    # Register from all sub-registries
    _register_components_from_registry(PROJECTION_REGISTRY, "projection")
    _register_components_from_registry(SCHEDULER_REGISTRY, "scheduler")
    _register_components_from_registry(HEAD_REGISTRY, "head")
    _register_components_from_registry(LOSS_REGISTRY, "loss")
    
    # Register model components (lazy import to avoid circular dependencies)
    from ..encoder import Encoder, VAEEncoder
    from ..decoder import Decoder, VAEDecoder
    from ..autoencoder import Autoencoder, VAE
    from .unet import UnetWithAttention
    
    model_components = {
        "Encoder": Encoder,
        "VAEEncoder": VAEEncoder,
        "Decoder": Decoder,
        "VAEDecoder": VAEDecoder,
        "Autoencoder": Autoencoder,
        "VAE": VAE,
        "UnetWithAttention": UnetWithAttention,
        "UNet": UnetWithAttention,  # Alias for backward compatibility
    }
    
    for name, cls in model_components.items():
        if name in COMPONENT_REGISTRY:
            warnings.warn(
                f"Component '{name}' already registered. "
                f"Overwriting {_COMPONENT_SOURCES.get(name, 'unknown')} with model component."
            )
        COMPONENT_REGISTRY[name] = cls
        _COMPONENT_SOURCES[name] = "model"


# Initialize the master registry
_register_all_components()


def create_component(cfg: Dict[str, Any], registry: Optional[Dict[str, Type]] = None) -> Any:

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict, got {type(cfg)}")
    
    comp_type = cfg.get("type")
    if comp_type is None:
        raise ValueError("Config must have 'type' field")
    
    # Use provided registry or search in master registry
    if registry is not None:
        if comp_type not in registry:
            raise ValueError(
                f"Unknown type '{comp_type}' in provided registry. "
                f"Available types: {list(registry.keys())}"
            )
        comp_cls = registry[comp_type]
    else:
        # Search in master registry
        if comp_type not in COMPONENT_REGISTRY:
            # Provide helpful error message
            source = _COMPONENT_SOURCES.get(comp_type, "unknown")
            available = list(COMPONENT_REGISTRY.keys())
            similar = [k for k in available if comp_type.lower() in k.lower() or k.lower() in comp_type.lower()]
            
            error_msg = (
                f"Unknown component type '{comp_type}'. "
                f"Available types: {available[:10]}"
            )
            if len(available) > 10:
                error_msg += f" (and {len(available) - 10} more)"
            
            if similar:
                error_msg += f"\nDid you mean one of: {similar[:5]}?"
            
            raise ValueError(error_msg)
        
        comp_cls = COMPONENT_REGISTRY[comp_type]
    
    # Create instance
    comp_cfg = {k: v for k, v in cfg.items() if k != "type"}
    instance = comp_cls(**comp_cfg)
    
    # Load checkpoint if specified
    ckpt_path = cfg.get("checkpoint")
    if ckpt_path:
        from .base_component import load_checkpoint_weights
        weights = load_checkpoint_weights(ckpt_path, map_location="cpu")
        instance.load_state_dict(weights, strict=False)
    
    # Freeze if requested
    if cfg.get("frozen", False):
        instance.freeze()
    
    return instance


def register_component(name: str, cls: Type, source: str = "custom"):
    """
    Manually register a component in the master registry.
    
    Args:
        name: Component name
        cls: Component class
        source: Source category (for tracking)
    """
    COMPONENT_REGISTRY[name] = cls
    _COMPONENT_SOURCES[name] = source


def get_component_type(component_name: str) -> Optional[str]:
    """Get the source registry type for a component."""
    return _COMPONENT_SOURCES.get(component_name)


def list_components(category: Optional[str] = None) -> Dict[str, str]:
    """
    List all registered components, optionally filtered by category.
    
    Args:
        category: Optional category filter ("projection", "scheduler", "head", "loss")
    
    Returns:
        Dict mapping component names to their source categories
    """
    if category:
        return {name: src for name, src in _COMPONENT_SOURCES.items() if src == category}
    return _COMPONENT_SOURCES.copy()

