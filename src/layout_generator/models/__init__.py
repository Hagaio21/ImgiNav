# This is the central registry that will store a mapping from
# model names (e.g., "UNet") to their actual Python classes.
MODEL_REGISTRY = {}

def register_model(name):
    """
    A decorator function that allows models to register themselves.
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

# --- NEW: Import all model modules here ---
# This is the crucial part. By importing them here, we ensure that their
# @register_model decorators run and populate the MODEL_REGISTRY.
from . import autoencoder
from . import unet
from . import bert_encoder
# Add any new models you create here in the future