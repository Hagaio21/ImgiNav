# This is the central registry for conditioning modules.
CONDITIONING_REGISTRY = {}

def register_conditioning(name):
    """Decorator to add a conditioning module class to the registry."""
    def decorator(cls):
        CONDITIONING_REGISTRY[name] = cls
        return cls
    return decorator

# --- NEW: Import all conditioning modules here ---
# This ensures their @register_conditioning decorators run.
from . import vector_conditioning
from . import image_conditioning
from . import text_conditioning
# Add any new conditioning modules you create here in the future.