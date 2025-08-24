# src/layout_generator/models/__init__.py

# This is the central registry that will store a mapping from
# model names (e.g., "UNet") to their actual Python classes.
MODEL_REGISTRY = {}

def register_model(name):
    """
    A decorator function that allows models to register themselves.

    Usage:
        @register_model("UNet")
        class UNet(BaseModel):
            ...
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator