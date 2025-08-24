CONDITIONING_REGISTRY = {}

def register_conditioning(name):
    """Decorator to add a conditioning module class to the registry."""
    def decorator(cls):
        CONDITIONING_REGISTRY[name] = cls
        return cls
    return decorator