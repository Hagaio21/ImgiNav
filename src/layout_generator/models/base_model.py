# src/layout_generator/models/base_model.py
import torch.nn as nn
import json

class BaseModel(nn.Module):
    """
    A base class for all models, enforcing initialization via a config dict.
    This config is stored as a serializable 'info' attribute for reproducibility.
    """
    def __init__(self, config: dict):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("Model must be initialized with a config dictionary.")
        self.info = config

    def get_info_as_json(self, indent=4):
        """Returns the model's architecture info as a formatted JSON string."""
        return json.dumps(self.info, indent=indent)