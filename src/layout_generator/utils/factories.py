import yaml
import torch
import torch.nn as nn
from ..models import MODEL_REGISTRY
from ..models.base_model import BaseModel
from ..models.diffusion import DiffusionModel
from ..modules.scheduler import NoiseScheduler

def create_conditioning_module(config: dict, unet_params: dict) -> nn.Module:
    """
    A helper factory that builds and assembles a complete conditioning module.
    It handles dependencies like encoders by calling the main create_model factory.
    """
    # By importing the registry here, locally, the circular dependency is broken.
    from ..modules.conditioning import CONDITIONING_REGISTRY

    name = config['name']
    params = config.get('params', {})
    module_class = CONDITIONING_REGISTRY.get(name)
    if module_class is None:
        raise ValueError(f"Unknown conditioning module name: '{name}'. Is it registered?")

    # --- Assembly Logic for Different Conditioning Types ---

    if name == "VectorAdditive":
        # Inject the UNet's channel architecture into the module's config
        # so it can pre-build the necessary projection layers.
        params['unet_channel_sizes'] = unet_params['architecture']['encoder']
        return module_class(config=params)

    elif name in ["ImageCrossAttention", "TextCrossAttention"]:
        # 1. Build the dependency (the encoder model) first
        encoder_config_path = params.pop('encoder_config_path')
        # We can safely call the main factory recursively here.
        encoder = create_model(encoder_config_path, device='cpu')
        
        # 2. Load the encoder's checkpoint if one is specified
        checkpoint = params.pop('encoder_checkpoint', None)
        if checkpoint:
            encoder.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        
        # 3. Inject the pre-built encoder into the conditioning module's constructor
        if name == 'ImageCrossAttention':
            return module_class(config=params, vision_encoder=encoder)
        else: # TextCrossAttention
            return module_class(config=params, text_encoder=encoder)
    
    else:
        # For any other simple conditioning types that have no dependencies
        return module_class(config=params)
    
def create_model(config_input, device='cuda') -> BaseModel:
    if isinstance(config_input, str):
        with open(config_input, 'r') as f:
            config = yaml.safe_load(f)
    elif isinstance(config_input, dict):
        config = config_input
    else:
        raise TypeError(f"config_input must be a path string or a dictionary, not {type(config_input)}")

    model_name = config['model']['name']
    model_params = config['model']['params']

    if model_name == "UNet":
        unet_class = MODEL_REGISTRY.get(model_name)
        if unet_class is None: raise ValueError("UNet model not found in registry.")
        conditioning_config = model_params['conditioning']
        # Pass the main unet_params to the helper function
        conditioning_module = create_conditioning_module(conditioning_config, model_params)
        
        # Pop conditioning after its params have been used
        model_params.pop('conditioning')
        
        return unet_class(config=model_params, conditioning_module=conditioning_module).to(device)
    else:
        model_class = MODEL_REGISTRY.get(model_name)
        if model_class is None:
            raise ValueError(f"Unknown model name: '{model_name}'. Ensure it is registered.")
        return model_class(config=model_params).to(device)