import os
import json
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from torchvision.utils import save_image
import re
from .base_model import BaseModel

import torch.nn as nn
import torch
from .base_model import BaseModel
from . import register_model

@register_model("Encoder")
class Encoder(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        
        layers = []
        channel_list = self.info['architecture'] # Read explicit channel list
        
        for i in range(len(channel_list) - 2):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i+1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            
        # Final layer to latent space
        layers.append(nn.Conv2d(channel_list[-2], channel_list[-1], kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

@register_model("Decoder")
class Decoder(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        
        layers = []
        channel_list = self.info['architecture'] # Read explicit channel list

        # First layer from latent space
        layers.append(nn.ConvTranspose2d(channel_list[0], channel_list[1], kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())

        # Intermediate upsampling layers
        for i in range(1, len(channel_list) - 2):
            layers.append(nn.ConvTranspose2d(channel_list[i], channel_list[i + 1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())

        # Final layer to output image
        layers.append(nn.ConvTranspose2d(channel_list[-2], channel_list[-1], kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)