import os
import json
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from torchvision.utils import save_image
import re


class Encoder(nn.Module):
    def __init__(self, channel_list=None, in_channels=3, base_channels=64, latent_channels=4,
                 depth=3, channel_multiplier=2, name="autoencoder"):
        super().__init__()
        self.name = name

        if channel_list:
            self.channel_list = channel_list
        else:
            self.channel_list = [in_channels]
            channels = base_channels
            for _ in range(depth):
                self.channel_list.append(channels)
                channels *= channel_multiplier
            self.channel_list.append(latent_channels)

        self.config = {
            "name": name,
            "channel_list": self.channel_list,
            "mode": "explicit" if channel_list else "parametric",
            "in_channels": in_channels,
            "base_channels": base_channels,
            "latent_channels": latent_channels,
            "depth": depth,
            "channel_multiplier": channel_multiplier
        }

        layers = []
        for in_c, out_c in zip(self.channel_list[:-2], self.channel_list[1:-1]):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())

        # Final layer to latent
        layers.append(nn.Conv2d(self.channel_list[-2], self.channel_list[-1], kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))

    def info(self):
        return {
            "name": self.name,
            "model": self.__class__.__name__,
            "config": self.config,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "layers": str(self.encoder)
        }


class Decoder(nn.Module):
    def __init__(self, channel_list=None, out_channels=3, base_channels=64, latent_channels=4,
                 depth=3, channel_multiplier=2, name="autoencoder"):
        super().__init__()
        self.name = name

        if channel_list:
            self.channel_list = channel_list
        else:
            self.channel_list = [latent_channels]
            channels = base_channels * (channel_multiplier ** (depth - 1))
            for _ in range(depth):
                self.channel_list.append(channels)
                channels //= channel_multiplier
            self.channel_list.append(out_channels)

        self.config = {
            "name": name,
            "channel_list": self.channel_list,
            "mode": "explicit" if channel_list else "parametric",
            "out_channels": out_channels,
            "base_channels": base_channels,
            "latent_channels": latent_channels,
            "depth": depth,
            "channel_multiplier": channel_multiplier
        }

        layers = []

        # First layer
        layers.append(nn.ConvTranspose2d(self.channel_list[0], self.channel_list[1], kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())

        # Intermediate
        for i in range(1, len(self.channel_list) - 2):
            layers.append(nn.ConvTranspose2d(self.channel_list[i], self.channel_list[i + 1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())

        # Final layer
        layers.append(nn.ConvTranspose2d(self.channel_list[-2], self.channel_list[-1], kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))

    def info(self):
        return {
            "name": self.name,
            "model": self.__class__.__name__,
            "config": self.config,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "layers": str(self.decoder)
        }
