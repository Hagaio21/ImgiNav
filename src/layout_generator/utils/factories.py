import json
from models.autoencoder import Decoder,Encoder
from models.diffusion import DiffusionModel
from models.unet import UNet
from modules.scheduler import NoiseScheduler
from utils.logger import Logger


class AutoEncoderFactory:

    def __init__(self, name="autoencoder", channel_list=None,
                 in_channels=3, out_channels=3, base_channels=64,
                 latent_channels=4, depth=3, channel_multiplier=2,
                 logger=None):
        self.name = name
        self.channel_list = channel_list
        self.logger = logger or Logger(source=self.__class__.__name__)
        self.config = {
            "name": name,
            "channel_list": channel_list,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "latent_channels": latent_channels,
            "depth": depth,
            "channel_multiplier": channel_multiplier
        }

        self.logger.info(f"AutoEncoderFactory initialized for model '{name}'")
        self.logger.debug(f"Config: {json.dumps(self.config, indent=4)}")

    def build(self):
        self.logger.info("Building Encoder and Decoder models...")
        encoder = Encoder(
            channel_list=self.channel_list,
            in_channels=self.config["in_channels"],
            base_channels=self.config["base_channels"],
            latent_channels=self.config["latent_channels"],
            depth=self.config["depth"],
            channel_multiplier=self.config["channel_multiplier"],
            name=self.name
        )
        decoder = Decoder(
            channel_list=self.channel_list[::-1] if self.channel_list else None,
            out_channels=self.config["out_channels"],
            base_channels=self.config["base_channels"],
            latent_channels=self.config["latent_channels"],
            depth=self.config["depth"],
            channel_multiplier=self.config["channel_multiplier"],
            name=self.name
        )
        self.logger.info("✓ Encoder and Decoder built successfully")
        self.logger.debug(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
        self.logger.debug(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
        return encoder, decoder


class DiffusionFactory:
    def __init__(self,
                 name="diffusion_v1",
                 in_channels=3,
                 out_channels=3,
                 base_channels=64,
                 depth=4,
                 cond_dim=128,
                 time_dim=128,
                 group_norm_groups=8, # Refactored
                 time_embedding_scale=10000.0, # Refactored
                 schedule_type="linear",
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 logger=None):
        self.name = name
        self.logger = logger or Logger(source=self.__class__.__name__)
        self.config = {
            "name": name,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "depth": depth,
            "cond_dim": cond_dim,
            "time_dim": time_dim,
            "group_norm_groups": group_norm_groups, # Refactored
            "time_embedding_scale": time_embedding_scale, # Refactored
            "schedule_type": schedule_type,
            "timesteps": timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end
        }
        self.logger.info(f"DiffusionFactory initialized for model '{name}'")
        self.logger.debug(f"Config: {json.dumps(self.config, indent=4)}")


    def build(self, device='cuda') -> DiffusionModel:
        self.logger.info("Building Diffusion model components...")
        unet = UNet(
            in_channels=self.config["in_channels"],
            out_channels=self.config["out_channels"],
            base_channels=self.config["base_channels"],
            depth=self.config["depth"],
            time_dim=self.config["time_dim"],
            cond_dim=self.config["cond_dim"],
            group_norm_groups=self.config["group_norm_groups"], # Refactored
            time_embedding_scale=self.config["time_embedding_scale"], # Refactored
            name=self.name + "_unet"
        ).to(device)
        self.logger.debug(f"UNet params: {sum(p.numel() for p in unet.parameters()):,}")

        scheduler = NoiseScheduler(
            num_timesteps=self.config["timesteps"],
            beta_start=self.config["beta_start"],
            beta_end=self.config["beta_end"],
            schedule=self.config["schedule_type"],
            name=self.name + "_scheduler"
        ).to(device)
        self.logger.debug(f"Scheduler: {self.config['schedule_type']} with {self.config['timesteps']} timesteps")

        model = DiffusionModel(unet=unet, scheduler=scheduler, name=self.name).to(device)
        self.logger.info("✓ DiffusionModel built successfully")
        return model

    def info(self):
        return {
            "factory": self.__class__.__name__,
            "name": self.name,
            "config": self.config
        }

