import torch
import torch.nn as nn
from pathlib import Path
import yaml

from models.components.base_component import BaseComponent
from models.autoencoder import Autoencoder
from models.components.unet import DualUNet
from models.components.scheduler import SCHEDULER_REGISTRY


class DiffusionModel(BaseComponent):
    """Minimal end-to-end diffusion model for training."""

    def _build(self):
        ae_cfg = self._init_kwargs.get("autoencoder", {})
        unet_cfg = self._init_kwargs.get("unet", {})
        sched_cfg = self._init_kwargs.get("scheduler", {})

        # autoencoder
        self.autoencoder = Autoencoder.from_config(ae_cfg)

        # UNet backbone
        self.unet = DualUNet.from_config(unet_cfg)

        # noise scheduler
        sched_type = sched_cfg.get("type", "CosineScheduler")
        if sched_type not in SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown scheduler: {sched_type}")
        self.scheduler = SCHEDULER_REGISTRY[sched_type].from_config(sched_cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x0, t, cond=None, noise=None):
        """Forward diffusion training step."""
        if noise is None:
            noise = torch.randn_like(x0)

        # encode image
        latents = self.autoencoder.encoder(x0)

        # add noise
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # predict noise
        pred_noise = self.unet(noisy_latents, t, cond)

        # decode reconstruction
        decoded = self.autoencoder.decoder(noisy_latents)

        return {
            "latent": latents,
            "noisy_latent": noisy_latents,
            "pred_noise": pred_noise,
            **decoded,
        }

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------
    @classmethod
    def load_config(cls, cfg_path):
        cfg_path = Path(cfg_path)
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls.from_config(cfg)

    def to_config(self):
        return {
            "type": "DiffusionModel",
            "autoencoder": self.autoencoder.to_config(),
            "unet": self.unet.to_config(),
            "scheduler": self.scheduler.to_config(),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path):
        path = Path(path)
        ckpt = {
            "config": self.to_config(),
            "state_dict": self.state_dict(),
        }
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model = cls.from_config(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
