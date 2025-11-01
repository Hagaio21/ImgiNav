import torch
import torch.nn as nn
from pathlib import Path
import yaml

from models.components.base_model import BaseModel
from models.autoencoder import Autoencoder
from models.decoder import Decoder
from models.components.unet import DualUNet
from models.components.scheduler import SCHEDULER_REGISTRY


class DiffusionModel(BaseModel):
    """Minimal end-to-end diffusion model for training."""

    def _build(self):
        ae_cfg = self._init_kwargs.get("autoencoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)
        unet_cfg = self._init_kwargs.get("unet", {})
        sched_cfg = self._init_kwargs.get("scheduler", {})

        # Accept either full autoencoder or just decoder
        if ae_cfg:
            # Full autoencoder provided
            self.autoencoder = Autoencoder.from_config(ae_cfg)
            self.decoder = self.autoencoder.decoder
            self.encoder = self.autoencoder.encoder
            self._has_encoder = True
        elif decoder_cfg:
            # Only decoder provided (for inference or pre-encoded training)
            self.decoder = Decoder.from_config(decoder_cfg)
            self.encoder = None
            self.autoencoder = None
            self._has_encoder = False
        else:
            raise ValueError("DiffusionModel requires either 'autoencoder' or 'decoder' config")

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
    def forward(self, x0_or_latents, t, cond=None, noise=None):
        """
        Forward diffusion training step.
        
        Args:
            x0_or_latents: Either images (if encoder available) or latents (if decoder-only)
            t: Timestep tensor
            cond: Optional conditioning
            noise: Optional noise tensor
        """
        # Encode image to latents if encoder is available, otherwise assume input is already latents
        if self._has_encoder:
            # x0 is images, encode them
            latents = self.encoder(x0_or_latents)
            if noise is None:
                noise = torch.randn_like(latents)
            elif noise.shape != latents.shape:
                # If noise is in image space, encode it to latent space
                if noise.shape == x0_or_latents.shape:
                    noise = self.encoder(noise)
                else:
                    noise = torch.randn_like(latents)
        else:
            # Input is already latents (decoder-only mode)
            latents = x0_or_latents
            if noise is None:
                noise = torch.randn_like(latents)

        # add noise
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # predict noise
        pred_noise = self.unet(noisy_latents, t, cond)

        # decode reconstruction
        decoded = self.decoder(noisy_latents)

        return {
            "latent": latents,
            "noisy_latent": noisy_latents,
            "pred_noise": pred_noise,
            "noise": noise,  # Return the noise used (in latent space)
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
        cfg = {
            "type": "DiffusionModel",
            "unet": self.unet.to_config(),
            "scheduler": self.scheduler.to_config(),
        }
        if self._has_encoder:
            cfg["autoencoder"] = self.autoencoder.to_config()
        else:
            cfg["decoder"] = self.decoder.to_config()
        return cfg

    # ------------------------------------------------------------------
    # Checkpointing (inherited from BaseModel, can override if needed)
    # ------------------------------------------------------------------
