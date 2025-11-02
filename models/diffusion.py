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

        # Load decoder from checkpoint (cannot train decoder, only load from checkpoint)
        # Accept either autoencoder checkpoint or decoder checkpoint
        if ae_cfg:
            # Autoencoder checkpoint provided
            ae_checkpoint = ae_cfg.get("checkpoint")
            if not ae_checkpoint:
                raise ValueError("'autoencoder' config must provide a 'checkpoint' path (diffusion cannot train autoencoder)")
            # Load autoencoder from checkpoint and extract decoder
            autoencoder = Autoencoder.load_checkpoint(ae_checkpoint, map_location="cpu")
            self.decoder = autoencoder.decoder
            self.encoder = None  # Not needed for pre-embedded latents
            self.autoencoder = None
            self._has_encoder = False
            # Freeze decoder if requested
            if ae_cfg.get("frozen", False):
                self.decoder.freeze()
        elif decoder_cfg:
            # Decoder checkpoint provided (should be autoencoder checkpoint that contains decoder)
            decoder_checkpoint = decoder_cfg.get("checkpoint")
            if not decoder_checkpoint:
                raise ValueError("'decoder' config must provide a 'checkpoint' path (diffusion cannot train decoder)")
            # Load autoencoder from checkpoint and extract decoder
            # Note: decoder checkpoint should actually be an autoencoder checkpoint
            autoencoder = Autoencoder.load_checkpoint(decoder_checkpoint, map_location="cpu")
            self.decoder = autoencoder.decoder
            self.encoder = None
            self.autoencoder = None
            self._has_encoder = False
            # Freeze decoder if requested
            if decoder_cfg.get("frozen", False):
                self.decoder.freeze()
        else:
            raise ValueError("DiffusionModel requires either 'autoencoder' or 'decoder' config with 'checkpoint' path")

        # UNet backbone
        self.unet = DualUNet.from_config(unet_cfg)
        # Freeze UNet if requested (or use special ControlNet freezing)
        if unet_cfg.get("frozen", False):
            self.unet.freeze()
        # Support ControlNet-style freezing via config
        if unet_cfg.get("freeze_downblocks", False):
            self.unet.freeze_downblocks()
        if unet_cfg.get("freeze_upblocks", False):
            self.unet.freeze_upblocks()
        freeze_blocks = unet_cfg.get("freeze_blocks", None)
        if freeze_blocks:
            self.unet.freeze_blocks(freeze_blocks)

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
            encoder_out = self.encoder(x0_or_latents)  # Returns dict
            # Extract latent z from dict
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out and "logvar" in encoder_out:
                # VAE mode: use mu as deterministic representation for diffusion
                # (diffusion will add its own noise)
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'/'logvar'. Got: {list(encoder_out.keys())}")
            
            if noise is None:
                noise = torch.randn_like(latents)
            elif noise.shape != latents.shape:
                # If noise is in image space, encode it to latent space
                if noise.shape == x0_or_latents.shape:
                    noise_out = self.encoder(noise)
                    # Extract latent from encoder output
                    if "latent" in noise_out:
                        noise = noise_out["latent"]
                    elif "mu" in noise_out:
                        noise = noise_out["mu"]
                    else:
                        noise = torch.randn_like(latents)
                else:
                    noise = torch.randn_like(latents)
        else:
            # Input is already latents (decoder-only mode)
            # Can be dict or tensor - handle both
            if isinstance(x0_or_latents, dict):
                if "latent" in x0_or_latents:
                    latents = x0_or_latents["latent"]
                elif "mu" in x0_or_latents:
                    latents = x0_or_latents["mu"]
                else:
                    raise ValueError(f"Latent dict must contain 'latent' or 'mu'. Got: {list(x0_or_latents.keys())}")
            else:
                latents = x0_or_latents  # Direct tensor
            if noise is None:
                noise = torch.randn_like(latents)

        # add noise
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # predict noise
        pred_noise = self.unet(noisy_latents, t, cond)

        # decode reconstruction - decoder expects dict
        decoded = self.decoder({"latent": noisy_latents})

        return {
            "latent": latents,
            "noisy_latent": noisy_latents,
            "pred_noise": pred_noise,
            "noise": noise,  # Return the noise used (in latent space)
            **decoded,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, batch_size=1, latent_shape=None, cond=None, num_steps=50, 
               method="ddim", eta=0.0, device=None, return_history=False, verbose=False):
 
        if device is None:
            device = next(self.parameters()).device
        
        # Infer latent shape from decoder config if not provided
        if latent_shape is None:
            latent_ch = self.decoder._init_kwargs.get('latent_channels', 4)
            up_steps = self.decoder._init_kwargs.get('upsampling_steps', 4)
            spatial_res = 512 // (2 ** up_steps)
            latent_shape = (latent_ch, spatial_res, spatial_res)
        
        self.scheduler = self.scheduler.to(device)
        latents = torch.randn((batch_size, *latent_shape), device=device)
        
        # Create timestep schedule
        if method == "ddim":
            step_size = self.scheduler.num_steps // num_steps
            timesteps = torch.arange(0, self.scheduler.num_steps, step_size, device=device).long()
        else:
            timesteps = torch.arange(self.scheduler.num_steps - 1, -1, -1, device=device).long()
        
        history = [] if return_history else None
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            if verbose and (i % max(1, len(timesteps) // 10) == 0 or i == len(timesteps) - 1):
                print(f"  Sampling step {i+1}/{len(timesteps)} (t={t.item()})")
            
            t_batch = t.expand(batch_size)
            
            with torch.no_grad():
                pred_noise = self.unet(latents, t_batch, cond)
            
            # DDIM step
            if method == "ddim":
                alpha_bar_t = self.scheduler.alpha_bars[t].view(-1, 1, 1, 1)
                alpha_bar_prev = self.scheduler.alpha_bars[timesteps[i-1]].view(-1, 1, 1, 1) if i > 0 else torch.tensor(1.0, device=device).view(-1, 1, 1, 1)
                
                # Predict x0
                pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                
                # DDIM update (deterministic if eta=0)
                if eta > 0 and i < len(timesteps) - 1:
                    sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
                    noise = sigma * torch.randn_like(latents)
                else:
                    noise = 0
                
                latents = alpha_bar_prev.sqrt() * pred_x0 + (1 - alpha_bar_prev).sqrt() * pred_noise + noise
            
            else:
                # DDPM step
                alpha_t = self.scheduler.alphas[t].view(-1, 1, 1, 1)
                beta_t = self.scheduler.betas[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    alpha_bar_prev = self.scheduler.alpha_bars[timesteps[i+1]].view(-1, 1, 1, 1)
                    pred_mean = (1 / alpha_t.sqrt()) * (latents - beta_t / (1 - self.scheduler.alpha_bars[t]).sqrt().view(-1, 1, 1, 1) * pred_noise)
                    latents = pred_mean + beta_t.sqrt() * torch.randn_like(latents)
            
            if return_history:
                history.append(latents.clone())
        
        # Build output dict
        result = {"latent": latents}
        
        # Decode to RGB if decoder is available
        with torch.no_grad():
            decoded_out = self.decoder({"latent": latents})
            if "rgb" in decoded_out:
                rgb = decoded_out["rgb"]
                # Denormalize from [-1, 1] to [0, 1]
                rgb = (rgb + 1.0) / 2.0
                rgb = torch.clamp(rgb, 0.0, 1.0)
                result["rgb"] = rgb
        
        if return_history:
            result["history"] = history
        
        return result

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
