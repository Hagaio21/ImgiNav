import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
import yaml

from modules.scheduler import *
from modules.unet import UNet
from modules.autoencoder import AutoEncoder
from tqdm import tqdm

class LatentDiffusion(nn.Module):
    """
    Inference-only Latent Diffusion wrapper.
    Loads scheduler, UNet, and AutoEncoder (decoder only).
    Uses latent size specified in the config.
    """

    def __init__(
        self,
        unet: UNet,
        scheduler: NoiseScheduler,
        autoencoder: Optional[AutoEncoder] = None,
        latent_shape: Optional[tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.autoencoder = autoencoder
        self.latent_shape = latent_shape

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        image: bool = False,
        cond: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_latents: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        guidance_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        start_noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        """
        Generate samples from pure noise in latent space using the same DDPM
        update rule as training.

        Args:
            batch_size: number of samples
            image: if True, decode latents to RGB images
            cond: conditioning tensor [B, C_cond, H, W] or None
            num_steps: optional override for scheduler steps
            device: device string
        Returns:
            Latents [B, C, H, W] or decoded images [B, 3, H*, W*]
        """
        assert self.latent_shape is not None, "latent_shape must be set"

        self.eval()
        self.unet.eval()

        C, H, W = self.latent_shape
        if start_noise is None:
            x_t = torch.randn(batch_size, C, H, W, device=device)
        else:
            x_t = start_noise.to(device)

        steps = num_steps or self.scheduler.num_steps # set number of steps 
        timesteps = torch.linspace(steps - 1, 0, steps, dtype=torch.long, device=device) # set actual steps vector
        if return_latents:
            noise_history = []
            latents_history = []
        print("Generating...")
        for t in tqdm(timesteps, desc="Diffusion sampling", total=len(timesteps)): # for each step in reverse
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if guidance_scale == 1.0 or uncond_cond is None:
                noise_pred = self.unet(x_t, t_batch, cond)
            else:
                noise_pred_cond = self.unet(x_t, t_batch, cond)
                noise_pred_uncond = self.unet(x_t, t_batch, uncond_cond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if return_latents:
                noise_history.append(noise_pred)

            if t > 0:
                # get cooeficcients from scheduler 
                alpha_t = self.scheduler.alphas[t]
                alpha_bar_t = self.scheduler.alpha_bars[t]
                alpha_bar_prev = self.scheduler.alpha_bars[t - 1]
                beta_t = self.scheduler.betas[t]

                # Predict x0 and clamp for stability
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, -3, 3)

                # Compute DDPM mean
                coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
                coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                mean = coef1 * pred_x0 + coef2 * x_t

                # Add variance noise
                noise = torch.randn_like(x_t)
                variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                x_t = mean + torch.sqrt(variance) * noise
                if return_latents:
                    latents_history.append(x_t.clone())
            else:
                alpha_bar_t = self.scheduler.alpha_bars[t]
                x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                if return_latents:
                    latents_history.append(x_t.clone())

        if image:
            assert self.autoencoder is not None, "AutoEncoder required for image decoding"
            print("Decoding...")
            x_t = self.autoencoder.decoder(x_t)

        if return_latents:
            history = {
                "latents": latents_history,
                "noise": noise_history
            }
            return x_t , history
        else:
            return x_t 


    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Build inference model from master YAML config:
        ---
        latent:
          base: 64
          channels: 4

        scheduler:
          type: LinearScheduler
          num_steps: 1000

        autoencoder:
          config: configs/ae.yaml
          checkpoint: checkpoints/ae.pt

        unet:
          config: configs/unet.yaml
          checkpoint: checkpoints/unet.pt
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Scheduler
        sched_cfg = config["scheduler"]
        sched_class = globals()[sched_cfg["type"]]
        scheduler = sched_class(num_steps=sched_cfg["num_steps"]).to(device)

        # Autoencoder
        ae_cfg_path = config["autoencoder"]["config"]
        ae_ckpt_path = config["autoencoder"].get("checkpoint", None)
        autoencoder = AutoEncoder.from_config(ae_cfg_path).to(device)
        if ae_ckpt_path:
            autoencoder.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
        autoencoder.eval()

        # UNet
        unet_cfg_path = config["unet"]["config"]
        unet_ckpt_path = config["unet"].get("checkpoint", None)
        with open(unet_cfg_path, "r", encoding="utf-8") as f:
            unet_cfg = yaml.safe_load(f)
            if "unet" in unet_cfg:
                unet_cfg = unet_cfg["unet"]

        unet = UNet(**unet_cfg).to(device)
        if unet_ckpt_path:
            ckpt = torch.load(unet_ckpt_path, map_location=device)
            # handle both plain and wrapped checkpoints
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "unet_state_dict" in ckpt:
                state = ckpt["unet_state_dict"]
            else:
                state = ckpt
            unet.load_state_dict(state, strict=False)
        unet.eval()


        # Latent shape from config (preferred) or AE config
        if "latent" in config:
            C = config["latent"]["channels"]
            H = W = config["latent"]["base"]
        else:
            with open(ae_cfg_path, "r", encoding="utf-8") as f:
                ae_cfg = yaml.safe_load(f)
            C = ae_cfg["encoder"]["latent_channels"]
            H = W = ae_cfg["encoder"]["latent_base"]

        return cls(unet=unet, scheduler=scheduler, autoencoder=autoencoder, latent_shape=(C, H, W)).to(device)
