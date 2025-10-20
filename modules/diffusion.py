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

    def to_config(self, save_path: Union[str, Path]):
        ckpt_dir = Path(save_path).parent
        config = {
            "latent": {
                "channels": self.latent_shape[0],
                "base": self.latent_shape[1],
            },
            "scheduler": {
                "type": type(self.scheduler).__name__,
                "num_steps": self.scheduler.num_steps,
            },
            "autoencoder": {
                "config": str((ckpt_dir / "autoencoder_config.yaml").as_posix()),
                "checkpoint": str((ckpt_dir / "ae_latest.pt").as_posix()),
            },
            "unet": {
                "config": str((ckpt_dir / "unet_config.yaml").as_posix()),
                "checkpoint": str((ckpt_dir / "unet_latest.pt").as_posix()),
            },
        }
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        print(f"[Config] LatentDiffusion saved → {save_path}")

