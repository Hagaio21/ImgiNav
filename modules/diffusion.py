import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
import yaml

from scheduler import NoiseScheduler, LinearScheduler
from unet import UNet


class LatentDiffusion(nn.Module):
    """
    Latent Diffusion Model
    
    Diffusion model that operates on latent embeddings.
    Handles noise scheduling and denoising only.
    """
    
    def __init__(self, unet: UNet, scheduler: NoiseScheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
    
    def forward(self, latents: torch.Tensor, t: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None):
        """
        Training forward pass: add noise to latents, predict the noise
        
        Args:
            latents: Clean latent embeddings [B, C, H, W]
            t: Timesteps [B]. If None, samples random timesteps
            noise: Noise to add [B, C, H, W]. If None, samples random noise
        
        Returns:
            noise_pred: Predicted noise [B, C, H, W]
            noise: Actual noise [B, C, H, W]
        """
        B = latents.shape[0]
        device = latents.device
        
        # Random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.scheduler.num_steps, (B,), device=device)
        
        # Random noise if not provided
        if noise is None:
            noise = torch.randn_like(latents)
        
        # Add noise
        noisy_latents, _ = self.scheduler.add_noise(latents, t, noise)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, t, cond=None)
        
        return noise_pred, noise
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        batch_size: int = 1,
        num_steps: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> torch.Tensor:
        """
        Sample latents from noise using DDPM sampling
        
        Args:
            shape: Latent shape (C, H, W)
            batch_size: Number of samples
            num_steps: Denoising steps (default: scheduler's num_steps)
            device: Device
        
        Returns:
            Sampled latents [B, C, H, W]
        """
        self.unet.eval()
        
        # Start from noise
        x_t = torch.randn(batch_size, *shape, device=device)
        
        steps = num_steps or self.scheduler.num_steps
        timesteps = torch.linspace(steps - 1, 0, steps, dtype=torch.long, device=device)
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(x_t, t_batch, cond=None)
            
            # Denoise
            alpha_bar = self.scheduler.alpha_bars[t]
            
            if t > 0:
                alpha_bar_prev = self.scheduler.alpha_bars[t - 1]
                beta = self.scheduler.betas[t]
                
                # Predict x0
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
                
                # Sample next step
                sigma = torch.sqrt(beta)
                noise = torch.randn_like(x_t)
                x_t = (
                    torch.sqrt(alpha_bar_prev) * pred_x0 +
                    torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred +
                    sigma * noise
                )
            else:
                # Final step
                x_t = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        
        self.unet.train()
        return x_t
    
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        unet_checkpoint: Optional[Union[str, Path]] = None
    ):
        """
        Load from YAML config
        
        Config format:
        ```yaml
        scheduler:
          type: LinearScheduler
          num_steps: 1000
        
        unet:
          in_channels: 8
          out_channels: 8
          base_channels: 64
          depth: 4
          num_res_blocks: 2
          time_dim: 128
        ```
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Scheduler
        scheduler_cfg = config['scheduler']
        scheduler_class = globals()[scheduler_cfg['type']]
        scheduler = scheduler_class(num_steps=scheduler_cfg['num_steps'])
        
        # UNet
        unet = UNet(**config['unet'])
        if unet_checkpoint:
            unet.load_state_dict(torch.load(unet_checkpoint, map_location='cpu'))
        
        return cls(unet=unet, scheduler=scheduler)
    
    def save(self, path: Union[str, Path]):
        """Save UNet weights"""
        torch.save(self.unet.state_dict(), path)
    
    def load(self, path: Union[str, Path], device: str = 'cpu'):
        """Load UNet weights"""
        self.unet.load_state_dict(torch.load(path, map_location=device))