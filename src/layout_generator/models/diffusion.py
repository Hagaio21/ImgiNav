import torch
import torch.nn as nn
import tqdm

class DiffusionModel(nn.Module):
    def __init__(self, unet: nn.Module, scheduler: nn.Module, name='diffusion_model'):
        super().__init__()
        self.name = name
        self.unet = unet
        self.scheduler = scheduler

        self.config = {
            "name": name,
            "unet": self.unet.config,
            "scheduler": self.scheduler.config
        }

    def forward(self, x_0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        Training forward pass: generate x_t and target noise.
        Returns: predicted noise, true noise, x_t
        """
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.add_noise(x_0, noise, t)
        pred_noise = self.predict_noise(x_t, t, cond)
        return pred_noise, noise, x_t

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Calls U-Net to predict noise from x_t, timestep, and condition.
        """
        return self.unet(x_t, t, cond)

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Utility: forward diffusion step
        """
        return self.scheduler.add_noise(x_0, noise, t)

    @torch.no_grad()
    def sample(self, shape: tuple, cond: torch.Tensor, device='cuda'):
        """
        Run the full reverse process starting from noise.
        shape: (B, C, H, W)
        cond: [B, cond_dim]
        """
        x_t = torch.randn(shape, device=device)

        num_timesteps = self.scheduler.config["num_timesteps"]
        for t in tqdm(reversed(range(num_timesteps)), desc="Sampling", total=num_timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(x_t, t_batch, cond)
            x_t = self.scheduler.reverse_step(x_t, t_batch, pred_noise)

        return x_t.clamp(-1, 1)

    def info(self):
        return {
            "name": self.name,
            "model": self.__class__.__name__,
            "config": self.config,
            "unet_params": sum(p.numel() for p in self.unet.parameters()),
            "scheduler_type": self.scheduler.config["schedule"]
        }
    
    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))

