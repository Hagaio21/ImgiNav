import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear', name='noise_scheduler'):
        super().__init__()
        self.name = name
        self.config = {
            "name": name,
            "num_timesteps": num_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "schedule": schedule
        }

        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise NotImplementedError(f"Schedule '{schedule}' is not implemented.")

        # Register buffers for proper `.to(device)` behavior
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alpha_cumprod', torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer('alpha_cumprod_prev', torch.cat([
            torch.tensor([1.0], dtype=torch.float32), torch.cumprod(1.0 - betas, dim=0)[:-1]
        ]))

        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - self.alpha_cumprod))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alphas = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: x_t = sqrt(ᾱ_t) * x₀ + sqrt(1 - ᾱ_t) * ε
        """
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def scale_noise(self, t: torch.Tensor):
        """
        Get scaling factors for converting model prediction to x₀:
        Returns (1 / sqrt(ᾱ_t), 1 / sqrt(1 - ᾱ_t))
        """
        return (
            1.0 / self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1),
            1.0 / self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        )

    def reverse_step(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion step (DDPM-style sampling)
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)

        pred_x0 = (x_t - noise_pred * self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)) / \
                  self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)

        posterior_mean = (
            beta_t * pred_x0 + (1.0 - beta_t) * x_t
        ) / (1.0 - alpha_t)

        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        variance = self.posterior_variance[t].view(-1, 1, 1, 1)

        return posterior_mean + torch.sqrt(variance) * noise

    def info(self):
        return {
            "name": self.name,
            "model": self.__class__.__name__,
            "config": self.config,
            "timesteps": self.config["num_timesteps"]
        }
