import torch
from .base_component import BaseComponent


class NoiseScheduler(BaseComponent):
    def _build(self):
        num_steps = self._init_kwargs.get("num_steps", 1000)
        self.num_steps = num_steps
        alphas, betas = self.build_schedule(num_steps)
        alpha_bars = torch.cumprod(alphas, dim=0)
        # Register as buffers so they're included in state_dict
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def build_schedule(self, num_steps: int):
        raise NotImplementedError

    def to(self, device):
        # Buffers are automatically moved by nn.Module.to(), but we override
        # to ensure immediate movement and return self for chaining
        super().to(device)
        return self

    def add_noise(self, x0, noise, t, return_scaled_noise=False):
        """
        Add noise to x0 using the diffusion schedule.
        
        Args:
            x0: Clean latents (should be normalized to ~N(0,1))
            noise: Noise tensor (standard normal N(0,1) from randn_like)
            t: Timestep tensor
            return_scaled_noise: If True, also return the noise used
        
        Returns:
            noisy_x: Noisy latents
            (noise): Noise used (only if return_scaled_noise=True)
        """
        t = t.long().view(-1)
        alpha_bars = self.alpha_bars.to(t.device)
        sqrt_alpha_bar = alpha_bars[t].sqrt().to(x0.device)
        sqrt_one_minus = (1 - alpha_bars[t]).sqrt().to(x0.device)
        # Reshape to broadcast correctly: [B] -> [B, 1, 1, 1, ...]
        while sqrt_alpha_bar.dim() < x0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        
        # Noise is already in correct distribution (from randn_like)
        noisy_x = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        
        if return_scaled_noise:
            return noisy_x, noise
        return noisy_x
    
    def randn_like(self, tensor):
        """Generate standard normal noise N(0, 1)."""
        return torch.randn_like(tensor)


class LinearScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        return alphas, betas


class CosineScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        steps = torch.arange(0, num_steps + 1, dtype=torch.float32)
        alpha_bars = torch.cos(((steps / num_steps) + 0.008) / 1.008 * 3.14159 / 2) ** 2
        # Ensure alpha_bars are monotonically decreasing and in valid range
        alpha_bars = torch.clamp(alpha_bars, min=1e-6, max=1.0)
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        # Clamp betas more aggressively to match linear scheduler range for stability
        # This prevents extreme beta values that cause network collapse
        betas = torch.clamp(betas, min=1e-4, max=0.02)  # Match linear scheduler max (was 0.999)
        alphas = 1 - betas
        return alphas, betas


class SquaredCosineScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        steps = torch.arange(0, num_steps + 1, dtype=torch.float32)
        alpha_bars = torch.cos(((steps / num_steps) + 0.008) / 1.008 * 3.14159 / 2) ** 4
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        alphas = 1 - betas
        return alphas, betas


class SigmoidScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        steps = torch.linspace(-6, 6, num_steps)
        alpha_bars = torch.sigmoid(steps)
        alpha_bars = (alpha_bars - alpha_bars.min()) / (alpha_bars.max() - alpha_bars.min())
        alpha_bars = torch.flip(alpha_bars, [0])
        betas = torch.zeros(num_steps)
        betas[0] = 1 - alpha_bars[0]
        betas[1:] = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        alphas = 1 - betas
        return alphas, betas


class ExponentialScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        # Use larger decay rate to ensure alpha_bar_T ≈ 0
        # exp(-10) ≈ 4.5e-5, ensuring proper noise destruction
        alpha_bars = torch.exp(-10 * steps / num_steps)
        # Ensure final alpha_bar is small
        alpha_bars = torch.clamp(alpha_bars, min=1e-6, max=1.0)
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, min=1e-4, max=0.999)
        alphas = 1 - betas
        return alphas, betas


class QuadraticScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        betas = torch.linspace(1e-4**0.5, 0.02**0.5, num_steps) ** 2
        alphas = 1.0 - betas
        return alphas, betas


SCHEDULER_REGISTRY = {
    "LinearScheduler": LinearScheduler,
    "CosineScheduler": CosineScheduler,
    "SquaredCosineScheduler": SquaredCosineScheduler,
    "SigmoidScheduler": SigmoidScheduler,
    "ExponentialScheduler": ExponentialScheduler,
    "QuadraticScheduler": QuadraticScheduler,
}
