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
        
        # Noise scaling parameters (set by diffusion model based on latent statistics)
        # noise_scale: scale factor for noise (typically latent std)
        # noise_offset: offset for noise (typically latent mean, default 0)
        noise_scale = self._init_kwargs.get("noise_scale", None)
        noise_offset = self._init_kwargs.get("noise_offset", None)
        if noise_scale is not None:
            self.register_buffer("noise_scale", noise_scale)
        else:
            self.noise_scale = None
        if noise_offset is not None:
            self.register_buffer("noise_offset", noise_offset)
        else:
            self.noise_offset = None

    def build_schedule(self, num_steps: int):
        raise NotImplementedError

    def set_noise_scale(self, noise_scale):
        """Set noise scale (typically latent std)."""
        if noise_scale is not None:
            if not hasattr(self, 'noise_scale') or self.noise_scale is None:
                self.register_buffer("noise_scale", noise_scale)
            else:
                self.noise_scale.data.copy_(noise_scale)
        else:
            self.noise_scale = None
    
    def set_noise_offset(self, noise_offset):
        """Set noise offset (typically latent mean)."""
        if noise_offset is not None:
            if not hasattr(self, 'noise_offset') or self.noise_offset is None:
                self.register_buffer("noise_offset", noise_offset)
            else:
                self.noise_offset.data.copy_(noise_offset)
        else:
            self.noise_offset = None

    def to(self, device):
        # Buffers are automatically moved by nn.Module.to(), but we override
        # to ensure immediate movement and return self for chaining
        super().to(device)
        return self

    def add_noise(self, x0, noise, t):
        t = t.long().view(-1)
        # Move alpha_bars to same device as t if needed
        alpha_bars = self.alpha_bars.to(t.device)
        sqrt_alpha_bar = alpha_bars[t].sqrt().to(x0.device)
        sqrt_one_minus = (1 - alpha_bars[t]).sqrt().to(x0.device)
        # Reshape to broadcast correctly: [B] -> [B, 1, 1, 1, ...]
        # Add dimensions to match spatial dimensions of x0
        while sqrt_alpha_bar.dim() < x0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        
        # Apply noise scaling if set (ensure noise_scale is on same device as noise)
        if self.noise_scale is not None:
            noise_scale = self.noise_scale.to(noise.device)
            noise = noise * noise_scale
        if self.noise_offset is not None:
            noise_offset = self.noise_offset.to(noise.device)
            noise = noise + noise_offset
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise
    
    def scale_noise(self, noise):
        """Scale noise using noise_scale and noise_offset."""
        if self.noise_scale is not None:
            # Ensure noise_scale is on same device as noise
            noise_scale = self.noise_scale.to(noise.device)
            noise = noise * noise_scale
        if self.noise_offset is not None:
            # Ensure noise_offset is on same device as noise
            noise_offset = self.noise_offset.to(noise.device)
            noise = noise + noise_offset
        return noise


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
        # Clamp betas to avoid numerical issues (like other schedulers do)
        betas = torch.clamp(betas, min=1e-4, max=0.999)
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
