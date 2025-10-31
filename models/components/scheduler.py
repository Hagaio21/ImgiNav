import torch
from .base_component import BaseComponent


class NoiseScheduler(BaseComponent):
    def _build(self):
        num_steps = self._init_kwargs.get("num_steps", 1000)
        self.num_steps = num_steps
        self.alphas, self.betas = self.build_schedule(num_steps)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def build_schedule(self, num_steps: int):
        raise NotImplementedError

    def to(self, device):
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0, noise, t):
        t = t.long().view(-1)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().to(x0.device)
        sqrt_one_minus = (1 - self.alpha_bars[t]).sqrt().to(x0.device)
        # Reshape to broadcast correctly: [B] -> [B, 1, 1, 1, ...]
        # Add dimensions to match spatial dimensions of x0
        while sqrt_alpha_bar.dim() < x0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


class LinearScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        return alphas, betas


class CosineScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        steps = torch.arange(0, num_steps + 1, dtype=torch.float32)
        alpha_bars = torch.cos(((steps / num_steps) + 0.008) / 1.008 * 3.14159 / 2) ** 2
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
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
        steps = torch.arange(num_steps, dtype=torch.float32)
        alpha_bars = torch.exp(-5 * steps / num_steps)
        betas = torch.zeros(num_steps)
        betas[0] = 1 - alpha_bars[0]
        betas[1:] = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, 1e-4, 0.999)
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
