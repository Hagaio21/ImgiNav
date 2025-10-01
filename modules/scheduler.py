import torch
from abc import ABC, abstractmethod

class NoiseScheduler(ABC):
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.alphas, self.betas = self.build_schedule(num_steps)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @abstractmethod
    def build_schedule(self, num_steps):
        pass

    def add_noise(self, x0, t, noise):
        sqrt_alpha_bar = self.alpha_bars[t].sqrt()
        sqrt_one_minus = (1 - self.alpha_bars[t]).sqrt()
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise


class LinearScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        return alphas, betas

class CosineScheduler(NoiseScheduler):
    def build_schedule(self, num_steps):
        # cosine decay formula (simplified)
        steps = torch.arange(0, num_steps + 1, dtype=torch.float32)
        alpha_bars = torch.cos(((steps / num_steps) + 0.008) / 1.008 * 3.14159 / 2) ** 2
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        alphas = 1 - betas
        return alphas, betas
