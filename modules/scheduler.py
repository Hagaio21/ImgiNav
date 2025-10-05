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
    
    def to(self, device):
        """Move scheduler tensors to device"""
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0, t, noise):
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
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




class SquaredCosineScheduler(NoiseScheduler):
    """Improved Cosine scheduler - better signal preservation"""
    def build_schedule(self, num_steps):
        steps = torch.arange(0, num_steps + 1, dtype=torch.float32)
        alpha_bars = torch.cos(((steps / num_steps) + 0.008) / 1.008 * 3.14159 / 2) ** 4
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        alphas = 1 - betas
        return alphas, betas


class SigmoidScheduler(NoiseScheduler):
    """Sigmoid-based schedule - smooth transition"""
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
    """Exponential decay - fast at start, slow at end"""
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
    """Quadratic schedule - middle ground between linear and cosine"""
    def build_schedule(self, num_steps):
        betas = torch.linspace(1e-4**0.5, 0.02**0.5, num_steps) ** 2
        alphas = 1.0 - betas
        return alphas, betas