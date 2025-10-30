# Losses package
from .custom_loss import VAELoss, StandardVAELoss, SegmentationVAELoss, VGGPerceptualLoss, DiffusionLoss

__all__ = [
    'VAELoss', 'StandardVAELoss', 'SegmentationVAELoss', 'VGGPerceptualLoss', 'DiffusionLoss'
]
