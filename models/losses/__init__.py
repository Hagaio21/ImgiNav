# Losses package
from .custom_loss import VAELoss, StandardVAELoss, SegmentationVAELoss, VGGPerceptualLoss

__all__ = [
    'VAELoss', 'StandardVAELoss', 'SegmentationVAELoss', 'VGGPerceptualLoss'
]
