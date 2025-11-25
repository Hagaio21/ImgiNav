from .base_loss import (
    LOSS_REGISTRY,
    LossComponent,
    MSELoss,
    ColorWeightedMSELoss,
    KLDLoss,
    LatentStandardizationLoss,
    LatentStructuralLossAE,
    CompositeLoss,
)
from .clip_loss import CLIPLoss

__all__ = [
    "LOSS_REGISTRY",
    "LossComponent",
    "MSELoss",
    "ColorWeightedMSELoss",
    "KLDLoss",
    "LatentStandardizationLoss",
    "LatentStructuralLossAE",
    "CompositeLoss",
    "CLIPLoss",
]

