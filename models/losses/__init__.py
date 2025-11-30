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
# Import reconstruction losses to register them
from . import reconstruction_loss

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
    "L1Loss",
    "PerceptualLoss",
    "SSIMLoss",
    "CombinedReconstructionLoss",
    "GradientDifferenceLoss",
]

