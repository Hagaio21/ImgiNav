from .base_loss import (
    LOSS_REGISTRY,
    LossComponent,
    L1Loss,
    MSELoss,
    KLDLoss,
    LatentStandardizationLoss,
    CrossEntropyLoss,
    PerceptualLoss,
    CompositeLoss,
)
from .semantic_loss import SemanticLoss
from .diffusion_losses import SNRWeightedNoiseLoss, DiscriminatorLoss, LatentStructuralLoss

__all__ = [
    "LOSS_REGISTRY",
    "LossComponent",
    "L1Loss",
    "MSELoss",
    "KLDLoss",
    "LatentStandardizationLoss",
    "CrossEntropyLoss",
    "PerceptualLoss",
    "CompositeLoss",
    "SemanticLoss",
    "SNRWeightedNoiseLoss",
    "DiscriminatorLoss",
    "LatentStructuralLoss",
]

