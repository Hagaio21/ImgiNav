from .base_loss import (
    LOSS_REGISTRY,
    LossComponent,
    L1Loss,
    MSELoss,
    KLDLoss,
    LatentStandardizationLoss,
    CrossEntropyLoss,
    PerceptualLoss,
    LatentStructuralLossAE,
    CompositeLoss,
)
from .semantic_loss import SemanticLoss
from .diffusion_losses import SNRWeightedNoiseLoss, DiscriminatorLoss, LatentStructuralLoss
from .adversarial_loss import DiscriminatorLossWithReconnection
from .advanced_losses import (
    FrequencyDomainLoss,
    CharbonnierLoss,
    FeatureMatchingLoss,
    ConsistencyLoss,
    StyleLoss,
)

__all__ = [
    "LOSS_REGISTRY",
    "LossComponent",
    "L1Loss",
    "MSELoss",
    "KLDLoss",
    "LatentStandardizationLoss",
    "CrossEntropyLoss",
    "PerceptualLoss",
    "LatentStructuralLossAE",
    "CompositeLoss",
    "SemanticLoss",
    "SNRWeightedNoiseLoss",
    "DiscriminatorLoss",
    "DiscriminatorLossWithReconnection",
    "LatentStructuralLoss",
    "FrequencyDomainLoss",
    "CharbonnierLoss",
    "FeatureMatchingLoss",
    "ConsistencyLoss",
    "StyleLoss",
]

