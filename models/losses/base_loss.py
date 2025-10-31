import torch
import torch.nn as nn
from ..components.base_component import BaseComponent


class LossComponent(BaseComponent):
    """
    Base class for all loss functions.
    Provides config-driven instantiation and consistent forward signature.
    """
    def forward(self, preds: dict, targets: dict):
        """
        Each loss gets model outputs (preds) and ground truths (targets).
        Returns scalar loss tensor and optional dict of sublosses.
        """
        raise NotImplementedError
