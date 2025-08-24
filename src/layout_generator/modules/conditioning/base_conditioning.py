import torch.nn as nn

class BaseConditioningModule(nn.Module):
    """
    Abstract base class for all conditioning modules.
    It defines the interface for injecting a condition into the UNet.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.info = config

    def forward(self, x: nn.Module, time_emb: nn.Module, condition: nn.Module):
        """

        Injects the condition into the model.
        Args:
            x (nn.Module): The input feature map from the UNet (e.g., of shape [B, C, H, W]).
            time_emb (nn.Module): The time embedding tensor (e.g., of shape [B, D_emb]).
            condition (nn.Module): The raw conditioning tensor (e.g., of shape [B, D_cond]).

        Returns:
            The modified feature map `x` or time embedding `time_emb`.
        """
        raise NotImplementedError