import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionMixer(nn.Module):
    def __init__(self, out_channels: int, target_size: tuple[int, int]):
        super().__init__()
        self.out_channels = out_channels
        self.target_size = target_size  # (H, W)

    def project_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Project any condition tensor to [B, C_out, H, W]."""
        B = x.shape[0]
        H, W = self.target_size

        if x.ndim == 2:  # [B, D]
            proj = nn.Linear(x.shape[1], self.out_channels, bias=False).to(x.device)
            x = proj(x).view(B, self.out_channels, 1, 1)
            x = F.interpolate(x, size=(H, W), mode="nearest")

        elif x.ndim == 4:  # [B, C, h, w]
            conv = nn.Conv2d(x.shape[1], self.out_channels, kernel_size=1, bias=False).to(x.device)
            x = conv(x)
            if (x.shape[2], x.shape[3]) != (H, W):
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unsupported condition shape: {x.shape}")

        return x

    def forward(self, conds: list[torch.Tensor], weights: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class ConcatMixer(ConditionMixer):
    def __init__(self, out_channels: int, target_size: tuple[int, int]):
        super().__init__(out_channels, target_size)

    def forward(self, conds: list[torch.Tensor], weights: torch.Tensor | None = None) -> torch.Tensor:
        projected = [self.project_condition(c) for c in conds]
        return torch.cat(projected, dim=1)


class WeightedMixer(ConditionMixer):
    def __init__(self, out_channels: int, target_size: tuple[int, int]):
        super().__init__(out_channels, target_size)

    def forward(self, conds: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim == 1:
            weights = weights.view(-1, 1, 1, 1)
        projected = [self.project_condition(c) * w for c, w in zip(conds, weights)]
        return sum(projected)


class LearnedWeightedMixer(ConditionMixer):
    def __init__(self, num_conds: int, out_channels: int, target_size: tuple[int, int]):
        super().__init__(out_channels, target_size)
        self.raw_weights = nn.Parameter(torch.zeros(num_conds))

    def forward(self, conds: list[torch.Tensor], weights: torch.Tensor | None = None) -> torch.Tensor:
        norm_weights = F.softmax(self.raw_weights, dim=0)
        projected = [self.project_condition(c) * w for c, w in zip(conds, norm_weights)]
        return sum(projected)
