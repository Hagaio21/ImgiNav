import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionMixer(nn.Module):
    def __init__(self, out_channels: int, target_size: tuple[int, int]):
        super().__init__()
        self.out_channels = out_channels
        self.target_size = target_size  # (H, W)

    def project_condition(self, x: torch.Tensor, out_channels: int = None) -> torch.Tensor:
        """Project any condition tensor to [B, C_out, H, W]."""
        if out_channels is None:
            out_channels = self.out_channels
            
        B = x.shape[0]
        H, W = self.target_size

        if x.ndim == 2:  # [B, D]
            proj = nn.Linear(x.shape[1], out_channels * H * W, bias=False).to(x.device)
            x = proj(x).view(B, out_channels, 1, 1)
            x = F.interpolate(x, size=(H, W), mode="nearest")

        elif x.ndim == 4:  # [B, C, h, w]
            conv = nn.Conv2d(x.shape[1], out_channels, kernel_size=1, bias=False).to(x.device)
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

    def forward(self, conds: list[torch.Tensor | None], weights: torch.Tensor | None = None) -> torch.Tensor:
        valid = [c for c in conds if c is not None]
        if not valid:
            raise ValueError("All conditioning inputs are None.")
        
        # Split channels among conditions
        num_valid = len(valid)
        channels_per_cond = self.out_channels // num_valid
        remainder = self.out_channels % num_valid
        
        projected = []
        for i, c in enumerate(valid):
            c_channels = channels_per_cond + (1 if i < remainder else 0)
            proj = self.project_condition(c, out_channels=c_channels)
            projected.append(proj)
        
        return torch.cat(projected, dim=1)


class WeightedMixer(ConditionMixer):
    def __init__(self, out_channels: int, target_size: tuple[int, int]):
        super().__init__(out_channels, target_size)

    def forward(self, conds: list[torch.Tensor | None], weights: torch.Tensor) -> torch.Tensor:
        valid = [(c, w) for c, w in zip(conds, weights) if c is not None]
        if not valid:
            raise ValueError("All conditioning inputs are None.")
        projected = [self.project_condition(c) * w for c, w in valid]
        return sum(projected)


class LearnedWeightedMixer(ConditionMixer):
    def __init__(self, num_conds: int, out_channels: int, target_size: tuple[int, int]):
        super().__init__(out_channels, target_size)
        self.raw_weights = nn.Parameter(torch.zeros(num_conds))

    def forward(self, conds: list[torch.Tensor | None], weights: torch.Tensor | None = None) -> torch.Tensor:
        valid = [c for c in conds if c is not None]
        if not valid:
            raise ValueError("All conditioning inputs are None.")
        norm_weights = F.softmax(self.raw_weights[: len(valid)], dim=0)
        projected = [self.project_condition(c) * w for c, w in zip(valid, norm_weights)]
        return sum(projected)