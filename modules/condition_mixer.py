import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionMixer(nn.Module):
    def __init__(self, out_channels: int, target_size: tuple[int, int],
                 pov_channels: int = None, graph_channels: int = None):
        super().__init__()
        self.out_channels = out_channels
        self.target_size = target_size
        
        # Save current RNG state
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        
        self.pov_proj = self._make_projection(pov_channels, out_channels) if pov_channels else None
        self.graph_proj = self._make_projection(graph_channels, out_channels) if graph_channels else None
        
        # Restore RNG state
        torch.set_rng_state(rng_state)

    def _make_projection(self, in_channels: int, out_channels: int) -> nn.ModuleDict:
        H, W = self.target_size
        return nn.ModuleDict({
            'linear': nn.Linear(in_channels, out_channels * H * W, bias=False),
            'conv': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        })

    def project_condition(self, x: torch.Tensor, proj_module: nn.ModuleDict, out_channels: int) -> torch.Tensor:
        B, H, W = x.shape[0], *self.target_size
        if x.ndim == 2:
            return proj_module['linear'](x).view(B, out_channels, H, W)
        elif x.ndim == 4:
            x = proj_module['conv'](x)
            return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) if x.shape[2:] != (H, W) else x
        raise ValueError(f"Unsupported condition shape: {x.shape}")

    def forward(self, conds: list[torch.Tensor | None], weights: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class ConcatMixer(ConditionMixer):
    def __init__(self, out_channels: int, target_size: tuple[int, int],
                 pov_channels: int = None, graph_channels: int = None):
        # Split channels for concatenation
        self.pov_out_channels = out_channels // 2
        self.graph_out_channels = out_channels - self.pov_out_channels
        
        # Call parent init but don't create projections yet
        super().__init__(out_channels, target_size)
        
        # Save current RNG state
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        
        # Create projections with correct output sizes
        self.pov_proj = self._make_projection(pov_channels, self.pov_out_channels) if pov_channels else None
        self.graph_proj = self._make_projection(graph_channels, self.graph_out_channels) if graph_channels else None
        
        # Restore RNG state
        torch.set_rng_state(rng_state)
    
    def forward(self, conds: list[torch.Tensor | None], weights=None) -> torch.Tensor:
        pov, graph = conds
        B = (pov if pov is not None else graph).shape[0]
        device = (pov if pov is not None else graph).device
        H, W = self.target_size
        
        pov_out = self.project_condition(pov, self.pov_proj, self.pov_out_channels) if pov is not None else torch.zeros(B, self.pov_out_channels, H, W, device=device)
        graph_out = self.project_condition(graph, self.graph_proj, self.graph_out_channels) if graph is not None else torch.zeros(B, self.graph_out_channels, H, W, device=device)
        return torch.cat([pov_out, graph_out], dim=1)

    
class WeightedMixer(ConditionMixer):
    def forward(self, conds: list[torch.Tensor | None], weights: torch.Tensor = None) -> torch.Tensor:
        pov, graph = conds
        B = (pov if pov is not None else graph).shape[0]
        device = (pov if pov is not None else graph).device
        H, W = self.target_size
        
        # Default to equal weights if not provided
        if weights is None:
            weights = torch.tensor([0.5, 0.5], device=device)
        
        pov_out = self.project_condition(pov, self.pov_proj, self.out_channels) * weights[0] if pov is not None else torch.zeros(B, self.out_channels, H, W, device=device)
        graph_out = self.project_condition(graph, self.graph_proj, self.out_channels) * weights[1] if graph is not None else torch.zeros(B, self.out_channels, H, W, device=device)
        return pov_out + graph_out