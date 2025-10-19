import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim 
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class BaseMixer(nn.Module):

    def __init__(self, out_channels: int, target_size: tuple[int, int],
                 pov_channels: Optional[int], graph_channels: Optional[int]):
        super().__init__()
        self.out_channels = out_channels
        self.target_size = target_size
        self.pov_channels = pov_channels
        self.graph_channels = graph_channels

        if pov_channels is None and graph_channels is None:
             raise ValueError("Must provide at least one input channel dimension (pov_channels or graph_channels).")

        # Determine output channels per branch (assuming concatenation for now)
        # Subclasses can override if they mix differently (e.g., weighted sum)
        self.pov_out_channels = out_channels // 2
        self.graph_out_channels = out_channels - self.pov_out_channels

        # --- Subclasses MUST define these projectors ---
        self.pov_projector: nn.Module = nn.Identity()
        self.graph_projector: nn.Module = nn.Identity()

    def _project_and_reshape(self, x: Optional[torch.Tensor], projector: nn.Module, out_channels_branch: int) -> torch.Tensor:

        B = 1 
        device = 'cpu' # Default device
        H, W = self.target_size
        
        if x is not None:
             B = x.shape[0]
             device = x.device
        
        if x is None or isinstance(projector, nn.Identity):
             return torch.zeros(B, out_channels_branch, H, W, device=device)

        if x.ndim == 2: # Embedding vector [B, C_in]
            # Ensure correct dtype
            dtype = next(projector.parameters()).dtype if list(projector.parameters()) else torch.float32
            x = x.to(dtype)

            projected = projector(x) # Apply the specific projector (Linear or MLP)
            
            expected_flat_dim = out_channels_branch * H * W
            if projected.shape[-1] != expected_flat_dim:
                 raise ValueError(f"Projector output dimension mismatch. Expected {expected_flat_dim}, got {projected.shape[-1]}")
                 
            output = projected.view(B, out_channels_branch, H, W)
        else:
            raise ValueError(f"This mixer only supports 2D embedding inputs (shape [B, C_in]), got {x.shape}")

        return output

    def forward(self, conds: list[Optional[torch.Tensor]], weights=None) -> torch.Tensor:

        raise NotImplementedError("Subclasses must implement the forward method.")

class LinearConcatMixer(BaseMixer):

    def __init__(self, out_channels: int, target_size: tuple[int, int],
                 pov_channels: Optional[int] = None, graph_channels: Optional[int] = None):
        super().__init__(out_channels, target_size, pov_channels, graph_channels)

        H, W = target_size

        pov_target_dim = self.pov_out_channels * H * W
        graph_target_dim = self.graph_out_channels * H * W

        if self.pov_channels is not None:
             print(f"LinearConcatMixer: Creating Linear POV projector ({self.pov_channels} -> {pov_target_dim})", flush=True)
             self.pov_projector = nn.Linear(self.pov_channels, pov_target_dim, bias=False)

        if self.graph_channels is not None:
             print(f"LinearConcatMixer: Creating Linear Graph projector ({self.graph_channels} -> {graph_target_dim})", flush=True)
             self.graph_projector = nn.Linear(self.graph_channels, graph_target_dim, bias=False)

        torch.set_rng_state(rng_state)

    def forward(self, conds: list[Optional[torch.Tensor]], weights=None) -> torch.Tensor:
        pov, graph = conds

        pov_out = self._project_and_reshape(pov, self.pov_projector, self.pov_out_channels)

        graph_out = self._project_and_reshape(graph, self.graph_projector, self.graph_out_channels)

        return torch.cat([pov_out, graph_out], dim=1)

class NonLinearConcatMixer(BaseMixer):

    def __init__(self, out_channels: int, target_size: tuple[int, int],
                 pov_channels: Optional[int] = None, graph_channels: Optional[int] = None,
                 hidden_dim_mlp: Optional[int] = None):
        super().__init__(out_channels, target_size, pov_channels, graph_channels)

        H, W = target_size
        pov_target_dim = self.pov_out_channels * H * W
        graph_target_dim = self.graph_out_channels * H * W


        if self.pov_channels is not None:
             print(f"NonLinearConcatMixer: Creating MLP POV projector ({self.pov_channels} -> {pov_target_dim})", flush=True)
             self.pov_projector = ProjectionMLP(self.pov_channels, pov_target_dim, hidden_dim=hidden_dim_mlp)

        if self.graph_channels is not None:
             print(f"NonLinearConcatMixer: Creating MLP Graph projector ({self.graph_channels} -> {graph_target_dim})", flush=True)
             self.graph_projector = ProjectionMLP(self.graph_channels, graph_target_dim, hidden_dim=hidden_dim_mlp)


    def forward(self, conds: list[Optional[torch.Tensor]], weights=None) -> torch.Tensor:
        """Projects conditions using ProjectionMLP and concatenates."""
        pov, graph = conds

        pov_out = self._project_and_reshape(pov, self.pov_projector, self.pov_out_channels)

        graph_out = self._project_and_reshape(graph, self.graph_projector, self.graph_out_channels)

        return torch.cat([pov_out, graph_out], dim=1)