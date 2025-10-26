import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def create_norm(norm_type: str, num_channels: int, target_size: tuple[int, int]) -> nn.Module:
    """Stable normalization for mixer outputs."""
    if norm_type in (None, "group", "default"):
        return nn.GroupNorm(1, num_channels, affine=True)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_channels, affine=True)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm_type == "layer":
        H, W = target_size
        return nn.LayerNorm([num_channels, H, W], elementwise_affine=True)
    elif norm_type in ("none", "identity"):
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")



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

    def _get_batch_size_and_device(self, *conds, B_hint=None, device_hint=None) -> tuple[int, torch.device]:
        """Determines batch size and device from the first available tensor."""
        for c in conds:
            if c is not None:
                return c.shape[0], c.device

        # Fallback if all conditions are None (e.g. full dropout)
        if B_hint is not None and device_hint is not None:
            return B_hint, device_hint

        try:
            device = device_hint or next(self.parameters()).device
        except StopIteration:
            device = device_hint or ('cuda' if torch.cuda.is_available() else 'cpu')

        if B_hint is not None:
            return B_hint, device

        print("Warning: Mixer could not determine batch size from inputs. Defaulting to B=1.")
        return 1, device


    def _project_and_reshape(self, x: Optional[torch.Tensor], projector: nn.Module, 
                                out_channels_branch: int, B: int, device: torch.device) -> torch.Tensor:
            """
            Projects a tensor (if not None) or returns zeros, using a *provided*
            batch size and device.
            """
            H, W = self.target_size
            
            if x is None or isinstance(projector, nn.Identity):
                # Use B and device passed in
                return torch.zeros(B, out_channels_branch, H, W, device=device)

            # Handle the placeholder tensor [B, 1] from collate_fn
            if x.ndim == 2 and x.shape[1] == 1 and x.std() < 1e-6:
                return torch.zeros(B, out_channels_branch, H, W, device=device)

            if x.ndim == 2: # Embedding vector [B, C_in]
                if x.shape[0] != B:
                    print(f"Warning: Mixer input tensor batch size {x.shape[0]} != determined batch size {B}. Using {x.shape[0]}.")
                    B = x.shape[0] # Trust the tensor
                try:
                    # Get the dtype (e.g., float32) from the projector's parameters
                    projector_dtype = next(projector.parameters()).dtype
                    x = x.to(projector_dtype)
                except StopIteration:
                    # Projector has no parameters (e.g., nn.Identity), do nothing
                    pass

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
    
    def get_projected_conditions(self, cond_pov: Optional[torch.Tensor], cond_graph: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            
            B, device = self._get_batch_size_and_device(cond_pov, cond_graph)
            
            pov_out = self._project_and_reshape(cond_pov, self.pov_projector, self.pov_out_channels, B, device)
            graph_out = self._project_and_reshape(cond_graph, self.graph_projector, self.graph_out_channels, B, device)
            return pov_out, graph_out

class LinearConcatMixer(BaseMixer):

    def __init__(self, out_channels: int, target_size: tuple[int, int],
                 pov_channels: Optional[int] = None, graph_channels: Optional[int] = None, norm_type=None):
        super().__init__(out_channels, target_size, pov_channels, graph_channels)

        self.norm = create_norm(norm_type, out_channels, target_size)

        H, W = target_size

        pov_target_dim = self.pov_out_channels * H * W
        graph_target_dim = self.graph_out_channels * H * W

        if self.pov_channels is not None:
             print(f"LinearConcatMixer: Creating Linear POV projector ({self.pov_channels} -> {pov_target_dim})", flush=True)
             self.pov_projector = nn.Linear(self.pov_channels, pov_target_dim, bias=False)

        if self.graph_channels is not None:
             print(f"LinearConcatMixer: Creating Linear Graph projector ({self.graph_channels} -> {graph_target_dim})", flush=True)
             self.graph_projector = nn.Linear(self.graph_channels, graph_target_dim, bias=False)


    def forward(self, conds: list[Optional[torch.Tensor]], B_hint=None, device_hint=None, weights=None) -> torch.Tensor:
        pov, graph = conds
        B, device = self._get_batch_size_and_device(pov, graph, B_hint=B_hint, device_hint=device_hint)

        pov_out = self._project_and_reshape(pov, self.pov_projector, self.pov_out_channels, B, device)
        graph_out = self._project_and_reshape(graph, self.graph_projector, self.graph_out_channels, B, device)

        out = torch.cat([pov_out, graph_out], dim=1)
        out = self.norm(out)

        # --- safety normalization ---
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        std = out.std(dim=(1, 2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5, max=1e5)
        out = (out / std) 
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        return out


class NonLinearConcatMixer(BaseMixer):

    def __init__(self, out_channels: int, target_size: tuple[int, int],
                 pov_channels: Optional[int] = None, graph_channels: Optional[int] = None,
                 hidden_dim_mlp: Optional[int] = None, norm_type=None):
        super().__init__(out_channels, target_size, pov_channels, graph_channels)

        self.norm = create_norm(norm_type, out_channels, target_size)

        H, W = target_size
        pov_target_dim = self.pov_out_channels * H * W
        graph_target_dim = self.graph_out_channels * H * W


        if self.pov_channels is not None:
             print(f"NonLinearConcatMixer: Creating MLP POV projector ({self.pov_channels} -> {pov_target_dim})", flush=True)
             self.pov_projector = ProjectionMLP(self.pov_channels, pov_target_dim, hidden_dim=hidden_dim_mlp)

        if self.graph_channels is not None:
             print(f"NonLinearConcatMixer: Creating MLP Graph projector ({self.graph_channels} -> {graph_target_dim})", flush=True)
             self.graph_projector = ProjectionMLP(self.graph_channels, graph_target_dim, hidden_dim=hidden_dim_mlp)


    def forward(self, conds: list[Optional[torch.Tensor]], B_hint=None, device_hint=None, weights=None) -> torch.Tensor:
        pov, graph = conds
        B, device = self._get_batch_size_and_device(pov, graph, B_hint=B_hint, device_hint=device_hint)

        pov_out = self._project_and_reshape(pov, self.pov_projector, self.pov_out_channels, B, device)
        graph_out = self._project_and_reshape(graph, self.graph_projector, self.graph_out_channels, B, device)

        out = torch.cat([pov_out, graph_out], dim=1)
        out = self.norm(out)

        # --- safety normalization ---
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        std = out.std(dim=(1, 2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5, max=1e5)
        out = (out / std)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Print output shape on first forward pass only
        if not hasattr(self, '_printed_output_shape'):
            print(f"NonLinearConcatMixer output shape: {out.shape} [B={B}, C={self.out_channels}, H={self.target_size[0]}, W={self.target_size[1]}]", flush=True)
            print(f"  - POV contribution: {pov_out.shape}", flush=True)
            print(f"  - Graph contribution: {graph_out.shape}", flush=True)
            self._printed_output_shape = True

        return out

        