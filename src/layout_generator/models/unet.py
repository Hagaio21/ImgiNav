import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim, embedding_scale=10000.0): # Refactored
        super().__init__()
        self.embedding_scale = embedding_scale # Refactored
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half_dim = self.mlp[0].in_features // 2
        # Use the configured embedding_scale
        emb = torch.log(torch.tensor(self.embedding_scale)) / (half_dim - 1) # Refactored
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64,
                 depth=4, time_dim=128, cond_dim=128, 
                 group_norm_groups=8, time_embedding_scale=10000.0, # Refactored
                 name="unet_conditioned"):
        super().__init__()
        assert cond_dim > 0 and time_dim > 0

        self.name = name
        self.config = {
            "name": name,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "depth": depth,
            "time_dim": time_dim,
            "cond_dim": cond_dim,
            "group_norm_groups": group_norm_groups, # Refactored
            "time_embedding_scale": time_embedding_scale # Refactored
        }
        
        # Pass the configurable scale to TimeEmbedding
        self.time_embed = TimeEmbedding(time_dim, embedding_scale=time_embedding_scale) # Refactored
        self.condition_proj = nn.Linear(cond_dim, time_dim)

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        channels = [base_channels * (2 ** i) for i in range(depth)]
        for i in range(depth):
            in_ch = channels[i - 1] if i > 0 else base_channels
            out_ch = channels[i]
            # Pass the configurable group count to the block builder
            self.down_blocks.append(self._res_block(in_ch, out_ch, time_dim, num_groups=group_norm_groups)) # Refactored

        for i in reversed(range(depth)):
            in_ch = channels[i + 1] if i + 1 < depth else channels[-1]
            out_ch = channels[i]
            # Pass the configurable group count to the block builder
            self.up_blocks.append(self._res_block(in_ch, out_ch, time_dim, num_groups=group_norm_groups)) # Refactored

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _res_block(self, in_ch, out_ch, embed_dim, num_groups): # Refactored
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch), # Refactored
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch), # Refactored
            nn.SiLU()
        )

    def predict(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]        - noisy image
        t: [B]                 - diffusion timestep
        cond: [B, cond_dim]    - external condition (e.g. from frozen encoder)
        """
        assert cond.ndim == 2 and t.ndim == 1 and x.ndim == 4, "Invalid input shapes"

        # Embed timestep and condition
        t_emb = self.time_embed(t)        # [B, time_dim]
        c_emb = self.condition_proj(cond) # [B, time_dim]
        emb = t_emb + c_emb               # [B, time_dim]

        x = self.input_proj(x)

        # Down path
        for block in self.down_blocks:
            x = x + emb[:, :, None, None]  # broadcast and inject
            x = block(x)

        # Up path
        for block in self.up_blocks:
            x = x + emb[:, :, None, None]
            x = block(x)

        return self.final(x)

    def forward(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def info(self):
        return {
            "name": self.name,
            "model": self.__class__.__name__,
            "config": self.config,
            "total_params": sum(p.numel() for p in self.parameters())
        }