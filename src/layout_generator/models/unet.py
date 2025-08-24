import torch
import torch.nn as nn
from .base_model import BaseModel
from . import register_model
from ..modules.conditioning.base_conditioning import BaseConditioningModule
from ..modules.blocks import DoubleConv, DownBlock, UpBlock

class TimeEmbedding(nn.Module):
    def __init__(self, dim, embedding_scale=10000.0):
        super().__init__()
        self.embedding_scale = embedding_scale
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, t):
        half_dim = self.mlp[0].in_features // 2
        emb = torch.log(torch.tensor(self.embedding_scale, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


@register_model("UNet")
class UNet(BaseModel):
    def __init__(self, config: dict, conditioning_module: BaseConditioningModule):
        super().__init__(config)
        self.conditioning_module = conditioning_module
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        time_dim = self.info['time_dim']
        self.time_embed = TimeEmbedding(time_dim, embedding_scale=self.info.get('time_embedding_scale', 10000.0))

        # --- Build Encoder from config ---
        encoder_channels = self.info['architecture']['encoder']
        self.inc = DoubleConv(self.info['in_channels'], encoder_channels[0])
        for i in range(len(encoder_channels) - 1):
            self.encoder_blocks.append(DownBlock(encoder_channels[i], encoder_channels[i+1]))

        # --- Build Decoder from config (Corrected Logic) ---
        decoder_channels = self.info['architecture']['decoder']
        
        # First up-sampling block (from the bottleneck)
        self.decoder_blocks.append(
            UpBlock(
                in_channels_up=encoder_channels[-1],      # Channels from bottleneck
                in_channels_skip=encoder_channels[-2],    # Channels from the last skip connection
                out_channels=decoder_channels[0]
            )
        )
        
        # Subsequent up-sampling blocks
        for i in range(len(decoder_channels) - 1):
            self.decoder_blocks.append(
                UpBlock(
                    in_channels_up=decoder_channels[i],        # Channels from previous decoder block
                    in_channels_skip=encoder_channels[-i-3], # Channels from the corresponding skip connection
                    out_channels=decoder_channels[i+1]
                )
            )
            
        self.outc = nn.Conv2d(decoder_channels[-1], self.info['out_channels'], kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        t_emb = self.time_embed(t)
        
        # --- Encoder Path ---
        x = self.inc(x)
        skip_connections.append(x)
        
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
        
        # --- Decoder Path (Corrected Logic) ---
        # Reverse skip connections for easy access
        skip_connections = skip_connections[::-1]
        
        # The bottleneck (x) and the first skip connection are used by the first UpBlock
        x = self.decoder_blocks[0](x, skip_connections[1]) # Use skip_connections[1], as [0] is the bottleneck itself
        
        for i in range(1, len(self.decoder_blocks)):
            # The current feature map (x) and the next skip connection
            x = self.decoder_blocks[i](x, skip_connections[i+1])
        
        # Apply conditioning on the final feature map before output
        x = self.conditioning_module(x, t_emb, cond)
            
        return self.outc(x)