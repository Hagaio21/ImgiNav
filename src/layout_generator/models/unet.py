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


from .base_model import BaseModel
from . import register_model
from ..modules.blocks import DoubleConv, DownBlock, UpBlock 
from ..modules.conditioning.base_conditioning import BaseConditioningModule

@register_model("UNet")
class UNet(BaseModel):
    def __init__(self, config: dict, conditioning_module: BaseConditioningModule):
        super().__init__(config)
        
        # The UNet now receives a ready-to-use conditioning module
        self.conditioning_module = conditioning_module
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        time_dim = self.info['time_dim']
        cond_dim = self.info['cond_dim']
        
        self.time_embed = TimeEmbedding(time_dim, embedding_scale=self.info.get('time_embedding_scale', 10000.0))
        self.condition_proj = nn.Linear(cond_dim, time_dim)

        # --- Build Encoder from config ---
        encoder_channels = self.info['architecture']['encoder']
        self.inc = DoubleConv(self.info['in_channels'], encoder_channels[0])
        
        for i in range(len(encoder_channels) - 1):
            self.encoder_blocks.append(DownBlock(encoder_channels[i], encoder_channels[i+1]))

        # --- Build Decoder from config ---
        decoder_channels = self.info['architecture']['decoder']
        
        # The bridge (bottleneck) connection
        bridge_in = encoder_channels[-1] + encoder_channels[-2]
        self.decoder_blocks.append(UpBlock(bridge_in, decoder_channels[0]))
        
        for i in range(len(decoder_channels) - 1):
            # Calculate input channels for the UpBlock: (skip_connection + previous_decoder_output)
            skip_ch = encoder_channels[-i-2]
            prev_dec_ch = decoder_channels[i]
            in_ch = skip_ch + prev_dec_ch
            self.decoder_blocks.append(UpBlock(in_ch, decoder_channels[i+1]))
            
        self.outc = nn.Conv2d(decoder_channels[-1], self.info['out_channels'], kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        t_emb = self.time_embed(t)
        c_emb = self.condition_proj(cond)
        emb = t_emb + c_emb

        x = self.inc(x)
        skip_connections.append(x)
        
        for block in self.encoder_blocks:
            x = block(x)
            x = self.conditioning_module(x, t_emb, cond)
            skip_connections.append(x)
        
        # The forward pass in your original UNet added embeddings at each block.
        # This is a more standard UNet implementation with skip connections.
        # We can add the embedding injection back if needed.
        
        x = self.decoder_blocks[0](x, skip_connections[-2])
        
        for i in range(1, len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, skip_connections[-i-2])
            
        return self.outc(x)