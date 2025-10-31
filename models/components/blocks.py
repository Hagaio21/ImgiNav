import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(1, dim * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t = self.act(self.fc1(t))
        return self.fc2(t)
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_emb = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_dim)
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb):
        for res in self.res_blocks:
            x = res(x, t_emb)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks=1):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_ch + out_ch if i == 0 else out_ch, out_ch, time_dim)
            for i in range(num_res_blocks)
        ])

    def forward(self, x, skip, t_emb):

        x = self.upsample(x)
        
        x = torch.cat([x, skip], dim=1)
        
        for res in self.res_blocks:
            x = res(x, t_emb)
        return x
