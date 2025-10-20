import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ConstructiveDiffusionLoss(nn.Module):
    def __init__(self, in_channels, height, width, alpha=0.5, gamma=0.1, dropout_p=0.3):
        """
        in_channels : number of latent channels (e.g. 4)
        height, width : spatial dimensions (e.g. 64, 64)
        alpha : balance between MSE and cosine inside L_c
        gamma : global weight of L_c
        dropout_p : probability to skip L_c each forward pass
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dropout_p = dropout_p

        # small projection head h
        self.h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, pred_noise, true_noise, x0_hat, z_c):
        L_d = F.mse_loss(pred_noise, true_noise)

        # Either mixer dropout (zero tensor) OR stochastic dropout for loss
        if (
            z_c is None
            or torch.all(z_c == 0)
            or random.random() < self.dropout_p
        ):
            return L_d, {
                "L_d": L_d.detach(),
                "L_c": torch.tensor(0.0, device=L_d.device),
                "L_mse": torch.tensor(0.0, device=L_d.device),
                "L_cos": torch.tensor(0.0, device=L_d.device),
            }

        x_proj = self.h(torch.nan_to_num(x0_hat, nan=0.0))
        z_proj = self.h(torch.nan_to_num(z_c, nan=0.0))

        x_proj = x_proj / (x_proj.norm(dim=1, keepdim=True) + 1e-8)
        z_proj = z_proj / (z_proj.norm(dim=1, keepdim=True) + 1e-8)

        L_mse = F.mse_loss(x_proj, z_proj)
        L_cos = 1 - (x_proj * z_proj).sum(dim=1).mean()

        L_c = self.alpha * L_mse + (1 - self.alpha) * L_cos
        total_loss = L_d + self.gamma * L_c

        return total_loss, {
            "L_d": L_d.detach(),
            "L_c": L_c.detach(),
            "L_mse": L_mse.detach(),
            "L_cos": L_cos.detach(),
        }


