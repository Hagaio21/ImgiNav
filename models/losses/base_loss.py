import torch
import torch.nn as nn
import numpy as np
from ..components.base_component import BaseComponent

LOSS_REGISTRY = {}

def register_loss(cls):
    """Decorator to register loss classes."""
    LOSS_REGISTRY[cls.__name__] = cls
    return cls

class LossComponent(BaseComponent):
    def _build(self):
        self.key = self._init_kwargs.get("key", None)
        self.target_key = self._init_kwargs.get("target", self.key)
        self.weight = self._init_kwargs.get("weight", 1.0)

    def forward(self, preds, targets):
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg):
        """Factory method used by CompositeLoss."""
        return cls(**cfg)



@register_loss
class L1Loss(LossComponent):
    def _build(self):
        super()._build()
        self.criterion = nn.L1Loss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(self.criterion.parameters(), torch.zeros(1)).device
            return torch.tensor(0.0, device=device), {}
        loss = self.criterion(preds[self.key], targets[self.target_key]) * self.weight
        return loss, {f"L1_{self.key}": loss.detach()}


@register_loss
class MSELoss(LossComponent):
    def _build(self):
        super()._build()
        self.criterion = nn.MSELoss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(self.criterion.parameters(), torch.zeros(1)).device
            return torch.tensor(0.0, device=device), {}
        loss = self.criterion(preds[self.key], targets[self.target_key]) * self.weight
        return loss, {f"MSE_{self.key}": loss.detach()}


@register_loss
class KLDLoss(LossComponent):
    def forward(self, preds, targets=None):
        mu = preds.get("mu")
        logvar = preds.get("logvar")
        if mu is None or logvar is None:
            device = mu.device if mu is not None else torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld * self.weight
        return kld, {"KLD": kld.detach()}


@register_loss
class LatentStandardizationLoss(LossComponent):
    """
    Loss that encourages latents to be approximately N(0,1).
    Penalizes mean deviation from 0 and std deviation from 1.
    """
    def forward(self, preds, targets=None):
        # Extract latent from encoder output
        latent = preds.get(self.key)  # key should be "latent" or "mu"
        if latent is None:
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            return torch.tensor(0.0, device=device), {}
        
        # Flatten latents to compute global mean and std
        latent_flat = latent.reshape(latent.shape[0], -1)
        
        # Compute mean and std
        latent_mean = latent_flat.mean()
        latent_std = latent_flat.std()
        
        # Penalize mean ≠ 0 and std ≠ 1
        mean_loss = latent_mean.pow(2)  # L2 penalty on mean
        std_loss = (latent_std - 1.0).pow(2)  # L2 penalty on std deviation from 1
        
        # Combined loss
        loss = (mean_loss + std_loss) * self.weight
        
        return loss, {
            f"LatentStd_Mean": mean_loss.detach(),
            f"LatentStd_Std": std_loss.detach(),
            f"LatentStd_MeanVal": latent_mean.detach(),
            f"LatentStd_StdVal": latent_std.detach(),
        }


@register_loss
class CrossEntropyLoss(LossComponent):
    def _build(self):
        super()._build()
        self.ignore_index = self._init_kwargs.get("ignore_index", -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, preds, targets):
        pred = preds[self.key]
        tgt = targets[self.target_key]

        # Handle string labels FIRST (before checking tensor attributes)
        # Convert sample_type string ("room"/"scene") to class index for binary classification
        if isinstance(tgt, str) or (isinstance(tgt, list) and len(tgt) > 0 and isinstance(tgt[0], str)):
            from .loss_utils import sample_type_to_class_index
            tgt = sample_type_to_class_index(tgt, ignore_index=self.ignore_index)
            if isinstance(tgt, torch.Tensor):
                tgt = tgt.to(pred.device)
        # Convert RGB layout to segmentation mask (4D with 3 channels = RGB image)
        elif isinstance(tgt, torch.Tensor) and tgt.ndim == 4 and tgt.shape[1] == 3:
            from .loss_utils import create_seg_mask
            tgt = create_seg_mask(tgt, ignore_index=self.ignore_index).to(tgt.device)
        elif isinstance(tgt, torch.Tensor) and tgt.ndim == 4 and tgt.shape[1] == 1:
            tgt = tgt.squeeze(1)
        # Convert numeric labels (legacy room_id support) - defaults to room (0) if not found
        elif isinstance(tgt, torch.Tensor) and (tgt.ndim == 0 or tgt.ndim == 1):
            # For numeric values, assume 0 or "0000" means scene (1), others are room (0)
            tgt = (tgt == 0).long().to(pred.device)  # 0 -> 1 (scene), others -> 0 (room)
        elif not isinstance(tgt, torch.Tensor):
            # Handle non-tensor numeric values
            tgt = torch.tensor(1 if (isinstance(tgt, (int, float)) and (tgt == 0 or str(tgt) == "0000")) else 0, 
                             dtype=torch.long, device=pred.device)
        
        if tgt.dtype != torch.long:
            tgt = tgt.long()

        loss = self.criterion(pred, tgt) * self.weight
        logs = {f"CE_{self.key}": loss.detach()}
        return loss, logs


@register_loss
class PerceptualLoss(LossComponent):
    def _build(self):
        super()._build()
        from torchvision.models import vgg16
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self._device_set = False

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(self.vgg.parameters()).device if self._device_set else preds.get(self.key, targets.get(self.target_key, torch.zeros(1)))
            if isinstance(device, torch.Tensor):
                device = device.device
            return torch.tensor(0.0, device=device), {}
        
        # Ensure VGG is on the same device as inputs
        pred_tensor = preds[self.key]
        target_tensor = targets[self.target_key]
        device = pred_tensor.device
        
        if not self._device_set or next(self.vgg.parameters()).device != device:
            self.vgg = self.vgg.to(device)
            self._device_set = True
        
        f_pred = self.vgg(pred_tensor)
        f_tgt = self.vgg(target_tensor)
        loss = self.criterion(f_pred, f_tgt) * self.weight
        return loss, {f"Perceptual_{self.key}": loss.detach()}


# === Composite loss ==========================================================
@register_loss
class CompositeLoss(LossComponent):
    def _build(self):
        self.losses = nn.ModuleList()
        for sub_cfg in self._init_kwargs.get("losses", []):
            loss_type = sub_cfg["type"]
            if loss_type not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss type: {loss_type}")
            self.losses.append(LOSS_REGISTRY[loss_type].from_config(sub_cfg))

    def forward(self, preds, targets):
        total, logs = 0.0, {}
        for loss_fn in self.losses:
            loss, sublog = loss_fn(preds, targets)
            total += loss
            logs.update(sublog)
        return total, logs
