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
class CrossEntropyLoss(LossComponent):
    def _build(self):
        super()._build()
        self.ignore_index = self._init_kwargs.get("ignore_index", -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, preds, targets):
        pred = preds[self.key]
        tgt = targets[self.target_key]

        # Convert RGB layout to segmentation mask (4D with 3 channels = RGB image)
        if tgt.ndim == 4 and tgt.shape[1] == 3:
            from .loss_utils import create_seg_mask
            tgt = create_seg_mask(tgt, ignore_index=self.ignore_index).to(tgt.device)
        elif tgt.ndim == 4 and tgt.shape[1] == 1:
            tgt = tgt.squeeze(1)
        # Convert room_id label to class index for classification
        # Labels are scalars (0D) or 1D tensors (batched) - always convert to class index
        elif tgt.ndim == 0 or tgt.ndim == 1:
            from .loss_utils import room_id_to_class_index
            tgt = room_id_to_class_index(tgt, ignore_index=self.ignore_index).to(pred.device)
        
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

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(self.vgg.parameters()).device
            return torch.tensor(0.0, device=device), {}
        f_pred = self.vgg(preds[self.key])
        f_tgt = self.vgg(targets[self.target_key])
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
