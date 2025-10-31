import torch
import torch.nn as nn
from ..components.base_component import BaseComponent


class LossComponent(BaseComponent):
    def _build(self):
        self.key = self._init_kwargs.get("key", None)
        self.target_key = self._init_kwargs.get("target", self.key)
        self.weight = self._init_kwargs.get("weight", 1.0)

    def forward(self, preds, targets):
        raise NotImplementedError

class L1Loss(LossComponent):
    def _build(self):
        super()._build()
        self.criterion = nn.L1Loss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            return torch.tensor(0.0, device=next(self.parameters()).device), {}
        loss = self.criterion(preds[self.key], targets[self.target_key]) * self.weight
        return loss, {f"L1_{self.key}": loss.detach()}

class MSELoss(LossComponent):
    def _build(self):
        super()._build()
        self.criterion = nn.MSELoss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            return torch.tensor(0.0, device=next(self.parameters()).device), {}
        loss = self.criterion(preds[self.key], targets[self.target_key]) * self.weight
        return loss, {f"MSE_{self.key}": loss.detach()}

class CompositeLoss(LossComponent):
    def _build(self):
        from .losses import LOSS_REGISTRY
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

class KLDLoss(LossComponent):
    def forward(self, preds, targets=None):
        mu = preds.get("mu")
        logvar = preds.get("logvar")
        if mu is None or logvar is None:
            return torch.tensor(0.0, device=next(self.parameters()).device), {}
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld * self.weight
        return kld, {"KLD": kld.detach()}

class CrossEntropyLoss(LossComponent):
    def _build(self):
        super()._build()
        self.ignore_index = self._init_kwargs.get("ignore_index", -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            return torch.tensor(0.0, device=next(self.parameters()).device), {}
        pred = preds[self.key]
        tgt = targets[self.target_key].long()
        loss = self.criterion(pred, tgt) * self.weight
        return loss, {f"CE_{self.key}": loss.detach()}
    
from torchvision.models import vgg16
class PerceptualLoss(LossComponent):
    def _build(self):
        super()._build()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            return torch.tensor(0.0, device=next(self.vgg.parameters()).device), {}
        f_pred = self.vgg(preds[self.key])
        f_tgt = self.vgg(targets[self.target_key])
        loss = self.criterion(f_pred, f_tgt) * self.weight
        return loss, {f"Perceptual_{self.key}": loss.detach()}
    


