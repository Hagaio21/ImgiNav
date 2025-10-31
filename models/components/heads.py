import torch
import torch.nn as nn
from .base_component import BaseComponent


class DecoderHead(BaseComponent):
    def _build(self):
        in_ch = self._init_kwargs.get("in_channels", 64)
        out_ch = self._init_kwargs.get("out_channels", 3)
        activation = getattr(nn, self._init_kwargs.get("activation", "SiLU"))()
        norm_groups = self._init_kwargs.get("norm_groups", 8)
        final_act = self._init_kwargs.get("final_activation", None)

        layers = [
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(norm_groups, in_ch),
            activation,
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        ]

        if final_act == "tanh":
            layers.append(nn.Tanh())
        elif final_act == "sigmoid":
            layers.append(nn.Sigmoid())
        elif final_act == "softmax":
            layers.append(nn.Softmax(dim=1))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class RGBHead(DecoderHead):
    def _build(self):
        self._init_kwargs.setdefault("out_channels", 3)
        self._init_kwargs.setdefault("final_activation", "tanh")
        super()._build()


class SegmentationHead(DecoderHead):
    def _build(self):
        num_classes = self._init_kwargs.get("num_classes", 21)
        self._init_kwargs["out_channels"] = num_classes
        self._init_kwargs["final_activation"] = "softmax"
        super()._build()


class ClassificationHead(BaseComponent):
    def _build(self):
        in_ch = self._init_kwargs.get("in_channels", 64)
        num_classes = self._init_kwargs.get("num_classes", 1000)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


HEAD_REGISTRY = {
    "DecoderHead": DecoderHead,
    "RGBHead": RGBHead,
    "SegmentationHead": SegmentationHead,
    "ClassificationHead": ClassificationHead,
}
