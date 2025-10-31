import torch
import torch.nn as nn
from models.components.base_component import BaseComponent
from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(BaseComponent):
    def _build(self):
        encoder_cfg = self._init_kwargs.get("encoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)

        if encoder_cfg is None or decoder_cfg is None:
            raise ValueError("Autoencoder requires both 'encoder' and 'decoder' configs.")

        self.encoder = Encoder.from_config(encoder_cfg)
        self.decoder = Decoder.from_config(decoder_cfg)

    def forward(self, x):
        z = self.encoder(x)
        outputs = self.decoder(z)
        return {"latent": z, **outputs}

    def to_config(self):
        cfg = super().to_config()
        cfg["encoder"] = self.encoder.to_config()
        cfg["decoder"] = self.decoder.to_config()
        return cfg
