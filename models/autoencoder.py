import torch
import torch.nn as nn
from models.components.base_model import BaseModel
from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(BaseModel):
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
    
    def decode(self, z):
        return self.decoder(z)
    
    def encode(self, x):
        return self.encoder(x)

    def to_config(self):
        cfg = super().to_config()
        cfg["encoder"] = self.encoder.to_config()
        cfg["decoder"] = self.decoder.to_config()
        return cfg
    
    # -----------------------
    # Component-level checkpointing
    # -----------------------
    def save_encoder_checkpoint(self, path, include_config=True):
        """Save only the encoder as a separate checkpoint."""
        self.encoder.save_checkpoint(path, include_config=include_config)
    
    def save_decoder_checkpoint(self, path, include_config=True):
        """Save only the decoder as a separate checkpoint."""
        self.decoder.save_checkpoint(path, include_config=include_config)
    
    @classmethod
    def load_encoder_checkpoint(cls, path, map_location="cpu"):
        """Load only the encoder from a separate checkpoint."""
        return Encoder.load_checkpoint(path, map_location=map_location)
    
    @classmethod
    def load_decoder_checkpoint(cls, path, map_location="cpu"):
        """Load only the decoder from a separate checkpoint."""
        return Decoder.load_checkpoint(path, map_location=map_location)
    
    @classmethod
    def from_separate_checkpoints(cls, encoder_path, decoder_path, map_location="cpu"):
        """
        Build an Autoencoder from separate encoder and decoder checkpoints.
        
        Args:
            encoder_path: Path to encoder checkpoint
            decoder_path: Path to decoder checkpoint
            map_location: Device to load on
            
        Returns:
            Autoencoder instance with loaded encoder and decoder
        """
        encoder = cls.load_encoder_checkpoint(encoder_path, map_location)
        decoder = cls.load_decoder_checkpoint(decoder_path, map_location)
        
        # Create autoencoder with loaded components
        ae = cls(encoder=encoder.to_config(), decoder=decoder.to_config())
        ae.encoder = encoder
        ae.decoder = decoder
        return ae
