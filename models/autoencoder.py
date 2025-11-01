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
        
        # Freeze components if requested in config
        if encoder_cfg.get("frozen", False):
            self.encoder.freeze()
        if decoder_cfg.get("frozen", False):
            self.decoder.freeze()

    def forward(self, x):
        """
        Forward pass. All components return dicts which are merged.
        
        Returns:
            Dictionary containing all keys from encoder and decoder outputs.
            Includes: "latent" (or "mu"/"logvar" for VAE), plus all decoder head outputs.
        """
        encoder_out = self.encoder(x)  # Dict: {"latent": z} or {"mu": mu, "logvar": logvar}
        decoder_out = self.decoder(encoder_out)  # Dict: {head_name: output}
        return {**encoder_out, **decoder_out}  # Merge all keys
    
    def decode(self, z_or_dict):
        """
        Decode from latent(s). Accepts dict or tensor for convenience.
        
        Args:
            z_or_dict: Either dict with "latent" key, or tensor (converted to dict)
        
        Returns:
            Dictionary with decoder head outputs
        """
        if isinstance(z_or_dict, dict):
            return self.decoder(z_or_dict)
        else:
            # Convert tensor to dict for convenience
            return self.decoder({"latent": z_or_dict})
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Returns:
            Dictionary: {"latent": z} or {"mu": mu, "logvar": logvar} for VAE
        """
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
