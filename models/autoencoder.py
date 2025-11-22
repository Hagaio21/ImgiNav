import torch
import torch.nn as nn
from pathlib import Path
from models.components.base_model import BaseModel
from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(BaseModel):
    def _build(self):
        encoder_cfg = self._init_kwargs.get("encoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)
        clip_projection_cfg = self._init_kwargs.get("clip_projection", None)

        if encoder_cfg is None or decoder_cfg is None:
            raise ValueError("Autoencoder requires both 'encoder' and 'decoder' configs.")

        self.encoder = Encoder.from_config(encoder_cfg)
        self.decoder = Decoder.from_config(decoder_cfg)
        
        # Optional CLIP projection layers (for joint embedding space training)
        self.clip_projections = None
        if clip_projection_cfg is not None:
            from models.losses.clip_loss import CLIPProjections
            # Create projection layers (these will be trainable)
            projection_dim = clip_projection_cfg.get("projection_dim", 256)
            text_dim = clip_projection_cfg.get("text_dim", 384)
            pov_dim = clip_projection_cfg.get("pov_dim", 512)
            latent_dim = clip_projection_cfg.get("latent_dim", None)
            self.clip_projections = CLIPProjections(
                projection_dim=projection_dim,
                text_dim=text_dim,
                pov_dim=pov_dim,
                latent_dim=latent_dim
            )
            # Mark that we're using CLIP projections
            self._has_clip_projections = True
        else:
            self._has_clip_projections = False

        if encoder_cfg.get("frozen", False):
            self.encoder.freeze()
        if decoder_cfg.get("frozen", False):
            self.decoder.freeze()
        
        # Write model statistics if save_path is available
        self._write_model_statistics()

    def forward(self, x):

        encoder_out = self.encoder(x)  # Dict: {"latent": z} or {"mu": mu, "logvar": logvar}
        decoder_out = self.decoder(encoder_out)  # Dict: {head_name: output}
        return {**encoder_out, **decoder_out}  # Merge all keys
    
    def decode(self, z_or_dict):

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
    
    def _write_model_statistics(self):
        """Write model parameter statistics to Statistics.txt file."""
        try:
            # Get save path from experiment config if available
            save_path = self._init_kwargs.get("save_path", None)
            if save_path is None:
                # Try to get from experiment config in parent kwargs
                exp_cfg = self._init_kwargs.get("experiment", {})
                save_path = exp_cfg.get("save_path", None)
            
            if save_path is None:
                return  # No save path available, skip writing
            
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            stats_file = save_path / "Statistics.txt"
            
            # Count parameters
            encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            encoder_total = sum(p.numel() for p in self.encoder.parameters())
            encoder_frozen = encoder_total - encoder_trainable
            
            decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            decoder_total = sum(p.numel() for p in self.decoder.parameters())
            decoder_frozen = decoder_total - decoder_trainable
            
            total_trainable = encoder_trainable + decoder_trainable
            total_params = encoder_total + decoder_total
            
            # Write statistics
            with open(stats_file, 'w') as f:
                f.write("Model Statistics\n")
                f.write("=" * 60 + "\n\n")
                f.write("Encoder Parameters:\n")
                f.write(f"  Trainable: {encoder_trainable:,} ({encoder_trainable / 1_000_000:.2f}M)\n")
                f.write(f"  Total: {encoder_total:,} ({encoder_total / 1_000_000:.2f}M)\n")
                f.write(f"  Frozen: {encoder_frozen:,} ({encoder_frozen / 1_000_000:.2f}M)\n")
                f.write(f"\nDecoder Parameters:\n")
                f.write(f"  Trainable: {decoder_trainable:,} ({decoder_trainable / 1_000_000:.2f}M)\n")
                f.write(f"  Total: {decoder_total:,} ({decoder_total / 1_000_000:.2f}M)\n")
                f.write(f"  Frozen: {decoder_frozen:,} ({decoder_frozen / 1_000_000:.2f}M)\n")
                f.write(f"\nTotal Trainable Parameters: {total_trainable:,} ({total_trainable / 1_000_000:.2f}M)\n")
                f.write(f"Total Parameters: {total_params:,} ({total_params / 1_000_000:.2f}M)\n")
        except Exception as e:
            # Don't fail model building if statistics writing fails
            import warnings
            warnings.warn(f"Failed to write model statistics: {e}")
