"""
Autoencoder and VAE models.

Consolidation notes:
- Uses `_extract_tensor_from_input()` from encoder module to avoid duplication
- Simplified CLIP projection handling with clearer logic
- Removed redundant config checks
- Uses plain dicts for inter-component communication
"""

import torch
import torch.nn as nn
from pathlib import Path
from models.components.base_model import BaseModel
from models.utils import reparameterize
from .encoder import Encoder, _extract_tensor_from_input
from .decoder import Decoder


class Autoencoder(BaseModel):
    """
    Standard autoencoder with encoder and decoder.
    
    Config:
        encoder: Encoder configuration dict
        decoder: Decoder configuration dict
        clip_projection: Optional CLIP projection configuration
    """
    
    def _build(self):
        encoder_cfg = self._init_kwargs.get("encoder")
        decoder_cfg = self._init_kwargs.get("decoder")

        if encoder_cfg is None or decoder_cfg is None:
            raise ValueError("Autoencoder requires both 'encoder' and 'decoder' configs.")

        # Build encoder and decoder using unified component registry
        self.add_component("encoder", self.create_component("encoder", default_type="Encoder"))
        self.add_component("decoder", self.create_component("decoder", default_type="Decoder"))
        
        # Optional CLIP projection - only if config is non-empty
        clip_projection_cfg = self._init_kwargs.get("clip_projection")
        if clip_projection_cfg and isinstance(clip_projection_cfg, dict) and len(clip_projection_cfg) > 0:
            self._setup_projection("clip_projection", default_type="CLIPProjections")
        
        # Write model statistics if save_path is available
        if getattr(self, 'save_path', None):
            self._write_model_statistics()

    def forward(self, x):
        """
        Forward pass through encoder and decoder.
        
        Args:
            x: Input tensor [B, C, H, W] or dict containing input
        
        Returns:
            Dict with encoder and decoder outputs merged
        """
        # Extract tensor from dict if needed
        x = _extract_tensor_from_input(x)
        
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        
        # Merge encoder and decoder outputs
        merged = dict(encoder_out) if isinstance(encoder_out, dict) else {"latent": encoder_out}
        if isinstance(decoder_out, dict):
            merged.update(decoder_out)
        return merged
    
    def decode(self, z_or_dict):
        """Decode latent representation to output."""
        if isinstance(z_or_dict, dict):
            return self.decoder(z_or_dict)
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
        cfg = self._components_to_config(cfg)
        cfg.pop("clip_projection", None)  # Exclude clip_projection from config
        return cfg
    
    def save_checkpoint(self, path, include_config=True, exclude_projections=True, use_compression=False, **extra_state):
        """
        Save autoencoder checkpoint, excluding projection components.
        
        Args:
            path: Path to save checkpoint
            include_config: Whether to include model config
            exclude_projections: If True, exclude any projection components from state_dict
            use_compression: If True, use gzip compression
            **extra_state: Additional state to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = self.state_dict()
        
        if exclude_projections:
            projection_prefixes = ["clip_projection.", "clip_projections.", "projection.", "projections."]
            state_dict = {
                k: v for k, v in state_dict.items()
                if not any(k.startswith(prefix) for prefix in projection_prefixes)
            }
        
        payload = {"state_dict": state_dict}
        if include_config:
            payload["config"] = self.to_config()
        payload.update(extra_state)
        
        if use_compression:
            import gzip
            import pickle
            with gzip.open(path, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            torch.save(payload, path)
    
    @classmethod
    def from_component_checkpoints(cls, component_paths, map_location="cpu"):
        """
        Build an Autoencoder from component checkpoints.
        
        Args:
            component_paths: Dict mapping component names to checkpoint paths
            map_location: Device to load on
            
        Returns:
            Autoencoder instance with loaded components
        """
        loaded_components = {}
        for name, path in component_paths.items():
            loaded_components[name] = cls.load_component_checkpoint(name, path, map_location)
        
        config = {name: comp.to_config() for name, comp in loaded_components.items()}
        model = cls(**config)
        
        for name, component in loaded_components.items():
            setattr(model, name, component)
            model.add_component(name, component)
        
        return model
    
    def _get_component_statistics(self):
        """Get parameter statistics for encoder and decoder components."""
        stats = {}
        
        encoder_stats = self._get_module_statistics(self.encoder, label="Encoder")
        if encoder_stats:
            stats["encoder"] = encoder_stats
        
        decoder_stats = self._get_module_statistics(self.decoder, label="Decoder")
        if decoder_stats:
            stats["decoder"] = decoder_stats
        
        return stats


class VAE(Autoencoder):
    """
    Variational Autoencoder - subclass of Autoencoder.
    
    Uses variational encoder that outputs mu and logvar instead of deterministic latents.
    """
    
    def _build(self):
        encoder_cfg = self._init_kwargs.get("encoder")
        decoder_cfg = self._init_kwargs.get("decoder")

        if encoder_cfg is None or decoder_cfg is None:
            raise ValueError("VAE requires both 'encoder' and 'decoder' configs.")

        # Ensure encoder and decoder use VAE types
        self._ensure_component_type("encoder", "VAEEncoder")
        self._ensure_component_type("decoder", "VAEDecoder")

        # Build components
        self.add_component("encoder", self.create_component("encoder", default_type="VAEEncoder"))
        self.add_component("decoder", self.create_component("decoder", default_type="VAEDecoder"))
        
        # Optional CLIP projection
        clip_projection_cfg = self._init_kwargs.get("clip_projection")
        if clip_projection_cfg:
            self._setup_projection("clip_projection", default_type="CLIPProjections")
        
        # Verify correct types
        from .encoder import VAEEncoder
        from .decoder import VAEDecoder
        self._validate_component_type(self.encoder, VAEEncoder, "encoder")
        self._validate_component_type(self.decoder, VAEDecoder, "decoder")
        
        # Write model statistics if save_path is available
        if getattr(self, 'save_path', None):
            self._write_model_statistics()
    
    def encode(self, x):
        """
        Encode input to variational latent representation.
        
        Returns:
            Dictionary: {"mu": mu, "logvar": logvar, "latent_features": features}
        """
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean tensor [B, C, H, W]
            logvar: Log variance tensor [B, C, H, W]
        
        Returns:
            Latent tensor z [B, C, H, W]
        """
        return reparameterize(mu, logvar)
    
    def sample(self, x, deterministic=False):
        """
        Sample from VAE latent distribution.
        
        Args:
            x: Input tensor [B, C, H, W]
            deterministic: If True, use mu as latent (no sampling)
        
        Returns:
            Dictionary with sampled latent and encoder outputs
        """
        encoder_out = self.encode(x)
        mu = encoder_out["mu"]
        logvar = encoder_out["logvar"]
        
        z = mu if deterministic else self.reparameterize(mu, logvar)
        
        return {"latent": z, **encoder_out}
