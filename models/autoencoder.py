import torch
import torch.nn as nn
from pathlib import Path
from models.components.base_model import BaseModel
from models.utils import reparameterize
from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(BaseModel):
    def _build(self):
        encoder_cfg = self._init_kwargs.get("encoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)

        if encoder_cfg is None or decoder_cfg is None:
            raise ValueError("Autoencoder requires both 'encoder' and 'decoder' configs.")

        # Use unified component registry for all components
        self.add_component("encoder", self.create_component("encoder", default_type="Encoder"))
        self.add_component("decoder", self.create_component("decoder", default_type="Decoder"))
        
        # Optional CLIP projection layers - only if explicitly provided in config
        # This allows training projections jointly with the VAE, but must be explicit
        clip_projection_cfg = self._init_kwargs.get("clip_projection", None)
        if clip_projection_cfg is not None:
            self._setup_projection("clip_projection", default_type="CLIPProjections")
        
        # Write model statistics if save_path is available
        if hasattr(self, 'save_path') and self.save_path:
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
        cfg = self._components_to_config(cfg)
        # Don't include clip_projection in config - it's a separate component
        cfg.pop("clip_projection", None)
        return cfg
    
    def save_checkpoint(self, path, include_config=True, exclude_projections=True, **extra_state):
        """
        Save autoencoder checkpoint, excluding projection components.
        
        Projections are separate components that should be saved separately by the trainer.
        This keeps the autoencoder checkpoint focused on encoder/decoder only.
        
        Args:
            path: Path to save checkpoint
            include_config: Whether to include model config
            exclude_projections: If True, exclude any projection components from state_dict
            **extra_state: Additional state to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get state dict, excluding projection components
        state_dict = self.state_dict()
        
        if exclude_projections:
            # Filter out any projection component keys (clip_projection, projection, etc.)
            filtered_state_dict = {}
            for key, value in state_dict.items():
                # Skip keys that start with common projection component names
                if not any(key.startswith(proj_prefix + ".") for proj_prefix in 
                          ["clip_projection", "clip_projections", "projection", "projections"]):
                    filtered_state_dict[key] = value
            state_dict = filtered_state_dict
        
        payload = {"state_dict": state_dict}
        if include_config:
            payload["config"] = self.to_config()
        payload.update(extra_state)
        torch.save(payload, path)
    
    @classmethod
    def from_component_checkpoints(cls, component_paths, map_location="cpu"):
        """
        Build an Autoencoder from component checkpoints.
        
        Args:
            component_paths: Dict mapping component names to checkpoint paths
                           (e.g., {"encoder": "path/to/encoder.pt", "decoder": "path/to/decoder.pt"})
            map_location: Device to load on
            
        Returns:
            Autoencoder instance with loaded components
        """
        # Load components
        loaded_components = {}
        for name, path in component_paths.items():
            loaded_components[name] = cls.load_component_checkpoint(name, path, map_location)
        
        # Build config from loaded components
        config = {name: comp.to_config() for name, comp in loaded_components.items()}
        
        # Create model with config
        model = cls(**config)
        
        # Replace with loaded components (in case of any state differences)
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
    
    VAE uses a variational encoder that outputs mu and logvar instead of deterministic latents.
    The encoder config will automatically have variational=True set.
    """
    
    def _build(self):
        encoder_cfg = self._init_kwargs.get("encoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)

        if encoder_cfg is None or decoder_cfg is None:
            raise ValueError("VAE requires both 'encoder' and 'decoder' configs.")

        # Ensure encoder and decoder configs use VAE types
        self._ensure_component_type("encoder", "VAEEncoder")
        self._ensure_component_type("decoder", "VAEDecoder")

        # Use unified component registry for all components
        self.add_component("encoder", self.create_component("encoder", default_type="VAEEncoder"))
        self.add_component("decoder", self.create_component("decoder", default_type="VAEDecoder"))
        
        # Optional CLIP projection layers - only if explicitly provided in config
        # This allows training projections jointly with the VAE, but must be explicit
        clip_projection_cfg = self._init_kwargs.get("clip_projection", None)
        if clip_projection_cfg is not None:
            self._setup_projection("clip_projection", default_type="CLIPProjections")
        
        # Verify encoder and decoder are correct types
        from .encoder import VAEEncoder
        from .decoder import VAEDecoder
        self._validate_component_type(self.encoder, VAEEncoder, "encoder")
        self._validate_component_type(self.decoder, VAEDecoder, "decoder")
        
        # Write model statistics if save_path is available
        if hasattr(self, 'save_path') and self.save_path:
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
            deterministic: If True, use mu as latent (no sampling). If False, sample from distribution.
        
        Returns:
            Dictionary with sampled latent and encoder outputs
        """
        encoder_out = self.encode(x)
        mu = encoder_out["mu"]
        logvar = encoder_out["logvar"]
        
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        
        return {"latent": z, **encoder_out}
    