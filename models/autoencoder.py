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
            # Create projection layers - BaseComponent accepts **kwargs
            if isinstance(clip_projection_cfg, dict):
                self.clip_projections = CLIPProjections(**clip_projection_cfg)
            else:
                # If it's already an instance, use it directly
                self.clip_projections = clip_projection_cfg
            # Ensure projections are registered as a submodule (for parameter tracking)
            # This is already done by assigning to self.clip_projections, but make it explicit
            self.add_module('clip_projections', self.clip_projections)
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
        # Include CLIP projection config if it exists (now uses BaseComponent.to_config())
        if hasattr(self, 'clip_projections') and self.clip_projections is not None:
            cfg["clip_projection"] = self.clip_projections.to_config()
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
    
    # -----------------------
    # VAE Metadata Management
    # -----------------------
    
    @staticmethod
    def save_metadata(output_dir: Path, exp_name: str, latent_stats: dict):
        """
        Save VAE metadata file with latent statistics for scale_factor and clamp values.
        
        Args:
            output_dir: Output directory path
            exp_name: Experiment name
            latent_stats: Dictionary with latent statistics (from compute_latent_statistics)
        """
        if not latent_stats or len(latent_stats) == 0:
            return
        
        import json
        
        # Extract statistics
        latent_std = latent_stats.get("LatentStats_Std", None)
        latent_mean = latent_stats.get("LatentStats_Mean", 0.0)
        latent_min = latent_stats.get("LatentStats_Min", None)
        latent_max = latent_stats.get("LatentStats_Max", None)
        
        # Calculate scale_factor (1.0 / std to normalize to unit variance)
        scale_factor = 1.0 / latent_std if latent_std and latent_std > 0 else 1.0
        
        # Calculate clamp values (based on std: typically ±6σ covers 99.7% of data)
        # Or use actual min/max if available
        if latent_min is not None and latent_max is not None:
            # Use actual min/max with some margin
            clamp_min = latent_min - 0.5  # Small margin
            clamp_max = latent_max + 0.5  # Small margin
        else:
            # Fallback to std-based clamping (±6σ)
            clamp_min = -6.0
            clamp_max = 6.0
        
        # Extract per-channel stats if available
        per_channel_mean = None
        per_channel_std = None
        per_channel_min = None
        per_channel_max = None
        
        if "LatentStats_MeanPerCh" in latent_stats:
            per_channel_mean = json.loads(latent_stats["LatentStats_MeanPerCh"])
        if "LatentStats_StdPerCh" in latent_stats:
            per_channel_std = json.loads(latent_stats["LatentStats_StdPerCh"])
        if "LatentStats_MinPerCh" in latent_stats:
            per_channel_min = json.loads(latent_stats["LatentStats_MinPerCh"])
        if "LatentStats_MaxPerCh" in latent_stats:
            per_channel_max = json.loads(latent_stats["LatentStats_MaxPerCh"])
        
        # Build metadata dictionary
        metadata = {
            "experiment_name": exp_name,
            "latent_statistics": {
                "global": {
                    "mean": float(latent_mean),
                    "std": float(latent_std) if latent_std else None,
                    "min": float(latent_min) if latent_min is not None else None,
                    "max": float(latent_max) if latent_max is not None else None,
                },
                "per_channel": {
                    "mean": per_channel_mean,
                    "std": per_channel_std,
                    "min": per_channel_min,
                    "max": per_channel_max,
                } if per_channel_mean is not None else None,
            },
            "recommended_values": {
                "scale_factor": float(scale_factor),
                "latent_clamp_min": float(clamp_min),
                "latent_clamp_max": float(clamp_max),
            },
            "notes": {
                "scale_factor": "1.0 / std, normalizes latents to unit variance",
                "latent_clamp_min": "Minimum value for clamping latents during diffusion",
                "latent_clamp_max": "Maximum value for clamping latents during diffusion",
            }
        }
        
        # Save metadata file
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / f"{exp_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_metadata(checkpoint_path: Path):
        """
        Load VAE metadata JSON file and extract recommended values.
        
        Args:
            checkpoint_path: Path to autoencoder checkpoint
        
        Returns:
            dict with 'scale_factor', 'latent_clamp_min', 'latent_clamp_max', or None if not found
        """
        import json
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return None
        
        # Metadata file is typically in the parent directory of checkpoints/
        # e.g., /work3/.../vae_clip/checkpoints/vae_clip_checkpoint_best.pt
        # -> /work3/.../vae_clip/vae_clip_metadata.json
        checkpoint_dir = checkpoint_path.parent  # checkpoints/
        vae_dir = checkpoint_dir.parent  # vae_clip/
        
        # Try to find metadata file - could be named {exp_name}_metadata.json
        # Look for any *_metadata.json in the VAE directory
        metadata_files = list(vae_dir.glob("*_metadata.json"))
        
        if not metadata_files:
            return None
        
        # Use the first metadata file found (or could match by experiment name)
        metadata_path = metadata_files[0]
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            recommended = metadata.get("recommended_values", {})
            if recommended:
                return {
                    "scale_factor": recommended.get("scale_factor"),
                    "latent_clamp_min": recommended.get("latent_clamp_min"),
                    "latent_clamp_max": recommended.get("latent_clamp_max")
                }
        except Exception as e:
            print(f"Warning: Failed to load VAE metadata from {metadata_path}: {e}")
        
        return None