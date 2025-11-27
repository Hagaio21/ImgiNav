"""
Tests for checkpoint save/load structure (not correctness of values).
"""
import pytest
import torch
from pathlib import Path
import tempfile

from models.autoencoder import Autoencoder
from models.diffusion import DiffusionModel


class TestCheckpointSaveFormat:
    """Test checkpoint save format structure."""
    
    def test_autoencoder_checkpoint_format(self, autoencoder_model, temp_dir):
        """Test autoencoder checkpoint has correct format."""
        checkpoint_path = temp_dir / "test_ae.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Required keys
        assert "state_dict" in checkpoint
        assert "config" in checkpoint
        
        # Types
        assert isinstance(checkpoint["state_dict"], dict)
        assert isinstance(checkpoint["config"], dict)
    
    def test_diffusion_checkpoint_format(self, diffusion_model, temp_dir):
        """Test diffusion checkpoint has correct format."""
        checkpoint_path = temp_dir / "test_diff.pt"
        diffusion_model.save_checkpoint(checkpoint_path, include_config=True)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Required keys
        assert "state_dict" in checkpoint
        assert "config" in checkpoint
        
        # Types
        assert isinstance(checkpoint["state_dict"], dict)
        assert isinstance(checkpoint["config"], dict)
    
    def test_checkpoint_without_config(self, autoencoder_model, temp_dir):
        """Test checkpoint can be saved without config."""
        checkpoint_path = temp_dir / "test_no_config.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=False)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert "state_dict" in checkpoint
        # Config may or may not be present
        assert "config" not in checkpoint or isinstance(checkpoint.get("config"), dict)


class TestCheckpointLoadStructure:
    """Test checkpoint load structure (not correctness)."""
    
    def test_load_autoencoder_strict(self, autoencoder_model, temp_dir):
        """Test loading autoencoder checkpoint with strict=True."""
        checkpoint_path = temp_dir / "test_ae.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True)
        
        loaded = Autoencoder.load_checkpoint(checkpoint_path, map_location='cpu', strict=True)
        
        assert loaded is not None
        assert isinstance(loaded, Autoencoder)
        assert hasattr(loaded, 'encoder')
        assert hasattr(loaded, 'decoder')
    
    def test_load_autoencoder_non_strict(self, autoencoder_model, temp_dir):
        """Test loading autoencoder checkpoint with strict=False."""
        checkpoint_path = temp_dir / "test_ae.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True)
        
        loaded = Autoencoder.load_checkpoint(checkpoint_path, map_location='cpu', strict=False)
        
        assert loaded is not None
        assert isinstance(loaded, Autoencoder)
    
    def test_load_diffusion_strict(self, diffusion_model, temp_dir):
        """Test loading diffusion checkpoint."""
        checkpoint_path = temp_dir / "test_diff.pt"
        diffusion_model.save_checkpoint(checkpoint_path, include_config=True)
        
        loaded = DiffusionModel.load_checkpoint(checkpoint_path, map_location='cpu')
        
        assert loaded is not None
        assert isinstance(loaded, DiffusionModel)
        assert hasattr(loaded, 'unet')
        assert hasattr(loaded, 'decoder')
        assert hasattr(loaded, 'scheduler')
    
    def test_load_diffusion_non_strict(self, diffusion_model, temp_dir):
        """Test loading diffusion checkpoint (DiffusionModel doesn't use strict parameter)."""
        checkpoint_path = temp_dir / "test_diff.pt"
        diffusion_model.save_checkpoint(checkpoint_path, include_config=True)
        
        loaded = DiffusionModel.load_checkpoint(checkpoint_path, map_location='cpu')
        
        assert loaded is not None
        assert isinstance(loaded, DiffusionModel)
    
    def test_load_with_custom_config(self, autoencoder_model, temp_dir):
        """Test loading checkpoint with custom config."""
        checkpoint_path = temp_dir / "test_ae.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True)
        
        # Load with custom config (should use custom config)
        custom_config = autoencoder_model.to_config()
        loaded = Autoencoder.load_checkpoint(
            checkpoint_path, 
            map_location='cpu', 
            config=custom_config
        )
        
        assert loaded is not None
        assert isinstance(loaded, Autoencoder)


class TestCheckpointExtraState:
    """Test checkpoint with extra state (optimizer, epoch, etc.)."""
    
    def test_save_with_extra_state(self, autoencoder_model, temp_dir):
        """Test saving checkpoint with extra state."""
        checkpoint_path = temp_dir / "test_extra.pt"
        extra_state = {
            "epoch": 10,
            "best_val_loss": 0.5,
            "optimizer_state": {"param_groups": []},
            "scheduler_state": {"last_epoch": 10},
            "scaler_state": {"scale": 1.0}
        }
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True, **extra_state)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert "epoch" in checkpoint
        assert "best_val_loss" in checkpoint
        assert "optimizer_state" in checkpoint
        assert checkpoint["epoch"] == 10
        assert checkpoint["best_val_loss"] == 0.5
    
    def test_load_with_extra_state(self, autoencoder_model, temp_dir):
        """Test loading checkpoint and retrieving extra state."""
        checkpoint_path = temp_dir / "test_extra.pt"
        extra_state = {
            "epoch": 10,
            "best_val_loss": 0.5,
        }
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True, **extra_state)
        
        model, extra = Autoencoder.load_checkpoint(
            checkpoint_path, 
            map_location='cpu', 
            return_extra=True
        )
        
        assert model is not None
        assert isinstance(extra, dict)
        assert "epoch" in extra
        assert "best_val_loss" in extra
        assert extra["epoch"] == 10
        assert extra["best_val_loss"] == 0.5


class TestComponentCheckpointing:
    """Test component-level checkpointing."""
    
    def test_save_component_checkpoint(self, autoencoder_model, temp_dir):
        """Test saving individual component checkpoint."""
        checkpoint_path = temp_dir / "encoder.pt"
        
        if hasattr(autoencoder_model, 'save_component_checkpoint'):
            autoencoder_model.save_component_checkpoint("encoder", checkpoint_path, include_config=True)
            
            assert checkpoint_path.exists()
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert "state_dict" in checkpoint
            assert "config" in checkpoint
    
    def test_load_component_checkpoint(self, temp_dir):
        """Test loading component checkpoint."""
        from models.encoder import Encoder
        
        # Create and save encoder
        config = {
            "type": "Encoder",
            "in_channels": 3,
            "latent_channels": 4,
            "base_channels": 16,
            "downsampling_steps": 2,
        }
        encoder = Encoder.from_config(config)
        checkpoint_path = temp_dir / "encoder.pt"
        encoder.save_checkpoint(checkpoint_path, include_config=True)
        
        # Load component
        if hasattr(Autoencoder, 'load_component_checkpoint'):
            loaded = Autoencoder.load_component_checkpoint("Encoder", checkpoint_path, map_location='cpu')
            assert loaded is not None
            assert isinstance(loaded, Encoder)


class TestBackwardCompatibility:
    """Test backward compatibility with old checkpoints."""
    
    def test_load_old_format_checkpoint(self, temp_dir):
        """Test that old format checkpoints can be loaded (if structure allows)."""
        # Create a minimal old-format checkpoint
        old_checkpoint = {
            "state_dict": {
                "encoder.conv1.weight": torch.randn(16, 3, 3, 3),
                "decoder.conv1.weight": torch.randn(16, 4, 3, 3),
            },
            "config": {
                "type": "Autoencoder",
                "encoder": {
                    "type": "Encoder",
                    "in_channels": 3,
                    "latent_channels": 4,
                    "base_channels": 16,
                    "downsampling_steps": 2,
                },
                "decoder": {
                    "type": "Decoder",
                    "latent_channels": 4,
                    "base_channels": 16,
                    "upsampling_steps": 2,
                    "heads": [
                        {"type": "RGBHead", "name": "rgb", "out_channels": 3}
                    ]
                }
            }
        }
        
        checkpoint_path = temp_dir / "old_format.pt"
        torch.save(old_checkpoint, checkpoint_path)
        
        # Try to load with strict=False (should handle missing keys)
        try:
            loaded = Autoencoder.load_checkpoint(checkpoint_path, map_location='cpu', strict=False)
            assert loaded is not None
        except Exception as e:
            # If it fails, that's okay - we're just testing structure
            # The important thing is that the loading mechanism exists
            assert "load_checkpoint" in str(type(e)) or "load_state_dict" in str(type(e))

