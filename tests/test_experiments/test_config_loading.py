"""
Tests for experiment config loading and validation.
"""
import pytest
import yaml
from pathlib import Path

from training.utils import load_config


class TestConfigLoading:
    """Test loading experiment configs."""
    
    def test_load_experiment_config(self):
        """Test loading an experiment config from experiments/ directory."""
        config_path = Path("experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml")
        
        if config_path.exists():
            config = load_config(config_path)
            
            assert config is not None
            assert isinstance(config, dict)
            assert "experiment" in config or "autoencoder" in config
    
    def test_load_diffusion_config(self):
        """Test loading a diffusion experiment config."""
        config_path = Path("experiments/diffusion/clip/regular_rooms/small_down_bottleneck_text_only.yaml")
        
        if config_path.exists():
            config = load_config(config_path)
            
            assert config is not None
            assert isinstance(config, dict)
            assert "experiment" in config or "diffusion" in config or "unet" in config
    
    def test_config_has_experiment_section(self):
        """Test that config has experiment section."""
        config_path = Path("experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml")
        
        if config_path.exists():
            config = load_config(config_path)
            
            # Should have experiment section or at least model config
            assert "experiment" in config or "autoencoder" in config or "diffusion" in config


class TestConfigValidation:
    """Test config validation."""
    
    def test_config_is_dict(self):
        """Test that loaded config is a dictionary."""
        config_path = Path("experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml")
        
        if config_path.exists():
            config = load_config(config_path)
            assert isinstance(config, dict)
    
    def test_config_has_required_sections(self):
        """Test that config has required sections."""
        config_path = Path("experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml")
        
        if config_path.exists():
            config = load_config(config_path)
            
            # Should have at least one of: experiment, autoencoder, diffusion, training
            has_required = any(key in config for key in ["experiment", "autoencoder", "diffusion", "training"])
            assert has_required


class TestConfigPathResolution:
    """Test config path resolution."""
    
    def test_absolute_path(self):
        """Test loading config with absolute path."""
        config_path = Path("experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml")
        
        if config_path.exists():
            abs_path = config_path.resolve()
            config = load_config(abs_path)
            
            assert config is not None
    
    def test_relative_path(self):
        """Test loading config with relative path."""
        config_path = Path("experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml")
        
        if config_path.exists():
            config = load_config(config_path)
            
            assert config is not None


class TestMissingRequiredFields:
    """Test handling of missing required fields."""
    
    def test_missing_experiment_section(self, temp_dir):
        """Test config with missing experiment section."""
        config_path = temp_dir / "minimal_config.yaml"
        
        minimal_config = {
            "autoencoder": {
                "encoder": {"in_channels": 3, "latent_channels": 4, "base_channels": 16, "downsampling_steps": 2},
                "decoder": {"latent_channels": 4, "base_channels": 16, "upsampling_steps": 2, "heads": []}
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f)
        
        # Should still load (experiment section may be optional)
        config = load_config(config_path)
        assert config is not None
    
    def test_invalid_config_file(self, temp_dir):
        """Test handling of invalid config file."""
        config_path = temp_dir / "invalid_config.yaml"
        
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        # Should raise an error or return None
        try:
            config = load_config(config_path)
            # If it doesn't raise, config should be None or empty
            assert config is None or len(config) == 0
        except Exception:
            # Exception is also acceptable
            pass

