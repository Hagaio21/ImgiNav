"""
Tests for Autoencoder model - focusing on building, class interactions, and component graphs.
"""
import pytest
import torch
from pathlib import Path
import tempfile
import shutil

from models.autoencoder import Autoencoder


class TestAutoencoderBuilding:
    """Test autoencoder model building from configs."""
    
    def test_build_from_config(self, test_autoencoder_config):
        """Test that autoencoder can be built from config."""
        ae_cfg = test_autoencoder_config.get("autoencoder", {})
        model = Autoencoder.from_config(ae_cfg)
        
        assert model is not None
        assert isinstance(model, Autoencoder)
    
    def test_components_exist(self, autoencoder_model):
        """Test that required components exist."""
        assert hasattr(autoencoder_model, 'encoder')
        assert hasattr(autoencoder_model, 'decoder')
        assert autoencoder_model.encoder is not None
        assert autoencoder_model.decoder is not None
    
    def test_component_types(self, autoencoder_model):
        """Test that components are correct types."""
        from models.encoder import Encoder
        from models.decoder import Decoder
        
        assert isinstance(autoencoder_model.encoder, Encoder)
        assert isinstance(autoencoder_model.decoder, Decoder)
    
    def test_clip_projection_optional(self, test_vae_config):
        """Test that CLIP projection is optional but accessible if configured."""
        ae_cfg = test_vae_config.get("autoencoder", {})
        model = Autoencoder.from_config(ae_cfg)
        
        # CLIP projection should exist if configured
        if "clip_projection" in ae_cfg:
            assert hasattr(model, 'clip_projections') or hasattr(model, 'clip_projection')
            # At least one should exist
            assert (hasattr(model, 'clip_projections') and model.clip_projections is not None) or \
                   (hasattr(model, 'clip_projection') and model.clip_projection is not None)
    
    def test_to_config(self, autoencoder_model):
        """Test that to_config() returns a valid config dict."""
        config = autoencoder_model.to_config()
        
        assert isinstance(config, dict)
        assert "type" in config or "encoder" in config or "decoder" in config
        # Should have encoder and decoder configs
        assert "encoder" in config or "decoder" in config


class TestAutoencoderComponentRelationships:
    """Test component relationships and interactions."""
    
    def test_encoder_decoder_relationship(self, autoencoder_model):
        """Test that encoder and decoder are properly connected."""
        # Both should exist
        assert hasattr(autoencoder_model, 'encoder')
        assert hasattr(autoencoder_model, 'decoder')
        
        # Encoder should output latents that decoder can accept
        # (We're not testing correctness, just that the relationship exists)
        assert autoencoder_model.encoder is not None
        assert autoencoder_model.decoder is not None
    
    def test_component_tracking(self, autoencoder_model):
        """Test that components are tracked in _component_names."""
        assert hasattr(autoencoder_model, '_component_names')
        assert isinstance(autoencoder_model._component_names, dict)
        
        # Encoder and decoder should be tracked
        components = list(autoencoder_model._component_names.values())
        assert 'encoder' in components or autoencoder_model.encoder in autoencoder_model._component_names
        assert 'decoder' in components or autoencoder_model.decoder in autoencoder_model._component_names


class TestAutoencoderComponentGraphs:
    """Test component graph generation."""
    
    def test_generate_component_graph_text(self, autoencoder_model):
        """Test that component graph can be generated in text format."""
        graph_text = autoencoder_model.generate_component_graph(format='text')
        
        assert isinstance(graph_text, str)
        assert len(graph_text) > 0
        # Should contain model name
        assert 'Autoencoder' in graph_text or 'Component' in graph_text
    
    def test_component_graph_structure(self, autoencoder_model):
        """Test that component graph has expected structure."""
        graph_text = autoencoder_model.generate_component_graph(format='text')
        
        # Should mention encoder and decoder
        assert 'encoder' in graph_text.lower() or 'Encoder' in graph_text
        assert 'decoder' in graph_text.lower() or 'Decoder' in graph_text
    
    def test_get_component_statistics(self, autoencoder_model):
        """Test that component statistics can be retrieved."""
        stats = autoencoder_model._get_component_statistics()
        
        assert isinstance(stats, dict)
        # Should have stats for encoder and decoder
        assert len(stats) > 0


class TestAutoencoderCheckpointing:
    """Test checkpoint save/load structure (not correctness)."""
    
    def test_save_checkpoint_structure(self, autoencoder_model, temp_dir):
        """Test that checkpoint is saved with correct structure."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True)
        
        assert checkpoint_path.exists()
        
        # Load and check structure
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert "state_dict" in checkpoint
        assert "config" in checkpoint
        assert isinstance(checkpoint["state_dict"], dict)
        assert isinstance(checkpoint["config"], dict)
    
    def test_checkpoint_keys(self, autoencoder_model, temp_dir):
        """Test that checkpoint contains expected keys."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        
        # Should have encoder and decoder keys
        keys = list(state_dict.keys())
        encoder_keys = [k for k in keys if k.startswith("encoder.")]
        decoder_keys = [k for k in keys if k.startswith("decoder.")]
        
        assert len(encoder_keys) > 0
        assert len(decoder_keys) > 0
    
    def test_load_checkpoint_structure(self, autoencoder_model, temp_dir):
        """Test that checkpoint can be loaded (structure only)."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True)
        
        # Load checkpoint
        loaded_model = Autoencoder.load_checkpoint(checkpoint_path, map_location='cpu')
        
        assert loaded_model is not None
        assert isinstance(loaded_model, Autoencoder)
        assert hasattr(loaded_model, 'encoder')
        assert hasattr(loaded_model, 'decoder')
    
    def test_checkpoint_with_extra_state(self, autoencoder_model, temp_dir):
        """Test checkpoint with extra state (optimizer, epoch, etc.)."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        extra_state = {
            "epoch": 10,
            "best_val_loss": 0.5,
            "optimizer_state": {"dummy": "state"}
        }
        autoencoder_model.save_checkpoint(checkpoint_path, include_config=True, **extra_state)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert "epoch" in checkpoint
        assert "best_val_loss" in checkpoint
        assert "optimizer_state" in checkpoint

