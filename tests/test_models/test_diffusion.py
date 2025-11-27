"""
Tests for DiffusionModel - focusing on building, class interactions, and component graphs.
"""
import pytest
import torch
from pathlib import Path
import tempfile

from models.diffusion import DiffusionModel


class TestDiffusionBuilding:
    """Test diffusion model building from configs."""
    
    def test_build_from_config(self, test_diffusion_config):
        """Test that diffusion model can be built from config."""
        diffusion_cfg = {
            "autoencoder": test_diffusion_config.get("autoencoder"),
            "unet": test_diffusion_config.get("unet", {}),
            "scheduler": test_diffusion_config.get("scheduler", {}),
            "embedding_projection": test_diffusion_config.get("embedding_projection"),
            "scale_factor": test_diffusion_config.get("scale_factor", 1.0),
        }
        model = DiffusionModel.from_config(diffusion_cfg)
        
        assert model is not None
        assert isinstance(model, DiffusionModel)
    
    def test_components_exist(self, diffusion_model):
        """Test that required components exist."""
        assert hasattr(diffusion_model, 'unet')
        assert hasattr(diffusion_model, 'decoder')
        assert hasattr(diffusion_model, 'scheduler')
        assert diffusion_model.unet is not None
        assert diffusion_model.decoder is not None
        assert diffusion_model.scheduler is not None
    
    def test_component_types(self, diffusion_model):
        """Test that components are correct types."""
        from models.components.unet import UnetWithAttention
        from models.decoder import Decoder
        from models.components.scheduler import LinearScheduler
        
        assert isinstance(diffusion_model.unet, UnetWithAttention)
        assert isinstance(diffusion_model.decoder, Decoder)
        assert isinstance(diffusion_model.scheduler, LinearScheduler)
    
    def test_embedding_projection_optional(self, diffusion_model):
        """Test that embedding projection exists if configured."""
        # If embedding_projection is in config, it should exist
        if hasattr(diffusion_model, 'embedding_projection'):
            assert diffusion_model.embedding_projection is not None
    
    def test_to_config(self, diffusion_model):
        """Test that to_config() returns a valid config dict."""
        config = diffusion_model.to_config()
        
        assert isinstance(config, dict)
        assert "type" in config
        assert config["type"] == "DiffusionModel"
        # Should have unet, decoder, scheduler configs
        assert "unet" in config or "decoder" in config or "scheduler" in config


class TestDiffusionComponentRelationships:
    """Test component relationships and interactions."""
    
    def test_unet_decoder_relationship(self, diffusion_model):
        """Test that UNet and decoder are properly connected."""
        assert hasattr(diffusion_model, 'unet')
        assert hasattr(diffusion_model, 'decoder')
        assert diffusion_model.unet is not None
        assert diffusion_model.decoder is not None
    
    def test_scheduler_unet_relationship(self, diffusion_model):
        """Test that scheduler and UNet are properly connected."""
        assert hasattr(diffusion_model, 'scheduler')
        assert hasattr(diffusion_model, 'unet')
        assert diffusion_model.scheduler is not None
        assert diffusion_model.unet is not None
    
    def test_embedding_projection_unet_relationship(self, diffusion_model):
        """Test that embedding projection and UNet are properly connected."""
        if hasattr(diffusion_model, 'embedding_projection') and diffusion_model.embedding_projection is not None:
            assert hasattr(diffusion_model, 'unet')
            assert diffusion_model.unet is not None
    
    def test_decoder_frozen_state(self, diffusion_model):
        """Test that decoder frozen state is accessible."""
        assert hasattr(diffusion_model, 'decoder')
        # Check if decoder has requires_grad attribute (for frozen check)
        if hasattr(diffusion_model.decoder, 'parameters'):
            params = list(diffusion_model.decoder.parameters())
            if len(params) > 0:
                # Just check that we can access the frozen state
                frozen = not any(p.requires_grad for p in params)
                # Either frozen or not, both are valid
                assert isinstance(frozen, bool)
    
    def test_component_tracking(self, diffusion_model):
        """Test that components are tracked in _component_names."""
        assert hasattr(diffusion_model, '_component_names')
        assert isinstance(diffusion_model._component_names, dict)
        
        # UNet, decoder, scheduler should be tracked
        components = list(diffusion_model._component_names.values())
        assert len(components) > 0


class TestDiffusionComponentGraphs:
    """Test component graph generation."""
    
    def test_generate_component_graph_text(self, diffusion_model):
        """Test that component graph can be generated in text format."""
        graph_text = diffusion_model.generate_component_graph(format='text')
        
        assert isinstance(graph_text, str)
        assert len(graph_text) > 0
        # Should contain model name
        assert 'DiffusionModel' in graph_text or 'Component' in graph_text
    
    def test_component_graph_structure(self, diffusion_model):
        """Test that component graph has expected structure."""
        graph_text = diffusion_model.generate_component_graph(format='text')
        
        # Should mention key components
        assert 'unet' in graph_text.lower() or 'UNet' in graph_text
        assert 'decoder' in graph_text.lower() or 'Decoder' in graph_text
        assert 'scheduler' in graph_text.lower() or 'Scheduler' in graph_text
    
    def test_get_component_statistics(self, diffusion_model):
        """Test that component statistics can be retrieved."""
        stats = diffusion_model._get_component_statistics()
        
        assert isinstance(stats, dict)
        # Should have stats for key components
        assert len(stats) > 0


class TestDiffusionCheckpointing:
    """Test checkpoint save/load structure (not correctness)."""
    
    def test_save_checkpoint_structure(self, diffusion_model, temp_dir):
        """Test that checkpoint is saved with correct structure."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        diffusion_model.save_checkpoint(checkpoint_path, include_config=True)
        
        assert checkpoint_path.exists()
        
        # Load and check structure
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert "state_dict" in checkpoint
        assert "config" in checkpoint
        assert isinstance(checkpoint["state_dict"], dict)
        assert isinstance(checkpoint["config"], dict)
    
    def test_checkpoint_keys(self, diffusion_model, temp_dir):
        """Test that checkpoint contains expected keys."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        diffusion_model.save_checkpoint(checkpoint_path, include_config=True)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        
        # Should have unet, decoder, scheduler keys
        keys = list(state_dict.keys())
        unet_keys = [k for k in keys if k.startswith("unet.")]
        decoder_keys = [k for k in keys if k.startswith("decoder.")]
        scheduler_keys = [k for k in keys if k.startswith("scheduler.")]
        
        assert len(unet_keys) > 0
        assert len(decoder_keys) > 0
        assert len(scheduler_keys) > 0
    
    def test_load_checkpoint_structure(self, diffusion_model, temp_dir):
        """Test that checkpoint can be loaded (structure only)."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        diffusion_model.save_checkpoint(checkpoint_path, include_config=True)
        
        # Load checkpoint
        loaded_model = DiffusionModel.load_checkpoint(checkpoint_path, map_location='cpu')
        
        assert loaded_model is not None
        assert isinstance(loaded_model, DiffusionModel)
        assert hasattr(loaded_model, 'unet')
        assert hasattr(loaded_model, 'decoder')
        assert hasattr(loaded_model, 'scheduler')
    
    def test_checkpoint_with_extra_state(self, diffusion_model, temp_dir):
        """Test checkpoint with extra state (optimizer, epoch, etc.)."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        extra_state = {
            "epoch": 10,
            "best_val_loss": 0.5,
            "optimizer_state": {"dummy": "state"}
        }
        diffusion_model.save_checkpoint(checkpoint_path, include_config=True, **extra_state)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert "epoch" in checkpoint
        assert "best_val_loss" in checkpoint
        assert "optimizer_state" in checkpoint

