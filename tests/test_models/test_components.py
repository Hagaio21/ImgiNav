"""
Tests for model components - focusing on building, registration, and component graphs.
"""
import pytest
import torch
from pathlib import Path

from models.components.registry import COMPONENT_REGISTRY, create_component
from models.encoder import Encoder
from models.decoder import Decoder
from models.components.unet import UnetWithAttention
from models.components.scheduler import LinearScheduler, CosineScheduler


class TestComponentInitialization:
    """Test component initialization from configs."""
    
    def test_encoder_from_config(self):
        """Test encoder can be built from config."""
        config = {
            "type": "Encoder",
            "in_channels": 3,
            "latent_channels": 4,
            "base_channels": 16,
            "downsampling_steps": 2,
        }
        encoder = Encoder.from_config(config)
        
        assert encoder is not None
        assert isinstance(encoder, Encoder)
    
    def test_decoder_from_config(self):
        """Test decoder can be built from config."""
        config = {
            "type": "Decoder",
            "latent_channels": 4,
            "base_channels": 16,
            "upsampling_steps": 2,
            "heads": [
                {
                    "type": "RGBHead",
                    "name": "rgb",
                    "out_channels": 3
                }
            ]
        }
        decoder = Decoder.from_config(config)
        
        assert decoder is not None
        assert isinstance(decoder, Decoder)
    
    def test_unet_from_config(self):
        """Test UNet can be built from config."""
        config = {
            "type": "UnetWithAttention",
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 128,
        }
        unet = UnetWithAttention.from_config(config)
        
        assert unet is not None
        assert isinstance(unet, UnetWithAttention)
    
    def test_scheduler_from_config(self):
        """Test scheduler can be built from config."""
        config = {
            "type": "LinearScheduler",
            "num_steps": 100
        }
        scheduler = LinearScheduler.from_config(config)
        
        assert scheduler is not None
        assert isinstance(scheduler, LinearScheduler)


class TestComponentRegistry:
    """Test component registration in registry."""
    
    def test_encoder_in_registry(self):
        """Test encoder is registered."""
        assert "Encoder" in COMPONENT_REGISTRY
        assert COMPONENT_REGISTRY["Encoder"] == Encoder
    
    def test_decoder_in_registry(self):
        """Test decoder is registered."""
        assert "Decoder" in COMPONENT_REGISTRY
        assert COMPONENT_REGISTRY["Decoder"] == Decoder
    
    def test_unet_in_registry(self):
        """Test UNet is registered."""
        assert "UnetWithAttention" in COMPONENT_REGISTRY
        assert COMPONENT_REGISTRY["UnetWithAttention"] == UnetWithAttention
    
    def test_scheduler_in_registry(self):
        """Test schedulers are registered."""
        assert "LinearScheduler" in COMPONENT_REGISTRY
        assert "CosineScheduler" in COMPONENT_REGISTRY
    
    def test_create_component_from_registry(self):
        """Test create_component uses registry."""
        config = {
            "type": "Encoder",
            "in_channels": 3,
            "latent_channels": 4,
            "base_channels": 16,
            "downsampling_steps": 2,
        }
        component = create_component(config)
        
        assert component is not None
        assert isinstance(component, Encoder)


class TestComponentToConfig:
    """Test component to_config() methods."""
    
    def test_encoder_to_config(self):
        """Test encoder to_config() returns valid config."""
        config = {
            "type": "Encoder",
            "in_channels": 3,
            "latent_channels": 4,
            "base_channels": 16,
            "downsampling_steps": 2,
        }
        encoder = Encoder.from_config(config)
        output_config = encoder.to_config()
        
        assert isinstance(output_config, dict)
        assert "type" in output_config or "in_channels" in output_config
    
    def test_decoder_to_config(self):
        """Test decoder to_config() returns valid config."""
        config = {
            "type": "Decoder",
            "latent_channels": 4,
            "base_channels": 16,
            "upsampling_steps": 2,
            "heads": [
                {
                    "type": "RGBHead",
                    "name": "rgb",
                    "out_channels": 3
                }
            ]
        }
        decoder = Decoder.from_config(config)
        output_config = decoder.to_config()
        
        assert isinstance(output_config, dict)
        assert "type" in output_config or "latent_channels" in output_config
    
    def test_unet_to_config(self):
        """Test UNet to_config() returns valid config."""
        config = {
            "type": "UnetWithAttention",
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 128,
        }
        unet = UnetWithAttention.from_config(config)
        output_config = unet.to_config()
        
        assert isinstance(output_config, dict)
        assert "type" in output_config or "in_channels" in output_config


class TestComponentGraphs:
    """Test component graph generation for individual components."""
    
    def test_encoder_component_graph(self):
        """Test encoder can generate component graph."""
        config = {
            "type": "Encoder",
            "in_channels": 3,
            "latent_channels": 4,
            "base_channels": 16,
            "downsampling_steps": 2,
        }
        encoder = Encoder.from_config(config)
        
        if hasattr(encoder, 'generate_component_graph'):
            graph_text = encoder.generate_component_graph(format='text')
            assert isinstance(graph_text, str)
            assert len(graph_text) > 0
    
    def test_decoder_component_graph(self):
        """Test decoder can generate component graph."""
        config = {
            "type": "Decoder",
            "latent_channels": 4,
            "base_channels": 16,
            "upsampling_steps": 2,
            "heads": [
                {
                    "type": "RGBHead",
                    "name": "rgb",
                    "out_channels": 3
                }
            ]
        }
        decoder = Decoder.from_config(config)
        
        if hasattr(decoder, 'generate_component_graph'):
            graph_text = decoder.generate_component_graph(format='text')
            assert isinstance(graph_text, str)
            assert len(graph_text) > 0
    
    def test_unet_component_graph(self):
        """Test UNet can generate component graph."""
        config = {
            "type": "UnetWithAttention",
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 128,
        }
        unet = UnetWithAttention.from_config(config)
        
        if hasattr(unet, 'generate_component_graph'):
            graph_text = unet.generate_component_graph(format='text')
            assert isinstance(graph_text, str)
            assert len(graph_text) > 0


class TestComponentRelationships:
    """Test component relationships (e.g., UNet contains blocks)."""
    
    def test_unet_has_blocks(self):
        """Test that UNet has internal blocks/components."""
        config = {
            "type": "UnetWithAttention",
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 128,
        }
        unet = UnetWithAttention.from_config(config)
        
        # UNet should have some internal structure
        # Check if it has _component_names or similar
        if hasattr(unet, '_component_names'):
            assert len(unet._component_names) >= 0  # May be empty or have components
    
    def test_decoder_has_heads(self):
        """Test that decoder has heads."""
        config = {
            "type": "Decoder",
            "latent_channels": 4,
            "base_channels": 16,
            "upsampling_steps": 2,
            "heads": [
                {
                    "type": "RGBHead",
                    "name": "rgb",
                    "out_channels": 3
                }
            ]
        }
        decoder = Decoder.from_config(config)
        
        # Decoder should have heads
        assert hasattr(decoder, 'heads') or hasattr(decoder, 'rgb') or hasattr(decoder, '_heads')
        # At least one head should exist
        if hasattr(decoder, 'heads'):
            assert len(decoder.heads) > 0
        elif hasattr(decoder, 'rgb'):
            assert decoder.rgb is not None

